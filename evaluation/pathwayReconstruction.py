import sys
import networkx as nx
import numpy as np
import os
from operator import itemgetter
import copy
import pickle
import time
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import logistic
from sklearn.model_selection import train_test_split, KFold
from evaluation.eval_util import sample_graph
from evaluation.metrics import precision_at_ks, mean_average_precision

PATHWAYS = [
        'Carbohydrate metabolism',
        'Energy metabolism',
        'Lipid metabolism',
        'Nucleotide metabolism',
        'Amino acid metabolism',
        'Metabolism of other amino acids',
        'Glycan biosynthesis and metabolism',
        'Metabolism of cofactors and vitamins',
        'Metabolism of terpenoids and polyketides',
        'Biosynthesis of other secondary metabolites',
        'Xenobiotics biodegradation and metabolism']

def graph_train_test_split_pathway(G, pname_to_pairs, pathways=None):
    nodelistMap = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, nodelistMap)
    print(nx.info(G))

    molecule_names = {i: n for n, i in nodelistMap.items()}
    nx.set_node_attributes(G, values=molecule_names, name='molecule')
    fingerprints = nx.get_node_attributes(G, name='fingerprint')    
    neg_G = nx.complement(G)
    neg_G.name = 'neg_G'
    print(nx.info(neg_G))
    
    if pathways is not None:
        pname_to_pairs = {p: e for p, e in pname_to_pairs.items() if p in pathways}


    for pathway, test_edges in pname_to_pairs.items():
        print('Testing pathway', pathway)
        test_edges_filtered = []
        for u, v in test_edges:
            if u in nodelistMap and v in nodelistMap and u != v:
                test_edges_filtered.append((nodelistMap[u], nodelistMap[v]))
        print('len(test_edges) before filtering', len(test_edges))
        print('len(test_edges) after filtering', len(test_edges_filtered))
        test_edges = test_edges_filtered
            
        train_G = G.copy()
        train_G.remove_edges_from(test_edges)
        train_G.name = "train_G"
        
        test_nodes = set([u for u, _ in test_edges]) | \
                     set([v for _, v in test_edges])
        print('Number of test nodes', len(test_nodes))
        print('%s edges between test nodes are also in train_G'\
              % (train_G.subgraph(test_nodes).number_of_edges())) 
        print(nx.info(train_G))
        neg_edges = neg_G.subgraph(test_nodes).edges()
        print('Number of neg edges:', len(neg_edges)) 
        yield {'train_G': train_G, 'test_edges': test_edges,
               'neg_edges': neg_edges, 'pathway': pathway}


def experimentPathwayReconstruction(G, model, resfile, pathway_map=None, 
                                    random_seed=None, **params):
    print('\nPathway reconstruction experiments')
    if random_seed:
        np.random.seed(random_seed)
    if pathway_map is None:
        pathway_map = '%s/kegg/pathway_map.pkl' % os.environ['DATAPATH']
    with open(pathway_map, 'rb') as f:
        pname_to_pairs = pickle.load(f)['pname_to_pairs']
    if not os.path.exists(resfile):
        with open(resfile, 'w') as f:
            f.write('pathway,AUC,precision_curve\n')
    else:
        processed_pathways = []
        with open(resfile) as f:
            f.readline()
            for line in f:
                processed_pathways.append(line.split(',')[0])
        pathways = [p for p in PATHWAYS if p not in processed_pathways]
        print('pathways left to process', pathways)
    for Gset in graph_train_test_split_pathway(G, pname_to_pairs, pathways):
        train_G, test_edges = Gset['train_G'], Gset['test_edges']
        neg_edges = Gset['neg_edges']
        model.learn_embedding(G=train_G)
    
        edges = np.concatenate((test_edges, neg_edges))
        ytrue = np.concatenate((np.ones(len(test_edges)), np.zeros(len(neg_edges))))
        yscore = model.get_edge_scores(edges, use_logistic=True)
        AUC = roc_auc_score(ytrue, yscore)
        prec_curve = precision_at_ks(edges, ytrue, test_edges)
        map_ = mean_average_precision(edges, ytrue, test_edges)
        print('AUC: %.2f' % AUC)
        print('Precision curve', prec_curve)
        print('MAP', map_)
        with open(resfile, 'a') as f:
            f.write('%s,%f,%s\n' 
                    % (Gset['pathway'], AUC, ';'.join([str(x) for x in prec_curve])))

