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
from evaluation.metrics import precision_at_ks, mean_average_precision

def graph_train_test_split_pathway(G, path_to_pairs, processed_pathways=None):
    nodelistMap = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, nodelistMap)
    print(nx.info(G))

    molecule_names = {i: n for n, i in nodelistMap.items()}
    nx.set_node_attributes(G, values=molecule_names, name='molecule')
    fingerprints = nx.get_node_attributes(G, name='fingerprint')    
    neg_G = nx.complement(G)
    neg_G.name = 'neg_G'
    print(nx.info(neg_G))


    for pathway, test_edges in path_to_pairs.items():
        if pathway in processed_pathways:
            continue
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
        neg_edges = np.array(list(neg_G.subgraph(test_nodes).edges()))
        print('Number of neg edges:', len(neg_edges)) 
        yield {'train_G': train_G, 
               'test_edges': test_edges, 
               'neg_edges': neg_edges, 
               'pathway_name': pathway[0], 
               'pathway_num': pathway[1]}


def experimentPathwayReconstruction(G, model, resfile, pathway_map=None, 
                                    random_seed=None, **params):
    print('\nPathway reconstruction experiments')
    print('Writing results to', resfile)
    if random_seed:
        np.random.seed(random_seed)
    if pathway_map is None:
        pathway_map = '%s/kegg/pathway_map.pkl' % os.environ['DATAPATH']
    with open(pathway_map, 'rb') as f:
        path_to_pairs = pickle.load(f)
    if not os.path.exists(resfile):
        with open(resfile, 'w') as f:
            f.write('pathway_name,pathway_num,AUC,precision_curve\n')
    else:
        processed_pathways = set([])
        with open(resfile) as f:
            f.readline()
            for line in f:
                pname, pnum = line.split(',')[:2]
                processed_pathways.add((pname, pnum))
        print('Processed pathways:', processed_pathways)
    for Gset in graph_train_test_split_pathway(G, path_to_pairs, processed_pathways):
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
            f.write('%s,%s,%f,%s\n' 
                    % (Gset['pathway_name'], 
                       Gset['pathway_num'], 
                       AUC, 
                       ';'.join([str(x) for x in prec_curve])))

