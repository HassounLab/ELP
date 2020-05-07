import sys
print("python version:", sys.version)
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

def graph_train_test_split_pathway(G, pname_to_pairs):
    nodelistMap = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, nodelistMap)
    print('Removing %d edges because of self loops' % nx.number_of_selfloops(G))
    self_loop_edges = nx.selfloop_edges(G)
    G.remove_edges_from(self_loop_edges)
    molecule_names = {i: n for n, i in nodelistMap.items()}
    nx.set_node_attributes(G, values=molecule_names, name='molecule')
    fingerprints = nx.get_node_attributes(G, name='fingerprint')    
    neg_G = nx.complement(G)

    for pathway, test_edges in pname_to_pairs.items():
        print('Testing pathway', pathway)
        print('Number of test edges', len(test_edges))
        
        train_G = G.copy()
        train_G.remove_edges_from(test_edges)
        train_G.name = "train_G"
        test_nodes = set([u for u, _ in test_edges]) | \
                     set([v for _, v in test_edges])
        
        print('%s edges between test nodes are also in train_G'\
              % (train_G.subgraph(test_nodes).number_of_edges())) 

        print(nx.info(train_G))
        neg_edges = neg_G.edges(test_nodes) 
        print('Number of neg edges:', len(neg_edges)) 
        yield {'train_G': train_G, 'test_edges': test_edges,
               'neg_edges': neg_edges, 'pathway': pathway}


def experimentPathwayReconstruction(G, model, resfile, pathway_map=None, 
                                    random_seed=None,  **params):
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
    
    for Gset in graph_train_test_split_pathway(G, pname_to_pairs):
        train_G, test_edges = Gset['train_G'], Gset['test_edges']
        neg_edges = Gset['neg_edges']
        model.learn_embedding(G=train_G)
    
        edges = np.concatenate((test_edges, neg_edges))
        ytrue = np.concatenate((np.ones(len(test_edges)), np.zeros(len(neg_edges))))
        yscore = model.get_edge_weights(edges, use_logistic=True)
        AUC = roc_auc_score(ytrue, yscore)
        prec_curve = precision_at_ks(edges, ytrue, test_edges)
        map_ = mean_average_precision(edges, ytrue, test_edges)
        print('AUC: %.2f', AUC)
        print('Precision curve', prec_curve)
        print('MAP', map_)
        with open(resfile, 'a') as f:
            f.write('%s,%f,%s\n' 
                    % (pathway, AUC, ';'.join([str(x) for x in prec_curve])))

