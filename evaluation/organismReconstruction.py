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

def graph_train_test_split_organism(G, organism_to_pairs, n_samples=1024, n_runs=5):
    nodelistMap = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, nodelistMap)
    print(nx.info(G))

    molecule_names = {i: n for n, i in nodelistMap.items()}
    nx.set_node_attributes(G, values=molecule_names, name='molecule')
    fingerprints = nx.get_node_attributes(G, name='fingerprint')    
    neg_G = nx.complement(G)
    neg_G.name = 'neg_G'
    print(nx.info(neg_G))
    for phylum, test_edges in organism_to_pairs.items():
   
        print('Testing phylum', phylum)

        test_edges = np.array([(nodelistMap[u], nodelistMap[v]) \
                               for u, v in test_edges])
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
        if len(neg_edges) < 1:
            print('Too few neg edges viable for testing')
            continue
        for i in range(n_runs):
            samp_idx = np.random.choice(np.arange(len(test_edges)), n_samples, replace=False)
            test_edges_samp = test_edges[samp_idx]   
            print('len(test_edges)', len(test_edges))
            
            """    
            train_G = G.copy()
            train_G.remove_edges_from(test_edges_samp)
            train_G.name = "train_G"
            test_nodes = set([u for u, _ in test_edges_samp]) | \
                         set([v for _, v in test_edges_samp])
            print('Number of test nodes', len(test_nodes))
            print('%s edges between test nodes are also in train_G'\
                  % (train_G.subgraph(test_nodes).number_of_edges())) 
            print(nx.info(train_G))
            neg_edges = np.array(list(neg_G.subgraph(test_nodes).edges()))
            if len(neg_edges) < 1:
                print('Too few neg edges viable for testing')
                continue
            """
   
            rand_idx = np.random.choice(
                np.arange(len(neg_edges)), len(test_edges), replace=False)
            neg_edges_samp = neg_edges[rand_idx]

            print('Number of neg edges:', len(neg_edges)) 
            yield {'train_G': train_G, 
                   'test_edges': test_edges_samp,
                   'neg_edges': neg_edges_samp, 
                   'phylum': phylum}
def evaluatePR(model, test_edges=None, 
               neg_edges=None, phylum=None, **kwargs):
    
    yscore_true = model.get_edge_scores(test_edges, use_logistic=True)
    yscore_false = model.get_edge_scores(neg_edges, use_logistic=True) 
    neg_edge_arange = np.arange(len(neg_edges))
    ytrue = np.concatenate((np.ones(len(test_edges)), np.zeros(len(neg_edges))))
    yscore = np.concatenate((yscore_true, yscore_false))
    AUC = roc_auc_score(ytrue, yscore)
    print('AUC: %.2f' % AUC)
    res = '%s,%d,%f\n' \
           % (phylum, len(test_edges), AUC)
    return res

def experimentOrganismReconstruction(G, model, res_prefix=None, organism_map=None, 
                                    random_seed=None, **params):
    resfile = 'logs/%s.results.txt' % res_prefix
    print('\nOrganism reconstruction experiments')
    print('Writing results to', resfile)
    if random_seed:
        np.random.seed(random_seed)
    if organism_map is None:
        organism_map = '%s/kegg/phylum_to_edges.pkl' % os.environ['DATAPATH']
    with open(organism_map, 'rb') as f:
        organism_to_pairs = pickle.load(f)
    if not os.path.exists(resfile):
        with open(resfile, 'w') as f:
            f.write('phylum,n_test_edges,AUC\n')
    for Gset in graph_train_test_split_organism(G, organism_to_pairs):
        model.learn_embedding(G=Gset['train_G'])
        print('Run complete')
        res = evaluatePR(model, **Gset)
        with open(resfile, 'a') as f:
            f.write(res)
