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

    for pname, pnum in list(path_to_pairs.keys()):
        if pname in processed_pathways:
            del path_to_pairs[(pname, pnum)]
    print('%d (pathway_name, pathway_num) left to process' \
          % (len(path_to_pairs)))   
    pathname_to_pairs = {}
    for (pname, pnum), edges in path_to_pairs.items():
        if pname not in pathname_to_pairs:
            pathname_to_pairs[pname] = {}
        pathname_to_pairs[pname][pnum] = set([])
        for u, v in edges:
            if u in nodelistMap and v in nodelistMap and u != v:
                u, v = min(u, v), max(u, v)
                pathname_to_pairs[pname][pnum].add(
                        (nodelistMap[u], nodelistMap[v]))
        pathname_to_pairs[pname][pnum] = list(pathname_to_pairs[pname][pnum])
        if len(pathname_to_pairs[pname][pnum]) == 0:
            print('Removing', pname, pnum, 'due to no test edges')
            del pathname_to_pairs[pname][pnum]

    for pname, pnum_to_pairs in pathname_to_pairs.items():
        print('Testing pathway', pname)
        test_edges = []
        for pnum, edges in pnum_to_pairs.items():
            test_edges.extend(edges)
        test_edges = list(set(test_edges))
        
        pnum_to_test_edge_idx = {}
        for pnum, edges in pnum_to_pairs.items():
            pnum_to_test_edge_idx[pnum] = []
            for e in edges:
                pnum_to_test_edge_idx[pnum].append(test_edges.index(e))

        test_edges = np.array(test_edges)
        print('len(test_edges)', len(test_edges))
        if len(test_edges) < 1:
            print('Too few test edges viable for testing')
            continue
            
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
        if len(neg_edges) > len(test_edges):
            rand_idx = np.random.choice(
                np.arange(len(neg_edges)), len(test_edges), replace=False)
            neg_edges = neg_edges[rand_idx]

        print('Number of neg edges:', len(neg_edges)) 
        yield {'train_G': train_G, 
               'test_edges': test_edges,
               'pnum_to_test_edge_idx': pnum_to_test_edge_idx, 
               'neg_edges': neg_edges, 
               'pathway_name': pname} 
def evaluatePR(model, pnum_to_test_edge_idx=None, test_edges=None, 
               neg_edges=None, pathway_name=None, **kwargs):
    
    yscore_true = model.get_edge_scores(test_edges, use_logistic=True)
    yscore_false = model.get_edge_scores(neg_edges, use_logistic=True) 
    neg_edge_arange = np.arange(len(neg_edges))
    res = ''
    for pnum, test_edge_idx in pnum_to_test_edge_idx.items(): 
        print(pathway_name, pnum)
        print('len(test_edges)', len(test_edge_idx))
        neg_edge_idx = np.random.choice(neg_edge_arange, len(test_edge_idx), replace=False)
        edges = np.concatenate((test_edges[test_edge_idx], neg_edges[neg_edge_idx]))
        ytrue = np.concatenate((np.ones(len(test_edge_idx)), np.zeros(len(test_edge_idx))))
        yscore = np.concatenate((yscore_true[test_edge_idx], yscore_false[neg_edge_idx]))
        AUC = roc_auc_score(ytrue, yscore)
        prec_curve = precision_at_ks(edges, yscore, test_edges[test_edge_idx])
        prec_curve = ';'.join(['%s-%s' % (str(x), str(y)) for (x, y) in prec_curve])
        map_ = mean_average_precision(edges, yscore, test_edges)
        print('AUC: %.2f' % AUC)
        print('Precision curve', prec_curve)
        print('MAP', map_)
        res += '%s,%s,%d,%f,%s,%s\n' \
               % (pathway_name, pnum, len(test_edge_idx), AUC, prec_curve, map_)
    return res

def experimentPathwayReconstruction(G, model, res_prefix=None, pathway_map=None, 
                                    random_seed=None, **params):
    resfile = 'logs/%s.results.txt' % res_prefix
    print('\nPathway reconstruction experiments')
    print('Writing results to', resfile)
    if random_seed:
        np.random.seed(random_seed)
    if pathway_map is None:
        pathway_map = '%s/kegg/pathway_map.pkl' % os.environ['DATAPATH']
    with open(pathway_map, 'rb') as f:
        path_to_pairs = pickle.load(f)
    processed_pathways = set([])
    if not os.path.exists(resfile):
        with open(resfile, 'w') as f:
            f.write('pathway_name,pathway_num,n_test_edges,AUC,precision_at_ks,MAP\n')
    else:
        with open(resfile) as f:
            f.readline()
            for line in f:
                pname = line.split(',')[0]
                processed_pathways.add(pname)
        print('Processed pathways:', processed_pathways)
    for Gset in graph_train_test_split_pathway(G, path_to_pairs, processed_pathways):
        model.learn_embedding(G=Gset['train_G'])
        print('Run complete')
        res = evaluatePR(model, **Gset)
        with open(resfile, 'a') as f:
            f.write(res)
