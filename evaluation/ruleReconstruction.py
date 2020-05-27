import sys
import networkx as nx
import numpy as np
import os
from operator import itemgetter
import copy
import pickle
import pandas as pd
import time
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import logistic
from sklearn.model_selection import train_test_split, KFold
from evaluation.metrics import precision_at_ks, mean_average_precision
def graph_train_test_split_rules(G, test_rules, test_edges_list):
    nodelistMap = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, nodelistMap)
    print(nx.info(G))

    molecule_names = {i: n for n, i in nodelistMap.items()}
    nx.set_node_attributes(G, values=molecule_names, name='molecule')
    fingerprints = nx.get_node_attributes(G, name='fingerprint')
    neg_G = nx.complement(G)
    neg_G.name = 'neg_G'
    print(nx.info(neg_G))
    
    for rule, test_edges in zip(test_rules, test_edges_list):
        print('Running test for rule', rule)
        print('len(test_edges) associated with rule', len(test_edges))
        test_edges_filtered = []
        for u, v in test_edges:
            if u in nodelistMap and v in nodelistMap and u != v:
                test_edges_filtered.append((nodelistMap[u], nodelistMap[v]))
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
        if len(neg_edges) < 1:
            print('Too few neg edges viable for testing')
            continue
        if len(neg_edges) > len(test_edges):
            rand_idx = np.random.choice(
                    np.arange(len(neg_edges)), len(test_edges), replace=False)
            neg_edges = neg_edges[rand_idx]
        yield {'train_G': train_G, 
               'test_edges': test_edges, 
               'neg_edges': neg_edges, 
               'rule_count': len(test_edges), 
               'rule_num': rule}

def sample_test_rules(rclass_to_edges, rule_prevalences, n_sample, 
                      test_rules_file, random_seed):
    if rule_prevalences is None:
        rule_prevalences = [0.4, 0.6]

    rclass_counts = [(rc, len(edges)) for rc, edges in rclass_to_edges.items()]
    rclass = pd.DataFrame(rclass_counts, columns=['rclass', 'freq'])
    rclass = rclass.sort_values('freq', ascending=False).reset_index(drop=True)
    rclass['cumsum'] = rclass['freq'].cumsum()
    with open(test_rules_file, 'w') as f:
        print('top rules')
        print(rclass[:n_sample])
        for r in rclass['rclass'].values[:n_sample]:
            f.write(r)
            f.write('\n')
        for rule_prevalence in rule_prevalences:
            mincut = rclass['freq'].sum() * (1 - rule_prevalence - 0.1)
            maxcut = rclass['freq'].sum() * (1 - rule_prevalence)
            rules = rclass[(rclass['cumsum'] >= mincut) & (rclass['cumsum'] <= maxcut)]
            if len(rules) > n_sample:
                rules = rules.sample(n=n_sample, replace=False, 
                                     random_state=random_seed)
                print('sampled rules')
            print(rules)
            for r in rules['rclass'].values:
                f.write(r)
                f.write('\n')




def experimentRuleReconstruction(G, model, res_prefix=None, test_rules_file='./test_rules.txt', 
                                 rclass_to_edges_file=None, random_seed=None, 
                                 rule_prevalences=None, n_sample=5, **params):
    resfile = 'logs/%s.results.txt' % res_prefix
    print('\nRule reconstruction experiments')
    print('Writing results to', resfile)
    if random_seed:
        np.random.seed(random_seed)
    if rclass_to_edges_file is None:
       rclass_to_edges_file = '%s/kegg/rclass_to_edges.pkl' % os.environ['DATAPATH']
        
    with open(rclass_to_edges_file, 'rb') as f:
        rclass_to_edges = pickle.load(f)
    
    if not os.path.exists(test_rules_file):
        sample_test_rules(rclass_to_edges, rule_prevalences, n_sample, 
                          test_rules_file, random_seed)
    with open(test_rules_file) as f:
        test_rules = [r[:-1] for r in f.readlines()]
    
    
    if os.path.exists(resfile):
        with open(resfile) as f:
            f.readline()
            for line in f:
                rule = line.split(',')[0]
                if rule in test_rules:
                    test_rules.remove(rule)
    else:
        with open(resfile, 'w') as f:
            f.write('rule_num,rule_count,AUC,precision_curve,MAP\n')
    test_edges_list = [rclass_to_edges[r] for r in test_rules]
    for Gset in graph_train_test_split_rules(G, test_rules, test_edges_list):
        train_G, test_edges = Gset['train_G'], Gset['test_edges']
        neg_edges = Gset['neg_edges']
        model.learn_embedding(G=train_G)
        edges = np.concatenate((test_edges, neg_edges))
        ytrue = np.concatenate((np.ones(len(test_edges)), np.zeros(len(neg_edges))))
        yscore = model.get_edge_scores(edges, use_logistic=True)
        AUC = roc_auc_score(ytrue, yscore)
        prec_curve = precision_at_ks(edges, yscore, test_edges)
        map_ = mean_average_precision(edges, yscore, test_edges)
        print('AUC: %.2f' % AUC)
        print('Precision curve', prec_curve)
        print('MAP', map_)
        with open(resfile, 'a') as f:
            f.write('%s,%s,%f,%s,%f\n' 
                    % (Gset['rule_num'], 
                       Gset['rule_count'], 
                       AUC, 
                       ';'.join(['%s-%s' % (str(x), str(y)) for x,y in prec_curve]),
                       map_))

