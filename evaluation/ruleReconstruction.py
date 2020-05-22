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
def graph_train_test_split_rules(G, df, rule_prevalences, processed_rules=None,
                                 n_sample=5, test_rules=None):
    nodelistMap = {n: i for i, n in enumerate(G.nodes())}
    G = nx.relabel_nodes(G, nodelistMap)
    print(nx.info(G))

    molecule_names = {i: n for n, i in nodelistMap.items()}
    nx.set_node_attributes(G, values=molecule_names, name='molecule')
    fingerprints = nx.get_node_attributes(G, name='fingerprint')
    neg_G = nx.complement(G)
    neg_G.name = 'neg_G'
    print(nx.info(neg_G))

    df = df.dropna()
    rc_vc = df['rclass'].value_counts().to_frame().reset_index()
    rc_vc.columns = ['rclass', 'freq']
    rc_vc['cumsum'] = rc_vc['freq'].cumsum()
    if test_rules is None:
        test_rules = []
        for rule_prevalence in rule_prevalences:
            mincut = rc_vc['freq'].sum() * (1 - rule_prevalence - 0.1)
            maxcut = rc_vc['freq'].sum() * (1 - rule_prevalence)
            rules = rc_vc[(rc_vc['cumsum'] >= mincut) & (rc_vc['cumsum'] <= maxcut)]
            rules = rules.reset_index(drop=True)
            print('Rules available in percentile %.1f:' % rule_prevalence)
            print(rules)
            rules = rules.sample(frac=1)
            i = 0
            for rule in rules['rclass'].values:
                if i == n_sample:
                    break
                if rule in processed_rules:
                    print('rule', rule, 'already processed, skipping')
                    i += 1
                    continue
                test_rules.append(rule)
    for rule in test_rules:
        print('Running test for rule', rule)
        test_edges = [(u, v) for u, v in \
                       df[df['rclass'] == rule][['reactant', 'product']].values]
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
        i += 1
        yield {'train_G': train_G, 
               'test_edges': test_edges, 
               'neg_edges': neg_edges, 
               'rule_count': len(test_edges), 
               'rule_num': rule}


def experimentRuleReconstruction(G, model, resfile, reaction_stats=None,                   
                                 random_seed=None, rule_prevalences=None, **params):
    print('\nRule reconstruction experiments')
    print('Writing results to', resfile)
    if random_seed:
        np.random.seed(random_seed)
    if reaction_stats is None:
        reaction_stats = '%s/kegg/kegg_reactions.csv' % os.environ['DATAPATH']
    df = pd.read_csv(reaction_stats)

    if rule_prevalences is None:
        rule_prevalences = [0.9, 0.4, 0]
    processed_rules = set([])
    
    if os.path.exists(resfile):
        with open(resfile) as f:
            f.readline()
            for line in f:
                rule = line.split(',')[0]
                processed_rules.add(rule)
    else:
        with open(resfile, 'w') as f:
            f.write('rule_num,rule_count,AUC,precision_curve,MAP\n')
    if os.path.exists('../test_rules.txt'):
        with open(resfile) as f:
            test_rules = [r[:-1] for r in f.readlines()]
        print('rules to run:', test_rules)
    else:
        test_rules = None
    for Gset in graph_train_test_split_rules(G, df, rule_prevalences, 
            processed_rules, test_rules):
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

