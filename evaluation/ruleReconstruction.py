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
def graph_train_test_split_rules(G, df, rule_prevalences, n_sample=5):
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
    for rule_prevalence in rule_prevalences:
        mincut = rc_vc['freq'].sum() * rule_prevalence
        maxcut = rc_vc['freq'].sum() * (rule_prevalence + 0.1)
        rules = rc_vc[(rc_vc['cumsum'] >= mincut) & (rc_vc['cumsum'] <= maxcut)]
        rules = rules.reset_index(drop=True)
        print('Rules available in percentile %.1f:' % rule_prevalence)
        print(rules)
        if len(rules) > n_sample:
            locs = np.random.choice(np.arange(len(rules)), size=n_sample, replace=False)
            rules = rules.loc[locs]
            print('Sampled rules')
            print(rules)
        else:
            print('too few rules - not sampling')
        for _, rule_count, rule in rules[['freq', 'rclass']].iterrows():
            print('Running test for rule', rule)
            test_edges = [(u, v) for u, v in \
                           df[df['rclass'] == rule][['reactants', 'products']].values]
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
            yield {'train_G': train_G, 
                   'test_edges': test_edges, 
                   'neg_edges': neg_edges, 
                   'rule_count': rule_count, 
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
    if not os.path.exists(resfile):
        with open(resfile, 'w') as f:
            f.write('pathway_name,pathway_num,AUC,precision_curve\n')
    for Gset in graph_train_test_split_rules(G, df, rule_prevalences):
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
                    % (Gset['rule_num'], 
                       Gset['rule_count'], 
                       AUC, 
                       ';'.join([str(x) for x in prec_curve])))

