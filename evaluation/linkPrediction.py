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

def print_graph_stats(G):
    print("%s has %d nodes, %d edges" %
          (G.name, G.number_of_nodes(), G.number_of_edges() // 2))

    def report_edge_attr(edge_name):
        attr = nx.get_edge_attributes(G, edge_name)
        if attr != {}:
            labels = []
            for k, v in attr.items():
                if v is not None:
                    labels.append(v)
            print("%d edges that have edge attribute %s" % 
                  (len(labels) // 2, edge_name))
            print("%d unique labels for %s" % 
                  (max(labels) + 1, edge_name))
    report_edge_attr("eclass_int")
    report_edge_attr("rclass_int")

def graph_test_train_split_inductive(G, test_ratio, remove_node_ratios=0.05, 
                                     undirected=True):
    train_G, test_G = G.copy(), nx.DiGraph()
    train_G.name, test_G.name = "train_G", "test_G"
   
    untrain_nodes = []
    remove_edges = set([]) 
    for n in train_G.nodes:
        if np.random.uniform() <= remove_node_ratios:
            for st, ed in train_G.edges(n):
                remove_edges.add((st, ed))
                if undirected:
                    remove_edges.add((ed, st))
            untrain_nodes.append(n)
    for st, ed in remove_edges:
        train_G.remove_edge(st, ed)
        test_G.add_edge(st, ed) 
    
    node_attr_dict = nx.get_node_attributes(train_G, name="molecule")
    train_neg_G = nx.complement(G)
    train_neg_G.name = "train_neg_G"
    nx.set_node_attributes(train_neg_G, values=node_attr_dict, name="molecule")
    print_graph_stats(train_G)
    print_graph_stats(test_G)
    print_graph_stats(train_neg_G)
    print("Number of untrain nodes", len(untrain_nodes))
    print("Number of edges incident to untrain nodes", len(remove_edges))
    return train_G, train_neg_G, test_G, untrain_nodes, \
           len(remove_edges)
def graph_test_train_split_folds(G, nfolds):
    print('Using folds')
    toremove = []
    for n in G.nodes:
        if G.nodes[n]['fingerprint'].sum() == 0:
            toremove.append(n)
    print('Removing %d nodes because of missing fingerprints' % len(toremove))
    G.remove_nodes_from(toremove)
    print('Removing %d edges because of self loops' % nx.number_of_selfloops(G))
    self_loop_edges = nx.selfloop_edges(G)
    G.remove_edges_from(self_loop_edges)
    
    edges = np.array(G.edges())
    neg_G = nx.complement(G)
    kf = KFold(n_splits=nfolds, shuffle=True)
    fgpt_attr = nx.get_node_attributes(G, name='fingerprint')
    for i, (train_idx, test_idx) in enumerate(kf.split(edges)):
        test_G = nx.Graph()
        test_G.add_edges_from(edges[test_idx])
        test_G.name = 'test_G_%d' % i
        
        train_G = nx.Graph()
        train_G.add_edges_from(edges[train_idx])
        train_G.name = 'train_G_%d' % i
        remove_nodes = [n for n, d in G.degree() if d == 0]
        print('Removing %d nodes from train_D due to no neighbors' % len(remove_nodes))
        train_G.remove_nodes_from(remove_nodes)
        """
        if not nx.is_connected(train_G):
            train_G = max(nx.weakly_connected_component_subgraphs(
                                train_G.to_directed()), 
                          key=len)
        """
        nx.set_node_attributes(train_G, values=fgpt_attr, name='fingerprint')
        nx.set_node_attributes(test_G, values=fgpt_attr, name='fingerprint')
        nodelist = train_G.nodes()
        nodelistMap = dict(zip(nodelist, range(len(nodelist))))    
        train_G = nx.relabel_nodes(train_G, nodelistMap)

        test_G = test_G.subgraph(nodelist).copy()
        test_G = nx.relabel_nodes(test_G, nodelistMap)
        molecule_names = {v: k for k, v in nodelistMap.items()}
        nx.set_node_attributes(train_G, values=molecule_names, name='molecule')  
        nx.set_node_attributes(test_G, values=molecule_names, name='molecule')  
        print_graph_stats(train_G)
        print_graph_stats(test_G)
        neg_G_ = neg_G.subgraph(nodelist).copy()
        neg_G_ = nx.relabel_nodes(neg_G_, nodelistMap)
        yield {'train_G': train_G, 'test_G': test_G, 'neg_G': neg_G_}

def graph_test_train_split(G, test_ratio, remove_node_ratios=0.05, 
                           undirected=True):
    toremove = []
    for n in G.nodes:
        if G.nodes[n]['fingerprint'].sum() == 0:
            toremove.append(n)
    print('Removing %d nodes because of missing fingerprints' % len(toremove))
    G.remove_nodes_from(toremove)

    self_loop_edges = nx.selfloop_edges(G)
    G.remove_edges_from(self_loop_edges)
    print('Removing %d edges because of self loops')

    train_G, test_G = G.copy(), G.copy()
    train_G.name, test_G.name = "train_G", "test_G"
    for st, ed in G.edges():
        if undirected and st >= ed:
            continue
        if np.random.uniform() <= test_ratio:
            train_G.remove_edge(st, ed)
            if undirected:
                train_G.remove_edge(ed, st)
        else:
            test_G.remove_edge(st, ed)
            if undirected:
                test_G.remove_edge(ed, st)
    if not nx.is_connected(train_G.to_undirected()):
        train_G = max(nx.weakly_connected_component_subgraphs(train_G), key=len)
    nodelist = train_G.nodes()
    nodelistMap = dict(zip(nodelist, range(len(nodelist))))    
    train_G = nx.relabel_nodes(train_G, nodelistMap)
    test_G = test_G.subgraph(nodelist).copy()
    test_G = nx.relabel_nodes(test_G, nodelistMap)
    
    node_attr_dict = nx.get_node_attributes(train_G, name="molecule")
    train_neg_G = nx.complement(train_G)
    train_neg_G.name = "train_neg_G"
    nx.set_node_attributes(train_neg_G, values=node_attr_dict, name="molecule")
    print_graph_stats(train_G)
    print_graph_stats(test_G)
    print_graph_stats(train_neg_G)

    return train_G, train_neg_G, test_G, nodelistMap


def run_one_LP(model, test_G, neg_G):
    if test_G.number_of_nodes() == 0:
        return 0, []
 
    true_edges = np.array(test_G.edges())

    rand_idx = np.random.choice(
            np.arange(neg_G.number_of_edges()), len(true_edges), replace=False)
    false_edges = np.array(neg_G.edges())[rand_idx]
    print("true edge shape", true_edges.shape)
    print("false edges shape", false_edges.shape)
    edges = np.concatenate((true_edges, false_edges))
    y_true = np.concatenate((np.ones(len(true_edges)), np.zeros(len(false_edges))))
    y_score = model.get_edge_weights(edges, use_logistic=True)
    AUC = roc_auc_score(y_true, y_score)
    print('AU')
    precision_curve = precision_at_ks(edges, y_score, true_edges)
    print("Precision curve", precision_curve)
    map_ = mean_average_precision(edges, y_score, true_edges)
    print('MAP', map_)
    return AUC, precision_curve, map_

def experimentLinkPrediction(G, model, resfile, rounds=3, 
          n_sampled_nodes=2048, sampling=True, nfolds=5,
          random_seed=None, inductive=False,  **params):
    print("\nLink Prediction experiments")

    if random_seed:
        np.random.seed(random_seed)
    # todo: comment back in
    #print("Original G has %d nodes, %d edges" % 
    #      (G.number_of_nodes(), G.number_of_edges()))
    print("Running LP", "inductively" if inductive else "transductively")
    if not os.path.exists(resfile):
        with open(resfile, 'w') as f:
            f.write('AUC,precision_curve,MAP\n')
    if inductive:
        test_ratios = [0.0]
        train_G, train_neg_G, test_G, untrain_nodes, \
            len_remove_edges  = graph_test_train_split_inductive(
                G, test_ratio=tr, undirected=undirected)
        Gs = [{'train_G': train_G, 'test_G': test_G, 'train_neg_G': train_neg_G}] 
    else:
        #train_G, train_neg_G, test_G, _ = graph_test_train_split(G, test_ratio=0.5)
        #Gs = [{'train_G': train_G, 'test_G': test_G}] #'train_neg_G': train_neg_G}] 
        # todo: comment back in
        #Gs = graph_test_train_split_folds(G, nfolds)
        pass
    for fold in range(nfolds):
        print('\nExperiment round', fold)
        fname = '%s/kegg/kfolds/kegg_graph_fold_%d_%d.pkl' % (os.environ['DATAPATH'], fold, nfolds)
        with open(fname, 'rb') as f:
            Gset =  pickle.load(f)
        """
    for tr, Gset in enumerate(Gs):
        fname = '%s/kegg/kfolds/kegg_graph_fold_%d_%d.pkl' % (os.environ['DATAPATH'], tr, nfolds)
        with open(fname, 'wb') as f:
            pickle.dump(Gset, f)
            print('Dumped Gset to', fname)
        continue
        print("Running LP round", tr)
        """
        """
            train_G, train_neg_G, test_G, nodelistMap = \
                graph_test_train_split(
                G, test_ratio=tr, undirected=undirected)
            untrain_nodes, len_remove_edges = [], 0

            if "walks" in datum and datum["walks"] is not None:  
                datum["walks"] = relabel_walks(datum["walks"], nodelistMap)
            if "cmty" in datum and datum["cmty"] is not None:
                datum["cmty"] = relabel_cmty(datum["cmty"], nodelistMap) 
        """
        train_G,  test_G, neg_G = Gset['train_G'], Gset['test_G'], Gset['neg_G']
        for i, n in enumerate(train_G.nodes):
            assert i == n, G_.name
        for n in test_G.nodes:
            assert n in train_G.nodes
        print('Removing %d edges because of self loops' % nx.number_of_selfloops(test_G))
        self_loop_edges = nx.selfloop_edges(test_G)
        test_G.remove_edges_from(self_loop_edges)
        print('Removing %d edges because of self loops' % nx.number_of_selfloops(train_G))
        self_loop_edges = nx.selfloop_edges(train_G)
        train_G.remove_edges_from(self_loop_edges)


        model.learn_embedding(G=train_G)#, val_G=test_G)
        untrain_nodes = []
        N = train_G.number_of_nodes()
        # todo: comment back in
        #if sampling and N > n_sampled_nodes:
        if sampling:
            AUC = []
            for r in range(rounds):
                print("sampling round ", r)
                node_list = sample_graph(train_G, n_sampled_nodes, 
                                         return_sampled_graph=False,
                                         undirected=undirected)
                sampled_test_G = test_G.subgraph(node_list).copy()
                sampled_neg_G = neg_G.subgraph(node_list).copy()

                AUCs, prec_curve, map_ = run_one_LP(
                    model, sampled_test_G, sampled_neg_G)
                AUC.append(AUCs)
            with open(resfile, 'a') as f:
                f.write('%s,%s,%f\n'
                        % (';'.join([str(x) for x in AUC]), 
                           ';'.join([str(x) for x in prec_curve]),
                           map_))
            
        else:
            print("Using all test_G edges, not sampling")
            AUC, prec_curve, map_ = run_one_LP(model, test_G, neg_G)
            with open(resfile, 'a') as f:
                f.write('%f,%s,%f\n'\
                        % (AUC, ';'.join([str(x) for x in prec_curve]), map_))



