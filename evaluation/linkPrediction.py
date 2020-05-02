import sys
print("python version:", sys.version)
import networkx as nx
import numpy as np
import os
import copy
import pickle
import time
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import logistic
from sklearn.model_selection import train_test_split, KFold
from evaluation.eval_util import sample_graph
from evaluation.metrics import computeMAP, computePrecisionCurve, \
    getPrecisionReport, getMetricsHeader

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
    
    edges = np.array(G.edges())
    kf = KFold(n_splits=nfolds, shuffle=True)
    fgpt_attr = nx.get_node_attributes(G, name='fingerprint')
    for i, (train_idx, test_idx) in enumerate(kf.split(edges)):
        test_G = nx.Graph()
        test_G.add_edges_from(edges[test_idx])
        test_G.name = 'test_G_%d' % i
        
        train_G = nx.Graph()
        train_G.add_edges_from(edges[train_idx])
        train_G.name = 'train_G_%d' % i
        if not nx.is_connected(train_G):
            train_G = max(nx.weakly_connected_component_subgraphs(
                                train_G.to_directed()), 
                          key=len)
        
        nx.set_node_attributes(train_G, values=fgpt_attr, name='fingerprint')
        nx.set_node_attributes(test_G, values=fgpt_attr, name='fingerprint')
        nodelist = train_G.nodes()
        nodelistMap = dict(zip(nodelist, range(len(nodelist))))    
        train_G = nx.relabel_nodes(train_G, nodelistMap)

        test_G = test_G.subgraph(nodelist).copy()
        test_G = nx.relabel_nodes(test_G, nodelistMap)
         
        node_attr_dict = nx.get_node_attributes(train_G, name="molecule")
        train_G = train_G.to_directed()
        test_G = test_G.to_directed()
        print_graph_stats(train_G)
        print_graph_stats(test_G)
        yield {'train_G': train_G, 'test_G': test_G}

def graph_test_train_split(G, test_ratio, remove_node_ratios=0.05, 
                           undirected=True):
    toremove = []
    for n in G.nodes:
        if G.nodes[n]['fingerprint'].sum() == 0:
            toremove.append(n)
    print('Removing %d nodes because of missing fingerprints' % len(toremove))
    G.remove_nodes_from(toremove)

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


def prec_at_ks(edges, pred_weights, true_edges, ks=None):
    if ks is None:
        ks = []
        for p in [10, 100, 500, 750, 1000]:
            if p < len(pred_weights):
                ks.append(p)
    
    pred_edgelist = [(st, ed, w) for ((st, ed), w) in zip(edges, pred_weights)]
    
    pred_edgelist = sorted(pred_edgelist, key=lambda x: x[2], reverse=True)
    
    true_edges_dict = {}
    
    for st, ed in true_edges:
        if st not in true_edges_dict:
            true_edges_dict[st] = []
        true_edges_dict[st].append(ed)
    
    prec_curve = []
    correct_edge = 0
    for i in range(ks[-1]):
        st, ed, _ = pred_edgelist[i]
        if st in true_edges_dict and ed in true_edges_dict[st]:
            assert st in true_edges_dict[ed]
            correct_edge += 1
        if (i + 1) in ks:
            prec_curve.append((i + 1, 1.0 * correct_edge / (i + 1)))
    
    return prec_curve


def run_one_LP(emb_model, test_G, neg_edges, untrain_nodes=None):
    if test_G.number_of_nodes() == 0:
        return 0, []
 
    if untrain_nodes is not None and untrain_nodes != []:
        print("Inductive prediction")
    true_edges = np.array(test_G.edges())

    rand_idx = np.random.choice(np.arange(len(neg_edges)), 
                                len(true_edges), replace=False)
    false_edges = np.array(neg_edges)[rand_idx]
    print("true edge shape", true_edges.shape)
    print("false edges shape", false_edges.shape)
    edges = np.concatenate((true_edges, false_edges))
    y_true = np.concatenate((np.ones(len(true_edges)), np.zeros(len(false_edges))))
    y_score = emb_model.get_edge_weights(edges, use_logistic=True)
    AUC = roc_auc_score(y_true, y_score)
    
    precision_curve = prec_at_ks(edges, y_score, true_edges)
    print("Precision curve", precision_curve)

    return AUC, precision_curve

def expLP(emb_model, verbose=1,  rounds=3, 
          n_sampled_nodes=2048, sampling=True, nfolds=5,
          random_seed=None, inductive=False,  **params):
    if verbose:
        print("\nLink Prediction")

    if random_seed:
        np.random.seed(random_seed)
    print("Original G has %d nodes, %d edges" % 
          (G.number_of_nodes(), G.number_of_edges()))
    res = {}
    print("Running LP", "inductively" if inductive else "transductively")
    if inductive:
        test_ratios = [0.0]
        train_G, train_neg_G, test_G, untrain_nodes, \
            len_remove_edges  = graph_test_train_split_inductive(
                G, test_ratio=tr, undirected=undirected)
        Gs = [{'train_G': train_G, 'test_G': test_G, 'train_neg_G': train_neg_G}] 
    else:
        #train_G, train_neg_G, test_G, _ = graph_test_train_split(G, test_ratio=0.5)
        #Gs = [{'train_G': train_G, 'test_G': test_G}] #'train_neg_G': train_neg_G}] 
        Gs = graph_test_train_split_folds(G, nfolds)
        neg_edges = nx.complement(G).edges()
    """
    for fold in range(nfolds):
        tr = fold
        fname = '%s/kegg/kegg_graph_fold_%d_%d.pkl' % (os.environ['STAGING'], fold, nfolds)
        with open(fname, 'rb') as f:
            Gset =  pickle.load(f)
    """
    res = []
    for tr, Gset in enumerate(Gs):
        fname = '%s/kegg/kegg_graph_fold_%d_%d.pkl' % (os.environ['STAGING'], tr, nfolds)
        with open(fname, 'wb') as f:
            pickle.dump(Gset, f)
            print('Dumped Gset to', fname)
        print("Running LP round", tr)
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
        train_G,  test_G = Gset['train_G'], Gset['test_G']
        emb_model.learn_embedding(G=train_G, val_G=test_G)
        untrain_nodes = []
        N = train_G.number_of_nodes()
        if sampling and N > n_sampled_nodes:
            AUCs = []
            for r in range(rounds):
                print("sampling round ", r)
                node_list = sample_graph(train_G, n_sampled_nodes, 
                                         return_sampled_graph=False,
                                         undirected=undirected)
                sampled_test_G = test_G.subgraph(node_list).copy()
                n_nodes[r] = sampled_test_G.number_of_nodes()
                n_edges[r] = sampled_test_G.number_of_edges()
                AUC, prec_curve = run_one_LP(
                    emb_model, sampled_test_G, neg_edges, 
                    untrain_nodes=untrain_nodes)
                AUC.append(AUCs)
            print('AUCs: %.2f' % AUCs)
            res.append(np.mean(AUCs))
        else:
            print("Using all test_G edges, not sampling")
            AUC, prec_curve = run_one_LP(
                emb_model, test_G, neg_edges
                untrain_nodes=untrain_nodes)
            print('AUC: %.2f' % AUC)
            res.append(AUC)
    return ','.join(res)




