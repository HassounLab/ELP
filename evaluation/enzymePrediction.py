import sys
print("python version:", sys.version)
import networkx as nx
import numpy as np
import os
import copy
import time
from sklearn.metrics import roc_auc_score, roc_curve
from scipy.stats import logistic
from sklearn.model_selection import train_test_split
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
    print_graph_stats(train_G)
    print_graph_stats(test_G)
    print("Number of untrain nodes", len(untrain_nodes))
    print("Number of edges incident to untrain nodes", len(remove_edges))
    return train_G,  test_G, untrain_nodes, \
           len(remove_edges)

def graph_test_train_split(G, test_ratio, remove_node_ratios=0.05, 
                           undirected=True):

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
    print_graph_stats(train_G)
    print_graph_stats(test_G)

    return train_G, test_G, nodelistMap

def expEP(datum, emb_model, pred_modes, verbose=1, test_ratios=[0.5], 
          undirected=True, n_sampled_nodes=2048, sampling=True, 
          random_seed=None, inductive=False, **params):
    if verbose:
        print("\nLink Prediction")

    G = datum["G"]
    datum = copy.deepcopy(datum) 
    res = {}
    print("Running enzyme label prediction", 
          "inductively" if inductive else "transductively")
    if inductive:
        test_ratios = [0.0]
    for tr in test_ratios:
        if random_seed:
            np.random.seed(random_seed)

        print("Running enzyme label predicion with test ratio", tr)
        if inductive:
            train_G, test_G, untrain_nodes, \
                len_remove_edges  = graph_test_train_split_inductive(
                    G, test_ratio=tr, undirected=undirected)
        else:
            train_G, test_G, nodelistMap = \
                graph_test_train_split(
                G, test_ratio=tr, undirected=undirected)
            untrain_nodes, len_remove_edges = [], 0
    
        print("Original G has %d nodes, %d edges" % 
              (G.number_of_nodes(), G.number_of_edges()))
        #datum = {k: v for k, v in datum.items()}
        datum["G"] = train_G
        datum["val_G"] = test_G
        print("learn embedding datum", datum)
        emb_model.learn_embedding(**datum)
        res[tr] = dict({
            ("n_train_G_nodes", train_G.number_of_nodes()),
            ("n_train_G_edges", train_G.number_of_edges()),
            ("n_test_G_nodes", test_G.number_of_nodes()),
            ("n_test_G_edges", test_G.number_of_edges()),
            ("n_untrain_nodes", len(untrain_nodes)),
            ("n_remove_edges", len_remove_edges)})

        emb_model.evaluate_enzyme_label_prediction(test_G)


    
