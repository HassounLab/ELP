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
import IPython

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

def relabel_walks(walks, nodelistMap):
    newwalks = []
    for n, w, aw in walks:
        if n in nodelistMap:
            newwalks.append((nodelistMap[n], w, aw))
    return np.array(newwalks)

def relabel_cmty(cmty, nodelistMap):
    newcmty = []
    for c, n in cmty:
        if n in nodelistMap:
            newcmty.append((c, nodelistMap[n]))
    return np.array(newcmty)

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


def run_one_LP(emb_model, test_G, neg_edges, pred_mode, meta="",
               savepath=None, untrain_nodes=None,
               roc_curve_savename=None, save_false_preds=False):
    if test_G.number_of_nodes() == 0:
        return 0, []
 
    if untrain_nodes is not None and untrain_nodes != []:
        print("Inductive prediction")
    true_edges = np.array(test_G.edges())

    rand_idx = np.random.choice(np.arange(len(neg_edges)), 
                                len(true_edges), replace=False)
    false_edges = np.array(train_neg_G.edges())[rand_idx]
    print("true edge shape", true_edges.shape)
    print("false edges shape", false_edges.shape)
    edges = np.concatenate((true_edges, false_edges))
    y_true = np.concatenate((np.ones(len(true_edges)), np.zeros(len(false_edges))))
    y_score = emb_model.get_edge_weights(edges, mode=pred_mode, use_logistic=True)
    AUC = roc_auc_score(y_true, y_score)
    
    precision_curve = prec_at_ks(edges, y_score, true_edges)
    print("Precision curve", precision_curve)
    """
    if savepath and roc_curve_savename:
        filename = os.path.join(savepath, roc_curve_savename)
        print("saving roc curve to", filename)
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        np.savez(filename, fpr=fpr, tpr=tpr, thresholds=thresholds, AUC=AUC) 
        
        if save_false_preds:
            cutoff = thresholds[np.abs(tpr - 0.9).argmin()]
            rclass_dict = nx.get_edge_attributes(test_G, name="rclass")
            if rclass_dict != {}:
                #assert rclass_dict != {}
                fname = os.path.join(savepath, "false_neg_cutoff%.5f_%s.txt" % (cutoff, meta))
                with open(fname, "w") as fp:
                    fp.write("mole1 mole2 rclass\n")
                    for edge, ys in zip(true_edges, y_score[:len(true_edges)]):
                        if ys < cutoff:
                            u, v = edge
                            u_mol = test_G.node[u]["molecule"]
                            v_mol = test_G.node[v]["molecule"]
                            if (u, v) in rclass_dict:
                                rclass = rclass_dict[(u, v)]
                            else:
                                rclass = None
                            fp.write("%s %s %s\n" % (u_mol, v_mol, rclass))
            
                #rclass_dict = nx.get_edge_attributes(train_neg_G, name="rclass")
                #assert rclass_dict != {}
                
                fname = os.path.join(savepath, "false_pos_cutoff%.5f_%s.txt" % (cutoff, meta))
                with open(fname, "w") as fp:
                    fp.write("mole1 mole2 rclass\n")
                    for edge, ys in zip(false_edges, y_score[len(true_edges):]):
                        if ys >= cutoff:
                            u, v = edge
                            u_mol = train_neg_G.node[u]["molecule"]
                            v_mol = train_neg_G.node[v]["molecule"]
                            #rclass = rclass_dict[(u, v)] if (u, v) in rclass_dict else None
                            #fp.write("%s %s %s\n" % (u_mol, v_mol, rclass))
                            fp.write("%s %s\n" % (u_mol, v_mol))
    
        """

    return AUC, precision_curve

def compute_and_save_scores(emb_model, G, neg_G, mode, savepath):
    true_edges = np.array(G.edges())
    false_edges = np.array(neg_G.edges())
    true_scores = emb_model.get_edge_weights(true_edges, mode=mode, use_logistic=True)
    false_scores = emb_model.get_edge_weights(false_edges, mode=mode, use_logistic=True)
    true_scores_dict = {e: s for e, s in zip(true_edges, true_scores)}
    false_scores_Dict = {e: s for e, s in zip(false_edges, false_scores)}
    nx.set_edge_attributes(G, true_scores_dict, "lp_score")
    nx.set_edge_attributes(neg_G, false_scores_dict, "lp_score")
    nx.write_gpickle(G, os.path.join(savepath, "graph_lp_score.pkl"))
    nx.write_gpickle(neg_G, os.path.join(savepath, "neg_graph_lp_score.pkl"))


def expLP(datum, emb_model, pred_modes, verbose=1, test_ratios=[0.5], rounds=3, 
          undirected=True, n_sampled_nodes=2048, sampling=True, nfolds=2,
          savepath=None, validation=True, val_ratio=0.1, random_seed=None, 
          inductive=False, save_scores=False, **params):
    if verbose:
        print("\nLink Prediction")

    if random_seed:
        np.random.seed(random_seed)
    G = datum["G"]
    datum = copy.deepcopy(datum) 
    res = {}
    print("Running LP", "inductively" if inductive else "transductively")
    if inductive:
        test_ratios = [0.0]
        train_G, train_neg_G, test_G, untrain_nodes, \
            len_remove_edges  = graph_test_train_split_inductive(
                G, test_ratio=tr, undirected=undirected)
        Gs = [{'train_G': train_G, 'test_G': test_G, 'train_neg_G': train_neg_G}] 
    else:
        #train_G, train_neg_G, test_G, _ = graph_test_train_split(G, test_ratio=0.1)
        #Gs = [{'train_G': train_G, 'test_G': test_G, 'train_neg_G': train_neg_G}] 
        Gs = graph_test_train_split_folds(G, nfolds)
        neg_edges = nx.complement(G).edges()
    for fold in range(nfolds):
        tr = fold
        fname = '%s/kegg/kegg_graph_fold_%d_%d.pkl' % (os.environ['STAGING'], fold, nfolds)
        with open(fname, 'rb') as f:
            Gset =  pickle.load(f)
        """
    for tr, Gset in enumerate(Gs):
        fname = '%s/kegg/kegg_graph_fold_%d_%d.pkl' % (os.environ['STAGING'], tr, nfolds)
        with open(fname, 'wb') as f:
            pickle.dump(Gset, f)
            print('Dumped Gset to', fname)
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
        print("Original G has %d nodes, %d edges" % 
              (G.number_of_nodes(), G.number_of_edges()))
        #datum = {k: v for k, v in datum.items()}
        train_G,  test_G = Gset['train_G'], Gset['test_G']
        datum["G"] = train_G
        datum["val_G"] = test_G
        print("learn embedding datum", datum)
        emb_model.learn_embedding(**datum)
        untrain_nodes = []
        res[tr] = dict({
            ("n_train_G_nodes", train_G.number_of_nodes()),
            ("n_train_G_edges", train_G.number_of_edges()),
            ("n_test_G_nodes", test_G.number_of_nodes()),
            ("n_test_G_edges", test_G.number_of_edges())})
            #("n_untrain_nodes", len(untrain_nodes)),
            #("n_remove_edges", len_remove_edges)})
        for pred_mode in pred_modes:
            print("using predictive mode", pred_mode)
            """
            if save_scores:
                compute_and_save_scores(emb_model, train_G, train_neg_G, pred_mode, savepath)
            """
            N = train_G.number_of_nodes()
            if sampling and N > n_sampled_nodes:
                AUCs = [None] * rounds
                prec_curves = [None] * rounds
                n_nodes = [None] * rounds
                n_edges = [None] * rounds
                
                roc_curves_files = []
                for r in range(rounds):
                    print("sampling round ", r)
                    node_list = sample_graph(train_G, n_sampled_nodes, 
                                             return_sampled_graph=False,
                                             undirected=undirected)
                    sampled_test_G = test_G.subgraph(node_list).copy()
                    n_nodes[r] = sampled_test_G.number_of_nodes()
                    n_edges[r] = sampled_test_G.number_of_edges()
                    roc_curve_savename = "roc_curve_tr%.2f_pm%s_r%d.npz" \
                                         % (tr, pred_mode, r)
                    roc_curves_files.append(
                        os.path.join(savepath, roc_curve_savename))
                    AUCs[r], prec_curve = run_one_LP(
                        emb_model, sampled_test_G, neg_edges, pred_mode, 
                        node_list=node_list,
                        untrain_nodes=untrain_nodes,
                        savepath=None,
                        roc_curve_savename=roc_curve_savename,
                        meta="tr%0.2f" % tr,
                        save_false_preds=False,
                        random_seed=random_seed)
                if verbose:
                    print("\tRound: %d\tAUC:%f" % (r, AUCs[r]))
                    print("mean_n_nodes: %d, mean_n_edges: %d " %
                          (np.mean(n_nodes), np.mean(n_edges)))
                    print("AUCs: %r" % AUCis)
                res[tr][pred_mode] = dict({
                    ("AUC", np.mean(AUCs)),
                    ("prec_curve", prec_curve),
                    ("std", np.std(AUCs)),
                    ("mean_n_nodes_sampled", int(np.mean(n_nodes))),
                    ("mean_n_edges_sampled", int(np.mean(n_edges)))})
                res[tr][pred_mode]["AUCs"] = list(AUCs)
            else:
                print("Using all test_G edges, not sampling")
                roc_curve_savename = "roc_curve_tr%.2f_pm%s.npz" \
                                     % (tr, pred_mode)
                roc_curves_files = [os.path.join(savepath, roc_curve_savename)]
                AUC, prec_curve = run_one_LP(
                    emb_model, test_G, train_neg_G, pred_mode,
                    savepath=savepath,
                    meta="tr%f" % tr,
                    untrain_nodes=untrain_nodes,
                    roc_curve_savename=roc_curve_savename,
                    save_false_preds=False)
 
                if verbose:
                    print("n_nodes: %d, n_edges: %d, AUC: %f" %
                      (N, test_G.number_of_edges(), AUC))
                res[tr][pred_mode] = dict({("AUC", AUC)})
                res[tr][pred_mode]["prec_curve"] = prec_curve
            """
            roc_curve_plot_file = os.path.join(
                savepath, "roc_curve_plot_tr%0.2f_pm%s.png" % (tr, pred_mode))
            plot_roc(roc_curve_plot_file,
                     roc_curves_files, 
                     title="Test Ratio %0.2f" % tr)
            """
    return res


    
