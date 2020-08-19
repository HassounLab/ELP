import networkx as nx
import numpy as np
import time
import os
import pickle
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import KFold
from evaluation.metrics import precision_at_ks, mean_average_precision
def graph_test_train_split_inductive(G, remove_node_ratios=0.05, nrepeat=5): 
    self_loop_edges = nx.selfloop_edges(G)
    print('Removing %d edges because of self loops' % nx.number_of_selfloops(G))
    G.remove_edges_from(self_loop_edges)
    nodelist = G.nodes()
    nodelistMap = dict(zip(nodelist, range(len(nodelist))))
    G = nx.relabel_nodes(G, nodelistMap)
    molecule_names = {v: k for k, v in nodelistMap.items()}
    neg_G = nx.complement(G)
    edges = np.array(G.edges())
    num_test_edges = int(len(edges) * remove_node_ratios + 1)
    print('Sampling %d edges for inductive testing' % num_test_edges)
    fgpt_attr = nx.get_node_attributes(G, name='fingerprint')
    edge_attr = nx.get_edge_attributes(G, name='edge_attr')
    for i in range(nrepeat):
        
        train_G, test_G = G.copy(), nx.Graph()
        train_G.name, test_G.name = "train_G_%d" % i, "test_G_%d" % i
        where = np.random.choice(np.arange(len(edges)), size=num_test_edges, replace=False)
        test_edges = edges[where] 
        train_G.remove_edges_from(test_edges)
        test_G.add_edges_from(test_edges)
        
        nx.set_node_attributes(train_G, values=fgpt_attr, name='fingerprint')
        nx.set_node_attributes(test_G, values=fgpt_attr, name='fingerprint')

        nx.set_edge_attributes(train_G, values=edge_attr, name='edge_attr')
        nx.set_edge_attributes(test_G, values=edge_attr, name='edge_attr')
        
        nx.set_node_attributes(train_G, values=molecule_names, name='molecule')  
        nx.set_node_attributes(test_G, values=molecule_names, name='molecule')  
        
        print(nx.info(train_G))
        print(nx.info(test_G))
        
        yield {'train_G': train_G, 'test_G': test_G, 'neg_G': neg_G}

def graph_test_train_split_folds(G, nfolds):
    print('Using folds')
    self_loop_edges = nx.selfloop_edges(G)
    print('Removing %d edges because of self loops' % nx.number_of_selfloops(G))
    G.remove_edges_from(self_loop_edges)

    
    edges = np.array(G.edges())
    neg_G = nx.complement(G)
    kf = KFold(n_splits=nfolds, shuffle=True)
    fgpt_attr = nx.get_node_attributes(G, name='fingerprint')
    edge_attr = nx.get_edge_attributes(G, name='edge_attr')
    for i, (train_idx, test_idx) in enumerate(kf.split(edges)):
        test_G = nx.Graph()
        test_G.add_edges_from(edges[test_idx])
        test_G.name = 'test_G_%d' % i
        
        train_G = nx.Graph()
        train_G.add_edges_from(edges[train_idx])
        train_G.name = 'train_G_%d' % i
        remove_nodes = [n for n, d in G.degree() if d == 0]
        print('Removing %d nodes from train_G due to no neighbors' % len(remove_nodes))
        train_G.remove_nodes_from(remove_nodes)
        
        nx.set_node_attributes(train_G, values=fgpt_attr, name='fingerprint')
        nx.set_node_attributes(test_G, values=fgpt_attr, name='fingerprint')
        
        nx.set_edge_attributes(train_G, values=edge_attr, name='edge_attr')
        nx.set_edge_attributes(test_G, values=edge_attr, name='edge_attr')
            
        nodelist = train_G.nodes()
        nodelistMap = dict(zip(nodelist, range(len(nodelist))))    
        
        train_G = nx.relabel_nodes(train_G, nodelistMap)

        test_G = test_G.subgraph(nodelist).copy()
        test_G = nx.relabel_nodes(test_G, nodelistMap)
        
        molecule_names = {v: k for k, v in nodelistMap.items()}
        nx.set_node_attributes(train_G, values=molecule_names, name='molecule')  
        nx.set_node_attributes(test_G, values=molecule_names, name='molecule')  
        
    
        neg_G_ = neg_G.subgraph(nodelist).copy()
        neg_G_ = nx.relabel_nodes(neg_G_, nodelistMap)
        
        print(nx.info(train_G))
        print(nx.info(test_G))
        
        yield {'train_G': train_G, 'test_G': test_G, 'neg_G': neg_G_}


def evaluateLinkPrediction(model, train_G=None, test_G=None, neg_G=None, 
                          debug=False, res_prefix=None, save_roc_curve=False):
    assert train_G and test_G and neg_G
    for i, n in enumerate(train_G.nodes):
        assert i == n, train_G.name
    if not debug:
        for n in test_G.nodes:
            assert n in train_G.nodes
    start_time = time.time()
    model.learn_embedding(G=train_G)#, val_G=test_G)
    time_taken = time.time() - start_time
    print('Time taken', time_taken)
    true_edges = np.array(test_G.edges())

    rand_idx = np.random.choice(
            np.arange(neg_G.number_of_edges()), len(true_edges), replace=False)
    false_edges = np.array(neg_G.edges())[rand_idx]
    print("true edge shape", true_edges.shape)
    print("false edges shape", false_edges.shape)
    edges = np.concatenate((true_edges, false_edges))
    y_true = np.concatenate((np.ones(len(true_edges)), np.zeros(len(false_edges))))
    y_score = model.get_edge_scores(edges, use_logistic=True)
    AUC = roc_auc_score(y_true, y_score)
    print('AUC', AUC)
    precision_curve = precision_at_ks(edges, y_score, true_edges)
    print("Precision curve", precision_curve)
    map_ = mean_average_precision(edges, y_score, true_edges)
    print('MAP', map_)
    if save_roc_curve:
        fpr, tpr, thresholds = roc_curve(y_true, y_score)
        roc_curve_file = '%s/%s_roc_curve.npz' % (os.environ['DATAPATH'], res_prefix)
        np.savez(roc_curve_file, fpr=fpr, tpr=tpr, thresholds=thresholds, AUC=AUC) 
        print('Saved roc curves', roc_curve_file)
    return AUC, precision_curve, map_, time_taken


def experimentLinkPrediction(G, model, res_prefix=None, nfolds=5, load_folds=True, start_from=0,
                             random_seed=None, inductive=False, save_roc_curve=False, **params):
    print("\nLink Prediction experiments")
    resfile = 'logs/%s.results.txt' % res_prefix
    print('Writing results to', resfile)
    if random_seed:
        np.random.seed(random_seed)
    print("Running LP", "inductively" if inductive else "transductively")
    if not os.path.exists(resfile):
        with open(resfile, 'w') as f:
            f.write('inductive,AUC,precision_curve,MAP,time\n')
    if inductive:
        Gs = graph_test_train_split_inductive(G)
        load_folds, start_from = False, 0 
    else:
        if not load_folds:
            Gs = graph_test_train_split_folds(G, nfolds)
            if start_from > 0:
                print('Not loading fold but also starting from nonzero')
                for _ in range(start_from):
                    next(Gs)

        print('Starting from fold', start_from)
    for fold in range(start_from, nfolds):
        print('\nExperiment fold', fold)
        fname = '%s/kegg/kfolds/kegg_graph%s_fold_%d_%d.pkl' \
                % (os.environ['DATAPATH'], 
                   '_' + params['fgpt_name'] if params['use_fgpt'] else '', 
                   fold, nfolds)
        if load_folds:
            print('Loading previously saved train/test set from', fname)
            with open(fname, 'rb') as f:
                Gset =  pickle.load(f)
        else:
            Gset = next(Gs)
            if not inductive:
                if os.path.exists(fname):
                    print(fname, 'already exists, not overwriting')
                else:
                    with open(fname, 'wb') as f:
                        pickle.dump(Gset, f)
                        print('Dumped Gset to', fname)
        
        AUC, prec_curve, map_, time_taken = evaluateLinkPrediction(
                model, 
                res_prefix=res_prefix, 
                save_roc_curve=save_roc_curve,
                **Gset)
        with open(resfile, 'a') as f:
            f.write('%r,%f,%s,%f,%d\n'\
                    % (inductive, AUC, ';'.join([str(x) for x in prec_curve]), map_, time_taken))
        print('Saved to', resfile)



