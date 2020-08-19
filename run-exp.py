import sys
print(sys.version)
from pprint import pprint
print('sys.path')
pprint(sys.path)
import numpy as np
import networkx as nx
import os
import json
from argparse import ArgumentParser
import pickle
from evaluation.linkPrediction import experimentLinkPrediction as expLP
from evaluation.pathwayReconstruction import experimentPathwayReconstruction as expPR
from evaluation.ruleReconstruction import experimentRuleReconstruction as expRR
from evaluation.visualization import experimentVisualization as expVIZ
from evaluation.organismReconstruction import experimentOrganismReconstruction as expOR

def read_graph(graph_f=None, use_fgpt=False, use_edge=False, 
               fgpt_name=None, edge_name=None, **kwargs):
    if graph_f.endswith(".pkl"):
        G = nx.read_gpickle(graph_f)
    else:
        G = nx.read_edgelist(graph_f)
    print('Original G from', graph_f)
    print('Removing %d edges because of self loops' % nx.number_of_selfloops(G))
    self_loop_edges = nx.selfloop_edges(G)
    G.remove_edges_from(self_loop_edges)
    G.name = 'G'
    print(nx.info(G))
    if use_fgpt:
        G = load_fingerprints(G, fgpt_name)
    if use_edge:
        G = load_edge_attr(G, edge_name)
    return G

def load_edge_attr(G, edge_name):
    edge_file = '%s/kegg/kegg_2020_%s_edge.pkl' \
                % (os.environ['DATAPATH'], edge_name)
    print('Loading edge labels from', edge_file)
    with open(edge_file, 'rb') as f:
        edge_attr = pickle.load(f)
    print('%d edges have edge attributes' % len(edge_attr))
    edge_attr = {(u, v): l for (u, v), l in edge_attr.items() if G.has_edge(u, v)}
    print('%d edge with edge attributes exist in G' % len(edge_attr))
    labels = sorted(set(edge_attr.values()))
    mapping = dict(zip(labels, range(len(labels))))
    print('%d unique edge labels' % len(mapping))
    edge_attr = {k: mapping[v] for k, v in edge_attr.items()}
    nx.set_edge_attributes(G, values=edge_attr, name='edge_attr')
    print('G loaded with edges')
    return G
     

def load_fingerprints(G, fgpt_name):
    fingerprint_file = '%s/kegg/kegg_2020_%s_fp.pkl' \
                       % (os.environ['DATAPATH'], fgpt_name)
    print('Loading fingerprints from', fingerprint_file)
    with open(fingerprint_file, 'rb') as f:
        fingerprints = pickle.load(f)
    lenfp = len(fingerprints)
    fingerprints = {n: fp for n, fp in fingerprints.items() if np.sum(fp) > 0}
    if len(fingerprints) < lenfp:
        print('Removed %d fingerprints because they are all zero' \
              % (lenfp - len(fingerprints)))
    G = G.subgraph(list(fingerprints.keys())).copy()
    nx.set_node_attributes(G, values=fingerprints, name='fingerprint')
    print('G loaded with fingerprint')
    print(nx.info(G))
    print('Fingerprint length', len(next(iter(fingerprints.values()))))
    return G

eval_methods = ['lp', 'pr', 'rr', 'viz']
def load_params(data, model_type, **kwargs):
    try:
        params = json.load(open("config.json", "r"))[data]
        print('params from config.json', params)
    except KeyError:
        print('Error loading data, using empty params')
        params = {}
    if model_type in params:
        for k in params[model_type]:
            params[k] = params[model_type][k]
    for eval_method in eval_methods:
        if eval_method in params:
            params.update(params[eval_method])
    for k, v in kwargs.items():
        if v is None:
            continue
        params[k] = v
    params['model_type'] = model_type
    print("custom params", params)
    return params
# wrap this in a function so we don't import certain packages unless we must
# namely tensorflow, which is only installed in a virtual enviroment
def get_model(model_type):
    if model_type == 'ep':
        from models.epEmbedding import EPEmbedding
        return EPEmbedding
    elif model_type == 'n2v' or model_type == 'node2vec':
        from models.node2vec import Node2vec
        return Node2vec
    elif model_type == 'dw' or model_type == 'deepwalk':
        from models.deepwalk import Deepwalk
        return Deepwalk
    elif model_type == 'js' or model_type == 'jaccard':
        from models.jaccardSimilarity import JaccardSimilarity
        return JaccardSimilarity
    elif model_type == 'svm':
        from models.l2svm import L2SVM
        return L2SVM
    elif model_type == 'logreg':
        from models.logisticRegression import LogisticRegression
        return LogisticRegression
    raise NameError('model type', model_type, 'not recognized')

def get_experiment(evaluation):
    if evaluation == 'lp':
        return expLP
    elif evaluation == 'pr':
        return expPR
    elif evaluation == 'rr':
        return expRR
    elif evaluation == 'or':
        return expOR
    elif evaluation == 'viz':
        return expVIZ
    raise NameError('Evaluation method', evalaution, 'not recognized')

def main(data=None, model_type=None, testmode=False, **kwargs):
    params = load_params(data, model_type, **kwargs)
    if params['evaluation'] == 'lp' and params['load_folds']:
        G = None
    else:
        G = read_graph(**params)
    
    Model = get_model(model_type)
    model = Model(**params)

    if not os.path.exists('logs'):
        os.mkdir('logs')

    params['res_prefix'] = '%s-%s-%s' \
                           % (data, model_type, params['evaluation'])
    exp = get_experiment(params['evaluation'])
    exp(G, model, **params)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('-m', '--model_type', default=None)
    parser.add_argument('-e', '--evaluation', default='lp')
    parser.add_argument(
            '--graph_f', 
            default='%s/kegg/kegg_2020_consolidated.edgelist' % os.environ['DATAPATH']) 
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--beta_edge", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--nn_lr", type=float, default=None)
    parser.add_argument("--use_node_embed", default=None, action="store_true")
    parser.add_argument("--use_fgpt", default=None, action="store_true")
    parser.add_argument("--use_edge", default=None, action="store_true")
    parser.add_argument("--edge_name", default=None)
    parser.add_argument('--random_seed', type=int, default=2020) 
    parser.add_argument("--inductive", default=None, action="store_true")
    parser.add_argument('--fgpt_name', default=None)
    parser.add_argument('--load_folds', action='store_true')
    parser.add_argument('--start_from', type=int, default=0)
    args = vars(parser.parse_args())
    print(args)
    main(**args)


