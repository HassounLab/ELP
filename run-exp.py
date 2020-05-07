import sys
print(sys.version)
import numpy as np
import networkx as nx
import os
import json
import glob
import time
from argparse import ArgumentParser
import pprint
from datetime import datetime
import pickle
from evaluation.linkPrediction import experimentLinkPrediction as expLP
from evaluation.pathwayReconstruction import experimentPathwayReconstruction as expPR

def read_graph(graph_f):
    if graph_f.endswith(".pkl"):
        return nx.read_gpickle(graph_f)
    return nx.read_edgelist(graph_f)

def load_fingerprints(G, fingerprint_file):
    with open(fingerprint_file, 'rb') as f:
        fingerprints = pickle.load(f)
    G = G.subgraph(list(fingerprints.keys())).copy()
    nx.set_node_attributes(G, values=fingerprints, name='fingerprint')
    return G

eval_methods = ['lp']
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
    raise NameError('model type', model_type, 'not recognized')
       
def main(data=None, model_type=None, testmode=False, **kwargs):
    params = load_params(data, model_type, **kwargs)
    model = get_model(model_type)
    """
    G = read_graph(params['graph_f'])
    if params['use_fgpt']:
        G = load_fingerprints(G, params['fingerprint_file'])
    """
    emb_model = model(**params)
    resfile = 'logs/%s-%s.results.txt' % (data, model_type)       
    if params['evaluation'] == 'lp':
        G = None
        expLP(G, emb_model, resfile, **params)
    if params['evaluation'] == 'pr':
        G = read_graph(params['graph_f'])
        if params['use_fgpt']:
            G = load_fingerprints(G, params['fingerprint_file'])
        expPR(G, emb_model, resfile, **params)
    
    print('results saved to', resfile)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('-m', '--model_type', default="baseline")
    parser.add_argument('-e', '--evaluation', default='lp') 
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--beta_edge", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--nn_lr", type=float, default=None)
    parser.add_argument("--use_node_embed", default=None, action="store_true")
    parser.add_argument("--use_fgpt", default=None, action="store_true")
    parser.add_argument("--fgpt_name", default="fingerprint")
    parser.add_argument("--use_edge_attr", default=None, action="store_true")
    parser.add_argument("--edge_name", default=None)
 
    parser.add_argument("--inductive", default=None, action="store_true")

    args = vars(parser.parse_args())
    print(args)
    main(**args)


