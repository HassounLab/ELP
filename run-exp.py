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
from models.emEmbedding import EMEmbedding as EE
from models.node2vec import node2vec as N2V
from models.deepwalk import deepwalk as DW
from models.jaccardSimilarity import JaccardSimilarity as JS
from models.l2svm import L2SVM as SVM
from evaluation.linkPrediction import expLP

def read_graph(graph_f):
    if graph_f.endswith(".pkl"):
        G = nx.read_gpickle(graph_f)
    else: 
        G = nx.read_edgelist(graph_f)
    return G

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

model_map = {'em': EE, 'js': JS,  'n2v': N2V, 'deepwalk': DW, 'svm': SVM}
def main(data=None, model_type=None, testmode=False, **kwargs):
    try:
        model = model_map[model_type]
    except KeyError:
        raise NameError('model type', model_type, 'not recognized')
    params = load_params(data, model_type, **kwargs)
    """
    G = read_graph(params['graph_f'])
    if params['use_fgpt']:
        G = load_fingerprints(G, params['fingerprint_file'])
    """
    G = None
    emb_model = model(**params)
    if params['evaluation'] == 'lp':
        res = expLP(G, emb_model, **params)
    
    resfile = '%s-%s.results.txt' % (data, model_type)       
    with open(resfile, 'a') as f:
        f.write(res + '\n')
    print('results saved to', resfile)
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('-m', '--model_type', default="baseline")
    
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


