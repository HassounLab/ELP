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

def read_graph(graph_f=None):
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
    G = load_fingerprints(G, 'pubchem')
    G = load_fingerprints(G, 'maccs')
    print(nx.info(G))
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



if __name__ == "__main__":
    G = read_graph('%s/kegg/kegg_2020.edgelist' % os.environ['DATAPATH'])
    Gfile = '%s/kegg/kegg_2020_consolidated.edgelist' % os.environ['DATAPATH']
    nx.write_edgelist(G, Gfile)
    print('Saved to', Gfile) 
