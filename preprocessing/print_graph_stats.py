import pandas as pd
import pickle
import networkx as nx
import os

path = '%s/kegg' % os.environ['DATAPATH']
G = nx.read_edgelist('%s/kegg_2020.edgelist' % path)
print('Full graph')
print(nx.info(G))
with open('%s/kegg_2020_maccs_fp.pkl' % path, 'rb') as f:
    maccs = pickle.load(f)

nodes = maccs.keys()
print('with MACCS')
print(nx.info(G.subgraph(nodes)))

with open('%s/kegg_2020_pubchem_fp.pkl' % path, 'rb') as f:
    pubchem = pickle.load(f)

nodes = pubchem.keys()
print('with PubChem')
print(nx.info(G.subgraph(nodes)))



