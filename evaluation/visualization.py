import networkx as nx
import numpy as np
import os
import pickle
def experimentVisualization(G, model, pathway_numbers, random_seed=None, **params):
    print('\n Visualization experiments')
    print('Visualization pathways', pathway_numbers)
    if random_seed:
        np.random.seed(random_seed)
    
    assert params['model_type'] == 'ep', params['model_type']
    
    with open('%s/kegg/pathway_map.pkl' % os.environ['DATAPATH'], 'rb') as f:
        pathway_map = pickle.load(f)

    test_nodes = set([])
    for (_, mapnum), edges in pathway_map.items():
        if mapnum in pathway_numbers:
            for u, v in edges:
                test_nodes.add(u)
                test_nodes.add(v)
    print('Number of test nodes', len(test_nodes))
    nodelist = G.nodes()
    nodelistMap = dict(zip(nodelist, range(len(nodelist))))
    G = nx.relabel_nodes(G, nodelistMap)
    molecule_map = {v: k for k, v in nodelistMap.items() if k in test_nodes}    
    model.learn_embedding(G=G)
    test_nodes = np.array(list(test_nodes))
    embeddings = model.get_embeddings(test_nodes) 
    fname = '%s/visualization_%s.npz'\
            % (os.environ['DATAPATH'], '_'.join(pathway_numbers))
    np.savez(fname, embeddings=embeddings, molecule_map=molecule_map)
    print('Embeddings saved to', fname) 
