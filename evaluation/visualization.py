import networkx as nx
import numpy as np
import os
import pickle
def experimentVisualization(G, model, pathway_numbers, res_prefix=None, random_seed=None, **params):
    print('\n Visualization experiments')
    print('Visualization pathways', pathway_numbers)
    if random_seed:
        np.random.seed(random_seed)
    
    assert params['model_type'] == 'ep', params['model_type']
    
    with open('%s/kegg/pathway_map.pkl' % os.environ['DATAPATH'], 'rb') as f:
        pathway_map = pickle.load(f)

    test_molecules = set([])
    for (_, mapnum), edges in pathway_map.items():
        if mapnum in pathway_numbers:
            for u, v in edges:
                test_molecules.add(u)
                test_molecules.add(v)
    test_molecules = [n for n in test_molecules if G.has_node(n)]
    print('Number of test nodes', len(test_molecules))
    nodelist = G.nodes()
    nodelistMap = dict(zip(nodelist, range(len(nodelist))))
    G = nx.relabel_nodes(G, nodelistMap)
    model.learn_embedding(G=G)
    test_nodes = np.array([nodelistMap[n] for n in test_molecules])
    embeddings = model.get_embeddings(test_nodes) 
    if not os.path.exists('%s/kegg/visualizations' % os.environ['DATAPATH']):
        os.mkdir('%s/kegg/visualizations' % os.environ['DATAPATH'])
    fname = '%s/kegg/visualizations/%s_%s.npz'\
            % (os.environ['DATAPATH'], res_prefix, '_'.join(pathway_numbers))
    np.savez(fname, embeddings=embeddings, molecules=test_molecules)
    print('Embeddings saved to', fname) 
