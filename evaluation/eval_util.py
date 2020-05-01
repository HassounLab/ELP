import numpy as np
import networkx as nx


def sample_graph(G, n_sampled_nodes=None, undirected=True, return_sampled_graph=True): 
    node_num = G.number_of_nodes()
    if n_sampled_nodes and node_num > n_sampled_nodes:
        node_list = np.random.choice(all_node_list, n_sampled_nodes, replace=False)
        if return_sampled_graph:
            return node_list
        sampled_G = G.copy()
        for st, ed in G.edges():
            if st not in node_list or ed not in node_list:
                sampled_G.remove_edge(st, ed)
        print("Sampled graph with %d nodes" % sampled_G.number_of_nodes())
        return sampled_G, node_list

    else:
        print("No need to sample graph, returning copy of original")

        all_node_list = np.array(list(G.nodes()))
        if return_sampled_graph:
            return G.copy(), all_node_list
        return all_node_list
def edgelist_from_nodelist(emb_model, pred_mode, node_list, threshold=0.):
    edgelist = []
    for u in node_list:
        for v in node_list:
            if u == v:
                continue
            edgelist.append([u, v])
    
    edgelist = np.array(edgelist)
    weights = emb_model.get_edge_weights(edgelist, mode=pred_mode)
    pred_edgelist = []
    for (u, v), w in zip(edgelist, weights):
        if w > threshold:
            pred_edgelist.append((u, v, w))
    
    return pred_edgelist


   
