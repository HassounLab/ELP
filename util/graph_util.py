import numpy as np
from collections import Counter
import networkx as nx

def read_graph(graph_f):
    if graph_f.endswith(".txt"):
        G = nx.read_edgelist(graph_f, create_using=nx.Graph(), nodetype=int)
        G = G.to_directed()
    elif graph_f.endswith(".pkl"):
        G = nx.read_gpickle(graph_f)
    else:
        raise IOError("Graph extension not supported" % graph_f)
    
    return G

def read_labels(labels_file, num_nodes, skip_isolated_nodes=False):
    labels = []
    with open(labels_file, "r") as f:
        for line in f:
            if line.startswith("#") or line == "\n":
                continue
            try:
                nodes = line.split(":")[1]
            except:
                nodes = line
            nodes = list(map(int, nodes.split()))
            if skip_isolated_nodes and len(nodes) == 1:
                print("skipping label because too few members")
                continue
            labels.append(nodes)
    labels = np.array(labels)
    num_labels = labels.shape[0]
    #print("num labels", num_labels)
    #print(labels)
    node_labels = np.zeros([num_nodes, num_labels])
    for i, l in enumerate(labels):
        for node in l:
            node_labels[node, i] = 1
    return node_labels

def read_cmty(cmty_file, num_nodes):
    cmty = []
    num_coms = 0
    with open(cmty_file, "r") as f:
        for line in f:
            if line.startswith("#") or line == "\n":
                continue
            try:
                nodes = line.split(":")[1]
            except:
                nodes = line
            nodes = list(map(int, nodes.split()))
            for n in nodes:
                cmty.append((num_coms, n))
            num_coms += 1
    return np.array(cmty), num_coms

def read_walks(walks_file, walk_length=5, **kwargs):
    walks = np.load(walks_file)
    walk_length = min(walk_length, walks.shape[1])
    
    all_awalks = generate_anony_walks(steps=walk_length - 1)
    assert walk_length == len(all_awalks[0])
    awalk_ids = {}
    for i, aw in enumerate(all_awalks):
        awalk_ids[tuple(aw)] = i
    
    anonymized_walks = Counter()
    appeared_awalks = set()
    for walk in walks:
        source = walk[0]
        try:
            walk_id = awalk_ids[anonymize_walk(walk[:walk_length])]
        except KeyError:
            print(walk[:walk_length])
            print(anonymize_walk(walk[:walk_length]))
            raise KeyError
        appeared_awalks.add(walk_id)
        anonymized_walks[(source, walk_id)] += 1
        
    print("%d/%d of all awalks appeared" % 
          (len(appeared_awalks), len(all_awalks)))

    anonymized_walks = np.array([(s, w, c) for (s, w), c, in \
                                 anonymized_walks.items()])
    return anonymized_walks, len(all_awalks)
        
def anonymize_walk(walk):
    idx = 0
    anony_walk = []
    d = {}
    for n in walk:
        if n not in d:
            d[n] = idx
            idx += 1
        anony_walk.append(d[n])
    return tuple(anony_walk)

def generate_anony_walks(steps, keep_last=True):
    '''Get all possible anonymous walks of length up to steps.'''
    paths = []
    last_step_paths = [[0, 1]]
    for i in range(2, steps+1):
        current_step_paths = []
        for j in range(i + 1):
            for walks in last_step_paths:
                if walks[-1] != j and j <= max(walks) + 1:
                    paths.append(walks + [j])
                    current_step_paths.append(walks + [j])
        last_step_paths = current_step_paths
    # filter only on n-steps walks
    if keep_last:
        paths = list(filter(lambda path: len(path) ==  steps + 1, paths))
    return paths
