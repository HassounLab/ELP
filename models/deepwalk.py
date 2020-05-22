import os
from subprocess import call
import networkx as nx
import numpy as np
from scipy.stats import logistic

from models.nnEmbeddingDecoder import NNEmbeddingDecoder
class Deepwalk:
    def __init__(self, embed_size=128, **kwargs):
        self.embed_size = embed_size
        self.nn_kwargs = kwargs["nn"] if "nn" in kwargs else {} # dictionary containing params for nn
        if "num_epochs" not in self.nn_kwargs:
            self.nn_kwargs["num_epochs"] = 40
        if "batch_size" not in self.nn_kwargs:
            self.nn_kwargs["batch_size"] = 2048
        if "beta" not in self.nn_kwargs:
            self.nn_kwargs["beta"] = 0
        if "verbose" not in self.nn_kwargs:
            self.nn_kwargs["verbose"] = 2
        if "lr" not in self.nn_kwargs:
            self.nn_kwargs["lr"] = 0.1
        if "hidden_sizes" not in self.nn_kwargs:
            self.nn_kwargs["hidden_sizes"] = [max(4, self.embed_size // 2),
                                              max(2, self.embed_size // 4)]
    def learn_embedding(self, G, neg_G=None, random_seed=None, **kwargs):
        assert G is not None
        self.G = G
        self.edges = np.array(G.edges())
        if neg_G is None:
            self.neg_edges = np.array(nx.complement(G).edges())
        else:
            self.neg_edges = np.array(neg_G.edges())
        nx.write_edgelist(
            G.to_undirected(), 
            os.path.join(os.environ["TMPDIR"], "tmpKeggGraph.txt"), 
            data=False)
               #"python3 -m deepwalk "\
        cmd =  'deepwalk ' \
               "--input %s/tmpKeggGraph.txt --output %s/tmpKeggGraph.emb "\
               "--format edgelist --representation-size %d"\
               % (os.environ["TMPDIR"], os.environ["TMPDIR"], self.embed_size)
        print("Running deep walk")
        call(cmd, shell=True)
        print("Deep walk run finished")
        self._load_embedding("%s/tmpKeggGraph.emb" % os.environ["TMPDIR"])
        self.num_nodes = self.embeddings.shape[0]
        self.nn_decoder = self._run_nn_decoder()


    def _load_embedding(self, file_name):
        with open(file_name, 'r') as f:
            n, d = f.readline().strip().split()
            self.embeddings = np.zeros((int(n), int(d)), dtype=np.float32)
            for line in f:
                emb = line.strip().split()
                emb_fl = [float(emb_i) for emb_i in emb[1:]]
                self.embeddings[int(emb[0]), :] = emb_fl
    def _run_nn_decoder(self, use_embeddings=False):
        decoder = NNEmbeddingDecoder(
            self.embeddings,
            self.edges,
            self.neg_edges,
            use_embeddings=use_embeddings,
            **self.nn_kwargs)
        return decoder

    def get_edge_scores(self, edges, use_logistic=False, **kwargs): 
        weights = self.nn_decoder.get_edge_logits(edges)
        if use_logistic:
            weights = logistic.cdf(weights)
        return weights

