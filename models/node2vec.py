import os
from subprocess import call
import networkx as nx
import numpy as np
from scipy.stats import logistic

from embedding.embedding import Embedding
from embedding.nnEmbeddingDecoder import NNEmbeddingDecoder
class node2vec(Embedding):
    def __init__(self, embed_size=128, **kwargs):
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
        self.embed_size = embed_size
    def learn_embedding(self, G, neg_G=None, random_seed=None, **kwargs):
        self.random_seed = random_seed
        assert G is not None
        self.G = G
        self.edges = np.array(G.edges())
        if neg_G is None:
            self.neg_edges = np.array(nx.complement(G).edges())
        else:
            self.neg_edges = np.array(neg_G.edges())
        args = [os.path.join(os.environ["DATAPATH"], "node2vec")]
        nx.write_edgelist(
            G, 
            os.path.join(os.environ["TMPDIR"], "tmpKeggGraph.txt"), 
            data=False)
        args.append("-i:%s/tmpKeggGraph.txt" % os.environ["TMPDIR"])
        args.append("-o:%s/tmpKeggGraph.emb" % os.environ["TMPDIR"])
        args.append("-d:%d" % self.embed_size)
        args.append("-e:200")
        args.append("-dr")
        args.append("-v")
        call(args)
        self.load_embedding("%s/tmpKeggGraph.emb" % os.environ["TMPDIR"])
        self.num_nodes = self.embeddings.shape[0]
        self.nn_decoder = None 

    def get_predicted_adjacency_matrix(self, **kwargs):
        return self.pre_nn_pred_adjmat

    def load_embedding(self, file_name):
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

    def get_edge_weights(self, edges, use_logistic=False, **kwargs): 
        if not self.nn_decoder:
            self.nn_decoder = self._run_nn_decoder()
        weights = self.nn_decoder.get_edge_logits(edges)
        if use_logistic:
            weights = logistic.cdf(weights)
        return weights

