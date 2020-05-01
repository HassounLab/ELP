import numpy as np 
import networkx as nx
from embedding.embedding import Embedding

class BaselineEmbedding(Embedding):
    def __init__(self, **kwargs):
        super(BaselineEmbedding, self).__init__(**kwargs)

    def print_summary(self):
        print("Running baseline embedding with the following parameters")
        self._print_params()
    
    def learn_embedding(self, G, undirected=True, embedding_savepath=None, 
                        neg_G=None, **kwargs):
        self._read_graph(G, undirected, neg_G)
        self._set_up_tf_graph() 
        self._run_tf_sess(embedding_savepath)

    def _generate_batch(self):
        np.random.shuffle(self.edges)
        for i in range(0, self.num_edges, self.batch_size):
            batch_in_nodes = self.edges[i:i + self.batch_size, 0]
            batch_ou_nodes = self.edges[i:i + self.batch_size, 1]
            batch_ou_nodes = batch_ou_nodes.reshape((len(batch_ou_nodes), 1))
            yield {self.in_nodes_plholder:batch_in_nodes, 
                   self.ou_nodes_plholder:batch_ou_nodes}

