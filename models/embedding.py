import numpy as np
import tensorflow as tf
import networkx as nx
from collections import OrderedDict
import math
import os
import time
from scipy.stats import logistic
from embedding.nnEmbeddingDecoder import NNEmbeddingDecoder

class Embedding:
    def __init__(self, embed_size=128, num_epochs=500, batch_size=1024, 
                 beta=0.00001, lr=1.0, num_samples=64, verbose=1, 
                 init_value=1.0, **kwargs):
        self.embed_size = embed_size
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.beta = beta #regularization parameter
        self.lr = lr #learning rate
        self.num_samples = num_samples
        self.verbose = verbose
        self.init_value = init_value 
        self.nn_kwargs = kwargs["nn"] if "nn" in kwargs else {} # dictionary containing params for nn
    def number_of_nodes(self):
        return self.num_nodes

    def get_embeddings(self):
        return self.embeddings

    def load_embedding(self, G, load):
        print("Loading pre trained embeddings from", load)
        self._read_graph(G)
        self.embeddings = np.load(load)
    
    def _print_params(self):
        print("\tembed_size=%d\n\tnum_epochs=%d\n\tbatch_size=%d\n\tbeta=%f\n\t"
              "lr=%f\n\tnum_samples=%d\n\tinit_value=%.1f"
              % (self.embed_size, self.num_epochs, self.batch_size, self.beta,
                 self.lr, self.num_samples, self.init_value))

    def _read_graph(self, G=None, undirected=True, neg_G=None, **kwargs):

        assert G is not None

        if self.verbose:
            self.print_summary()

        self.edges = np.array(G.edges())
        if neg_G:
            self.comp_edges = np.array(neg_G.edges())
        else:
            self.comp_edges = np.array(nx.complement(G).edges())
        self.num_edges = len(self.edges)
        self.num_nodes = G.number_of_nodes()
        self.undirected = undirected

        if self.verbose:
            print("Graph contains %d nodes and %d edges"
                  % (self.num_nodes, self.num_edges))

        assert self.batch_size <= self.num_edges
        assert 1 < self.num_samples < self.num_nodes
    
        self.nn_decoder, self.nn_inner_decoder = None, None
    def _print_eval_datum(self, node_embeddings):
        print("\t\tmean of emb vectors: %.3f, max: %.3f, min: %.3f" %
              (np.mean(node_embeddings), 
               np.max(node_embeddings), 
               np.min(node_embeddings)))
    
    
    def _set_up_tf_graph(self):
        tf.reset_default_graph()
        self.node_embeddings = tf.Variable(tf.random_uniform(
            [self.num_nodes, self.embed_size], 
            -self.init_value, self.init_value))
        self.node_biases = tf.Variable(tf.zeros([self.num_nodes]))
        
        self.in_nodes_plholder = tf.placeholder(tf.int64, shape=[None])
        self.ou_nodes_plholder = tf.placeholder(tf.int64, shape=[None, 1])
        
        self.in_nodes_embed = tf.nn.embedding_lookup(
            self.node_embeddings,
            self.in_nodes_plholder, 
            partition_strategy="div")
        
        sampled_values = tf.nn.uniform_candidate_sampler(
            true_classes=self.ou_nodes_plholder,
            num_true=1,
            num_sampled=self.num_samples,
            unique=True,
            range_max=self.num_nodes)
 
        self.loss = tf.reduce_mean(tf.nn.nce_loss(
            weights=self.node_embeddings,
            biases=self.node_biases,
            inputs=self.in_nodes_embed,
            labels=self.ou_nodes_plholder,
            num_sampled=self.num_samples,
            num_classes=self.num_nodes,
            sampled_values=sampled_values,
            remove_accidental_hits=True,
            partition_strategy="div"))

        self.reg = tf.nn.l2_loss(self.in_nodes_embed)
        self.obj = self.loss + self.beta * self.reg
        self.run_ops = [self.loss, self.reg]
        self.acc_loss_values = OrderedDict(
            {"total_loss": 0.0, "loss": 0.0, "reg": 0.0})

    def _run_tf_sess(self, embedding_savepath=None):
        
        optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.obj)
        self.run_ops.insert(0, self.obj)
        self.run_ops.insert(0, optimizer)
        starttime = time.time()
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            min_loss = float("inf")
            for epoch in range(self.num_epochs):
                if self.verbose:
                    print("Epoch %d/%d" % (epoch + 1, self.num_epochs))
                
                for k in self.acc_loss_values:
                    self.acc_loss_values[k] = 0
                
                for i, feed_dict in enumerate(self._generate_batch()):
                    
                    losses = sess.run(self.run_ops, feed_dict=feed_dict)
                    for j, k in enumerate(self.acc_loss_values):
                        self.acc_loss_values[k] += losses[j + 1]
                    
                    if self.verbose > 1:
                        print("\t\tBatch %d loss: %0.4f, reg: %0.4f" %
                              (i, self.acc_loss_values["loss"] / (i + 1), 
                               self.acc_loss_values["reg"] / (i + 1)))
                if self.verbose > 1:
                    embeddings = self.node_embeddings.eval()
                    self._print_eval_datum(embeddings)        
                
                for k in self.acc_loss_values:
                    self.acc_loss_values[k] /= (i + 1) 
                
                epoch_train_loss = self.acc_loss_values["total_loss"]

                if epoch_train_loss < min_loss or \
                        (self.verbose == 1 and (epoch + 1) % 25 == 0) or \
                        self.verbose > 1: 
                    print(", ".join(["%s: %f" % (k, v) for k, v \
                                     in self.acc_loss_values.items()]))
                else:
                    print("Total loss %0.4f" % epoch_train_loss)

                if epoch_train_loss < min_loss:
                    min_loss = epoch_train_loss
                    self.embeddings = self.node_embeddings.eval()
                    if self.verbose:
                        print("Found new best embedding")
        endtime = time.time()
        print("Time taken to learn graph embeddings: %fs" % (endtime - starttime))
        if embedding_savepath:
            saveas = os.path.join(embedding_savepath, "embedding-%d.npy" % self.embed_size)
            np.save(saveas, self.embeddings)
            print("Saved embedding to", embedding_savepath)
        

    def get_edge_weights(self, edges, mode="pre_nn", use_logistic=False):
        if mode == "pre_nn" or mode == 0:
            weights = np.sum(
                np.multiply(self.embeddings[edges[:, 0]], self.embeddings[edges[:, 1]]),
                axis=1)
        elif mode == "nn" or mode == 1:
            if not self.nn_decoder:
                self.nn_decoder = NNEmbeddingDecoder(
                    self.embeddings, 
                    self.edges, 
                    self.comp_edges, 
                    use_embeddings=False,
                    **self.nn_kwargs)
            weights = self.nn_decoder.get_edge_logits(edges) 
        elif mode == "nn_inner" or mode == 2:
            if not self.nn_inner_decoder:
                self.nn_inner_decoder = NNEmbeddingDecoder(
                    self.embeddings, 
                    self.edges, 
                    self.comp_edges, 
                    use_embeddings=True,
                    **self.nn_kwargs)
            weights = self.nn_inner_decoder.get_edge_logits(edges) 
        else:
            raise NotImplementedError(mode)
        if use_logistic:
            weights = logistic.cdf(weights)
        return weights
