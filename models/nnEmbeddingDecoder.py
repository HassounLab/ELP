import numpy as np
import  tensorflow as tf
print(tf.__version__)
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import networkx as nx
from collections import OrderedDict
import math
import os
import time

class NNEmbeddingDecoder:
    def __init__(self, embeddings, edges, comp_edges,
                 hidden_sizes=None, 
                 lr=1.0, verbose=1, beta=0.00001, num_epochs=10, 
                 batch_size=1028, **kwargs):

        if not isinstance(embeddings, list):
            self.embeddings = [embeddings]
        else:
            self.embeddings = embeddings
        self.embed_size = self.embeddings[0].shape[1] 

       

        self.edges = edges
        self.comp_edges = comp_edges
        
        self.num_edges = len(edges)        
            
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.beta = beta
        self.lr = lr
        self.verbose = verbose
        if hidden_sizes is None:
            self.hidden_sizes = [max(2, self.embed_size // 4)]
        else:
            self.hidden_sizes = hidden_sizes

        self._print_params()
        self._run_nn()
    def reset_graph(self):
        tf.reset_default_graph() 
    def _print_params(self):
        print("Neural network embedding decoder") 
        print("\tembed_size=%d\n\tnum_epochs=%d\n\tbatch_size=%d\n\tbeta=%f\n\t"
              "lr=%f\n\thidden_sizes=%r\n\tnum_embed_types=%d"
              % (self.embed_size, self.num_epochs, self.batch_size, self.beta,
                 self.lr, self.hidden_sizes, len(self.embeddings)))
    def _evaluate(self, sample_size=512, shuffle=False):
        if shuffle:
            np.random.shuffle(self.train_edges)
            np.random.shuffle(self.comp_edges)
        
        true_edges = self.edges[:sample_size]
        false_edges = self.comp_edges[:sample_size]
        
        true_logits = self._get_batch_logits(true_edges)
        false_logits = self._get_batch_logits(false_edges)
        
        print("True logits %f, false logits %f" 
              % (np.mean(true_logits), np.mean(false_logits)))
        
    def _get_batch_logits(self, batch_edges):

        logits_ = self.sess.run(
            self.logits,
            feed_dict = {self.uv_nodes_plholder: batch_edges})

        return logits_[:, 0]
    
    def get_edge_logits(self, edges):
        logits = np.empty(len(edges))
        for i in range(0, len(edges), self.batch_size):
            logits[i:i + self.batch_size] = self._get_batch_logits(edges[i: i + self.batch_size])
        return logits

       

    def _generate_batch(self):
        np.random.shuffle(self.edges)
        np.random.shuffle(self.comp_edges)
        
        for i in range(0, self.num_edges, self.batch_size // 2):
            true_uv_nodes = self.edges[i:i + self.batch_size // 2]
            false_uv_nodes = self.comp_edges[i:i + self.batch_size // 2]
            batch_uv_nodes = np.concatenate((true_uv_nodes, false_uv_nodes))
            batch_labels = np.concatenate(
                (np.ones(len(true_uv_nodes)), np.zeros(len(false_uv_nodes))))
            batch_labels = batch_labels.reshape((len(batch_labels), 1))
            yield batch_uv_nodes, batch_labels

    def _get_logits(self, embeds1, embeds2, scope_idx):
        with tf.variable_scope("params%d" % scope_idx, reuse=tf.AUTO_REUSE):
            h = tf.concat([embeds1, embeds2], axis=1)
            h_in_sizes = self.hidden_sizes[:-1]
            h_out_sizes = self.hidden_sizes[1:]
            for i, (h_in, h_out) in enumerate(zip(h_in_sizes, h_out_sizes)):
                W = tf.get_variable(
                    "W%d" % i,
                    shape=[h_in, h_out],
                    initializer=tf.glorot_normal_initializer())
                b = tf.get_variable(
                    "b%d" % i,
                    shape=[h_out],
                    initializer=tf.zeros_initializer())
                h = tf.add(tf.matmul(h, W), b)
                if i != len(self.hidden_sizes) - 2:
                    h = tf.sigmoid(h)
            return h

    def _regularizer(self):
        regs = []
        for scope_idx in range(len(self.embeddings)):
            with tf.variable_scope("params%d" % scope_idx, reuse=tf.AUTO_REUSE):
                h_in_sizes = self.hidden_sizes[:-1]
                h_out_sizes = self.hidden_sizes[1:]
                for i, (hin, hout) in enumerate(zip(h_in_sizes, h_out_sizes)):
                    W = tf.get_variable("W%d" % i)
                    r = tf.get_variable("r%d" % i, initializer=tf.nn.l2_loss(W))
                    regs.append(r)
        return regs

    def _run_nn(self):
        starttime = time.time()        
        
        
        # neural network weights
        self.hidden_sizes = [self.embed_size * 2] + self.hidden_sizes + [1]

        self.uv_nodes_plholder = tf.placeholder(tf.int64, [None, 2])

        labels_plholder = tf.placeholder(tf.float32, [None, 1])

        self.embeddings = [tf.constant(x) for x in self.embeddings]
        all_logits = []
        for et_i, embeddings in enumerate(self.embeddings):

            u_embeds = tf.nn.embedding_lookup(
                embeddings,
                self.uv_nodes_plholder[:, 0],
                partition_strategy="div")

            v_embeds = tf.nn.embedding_lookup(
                embeddings,
                self.uv_nodes_plholder[:, 1],
                partition_strategy="div")
    
            logits_uv = self._get_logits(u_embeds, v_embeds, et_i)
            logits_vu = self._get_logits(v_embeds, u_embeds, et_i)
            logits_ = tf.multiply(0.5, tf.add(logits_uv, logits_vu))
            all_logits.append(logits_)
        self.logits = tf.reduce_sum(all_logits, axis=0)

        
        
        cross_ent = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(
            labels=labels_plholder, logits=self.logits))
        regs = self._regularizer()
        loss = cross_ent + self.beta * sum(regs)

        #with tf.variable_scope("adamop", reuse=tf.AUTO_REUSE):
        op = tf.train.AdamOptimizer(self.lr).minimize(loss)
        
        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())
        run_ops = [op, loss, cross_ent] + regs
        
        acc_losses = OrderedDict({"total_loss": 0.0, "cross_ent": 0.0})
        for i in range(len(regs)):
            acc_losses["reg%d" % i] = 0.
            
        for epoch in range(self.num_epochs):

            print("NN epoch %d/%d" % (epoch + 1, self.num_epochs))
            
            for k in acc_losses:
                acc_losses[k] = 0.
            
            for i, (batch_uv_nodes, batch_labels) in \
                    enumerate(self._generate_batch()):

                losses = self.sess.run(
                    run_ops,
                    feed_dict={self.uv_nodes_plholder: batch_uv_nodes,
                               labels_plholder: batch_labels})
                for j, k in enumerate(acc_losses):
                    acc_losses[k] += losses[j + 1]

                if self.verbose > 1:
                    print("\tBatch %d total loss: %0.4f\r" %
                          (i, losses[1]), end="")
                
            for k in acc_losses:
                acc_losses[k] /= i
            
            print(", ".join(["%s: %f" % (k, v) for \
                             k, v in acc_losses.items()]))
            self._evaluate()    
        endtime = time.time()
        print("time taken to run NN: %fs" % (endtime - starttime))
 

