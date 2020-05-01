import numpy as np
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf
import networkx as nx
from collections import OrderedDict
from sklearn.metrics import accuracy_score
import math
import os
import time

class NNEmbeddingEnzymeDecoder:
    def __init__(self, embeddings, train_edges, train_labels,
                 test_edges, test_labels,  
                 lr=0.01, verbose=1, beta=0.00001, num_epochs=100, 
                 batch_size=1028, **kwargs):

        if not isinstance(embeddings, list):
            self.embeddings = [embeddings]
        else:
            self.embeddings = embeddings
        self.embed_size = self.embeddings[0].shape[1] 

        self.train_edges = train_edges
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.beta = beta
        self.lr = lr
        self.verbose = verbose
        self.nlabels = np.max(train_labels)
        train_labels -= 1
        self.train_labels = np.zeros((len(train_labels), self.nlabels))
        self.train_labels[np.arange(len(train_labels)), train_labels] = 1
        test_labels -= 1
        self.test_labels =  np.zeros((len(test_labels), self.nlabels))
        self.test_labels[np.arange(len(test_labels)), test_labels] = 1
        self.hidden_sizes = [max(self.embed_size, 
                                 self.nlabels), 
                                 self.nlabels]
        self._print_params()
        self._run_nn()

    def _print_params(self):
        print("Neural network embedding decoder") 
        print("\tembed_size=%d\n\tnum_epochs=%d\n\tbatch_size=%d\n\t"
              "beta=%f\n\tlr=%f\n\thidden_sizes=%r\n\tnum_embed_types=%d\n\t"
              "verbose=%d"
              % (self.embed_size, self.num_epochs, self.batch_size, self.beta,
                 self.lr, self.hidden_sizes, len(self.embeddings), 
                 self.verbose))

    def _get_logits(self, embeds1, embeds2, scope_idx):
        h = tf.keras.layers.concatenate([embeds1, embeds2])
        for h_out in self.hidden_sizes[1:-1]:
            h = tf.keras.layers.Dense(
                        h_out, activation='sigmoid',
                        kernel_initializer='glorot_normal',
                        kernel_regularizer=tf.keras.regularizers.l2(beta))(h)
        h = tf.keras.layers.Dense(
                self.hidden_sizes[-1], activation='softmax',
                kernel_initializer='glorot_normal',
                kernel_regularizer=tf.keras.regularizers.l2(beta))(h)
        return h

    def _run_nn(self):
        starttime = time.time()        
        
        
        # neural network weights
        self.hidden_sizes = [self.embed_size * 2] + self.hidden_sizes + [self.nlabels]

        self.uv_nodes = tf.keras.layers.Input(shape=(2,), dtype=tf.int64)


        self.embeddings = [tf.constant(x) for x in self.embeddings]
        all_logits = []
        for et_i, embeddings in enumerate(self.embeddings):

            u_embeds = tf.nn.embedding_lookup(
                embeddings,
                self.uv_nodes[:, 0])

            v_embeds = tf.nn.embedding_lookup(
                embeddings,
                self.uv_nodes[:, 1])
    
            logits_uv = self._get_logits(u_embeds, v_embeds, et_i)
            logits_vu = self._get_logits(v_embeds, u_embeds, et_i)
            logits_ = tf.multiply(0.5, tf.add(logits_uv, logits_vu))

            all_logits.append(logits_)
        self.logits = tf.reduce_sum(all_logits, axis=0)
       
        self.model = tf.keras.models.Model(inputs=self.uv_nodes, outputs=self.logits)

        optimizer = tf.keras.optimizers.Adam(lr)
        self.model.compile(
                optimizer=optimizer,
                loss='sparse_categorical_crossentropy',
                metrics=tf.keras.metrics.CategoricalAccuracy())
        self.model.fit(self.train_edges, self.train_labels,
            batch_size=self.batch_size, epochs=self.num_epochs,
            verbose=2, shuffle=True)
        print("time taken to run NN: %fs" % (endtime - starttime))
        print('\nEvaluation on test set')
        self.model.evaluate(self.test_edges, self.test_labels, verbose=2, 
                            batch_size=self.batch_size)
 
