import numpy as np
import tensorflow as tf
print(tf.__version__)
import networkx as nx
from collections import OrderedDict
import math
import os
import sys
import time
import pickle
from scipy.stats import logistic
from embedding.nnEmbeddingDecoder import NNEmbeddingDecoder
import IPython
NODE_EMBED = "node_embed"                                                       
NODE_ATTR_EMBED = "node_attr_embed"                                             
FGPT_EMBED = "fgpt_embed"                                                       
EDGE_ATTR_EMBED = "edge_attr_embed" 

default_params = { 
     'embed_size': 128, 
     'num_epochs': 500,
     'batch_size': 2048,
     'gamma': 10,
     'lr': 0.5,
     'verbose': 1,
     'use_fgpt': True,
     'use_edge_attr': False,
     'use_node_embed': True,
     'beta': 0,
     'fgpt_name': 'fingerprint',
     'edge_name': 'rclass_int',
     'edge_weight': 1.0,
     'random_seed': None,
     'nn': {
        'num_epochs': 40,
        'batch_size': 2048,
        'beta': 0,
        'verbose': 2,
        'lr': 0.1,
        'hidden_sizes': None,
        'random_seed': None
    }
}
class EMEmbedding:
    def __init__(self, **params):
        print('EmEmbedding version 2')
        for k, v in default_params.items():
            if k not in params:
                params[k] = v
        self.nn_kwargs = params['nn']
        if "hidden_sizes" not in self.nn_kwargs:                                
            self.nn_kwargs["hidden_sizes"] = [max(4, params['embed_size'] // 2),     
                                              max(2, params['embed_size'] // 4)]     
        if "random_seed" not in self.nn_kwargs:                                 
            self.nn_kwargs["random_seed"] = params['random_seed']

        print('\n'.join('\t%s: %r' % (k, v) for k, v in params.items()))
        self.p = params
        
        
    def number_of_nodes(self):
        return self.G.number_of_nodes()
    
    def learn_embedding(self, G, **kwargs):
        tf.random.set_seed(self.p['random_seed'])
        self.G = G
        self.embed_types, self.num_embeds = [], {}
      
        if self.p['use_node_embed']:
            self.embed_types.append(NODE_EMBED)
            self.num_embeds[NODE_EMBED] = G.number_of_nodes()
            adjmat = nx.adjacency_matrix(G).todense()
            assert not np.any(np.isnan(adjmat))
            assert np.all(np.sum(adjmat, axis=1) > 0)
            self.adjmat = tf.constant(adjmat, dtype=tf.float32)
        if self.p['use_fgpt']:
            fgpts = []                                                          
            for n in G.nodes:                                                   
                fgpts.append(G.nodes[n][self.p['fgpt_name']].reshape((-1, 1)))       
            fgpts = np.array(fgpts, dtype=np.float32)                           
            assert not np.any(np.isnan(fgpts))                                  
            self.fgpts = tf.constant(fgpts, dtype=tf.float32)                   
            self.num_embeds[FGPT_EMBED] = len(fgpts[0])
            print("Fgpt length", self.num_embeds[FGPT_EMBED]) 
            self.embed_types.append(FGPT_EMBED)
        self._run()
    def _get_embeds_h(self, batch_nodes, et):                 
        print('getting embeds_h') 
        if et == NODE_EMBED:                                                  
            print('retrieving node embedding') 
            return self.embeddings[et](batch_nodes) 
        elif et == FGPT_EMBED:                                                  
            fgpt_feats = tf.nn.embedding_lookup(self.fgpts, batch_nodes)
            embeddings = self.embeddings[et](tf.range(self.num_embeds[et]))
            print('attr embedding retrieved')
            embeds = tf.reduce_sum(fgpt_feats * embeddings, axis=1)        
            cardinality = tf.reduce_sum(fgpt_feats, axis=1)                     
            cardinality = tf.maximum(cardinality, 1)                            
            norm_embeds = embeds / cardinality                                  
            print('fgpt embeds shape', tf.shape(norm_embeds))
            return norm_embeds  
        else:
            raise ValueError('invalid embed type: ' + et)
    def _get_neigh_embeds_h(self, v_nodes, et): 
        if et not in self.embed_types:                                          
            raise ValueError("invalid embed type: " + et)                              
        def apply_neighbor(adj_row):                                             
            print('applying neighbor') 
            where = tf.greater(adj_row, 0)
            neigh_nodes = tf.squeeze(tf.where(where), axis=1)               
            num_neigh_nodes = tf.shape(neigh_nodes)[0]                      
            neigh_node_es = self._get_embeds_h(neigh_nodes, et)             
            e = tf.reduce_sum(neigh_node_es, axis=0)                        
            e /= tf.cast(num_neigh_nodes, dtype=tf.float32)                 
            print("e shape", tf.shape(e))                                   
            return e                                                        
        adjrows = tf.nn.embedding_lookup(self.adjmat, v_nodes)                           
        es = tf.map_fn(apply_neighbor, adjrows, dtype=tf.float32, parallel_iterations=10)
        return es 
    def _metric(self, h1, h2, et):                                              
        return tf.reduce_sum(h1 * h2, axis=1) # dot product
    
    def _rank_loss(self, latent, ypred):
        return tf.reduce_mean(tf.nn.relu(self.p['gamma'] + latent))

    def _min_output(self, latent, ypred):
        return tf.reduce_max(latent)

    def _max_output(self, latent, ypred):
        return tf.reduce_min(latent)

    def _run(self):
        print('runninEM embedding 2')
        IPython.embed()
        batch_nodes = tf.keras.layers.Input(shape=(), dtype=tf.int64)
        neg_nodes = tf.random.uniform(
                shape=tf.shape(batch_nodes),
                maxval=self.G.number_of_nodes(),
                dtype=tf.int64)
        print('batch_ndoes, neg nodes made')
        self.embeddings = {}
        outputs = []
        for et in self.embed_types:
            self.embeddings[et] = tf.keras.layers.Embedding(
                    self.num_embeds[et], self.p['embed_size'],
                    name='%s_embedding' % et,
                    embeddings_initializer='glorot_uniform',
                    embeddings_regularizer=tf.keras.regularizers.l2(
                            self.p['beta']))

            v_embeds_h = self._get_embeds_h(batch_nodes, et)
            print('v_embeds_h returned')
            u_embeds_h = self._get_embeds_h(neg_nodes, et)
            print('u_embeds_h returned')
            IPython.embed()
            v_neigh_embeds_h = self._get_neigh_embeds_h(batch_nodes, et)
            print('v_neigh_embeds_h returned')
            pos_latent = self._metric(v_neigh_embeds_h, v_embeds_h, et)
            neg_latent = self._metric(v_neigh_embeds_h, u_embeds_h, et)
            outputs.append(neg_latent - pos_latent)    
        
        model = tf.keras.models.Model(inputs=batch_nodes, outputs=outputs)
        print('model compiled')
        model.compile(tf.keras.optimizers.Adam(self.p['lr']), 
                      loss=self._rank_loss, 
                      metrics=[self._min_output, self._max_output]) 
        
        fake_y = np.zeros(len(nodes) * len(self.embed_types))\
                        .reshape(len(nodes), len(self.embed_types))
        train_ds = tf.data.Dataset.from_tensor_slices((nodes, fake_y))\
                        .batch(self.p['batch_size'])\
                        .shuffle(5000, reshuffle_each_iteration=True)\
                        .repeat(self.p['num_epochs'])
        print('fitting model') 
        model.fit(train_ds, verbose=2)
        print('Done Training')
        
        self.final_embeddings = []
        for et in self.embed_types:
            self.final_embeddings.append(self.embedding[et].get_weights()[0])
            print('final embeddings shape for', et, 
                  self.final_embeddings[-1].shape)
    def _run_nn_decoder(self, use_embeddings=False):                            
        decoder = NNEmbeddingDecoder(                                           
            self.final_embeddings,                                              
            np.array(self.G.edges()), 
            np.array(nx.complement(self.G).edges()),                                              
            use_embeddings=use_embeddings,                                      
            **self.nn_kwargs)                                                   
        return decode

    def get_edge_weights(self, edges, mode="nn", use_logistic=False):           
        self.nn_decoder = self._run_nn_decoder(use_embeddings=False)
        weights = self.nn_decoder.get_edge_logits(edges)                    
        if use_logistic:                                                        
            weights = logistic.cdf(weights)                                     
        return weights
