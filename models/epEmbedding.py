import numpy as np
#import tensorflow.compat.v1 as tf
#tf.disable_v2_behavior()
import tensorflow as  tf
print(tf.__version__)
import networkx as nx
from collections import OrderedDict
import math
import os
import sys
import time
import pprint
import pickle


from scipy.stats import logistic
from models.nnEmbeddingDecoder import NNEmbeddingDecoder
NODE_EMBED = "node_embed"
FGPT_EMBED = "fgpt_embed"
EDGE_ATTR_EMBED = "edge_attr_embed"
default_params = {
     'embed_size': 128,
     'num_epochs': 500,
     'batch_size': 2048,
     'gamma': 10,
     'lr': 0.01,
     'verbose': 1,
     'use_fgpt': True,
     'use_edge': False,
     'use_node_embed': True,
     'beta': 0.0002,
     'beta_edge': 0.0002,
     'random_seed': 2020,
     'early_stopping' : 50,
     'decoder': {
        'num_epochs': 40,
        'batch_size': 2048,
        'beta': 0,
        'verbose': 2,
        'lr': 0.01,
        'hidden_sizes': [32, 16],
        'random_seed': 2020
    }
}

class EPEmbedding:
    def __init__(self, **params):
        print('Embedding propagation')
        self.p = default_params
        for k in default_params:
            if k != 'decoder' and k in params:
                self.p[k] = params[k]
        if 'decoder' in params:
            for k in default_params['decoder']:
                if k in params['decoder']:
                    self.p['decoder'][k] = params['decoder'][k]
        
        assert self.p['use_node_embed'] or self.p['use_fgpt']

        print('\n'.join('\t%s: %r' % (k, v) for k, v in self.p.items()))
    def get_embeddings(self):
        return np.hstack(self.final_embeddings)
    
    def learn_embedding(self, G, neg_G=None, val_G=None):
        tf.reset_default_graph()
        self._read_graph(G, neg_G=neg_G, val_G=val_G) 
        self._run()


    def _read_graph(self, G=None, undirected=True, neg_G=None, val_G=None, 
                    **kwargs):

        assert G is not None

        self.edges = np.array(G.edges())
        if neg_G is None:
            self.neg_edges = np.array(nx.complement(G).edges())
        else:
            self.neg_edges = np.array(neg_G.edges())
        
        adjmat_np = nx.adjacency_matrix(G).todense()
        assert not np.any(np.isnan(adjmat_np))
        self.adjmat = tf.constant(adjmat_np, dtype=tf.float32) # TODO float problem?
        
        if val_G:
            print("using val edges")
            self.val_edges = np.array(val_G.to_undirected().edges())
            self.val_edges = self.val_edges[:self.p['batch_size']]
            for u, v in self.val_edges:
                adjmat_np[u, v] = adjmat_np[v, u] = -1
        else:
            print("no val edges")
            self.val_edges = None
        
        self.embed_types = []
        self.num_nodes = G.number_of_nodes()
        self.trainable_nodes = []
        for n in range(self.num_nodes):
            if G.degree(n) > 0:
                self.trainable_nodes.append(n)
        self.nodes = self.trainable_nodes = np.array(self.trainable_nodes)
        
        if self.p['verbose']:
            print("Graph contains %d nodes, %d trainable_nodes and %d edges"
                  % (self.num_nodes, len(self.trainable_nodes), G.number_of_edges()))
        
        self.num_embeds = {}

        if self.p['use_node_embed']:
            self.num_embeds[NODE_EMBED] = self.num_nodes
            self.embed_types.append(NODE_EMBED)
        
        if self.p['use_fgpt']:
            fgpts = []
            for n in G.nodes:
                fgpts.append(G.nodes[n]['fingerprint'].reshape((-1, 1)))
            fgpts = np.array(fgpts, dtype=np.float32)
            assert not np.any(np.isnan(fgpts))
            assert np.all(np.sum(fgpts, axis=1) > 0)
            self.fgpts = tf.constant(fgpts, dtype=tf.float32) 
            self.num_embeds[FGPT_EMBED] = len(fgpts[0])
            print("Fgpt length", self.num_embeds[FGPT_EMBED])
            self.embed_types.append(FGPT_EMBED)
        
        if self.p['use_edge']:
            num_edge_attrs = 0
            for u, v, ri in G.edges(data='edge_attr'):
                """
                if undirected and u > v:
                    continue
                """
                if ri is not None:
                    #assert isinstance(ri, int)
                    ri += 1
                    num_edge_attrs = max(num_edge_attrs, ri)
                    adjmat_np[u, v] = adjmat_np[v, u] = ri
                    """
                    assert adjmat_np[u, v] 
                    adjmat_np[u, v] = ri
                    if undirected:
                        assert adjmat_np[v, u]
                        adjmat_np[v, u] = ri
                    """
            assert num_edge_attrs > 0, "no edge attrs found"
            num_edge_attrs += 1
            self.num_embeds[EDGE_ATTR_EMBED] = num_edge_attrs
            print("Using edge attrs, %d classes" % num_edge_attrs)    
    
    def _get_embeds_h(self, batch_nodes, et, fgpt_feats=None):
        if et == NODE_EMBED:
            return tf.nn.embedding_lookup(self.embeddings[et], batch_nodes)
        elif et == FGPT_EMBED:
            if fgpt_feats is None:
                fgpt_feats =  tf.nn.embedding_lookup(self.fgpts, batch_nodes)
            embeds = tf.reduce_sum(
                fgpt_feats * self.embeddings[et], axis=1)
            cardinality = tf.reduce_sum(fgpt_feats, axis=1)
            cardinality = tf.maximum(cardinality, 1)
            norm_embeds = embeds / cardinality
            return norm_embeds
        else:
            raise ValueError("embed type not valid")
    
    def _get_edge_embeds_h(self, v, u_nodes):
        v = tf.expand_dims(v, 0)
       # v = tf.cast(v, dtype=tf.float32)
        v_nodes = tf.tile(v, [tf.shape(u_nodes)[0]])
        v_nodes = tf.cast(v_nodes, dtype=tf.int64)
        v_nodes = tf.reshape(v_nodes, (-1, 1))
        u_nodes = tf.reshape(u_nodes, (-1, 1))
        #v_nodes_p = tf.reshape(tf.minimum(v_nodes, u_nodes), (-1, 1))
        #u_nodes_p = tf.reshape(tf.maximum(v_nodes, u_nodes), (-1, 1))
        edges = tf.concat((v_nodes, u_nodes), axis=1)
        lookup_ids = tf.gather_nd(self.adjmat, edges)
        lookup_ids = tf.cast(lookup_ids, dtype=tf.int64)
        
        es = tf.nn.embedding_lookup(self.embeddings[EDGE_ATTR_EMBED], 
                                    lookup_ids)
        print("edge embeds", es)
        return es

    def _get_neigh_embeds_h(self, v_nodes, et, is_training): 
        if et not in self.embed_types:
            raise ValueError("invalid embed type")
        
        def apply_neighbor(x):
            v = x[0]
            adj_row = x[1]
            #print("apply neighbor to adj_row", adj_row)
           
            where = tf.cond(is_training,
                            lambda : tf.greater(adj_row, 0),
                            lambda : tf.not_equal(adj_row, 0))
            neigh_nodes = tf.squeeze(tf.where(where), axis=1)
            #print('neigh_nodes', neigh_nodes)
            num_neigh_nodes = tf.shape(neigh_nodes)[0]
            #print('num neigh nodes shape (expect singleton)', num_neigh_nodes)
            num_neigh_nodes = tf.maximum(num_neigh_nodes, 1) 
            neigh_node_es = self._get_embeds_h(neigh_nodes, et)
            if self.p['use_edge']:
                #v_tiled = tf.tile(v, [num_neigh_nodes])
                neigh_node_es += self._get_edge_embeds_h(v, neigh_nodes)
            e = tf.reduce_sum(neigh_node_es, axis=0)
            e /= tf.cast(num_neigh_nodes, dtype=tf.float32)
            #print("e shape", e)
            return e
        adjrows = tf.gather(self.adjmat, v_nodes) 
        es = tf.map_fn(apply_neighbor, (v_nodes, adjrows), dtype=tf.float32)
        #print('es shape', es)
        return es
    
    def _metric(self, h1, h2, et):
        return tf.reduce_sum(h1 * h2, axis=1) # dot product
        
    def _rank_loss(self, pos_latent, neg_latent, embed_type):
        return tf.reduce_mean(tf.nn.relu(self.p['gamma'] + neg_latent - pos_latent))
    
    def _edge_embed_assign_zeros_row_one(self):
        assert self.embeddings[EDGE_ATTR_EMBED]
        dtype = self.embeddings[EDGE_ATTR_EMBED].dtype
        zeros = tf.zeros(shape=[self.p['embed_size']], dtype=dtype) 
        self.embeddings[EDGE_ATTR_EMBED][0].assign(zeros)

    def _make_embed_variables(self, embed_type):
        if embed_type == EDGE_ATTR_EMBED:
            initializer = tf.zeros_initializer()
        else:
            initializer = tf.glorot_uniform_initializer()

        self.embeddings[embed_type] = tf.get_variable(
            "%s_embeds" % embed_type, 
            shape=[self.num_embeds[embed_type], self.p['embed_size']],
            initializer=initializer) 

        beta = self.p['beta_edge'] if embed_type == EDGE_ATTR_EMBED else self.p['beta']
        return beta * tf.nn.l2_loss(self.embeddings[embed_type])
    def _run(self):
        tf.keras.backend.set_learning_phase(1)
        if self.p['random_seed'] is not None:
            tf.set_random_seed(self.p['random_seed'])
        self.embeddings = {}
        regs = [] 
 
        for et in self.embed_types:
            regs.append(self._make_embed_variables(et))
        
        self.batch_nodes = tf.placeholder(tf.int64, shape=[None])
        #print('batch nodes shape', self.batch_nodes)
        batch_neg_nodes = tf.random_uniform(
                shape=tf.shape(self.batch_nodes),
                maxval=self.num_nodes,
                dtype=tf.int64)
        if self.p['use_edge']:
            self._make_embed_variables(EDGE_ATTR_EMBED)
        losses = {}
        self.run_ops_dict = OrderedDict({'obj': 0})
        is_training = tf.placeholder_with_default(True, shape=())
        for et in self.embed_types:
            v_embeds_h = self._get_embeds_h(self.batch_nodes, et)
            #print('v_embeds_h.shape', v_embeds_h)
            u_embeds_h = self._get_embeds_h(batch_neg_nodes, et)
            #print('u_embeds_h.shape', u_embeds_h)
            v_neigh_embeds_h = self._get_neigh_embeds_h(
                self.batch_nodes, et, is_training)
            #print('v_neigh_embeds_h.shape', v_neigh_embeds_h)
            pos_latent = self._metric(v_neigh_embeds_h, v_embeds_h, et)
            #print('pos latent shape', pos_latent)
            neg_latent = self._metric(v_neigh_embeds_h, u_embeds_h, et)
           
            losses[et] = self._rank_loss(pos_latent, neg_latent, et)
            #print('computed rank loss')
            self.run_ops_dict.update(OrderedDict([
                ("loss_%s" % et, losses[et]),
                ('v_embeds_h_%s' % et, tf.reduce_mean(v_embeds_h)),
                ('u_embeds_h_%s' % et, tf.reduce_mean(u_embeds_h)),
                ('v_neigh_embeds_h_%s' % et, tf.reduce_mean(v_neigh_embeds_h))
            ]))
          
            
       
        loss = tf.reduce_sum(list(losses.values()))
        self.run_ops_dict["reg"] = reg = tf.reduce_sum(regs)
        self.run_ops_dict['obj'] = obj = loss + reg
        
        optimizer = tf.train.AdamOptimizer(self.p['lr']).minimize(obj)
        if self.p['use_edge']: 
            self._edge_embed_assign_zeros_row_one()

            
        starttime = time.time()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        acc_loss_values = {k: 0 for k in self.run_ops_dict}
        
        sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        min_train_loss, main_val_loss = float("inf"), float("inf")
        best_loss_epoch = 0
        for epoch in range(self.p['num_epochs']):
            print("Epoch %d/%d" % (epoch + 1, self.p['num_epochs']))
        
            for i, feed_dict in enumerate(self._generate_batch()):
            
                losses = sess.run(list(self.run_ops_dict.values()), feed_dict=feed_dict)
                sess.run(optimizer, feed_dict=feed_dict)
                for j, k in enumerate(self.run_ops_dict):
                    acc_loss_values[k] += losses[j]
                if self.p['verbose'] > 2:
                    print('Batch', i + 1)
                    batch_losses_str = ', '.join(['%s: %.2f' % (k, l) for k, l in \
                                                   zip(self.run_ops_dict, losses)])
                    print('\t',  batch_losses_str)
                    
            for k in acc_loss_values:
                acc_loss_values[k] /= (i + 1) 
        
            epoch_train_loss = acc_loss_values["obj"]

            print("\n\t",
                  ", ".join(["%s: %.4f" % (k, v) for k, v in acc_loss_values.items()]))
            if (epoch + 1) % 50 == 0:
                print('Tme (%fs)' % (time.time() - starttime))
            
            if epoch_train_loss < min_train_loss:
                min_train_loss = epoch_train_loss
                if self.p['verbose'] > 1:
                    print("New best training loss")
                best_loss_epoch = epoch
            elif epoch - best_loss_epoch >= self.p['early_stopping']:
                print('Early stopping')
                break
            if self.val_edges is not None:
                losses = sess.run(
                    list(self.run_ops_dict.values()), 
                    feed_dict={self.batch_nodes: self.val_edges[:, 0],
                               is_training: False})
                self.val_edges = np.flip(self.val_edges, axis=1)
                print("validation losses")
                loss_dict = {k: losses[j] for j, k in enumerate(acc_loss_values)}
                print("\n\t", ", ".join(["%s: %.4f" % (k, v) for k, v in loss_dict.items()]))

        
        run_ops = []
        self.nodes = np.arange(self.num_nodes)
        self.final_embeddings = []
        for et in self.embed_types:
            run_ops.append(self._get_embeds_h(self.batch_nodes, et))
            self.final_embeddings.append(np.empty(
                (self.num_nodes, self.p['embed_size']), dtype=np.float32))
        for j, feed_dict in enumerate(self._generate_batch(use_orig=True)):
            batch_embeds = sess.run(run_ops, feed_dict=feed_dict)
            i = j * self.p['batch_size']
            for et_i in range(len(self.embed_types)):
                self.final_embeddings[et_i][i: i + self.p['batch_size']] = batch_embeds[et_i]
        endtime = time.time()
        print("Time taken to learn graph embeddings: %fs" % (endtime - starttime))
        
        tf.keras.backend.set_learning_phase(0)
        self.decoder = None
    def get_embeddings(self, nodes=None):
        if nodes is None:
            return np.concatenate(self.final_embeddings, axis=1)
        return np.concatenate([e[nodes] for e in self.final_embeddings], axis=1)
     
    def _generate_batch(self, shuffle=True, drop_last=False, use_orig=False):
        if shuffle and not use_orig:
            np.random.shuffle(self.nodes)
        elif use_orig:
            self.nodes = np.arange(self.num_nodes)
        for i in range(0, len(self.nodes), self.p['batch_size']):
            if drop_last and i + self.p['batch_size'] > self.num_nodes:
                break
            batch_nodes = self.nodes[i:i + self.p['batch_size']]
            if drop_last:
                assert len(batch_nodes) == self.p['batch_size']
            yield {self.batch_nodes: batch_nodes}
    

    def get_edge_scores(self, edges, use_logistic=False):
        if self.decoder is None:
            with tf.variable_scope("nn_decoder"):
                self.decoder = NNEmbeddingDecoder(
                    self.final_embeddings,
                    self.edges,
                    self.neg_edges,
                    **self.p['decoder'])
        weights = self.decoder.get_edge_logits(edges)
        if use_logistic:
            weights = logistic.cdf(weights)
        return weights
