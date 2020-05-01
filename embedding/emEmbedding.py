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
from embedding.nnEmbeddingDecoder import NNEmbeddingDecoder
from embedding.nnEmbeddingEnzymeDecoder import NNEmbeddingEnzymeDecoder
NODE_EMBED = "node_embed"
NODE_ATTR_EMBED = "node_attr_embed"
FGPT_EMBED = "fgpt_embed"
EDGE_ATTR_EMBED = "edge_attr_embed"
class EMEmbedding:
    def __init__(self, embed_size=128, num_epochs=500, batch_size=2048, 
                 gamma=10, lr=0.5, lr_mol_vae=0.0005, lr_node_attr=None, 
                 verbose=1, use_node_attr=True, use_fgpt=True, 
                 use_edge_attr=False, use_node_embed=True, load_node_embed=None,
                 beta=0.0, beta_edge=None, mol_vae_weight=1.0, stack_embeddings=False, 
                 classify_encodings=False, node_attr_postpone=0, 
                 fgpt_name="fingerprint", edge_name="rclass_int", edge_weight=1.0,
                 random_seed=None, **kwargs):
        self.embed_size = embed_size 
        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.gamma = gamma
        self.beta = beta
        self.beta_edge = beta if beta_edge is None else beta_edge
        self.lr = lr #learning rate
        self.lr_node_attr = lr_node_attr or lr
        self.lr_mol_vae = lr_mol_vae
        self.verbose = verbose
        self.use_node_embed = use_node_embed
        self.load_node_embed = load_node_embed
        self.use_fgpt = use_fgpt
        self.fgpt_name = fgpt_name
        self.edge_name = edge_name
        self.use_node_attr = use_node_attr
        self.node_attr_postpone = node_attr_postpone
        self.use_edge_attr = use_edge_attr
        self.edge_weight = edge_weight
        self.stack_embeddings = stack_embeddings
        self.mol_vae_kwargs = kwargs["mol_vae"] if "mol_vae" in kwargs else {}
        self.mol_vae_weight = mol_vae_weight
        self.classify_encodings = classify_encodings
        self.random_seed = random_seed
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
        if "random_seed" not in self.nn_kwargs:
            self.nn_kwargs["random_seed"] = random_seed
    def number_of_nodes(self):
        return self.num_nodes

    def get_embeddings(self):
        if self.stack_embeddings:
            return self.final_embeddings
        else:
            return np.hstack(self.final_embeddings)
    
    def learn_embedding(self, G, undirected=True, embedding_savepath=None, 
                        neg_G=None, val_G=None, **kwargs):
        tf.reset_default_graph()
        self._read_graph(G, undirected=undirected, neg_G=neg_G, val_G=val_G) 
        self._run(embedding_savepath)

    def load_embedding(self, G, load):
        print("Loading pre trained embeddings from", load)
        self._read_graph(G)
        self.final_embeddings = np.load(load)

    def print_summary(self):
        print("Running Embedding Propagation model with the following hyperparams:")
        self._print_params() 

    def _print_params(self):
        print("\tembed_size=%d\n\tnum_epochs=%d\n\tbatch_size=%d\n\tgamma=%d\n\t"
              "lr=%f\n\tlr_node_attr=%f\n\tlr_mol_vae=%f\n\tbeta=%f\n\t"
              "beta_edge=%f\n\t"
              "use_node_embed=%r\n\tload_node_embed=%r\n\tuse_fgpt=%r\n\t"
              "use_node_attr=%r\n\tuse_edge_attr=%r\n\t"
              "mol_vae_weight=%s\n\tstack_embeddings=%r\n\tfgpt_name=%s\n\t"
              "edge_name=%s\n\tedge_weight=%f\n\trandom_seed=%r"
              % (self.embed_size, self.num_epochs, self.batch_size, self.gamma,
                 self.lr, self.lr_node_attr, self.lr_mol_vae, self.beta, 
                 self.beta_edge,
                 self.use_node_embed, self.load_node_embed, self.use_fgpt,
                 self.use_node_attr, self.use_edge_attr, self.mol_vae_weight,
                 self.stack_embeddings, self.fgpt_name, self.edge_name, 
                 self.edge_weight, self.random_seed))

    def _read_graph(self, G=None, undirected=True, neg_G=None, val_G=None, 
                    **kwargs):

        assert G is not None
        self.G = G
        if self.verbose:
            self.print_summary()
        self.edges = np.array(G.edges())
        if neg_G is None:
            self.neg_edges = np.array(nx.complement(G).edges())
        else:
            self.neg_edges = np.array(neg_G.edges())
        self.num_edges = G.number_of_edges()
        adjmat_np = nx.adjacency_matrix(G).todense()
        assert np.all(np.sum(adjmat_np, axis=1) > 0)
        assert not np.any(np.isnan(adjmat_np))
        if val_G:
            print("using val edges")
            self.val_edges = np.array(val_G.to_undirected().edges())[:self.batch_size]
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
        self.trainable_nodes = np.array(self.trainable_nodes)
        self.undirected = undirected
        if self.verbose:
            print("Graph contains %d nodes, %d trainable_nodes and %d edges"
                  % (self.num_nodes, len(self.trainable_nodes), self.num_edges))
        self.num_embeds = {}

        if self.use_node_embed:
            self.num_embeds[NODE_EMBED] = self.num_nodes
            self.embed_types.append(NODE_EMBED)
        if self.use_node_attr or self.use_fgpt:
            fgpts = []
            for n in G.nodes:
                fgpts.append(G.nodes[n][self.fgpt_name].reshape((-1, 1)))
            fgpts = np.array(fgpts, dtype=np.float32)
            assert not np.any(np.isnan(fgpts))
            assert np.all(np.sum(fgpts, axis=1) > 0)
            self.fgpts = tf.constant(fgpts, dtype=tf.float32) 
            self.num_embeds[FGPT_EMBED] = len(fgpts[0])
            print("Fgpt length", self.num_embeds[FGPT_EMBED])
            self.embed_types.append(FGPT_EMBED)
        if self.use_edge_attr:
            num_edge_attrs = 0
            for u, v, ri in G.edges(data=self.edge_name):
                if undirected and u > v:
                    continue
                if ri is not None:
                    assert isinstance(ri, int)
                    ri += 1
                    num_edge_attrs = max(num_edge_attrs, ri)
                    assert adjmat_np[u, v] 
                    adjmat_np[u, v] = ri
                    if undirected:
                        assert adjmat_np[v, u]
                        adjmat_np[v, u] = ri
            assert num_edge_attrs > 0, "no edge attrs found"
            num_edge_attrs += 1
            self.num_embeds[EDGE_ATTR_EMBED] = num_edge_attrs
            print("Using edge attrs, %d classes" % num_edge_attrs)    
    
        self.adjmat = tf.constant(adjmat_np, dtype=tf.int64) # TODO float problem?
        self.nn_decoder, self.nn_inner_decoder = None, None 
        self.nn_enzyme_decoder = None
    def _get_embeds_h(self, batch_nodes, et, fgpt_feats=None):
        if et == NODE_EMBED:
            return tf.nn.embedding_lookup(self.embeddings[et], batch_nodes)
        elif et == NODE_ATTR_EMBED:
            x_in = tf.gather(self.smiles_hot, batch_nodes)
            return self.mol_vae_model.embed(x_in)
        elif et == FGPT_EMBED:
            if fgpt_feats is None:
                fgpt_feats =  tf.nn.embedding_lookup(self.fgpts, batch_nodes)
                #fgpt_feats = tf.gather(self.fgpts, batch_nodes)
            #embeds = tf.matmul(fgpt_feats, self.embeddings[et])
            embeds = tf.reduce_sum(
                fgpt_feats * self.embeddings[et], axis=1)
            cardinality = tf.reduce_sum(fgpt_feats, axis=1)
            cardinality = tf.maximum(cardinality, 1)
            #cardinality = tf.reshape(cardinality, (-1, 1))
            norm_embeds = embeds / cardinality
            return norm_embeds
        else:
            raise ValueError("embed type not valid")
    def _get_edge_embeds_h(self, v, u_nodes):
        #with tf.device('/gpu:0')i:
        if True:
            v = tf.cast(v, dtype=tf.float32)
            v_nodes = tf.tile(v, [tf.shape(u_nodes)[0]])
            v_nodes = tf.cast(v_nodes, dtype=tf.int64)
            v_nodes_p = tf.reshape(tf.minimum(v_nodes, u_nodes), (-1, 1))
            u_nodes_p = tf.reshape(tf.maximum(v_nodes, u_nodes), (-1, 1))
            edges = tf.concat((v_nodes_p, u_nodes_p), axis=1)
            lookup_ids = tf.gather_nd(self.adjmat, edges)
            lookup_ids = tf.cast(lookup_ids, dtype=tf.int64)
            
            es = tf.nn.embedding_lookup(self.embeddings[EDGE_ATTR_EMBED], 
                                        lookup_ids)
            print("edge embeds", es)
            return es
            #return tf.map_fn(apply_edges, edges, dtype=tf.float32)
        # return edge attr embeds, or 0 if no edge attr

    def _get_neigh_embeds_h(self, v_nodes, u_nodes, et, is_training, 
                            tf_device=0):
        if et not in self.embed_types:
            raise ValueError("invalid embed type")
        
        #with tf.device('/gpu:%d' % tf_device):
        if  True:
            def apply_neighbor(x):
                v = x[0]
                adj_row = x[1]
                print("apply neighbor v shape", tf.shape(v))
                v = tf.expand_dims(v, 0)
                #adj_row = tf.squeeze(tf.gather(self.adjmat, v))
                where = tf.cond(is_training,
                                lambda : tf.greater(adj_row, 0),
                                lambda : tf.not_equal(adj_row, 0))
                #import IPython
                #IPython.embed()
                neigh_nodes = tf.squeeze(tf.where(where), axis=1)
                num_neigh_nodes = tf.shape(neigh_nodes)[0]
                neigh_node_es = self._get_embeds_h(neigh_nodes, et)
                if self.use_edge_attr:
                    #v_tiled = tf.tile(v, [num_neigh_nodes])
                    neigh_node_es += self._get_edge_embeds_h(v, neigh_nodes)
                e = tf.reduce_sum(neigh_node_es, axis=0)
                e /= tf.cast(num_neigh_nodes, dtype=tf.float32)
                print("e shape", tf.shape(e))
                return e
            adjrows = tf.gather(self.adjmat, v_nodes) 
            es = tf.map_fn(apply_neighbor, (v_nodes, adjrows), dtype=tf.float32)
            print('es shape', tf.shape(es))
            return es
    
    def _metric(self, h1, h2, et):
        return tf.reduce_sum(h1 * h2, axis=1) # dot product
        
    def _rank_loss(self, pos_latent, neg_latent, embed_type):
        return tf.reduce_mean(tf.nn.relu(self.gamma + neg_latent - pos_latent))
    
    def _edge_embed_assign_zeros_row_one(self):
        assert self.embeddings[EDGE_ATTR_EMBED]
        dtype = self.embeddings[EDGE_ATTR_EMBED].dtype
        zeros = tf.zeros(shape=[self.embed_size], dtype=dtype) 
        self.embeddings[EDGE_ATTR_EMBED][0].assign(zeros)

    def _make_embed_variables(self, embed_type, make_regularizer=True):
        if embed_type == EDGE_ATTR_EMBED:
            initializer = tf.zeros_initializer()
        else:
            initializer = tf.glorot_uniform_initializer()

        self.embeddings[embed_type] = tf.get_variable(
            "%s_embeds" % embed_type, 
            shape=[self.num_embeds[embed_type], self.embed_size],
            initializer=initializer) 

        if make_regularizer:
            if embed_type == EDGE_ATTR_EMBED:
                beta = self.beta_edge
            else:
                beta = self.beta
                
            self.regs[embed_type] = beta * \
                tf.nn.l2_loss(self.embeddings[embed_type])

    def _run(self, embedding_savepath=None):
        tf.keras.backend.set_learning_phase(1)
        if self.random_seed is not None:
            tf.set_random_seed(self.random_seed)
        self.embeddings = {}
        self.regs = {}
        
        run_on_epoch_start = {}
 
        for et in self.embed_types:
            self._make_embed_variables(et)
        
        self.batch_nodes = tf.placeholder(tf.int64, shape=[None])

        batch_neg_nodes = tf.random_uniform(
                shape=tf.shape(self.batch_nodes),
                maxval=self.num_nodes,
                dtype=tf.int64)
        if self.use_edge_attr:
            self._make_embed_variables(EDGE_ATTR_EMBED)
        self.pos_latent, self.neg_latent, self.loss = {}, {}, {}
        self.run_ops_dict = OrderedDict({})
        self.run_ops_dict["total_loss"] = None
        self.eval_datums = {}
        self.is_training = tf.placeholder_with_default(True, shape=())
        for et in self.embed_types:
            v_embeds_h = self._get_embeds_h(self.batch_nodes, et)
            u_embeds_h = self._get_embeds_h(batch_neg_nodes, et)
            
            v_neigh_embeds_h = self._get_neigh_embeds_h(
                self.batch_nodes, batch_neg_nodes, et, self.is_training)
            print('got v_neigh_embeds_h')
            self.pos_latent[et] = self._metric(v_neigh_embeds_h, v_embeds_h, et)
            print('computed pos latent')
            self.neg_latent[et]  = self._metric(v_neigh_embeds_h, u_embeds_h, et)
           
            self.loss[et] = self._rank_loss(
                self.pos_latent[et], self.neg_latent[et], et)
            print('computed rank loss')
            self.run_ops_dict.update(OrderedDict([
                ("loss_%s" % et, self.loss[et])]))
            
            #if et != NODE_ATTR_EMBED: 
            """
            self.eval_datums["%s_embeddings_max" % et] = tf.reduce_max(self.embeddings[et])
            self.eval_datums['%s_embeddings_min' % et] = tf.reduce_min(self.embeddings[et])
            self.eval_datums['%s_pos_latent_max' % et] = tf.reduce_max(self.pos_latent[et])
            self.eval_datums['%s_pos_latent_min' % et] = tf.reduce_min(self.pos_latent[et])
            self.eval_datums['%s_neg_latent_max' % et] = tf.reduce_max(self.neg_latent[et])
            self.eval_datums['%s_neg_latent_min' % et] = tf.reduce_min(self.neg_latent[et])
            """
        self.run_opt = []
        if self.use_node_embed or self.use_fgpt:
            loss = tf.reduce_sum([self.loss[et] for et in self.embed_types \
                                  if et != NODE_ATTR_EMBED])
            reg = tf.reduce_sum(list(self.regs.values()))
            self.run_ops_dict["reg"] = reg
            self.obj = loss + reg
            total_loss = self.obj
            opt = tf.train.AdamOptimizer(self.lr).minimize(self.obj)
            self.run_opt.append(opt)
        else:
            total_loss = 0
        self.run_ops_dict["total_loss"] = total_loss 
        self.nodes = self.trainable_nodes
        if self.use_edge_attr: 
            self._edge_embed_assign_zeros_row_one()

        def print_losses(loss_dict):
            print("\n\t", ", ".join(["%s: %.4f" % (k, v) for k, v in loss_dict.items()]))
        def print_eval_data(data_name, eval_data):
            print(data_name, eval_data)
            #print("\t%s: max %.3f min %.3f mean %.3f"
            #      % (data_name, np.max(eval_data), np.min(eval_data), np.mean(eval_data)))
        
        
            
        starttime = time.time()
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True

        self.acc_loss_values = {k: 0 for k in self.run_ops_dict}
        if self.run_ops_dict == [] and self.run_opt == []:
            self.final_embeddings = []
        
        self.sess = sess = tf.Session()

        sess.run(tf.global_variables_initializer())
        min_train_loss, main_val_loss = float("inf"), float("inf")
        for epoch in range(self.num_epochs):
            if self.use_node_attr and epoch >= self.node_attr_postpone \
               and opt_node_attr not in self.run_opt:
                print("beginning to optimize node attr")
                self.run_opt.append(opt_node_attr)
            if self.verbose:
                print("Epoch %d/%d" % (epoch + 1, self.num_epochs))
            if run_on_epoch_start != {}:
                print(", ".join(["%s: %f" % (k, sess.run(v)) for \
                                 k, v in run_on_epoch_start.items()]))

            for k in self.acc_loss_values:
                self.acc_loss_values[k] = 0
        
            for i, feed_dict in enumerate(self._generate_batch()):
                sess.run(self.run_opt, feed_dict=feed_dict)
                losses = sess.run(list(self.run_ops_dict.values()), feed_dict=feed_dict)
                for j, k in enumerate(self.run_ops_dict):
                    self.acc_loss_values[k] += losses[j]
                
                if self.verbose > 3 or (self.verbose > 2 and (i == 0 or (i + 1) % 50  == 0)):
                    loss_dict = {k: losses[j] for j, k in enumerate(self.acc_loss_values)}
                    print("batch: %d " % (i + 1))
                    print_losses(loss_dict)
                    for data_name, data in self.eval_datums.items():
                        eval_data = sess.run(data, feed_dict=feed_dict)
                        print_eval_data(data_name, eval_data)
            if self.verbose > 1:
                for data_name, data in self.eval_datums.items():
                    eval_data = sess.run(data, feed_dict=feed_dict)
                    print_eval_data(data_name, eval_data)        
        
            for k in self.acc_loss_values:
                self.acc_loss_values[k] /= (i + 1) 
        
            epoch_train_loss = self.acc_loss_values["total_loss"]

            if self.verbose: 
                print_losses(self.acc_loss_values)
                if (epoch + 1) % 50 == 0:
                    print("Time so far: %fs" % (time.time() - starttime))
            else:
                    print("Total loss %0.4f" % epoch_train_loss)

            if epoch_train_loss < min_train_loss:
                min_train_loss = epoch_train_loss
                if self.verbose:
                    print("New best training loss")
            if self.val_edges is not None:
                losses = sess.run(
                    list(self.run_ops_dict.values()), 
                    feed_dict={self.batch_nodes: self.val_edges[:, 0],
                               self.is_training: False})
                self.val_edges = np.flip(self.val_edges, axis=1)
                print("validation losses")
                loss_dict = {k: losses[j] for j, k in enumerate(self.acc_loss_values)}
                print_losses(loss_dict)
                #save_path = saver.save(sess, "/tmp/model.ckpt")
                

        #saver.restore(sess, "/tmp/model.ckpt")
        
        run_ops = []
        for et in self.embed_types:
            run_ops.append(self._get_embeds_h(self.batch_nodes, et))
        self.nodes = np.arange(self.num_nodes)
        self.final_embeddings = []
        for et in self.embed_types:
            self.final_embeddings.append(np.empty(
            (self.num_nodes, self.embed_size), dtype=np.float32))
        for j, feed_dict in enumerate(self._generate_batch(use_orig=True)):
            batch_embeds = sess.run(run_ops, feed_dict=feed_dict)
            i = j * self.batch_size
            for et_i in range(len(self.embed_types)):
                self.final_embeddings[et_i][i: i + self.batch_size] = batch_embeds[et_i]
        endtime = time.time()
        print("Time taken to learn graph embeddings: %fs" % (endtime - starttime))
        
        if self.stack_embeddings:
            self.final_embeddings = np.concatenate(self.final_embeddings, axis=1)
        
        tf.keras.backend.set_learning_phase(0)
    
    def _get_fgpt_embeddings_new_nodes(self, nodes, fgpt_feats):
        embeddings = np.empty((len(nodes), self.embed_size))
        for i in range(0, len(nodes), self.batch_size):
            embeddings[i:i + self.batch_size] = sess.run(
                self.get_embeds_h(self.batch_nodes, FGPT_EMBED, fgpt_feats=fgpt_feats),
                feed_dict={self.batch_nodes: nodes[i:i + self.batch_size]})
        return embeddings

    def _generate_batch(self, shuffle=True, drop_last=False, use_orig=False):
        if shuffle and not use_orig:
            np.random.shuffle(self.nodes)
        elif use_orig:
            self.nodes = np.arange(self.num_nodes)
        for i in range(0, len(self.nodes), self.batch_size):
            if drop_last and i + self.batch_size > self.num_nodes:
                break
            batch_nodes = self.nodes[i:i + self.batch_size]
            if drop_last:
                assert len(batch_nodes) == self.batch_size
            #print(batch_nodes)
            yield {self.batch_nodes: batch_nodes}
                   #self.epsilon: np.random.normal(
                   #     size=(len(batch_nodes), self.embed_size))}
    
    def _run_nn_decoder(self, use_embeddings=False):
        decoder = NNEmbeddingDecoder(
            self.final_embeddings,
            self.edges,
            self.neg_edges,
            use_embeddings=use_embeddings,
            **self.nn_kwargs)
        return decoder
    def evaluate_enzyme_label_prediction(self, test_G):

        train_edges, train_labels = [], []
        for i, (u, v) in enumerate(self.G.edges):
            if self.G.edges[u, v][self.edge_name] is not None:
                train_edges.append((u, v))
                train_labels.append(self.G.edges[u, v][self.edge_name])

        test_edges, test_labels = [], []
        for i, (u, v) in enumerate(test_G.edges):
            if test_G.edges[u, v][self.edge_name] is not None and \
                    test_G.edges[u, v][self.edge_name] in train_labels:
                test_edges.append((u, v))
                test_labels.append(test_G.edges[u, v][self.edge_name])
        train_edges = np.array(train_edges)
        train_labels = np.array(train_labels) 
        test_edges = np.array(test_edges)
        test_labels = np.array(test_labels)
        NNEmbeddingEnzymeDecoder(self.final_embeddings, train_edges, 
                                 train_labels, test_edges, test_labels, 
                                 **self.nn_kwargs)


    def get_edge_weights(self, edges, mode="nn", use_logistic=False): 
        if mode == "nn":
            if not self.nn_decoder:
                with tf.variable_scope("nn_decoder"):
                    self.nn_decoder = self._run_nn_decoder(use_embeddings=False)
            weights = self.nn_decoder.get_edge_logits(edges)
        elif mode == "nn_inner":
            if not self.nn_inner_decoder:
                with tf.variable_scope("nn_inner_decoder"):
                    self.nn_inner_decoder = self._run_nn_decoder(use_embeddings=True)
                weights = self.nn_inner_decoder.get_edge_logits(edges)
            weights = np.hstack(weights)
        else:
            raise NotImplementedError(mode)
        if use_logistic:
            weights = logistic.cdf(weights)
        return weights
