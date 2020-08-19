from sklearn.svm import LinearSVC
from liblinearutil import train, predict, parameter
import numpy as np
import networkx as nx
from sklearn.metrics import roc_auc_score
class L2SVM:
    def __init__(self, C=1e-11, s=3, use_fgpt=True, random_seed=None, **kwargs):
        assert use_fgpt
        self.random_seed = random_seed
        self.C = C
        self.s = s
        print('L2SVM params -- C: %f\ts: %d\trandom_seed: %r' % (C, s, random_seed))
    def learn_embedding(self, G, **kwargs):
        print('l2svm G')
        print(nx.info(G))
        npairs = G.number_of_nodes() * (G.number_of_nodes() - 1)
        self.mapping = {}
        self.feature_vecs = np.zeros((npairs, len(G.nodes[0]['fingerprint']) * 3))
        labels = np.zeros(npairs) - 1
        
        k = 0
        nnodes = G.number_of_nodes()
        for i in range(nnodes - 1):
            fpi = G.nodes[i]['fingerprint']
            for j in range(i + 1, nnodes):
                fpj = G.nodes[j]['fingerprint']
                self.feature_vecs[k] = self._make_feature_vecs(fpi, fpj)
                labels[k] = labels[k + 1] = int(G.has_edge(i, j))
                self.mapping[(i, j)] = k
                k += 1
                self.feature_vecs[k] = self._make_feature_vecs(fpj, fpi)
                self.mapping[(j, i)] = k 
                k += 1
        assert np.all(labels >= 0)
        print('%d training instance' % (len(labels)))
        """
        self.svm = LinearSVC(loss='hinge', C=self.C, tol=0.1,
                             random_state=self.random_seed, verbose=1)
        self.svm.fit(self.feature_vecs, labels)
        print('Training completed')
        print('Accuracy on training set', self.svm.score(self.feature_vecs[:10], labels[:10]))
        """
        # -s 3 is l2 regularized l1 loss functino
        params = '-s 2 -C' # parameter selection
        params = '-s %d -c %f' % (self.s, self.C) 
        
        print('SVM param:', params)
        
        self.svm = train(labels, self.feature_vecs, params)
        print('SVM instance', self.svm)
        test_true_edges = list(G.edges())[:10]
        test_neg_edges = list(nx.complement(G).edges())[:10]
        yscore = self.get_edge_scores(test_true_edges + test_neg_edges)
        ylabel = np.concatenate((np.ones(10), np.zeros(10)))
        auc = roc_auc_score(ylabel, yscore)
        print('AUC on training set', auc)
        self.G = G

    def _get_feature_vecs(self, edge):
        try:
            return self.feature_vecs[self.mapping[edge]]
        except KeyError:
            i, j = edge
            fpi = self.G.nodes[i]['fingerprint']
            fpj = self.G.nodes[j]['fingerprint']
            return self._make_feature_vecs(fpi, fpj)

    def _make_feature_vecs(self, fpi, fpj):
        return np.concatenate((
            fpi & fpj, # common in both
            (fpi ^ fpj) & fpi, # only in fpi
            (fpi ^ fpj) & fpj)).astype(np.float32) # only in fpj                  
    
    def get_edge_scores(self, edges, **kwargs):
        fv1 = [self._get_feature_vecs((i, j)) for (i, j) in edges]
        fv2 = [self._get_feature_vecs((j, i)) for (i, j) in edges]
        """
        ypred1 = self.svm.decision_function(fv1)
        ypred2 = self.svm.decision_function(fv2) 
        """
        fakey = np.zeros(len(fv1))
        _, _, ypred1 = predict(fakey, fv1, self.svm)
        _, _, ypred2 = predict(fakey, fv2, self.svm)
        ypred1, ypred2 = np.array(ypred1), np.array(ypred2)
        #print('SVM decision function sample', ypred1[:20])
        ypred = 0.5 * (ypred1 + ypred2)
        return ypred        

