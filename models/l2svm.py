from sklearn.svm import LinearSVC
import numpy as np
import networkx as nx

class L2SVM:
    def __init__(self, C=1, use_fgpt=True, random_seed=None, **kwargs):
        assert use_fgpt
        self.random_seed = random_seed
        self.C = C
        print('L2SVM params -- C: %f\trandom_seed: %r' % (C, random_seed))
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
                self.feature_vecs[k] = np.concatenate((
                        fpi & fpj, # common in both
                        (fpi ^ fpj) & fpi, # only in fpi
                        (fpi ^ fpj) & fpj)) # only in fpj                  
                labels[k] = labels[k + 1] = int(G.has_edge(i, j))
                self.mapping[(i, j)] = k
                k += 1
                self.feature_vecs[k] = np.concatenate((
                        fpi & fpj,
                        (fpi ^ fpj) & fpj, 
                        (fpi ^ fpj) & fpi))
                self.mapping[(j, i)] = k 
                k += 1
        assert np.all(labels >= 0)

        self.feature_vecs = self.feature_vecs.astype(float) 
        labels = labels.astype(float)
        print('%d training instance' % (len(labels)))
        self.svm = LinearSVC(loss='hinge', C=self.C, tol=0.1,
                             random_state=self.random_seed, verbose=1)
        self.svm.fit(self.feature_vecs, labels)
        print('Training completed')
        print('Accuracy on training set', self.svm.score(self.feature_vecs[:50], labels[:50]))

    def get_edge_scores(self, edges, **kwargs):
        fv1 = [self.feature_vecs[self.mapping[(i, j)]] for (i, j) in edges]
        fv2 = [self.feature_vecs[self.mapping[(j, i)]] for (i, j) in edges]
        ypred1 = self.svm.decision_function(fv1)
        ypred2 = self.svm.decision_function(fv2) 
        print('SVM decision function sample', ypred1[:20])
        ypred = 0.5 * (ypred1 + ypred2)
        return ypred        

