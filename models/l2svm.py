import sklearn.svm import LinearSVC
import numpy as np

class L2SVM:
    def __init__(self, C=1, use_fgpt=True, random_seed=None, **kwargs):
        assert use_fgpt
        self.random_seed = random_seed
        self.C = C
        print('L2SVM params -- C: %f\trandom_seed: %.r' % (C, random_seed))

    def learn_embedding(self, G, **kwargs):
        npairs = G.number_of_nodes() * (G.number_of_nodes() - 1)
        self.mapping = {}
        self.feature_vecs = np.zeros((npairs, len(G.nodes[0]['fingerprint']) * 3))
        labels = np.zeros(npairs) - 1
        
        k = 0
        for i in range(G.number_of_nodes() - 1):
            fpi = G.nodes[i]['fingerprint']
            for j in range(i + 1, G.number_of_nodes()):
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

 
        self.svm = LinearSVC(loss='hinge', C=self.C,
                             random_state=self.random_seed, verbose=1)
        self.svm.fit(common_feature_vecs, labels)

    def get_edge_weights(self, edges, **kwargs):
        ks1 = [self.mapping[(i, j)] for (i, j) in edges]
        ks2 = [self.mapping[(j, i)] for (j, i) in edges]
        ypred1 = self.svm.predict(ks1)
        ypred2 = self.svm.predict(ks2) 
        ypred = 0.5 * (ypred1 + ypred2)
        return ypred        

