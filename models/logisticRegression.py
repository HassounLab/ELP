import sklearn.linear_model as lm
from sklearn.metrics import make_scorer, roc_auc_score
import numpy as np

class LogisticRegression:
    def __init__(self, C=1, Cs=10, use_fgpt=True, random_seed=None, use_cv=False, **kwargs):
        assert use_fgpt
        self.random_seed = random_seed
        self.C = C
        self.Cs = Cs
        self.use_cv = use_cv
        if use_cv: 
            print('Logistic Regression (using CV) params\n\tCs:', Cs)
        else:
            print('Logistic Regression (not using CV) params\n\tC:', C)
    
        print('\trandom_seed:', random_seed)
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
    
        scoring = make_scorer(roc_auc_score, needs_proba=True)
        if self.use_cv:
            self.clf = lm.LogisticRegressionCV(
                    Cs=self.Cs, cv=3, random_state=self.random_seed, verbose=1, 
                    scoring=scoring)
        else:
            self.clf = lm.LogisticRegression(
                    C=self.C, verbose=1, random_state=self.random_seed) 
        
        self.clf.fit(self.feature_vecs, labels)

    def get_edge_scores(self, edges, **kwargs):
        fv1 = [self.feature_vecs[self.mapping[(i, j)]] for (i, j) in edges]
        fv2 = [self.feature_vecs[self.mapping[(j, i)]] for (i, j) in edges]
        ypred1 = self.clf.predict_proba(fv1)
        ypred2 = self.clf.predict_proba(fv2) 
        ypred = 0.5 * (ypred1 + ypred2)
        ypred = ypred[:, list(self.clf.classes_).index(1)]
        return ypred        

