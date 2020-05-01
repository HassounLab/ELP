import sklearn
import numpy as np

class JaccardSimilarity:
    def __init__(self, fgpt_name="fingerprint", use_fgpt=True, **kwargs):
        self.fgpt_name = fgpt_name
        assert use_fgpt

    def learn_embedding(self, G, **kwargs):
        fgpts = []
        for n in G.nodes:
            fgpts.append(G.nodes[n][self.fgpt_name])
        self.fgpts = np.array(fgpts, dtype=np.float32)

    def get_edge_weights(self, edges, **kwargs):
        us, vs = edges[:, 0], edges[:, 1]
        fgpt_us = self.fgpts[us]
        fgpt_vs = self.fgpts[vs]
        jscore = []
        for fu, fv in zip(fgpt_us, fgpt_vs):
            jscore.append(sklearn.metrics.jaccard_score(fu, fv))
        return np.array(jscore)
