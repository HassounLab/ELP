from collections import defaultdict
import numpy as np
from operator import itemgetter

def precision_at_ks(edges, pred_weights, true_edges, ks=None):
    if ks is None:
        ks = [2 ** i for i in range(10)]

    pred_edgelist = [(u, v, w) for ((u, v), w) in zip(edges, pred_weights)]
    pred_edgelist = sorted(pred_edgelist, key=itemgetter(2), reverse=True)

    prec_curve = []
    correct_edge = 0

    true_edges = set([(u, v) for u, v in true_edges])

    for i in range(min(ks[-1], len(pred_edgelist))):
        u, v, _ = pred_edgelist[i]
        if (u, v) in true_edges:
            correct_edge += 1
        if (i + 1) in ks:
            prec_curve.append((i + 1, correct_edge / (i + 1)))

    return prec_curve

def mean_average_precision(edges, pred_weights, true_edges):
    adjlist = defaultdict(list)
    for (u, v), w in zip(edges, pred_weights):
        adjlist[u].append((v, w))
        adjlist[v].append((u, w))
    true_edges = set([(u, v) for u, v in true_edges] + [(v, u) for v, u in true_edges])
    AP = []
    for u, vws in adjlist.items():
        vws = sorted(vws, key=itemgetter(1), reverse=True)
        precs = []
        num_correct = 0
        for i, (v, w) in enumerate(vws):
            if (u, v) in true_edges:
                num_correct += 1
                precs.append(num_correct / (i + 1))
            else:
                precs.append(0)
        if num_correct == 0:
            AP.append(0)
        else:
            AP.append(sum(precs) / num_correct)
    return np.mean(AP)

