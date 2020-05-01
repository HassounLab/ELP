import numpy as np
import networkx as nx

precision_pos = [2, 4, 8, 16, 32, 64, 128]

def computePrecisionCurve(predicted_edge_list, true_digraph, max_k=-1):
    if max_k == -1:
        max_k = len(predicted_edge_list)
    else:
        max_k = min(max_k, len(predicted_edge_list))

    sorted_edges = sorted(predicted_edge_list, key=lambda x: x[2], reverse=True)

    precision_scores = []
    delta_factors = []
    correct_edge = 0
    for i in range(max_k):
        if true_digraph.has_edge(sorted_edges[i][0], sorted_edges[i][1]):
            correct_edge += 1
            delta_factors.append(1.0)
        else:
            delta_factors.append(0.0)
        precision_scores.append(1.0 * correct_edge / (i + 1))
    return precision_scores, delta_factors

def computeMAP(predicted_edge_list, true_digraph, max_k=-1):
    node_num = true_digraph.number_of_nodes()
    node_edges = {}
    for n in true_digraph.nodes():
        node_edges[n] = []
    for st, ed, w in predicted_edge_list:
        try:
            assert st in node_edges
        except AssertionError:
            print("Predicted node", st, "not in true_digraph")
        node_edges[st].append((st, ed, w))
    node_AP = [0.0] * node_num
    count = 0
    for i, n in enumerate(true_digraph.nodes()):
        if true_digraph.out_degree(n) == 0:
            continue
        count += 1
        precision_scores, delta_factors = computePrecisionCurve(
            node_edges[n], true_digraph, max_k)
        precision_rectified = [p * d for p, d in \
            zip(precision_scores, delta_factors)]
        if sum(delta_factors) == 0:
            node_AP[i] = 0
        else:
            node_AP[i] = float(sum(precision_rectified) / sum(delta_factors))
    return sum(node_AP) / count

def getPrecisionReport(prec_curv, edge_num):
    result_str = ''
    temp_pos = precision_pos[:] + [edge_num]
    for p in temp_pos:
        if p < len(prec_curv):
            result_str += "\t%.3f" % prec_curv[p - 1]
        else:
            result_str += "\t-"
    return result_str[1:]

def getMetricsHeader():
    header = "MAP\t" + "\t".join(["P@%d" % p for p in precision_pos])
    header += "\tP@EdgeNum"
    return header

