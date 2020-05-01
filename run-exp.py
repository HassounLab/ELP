import sys
print(sys.version)
import numpy as np
import networkx as nx
import os
import json
import glob
import time
from argparse import ArgumentParser
import pprint

from embedding.baselineEmbedding import BaselineEmbedding as BE
from embedding.emEmbedding import EMEmbedding as EE
from embedding.node2vec import node2vec as N2V
from embedding.deepwalk import deepwalk as DW

from evaluation.linkPrediction import expLP
from evaluation.jaccardSimilarity import JaccardSimilarity as JS
from evaluation.enzymePrediction import expEP
from util.graph_util import read_graph, read_labels, read_cmty, read_walks



def saveToJson(dataset, model_type, embed_size, savepath, exp_res):
    try:
        jsondata = json.load(open(savepath, "r"))
    except FileNotFoundError:
        jsondata = {}

    embed_size = str(embed_size)
        
    if dataset not in jsondata:
        jsondata[dataset] = {}
    if model_type not in jsondata[dataset]:
        jsondata[dataset][model_type] = {}
    if embed_size not in jsondata[dataset][model_type]:
        jsondata[dataset][model_type][embed_size] = {}
    for k, v in exp_res.items():
        if k not in jsondata[dataset][model_type][embed_size]:
            jsondata[dataset][model_type][embed_size][k] = {}
        jsondata[dataset][model_type][embed_size][k].update(v)
    savename = os.path.join(savepath, "results.json")
    print("Updating json results to", savename)
    json.dump(jsondata, open(savename, "w"))



def readdata(datadir, cmty_type, undirected=True, graph_f=None, graph_neg_f=None, **kwargs):
    if graph_f is None:
        graph_f = glob.glob("%s/graph.*" % datadir)
        graph_f = graph_f[0]
    else:
        graph_f = os.path.join(datadir, graph_f) 
    print("Reading graph", graph_f)
    
    G = read_graph(graph_f)
 
    if graph_neg_f is None:
        graph_neg_f = glob.glob("%s/neggraph.*" % datadir)
        graph_neg_f = graph_neg_f[0] if graph_neg_f != [] else None
    else:
        graph_neg_f = os.path.join(datadir, graph_neg_f)
    if graph_neg_f is None:
        neg_G = None
    else:
        print("Reading neg graph", graph_neg_f)
        neg_G = read_graph(graph_neg_f)   
 
    labels_f = "%s/labels.txt" % datadir
    cmty_f = "%s/%s.cmty.txt" % (datadir, cmty_type)
    walks_f = "%s/walks.npy" % datadir


    if os.path.exists(labels_f):
        labels = read_labels(labels_f, G.number_of_nodes())
    else:
        print("No labels file", labels_f)
        labels = None

    if os.path.exists(cmty_f):
        cmty, num_coms = read_cmty(cmty_f, G.number_of_nodes())
    else:
        print("No cmty file", cmty_f)
        cmty, num_coms = None, None

    if os.path.exists(walks_f):
        walks, num_walks = read_walks(walks_f, **kwargs)
    else:
        print("No walks file", walks_f)
        walks, num_walks = None, None

    return {"G": G, "neg_G": neg_G, "labels": labels, "cmty": cmty, "num_coms": num_coms,
            "walks": walks, "num_walks": num_walks}
    
def experiment(datum, model, dataset, model_type, embed_sizes=None, 
               pred_modes=None, evaluations=None, load=None, repeat=3, 
               savepath=None, nc_train_sizes=None, lp_test_ratios=None, 
               train_verbose=1, eval_verbose=1, save_embeddings=False, 
               save_lp_score=False, inductive=False, **params):

    if embed_sizes is None:
        embed_sizes = [2 ** i for i in range(1, 8)]
    if pred_modes is None:
        pred_modes = ["pre_nn", "nn", "nn_inner"]
    if evaluations is None:
        evaluations = ["lp", "gr", "nc"] 
    if nc_train_sizes is None:
        nc_train_sizes = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    if lp_test_ratios is None:
        lp_test_ratios = [0.5]

    if datum["labels"] is None and "nc" in evaluations:
        evaluations.remove("nc")
    exp_res = dict({
        ("dataset", dataset),
        ("model", model_type)})
    for i, e in enumerate(embed_sizes):
        
        emb_model = model(embed_size=e, verbose=train_verbose, **params)

        print("\n\nExperiment model %d" % (i + 1))
        if not train_verbose:
            emb_model.print_summary()
        MAPs = np.empty((len(pred_modes), repeat))
        micros = np.empty((len(nc_train_sizes), repeat))
        macros = np.empty((len(nc_train_sizes), repeat))
        exp_res[e] = {} 
        for r in range(repeat):
            print("repeat round %d/%d" % (r + 1, repeat))
            r_id = "repeat_%d" % (r + 1)
            exp_res[e][r_id] = {}
            if 'all'in evaluations:
                print('running all')
                emb_model.learn_embedding(datum['G'])
            if "lp" in evaluations:
                lpkwargs = dict(params)
                if "lp" in params:
                    lpkwargs.update(params["lp"])
                exp_res[e][r_id]["lp"] = expLP(
                    datum, 
                    emb_model, 
                    pred_modes, 
                    test_ratios=lp_test_ratios, 
                    verbose=eval_verbose,
                    savepath=savepath,
                    save_scores=save_lp_score,
                    **lpkwargs)
            if 'ep' in evaluations:
                expEP(datum, emb_model, pred_modes, test_ratios=lp_test_ratios, 
                      verbose=eval_verbose, inductive=inductive)
        print("Results of embed size %d over %d repeated exp" % (e, repeat))
        pprint.pprint(exp_res[e])
    print("Final results") 
    pprint.pprint(exp_res)

    
    if savepath is not None:
        savename = os.path.join(savepath, "results.json")
        json.dump(exp_res, open(savename, "w"))
        print("Dumped to json file", savename)

def main(data=None, model_type=None, cmty_type=None, testmode=False, savepath=None,
         **kwargs):
    valid_comm_types = ["oslom", "cfinder", "labelcmty"]
    if model_type == "baseline":
        model = BE
        cmty_type = None
    elif model_type == "node2vec":
        model = N2V
    elif model_type == "deepwalk":
        model = DW
    elif model_type == "em":
        model = EE
    elif model_type == "js":
        model = JS
    else:
        raise NameError('model type', model_type, 'not recognized')
    
    if testmode:
        data, datadir = "karate", "./test-data-karate"
    else:
        try:    
            datadir = os.path.join(os.environ["DATAPATH"], data)
        except KeyError:
            print("No data path detected, using test data")
            data, datadir = "karate", "./test-data-karate"
    try:
        params = json.load(open("config.json", "r"))[data]
    except KeyError:
        print('Error loading data, using empty params')
        params = {}

    if model_type in params:
        for k in params[model_type]:
            params[k] = params[model_type][k]

    if cmty_type in params:
        for k in params[cmty_type]:
            params[k] = params[cmty_type][k]

    for k, v in kwargs.items():
        if v is None:
            continue
        if k.startswith("nn_"):
            k = k[3:]
            params["nn"][k] = v
        else:
            params[k] = v
    params["savepath"] = savepath
    if "save_embeddings" in params and params["save_embeddings"]:
        params["embedding_savepath"] = params["savepath"]

    if model_type == "anony" and "walk_length" in params:
        model_type = "%s.%d" % (model_type, params["walk_length"])

    if testmode:
        embed_sizes = [2]
        params["num_epochs"] = 1
        params["nn_num_epochs"] = 1
        params["savepath"] = None

    print("custom params", params)
    datum = readdata(datadir, cmty_type, **params)

    if cmty_type is not None:
        model_type = "%s.%s" % (model_type, cmty_type)
    
    if "savepath" in params and params["savepath"] is not None:
        with open(os.path.join(params["savepath"], "hypers.txt"), "w") as f:
            f.write("%s\n" % (pprint.pformat(params)))   

    experiment(datum, model, data, model_type, **params)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('data')
    parser.add_argument('-m', '--model_type', default="baseline")
    parser.add_argument('-c', '--cmty_type', default="oslom")
    parser.add_argument('-t', '--test_mode', action="store_true")
    
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--beta", type=float, default=None)
    parser.add_argument("--beta_edge", type=float, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--nn_lr", type=float, default=None)
    parser.add_argument("--use_node_embed", default=None, action="store_true")
    parser.add_argument("--use_fgpt", default=None, action="store_true")
    parser.add_argument("--fgpt_name", default="fingerprint")
    parser.add_argument("--use_edge_attr", default=None, action="store_true")
    parser.add_argument("--edge_name", default=None)
    parser.add_argument("--savepath", default="./")
    parser.add_argument("--save_lp_scores", default=None, action="store_true")   
 
    parser.add_argument("--inductive", default=None, action="store_true")

    args = vars(parser.parse_args())
    print(args)
    main(**args)


