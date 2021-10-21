import numpy as np
import random
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import mean_absolute_error
from numpy import zeros, inf, array, argmin
import os
from scipy.optimize import linear_sum_assignment

# os.sysconf()
import networkx as nx, pickle, glob
try:
    from data.conjugate import conjugate_nx
except:
    from conjugate import conjugate_nx


def eval_accuracy(gt, hyp):
    """
    Eval accuracy on batch
    :param gt:
    :param hyp:
    :return:
    """
    hyp = np.where(hyp > 0.5, 1, 0)
    gt = np.where(gt > 0.5, 1, 0)
    acc = []
    p = []
    r = []
    f1 = []
    # accuracy: (tp + tn) / (p + n)
    for i in range(len(gt)):

        accuracy = accuracy_score(hyp[i], gt[i])

        # precision tp / (tp + fp)
        precision = precision_score(hyp[i], gt[i])
        # recall: tp / (tp + fn)
        recall = recall_score(hyp[i], gt[i])
        # f1: 2 tp / (2 tp + fp + fn)
        f1_ = f1_score(hyp[i], gt[i])
        acc.append(accuracy)
        p.append(precision)
        r.append(recall)
        f1.append(f1_)

    return acc, p, r, f1

def eval_graph(gt, hyp):
    """
    Eval accuracy on batch
    :param gt:
    :param hyp:
    :return:
    """
    hyp = np.exp(hyp)
    hyp = np.argmax(hyp, axis=1)
    # print(hyp)
    # gt = np.where(gt > 0.5, 1, 0)
    # accuracy: (tp + tn) / (p + n)
    # print(gt)
    # print(hyp)
    accuracy = accuracy_score(hyp, gt)

    # precision tp / (tp + fp)
    precision = precision_score(hyp, gt, average='macro')
    # recall: tp / (tp + fn)
    recall = recall_score(hyp, gt, average='macro')
    # f1: 2 tp / (2 tp + fp + fn)
    f1_ = f1_score(hyp, gt, average='macro')
    # print(recall, precision, f1_)
    return accuracy, precision, recall, f1_


def computePRF(nOk, nErr, nMiss):
    eps = 0.00001
    fP = nOk / (nOk + nErr + eps)
    fR = nOk / (nOk + nMiss + eps)
    fF = 2 * fP * fR / (fP + fR + eps)
    return fP, fR, fF

def jaccard_distance(x,y):
    """
        intersection over union
        x and y are of list or set or mixed of
        returns a cost (1-similarity) in [0, 1]
    """
    try:    
        setx = set(x)
        return  1 - (len(setx.intersection(y)) / len(setx.union(y)))
    except ZeroDivisionError:
        return 0.0

def evalHungarian(X,Y,thresh, func_eval=jaccard_distance):
        """
        https://en.wikipedia.org/wiki/Hungarian_algorithm
        """          
        cost = [func_eval(x,y) for x in X for y in Y]
        cost_matrix = np.array(cost, dtype=float).reshape((len(X), len(Y)))
        r1,r2 = linear_sum_assignment(cost_matrix)
        toDel=[]
        for a,i in enumerate(r2):
            # print (r1[a],ri)      
            if 1 - cost_matrix[r1[a],i] < thresh :
                toDel.append(a)                    
        r2 = np.delete(r2,toDel)
        r1 = np.delete(r1,toDel)
        _nOk, _nErr, _nMiss = len(r1), len(X)-len(r1), len(Y)-len(r1)
        return _nOk, _nErr, _nMiss

def get_all(path, ext="pkl"):
    file_names = glob.glob("{}*.{}".format(path,ext))
    return file_names

def read_results(fname, conjugate=True):
    """
    Since the differents methods tried save results in different formats,
    we try to load all possible formats.
    """
    results = {}
    if type(fname) == str: 
        f = open(fname, "r")
        lines = f.readlines()
        f.close()
        for line in lines[1:]:
            id_line, label, prediction = line.split(" ")
            id_line = id_line.split("/")[-1].split(".")[0]
            results[id_line] = (int(label), np.exp(float(prediction.rstrip())) )
    else:
        for id_line, label, prediction in fname:
            # print(id_line, label, prediction)
            id_line_ = id_line.split("/")[-1].split(".")[0]
            num = id_line = id_line.split("/")[-1].split(".")[1].split("_")[-1]
            prediction = np.argmax(np.exp(prediction))
            id_line = f'{id_line_}_{num}'
            results[id_line] = int(label), prediction
    # print(results)
    return results

def create_groups_span(gts:dict):
    ngroup = 0
    res = {}
    for key, list_v in gts.items():
        row, col = key
        # list_v.append(key)
        group_k = res.get((row, col), None)
        if group_k is None:
            group_k = ngroup
            ngroup += 1
            res[(row, col)] = group_k
        for k in list_v:
            row, col = k
            res[(row, col)] = group_k
    return res

def evaluate_graph_IoU(file_list, results, min_w = 0.5, th = 0.8, type_="COL", conjugate=True, all_edges=False, pruned=False, labels_fixed=None):
    ORACLE = False
    type_ = type_.lower()
    if type(results) != dict:
        results = read_results(results, conjugate=conjugate and not all_edges)
    nOk, nErr, nMiss = 0,0,0
    fP, fR, fF = 0,0,0
    res = []
    fname_search = "52684_002"
    num_files = 0
    for raw_path in file_list:  
        # if fname_search not in raw_path:
        #     continue
        f = open(raw_path, "rb")
        data_load = pickle.load(f)
        f.close()
        ids = data_load['ids']
        nodes = data_load['nodes']
        edges = data_load['edges']
        labels = data_load['labels']
        edge_features = data_load['edge_features']   
        file_name = raw_path.split("/")[-1].split(".")[0]
        if len(nodes) == 0:
            continue
        num_files += 1
        classes_hyp, classes_gt = {}, {}

        for i, node in enumerate(nodes):
            # hyp = resu
            # G.add_node(i, attr=node)
            # r, c = labels[i]['row'], labels[i]['col']
            # if r == -1 or c == -1:
            #     ctype = -1
            #     continue
            # if type_ == "col":
            #     ctype = c
            # elif type_ == "row":
            #     ctype = r
            # else:
            #     pass

            # key = f'{file_name}_{i}'
            key = f'{file_name}_{i}'
            # print(key)
            gt_i, hyp_i = results.get(key)
            group_hyp = classes_gt.get(gt_i, [])
            group_hyp.append(i)
            classes_gt[gt_i] = group_hyp

            group_hyp = classes_hyp.get(hyp_i, [])
            group_hyp.append(i)
            classes_hyp[hyp_i] = group_hyp
            # print(raw_path)
            # exit()
        cc, cc_gt = [], []
        keys = sorted(list(classes_gt.keys()))
        for k in keys:
            l = classes_gt.get(k)
            cc_gt.append(l)
            l = classes_hyp.get(k, [])
            cc.append(l)
       
        _nOk, _nErr, _nMiss = evalHungarian(cc, cc_gt, th, jaccard_distance)
        _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
        # if _fP < 0.99:
        #     print(file_name)
        # print(_nOk, _nErr, _nMiss, _fP, _fR, _fF)
        res.append([raw_path,  _nOk, _nErr, _nMiss, _fP, _fR, _fF])
        nOk += _nOk
        nErr += _nErr
        nMiss += _nMiss
        fP += _fP
        fR += _fR
        fF += _fF
        gt_edges_graph_dict = None
    # print(fF, num_files)
    fP, fR, fF = fP/num_files, fR/num_files, fF/num_files
    print("_nOk {}, _nErr {}, _nMiss {}, P: {} R: {} F1: {}".format(nOk, nErr, nMiss, fP, fR, fF))
    return fP, fR, fF, res
if __name__ == "__main__":
    test_samples()
