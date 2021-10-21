from __future__ import print_function
from builtins import range
import sys, pickle
from shutil import copyfile
import numpy as np
import cv2
import os,sys,inspect
import glob
import matplotlib.pyplot as plt
from tqdm import tqdm
import networkx as nx
currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0,parentdir)
try:
    from data.conjugate import conjugate_nx
except:
    from conjugate import conjugate_nx

try:
    from data.page import TablePAGE
except:
    from page import TablePAGE

try:
    from utils.metrics import evalHungarian, jaccard_distance, computePRF
except:
    from metrics import evalHungarian, jaccard_distance, computePRF

aux1 = 156
aux2 = 177


def read_lines(fpath):
    f = open(fpath, "r")
    lines = f.readlines()
    f.close()
    lines = [x.strip() for x in lines]
    return lines

def get_all(path, ext="pkl"):
    file_names = glob.glob(os.path.join(path, "*.{}".format(ext)))
    return file_names

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_results(fname):
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
        # print(fname)
        for id_line, label, prediction in fname:
            id_line = id_line.split("/")[-1].split(".")[0]
            results[id_line] = int(label), np.exp(float(prediction))
    return results

def get_dict_nodes_edges(nodes, edges):
    """
    for non directed graph
    :param edges:
    :return:
    """
    res = {}
    for i, _ in enumerate(nodes):
        res[i] = []
    for i,j,_ in edges:
        # i -> j
        aux = res.get(i, [])
        aux.append(j)
        res[i] = aux
        # j -> i
        aux = res.get(j, [])
        aux.append(i)
        res[j] = aux
    return res

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

def get_min(group, nodes, type="col"):
    """
    return the minium width or height line
    4 first elements of each node: x, y, w, h
    """
    group_sizes = []
    for node in group:
        # print(node)
        x, y, w, h = nodes[node][:4]
        x1 = x - (w/2) # x = mid point. x1 = arriba a la izquierda
        y1 = y - (h/2)
        if type == "col":
            group_sizes.append(x1)
        elif type == "row":
            group_sizes.append(y1)
        else:
            raise Exception(f'{type} not implemented')
    return min(group_sizes)

def get_order_ccs(ccs, nodes, type="col"):
    """
    Crea un orden para cada grupo a partir de su posicion. Simple.
    """
    ccs_ = []
    for i, group in enumerate(ccs):
        minium_ = get_min(group, nodes, type=type)
        ccs_.append([minium_, i])

    ccs_.sort()
    res = {}
    for num_group, (_, group) in enumerate(ccs_):
        for i in ccs[group]:
            res[i] = num_group
    return res

def idtl_to_list(ids_tl):
    res = []
    for id_ in ids_tl:
        id_line = id_.split("-")[-1]
        res.append(id_line)
    return res

def idtl_to_dict(ids_tl, dict_num_nodes:dict):
    res = {}
    sorted_keys = sorted(list(dict_num_nodes.keys()))
    i = 0
    for num_tl in sorted_keys:
        num_group = dict_num_nodes[num_tl]
        id_ = ids_tl[num_tl]
        res[id_] = num_group
        i += 1
    return res

def main():
    """
    
    """
    ORACLE = False
    conjugate = True
    all_edges = False
    min_w = 0.5
    type_ = sys.argv[2]
    path_PAGE = sys.argv[3]
    spans = len(sys.argv) > 3 and sys.argv[3].lower() in ["true", "si"]
    if spans:
        type_ = "span"
    dir_to_save_local = "connected_components_{}".format(type_)
  
    data_path = sys.argv[1]
    path_save = os.path.join(data_path, f'page_{type_}_hyp')
    files_to_use = "/data/HisClima/DatosHisclima/test.lst"
    if files_to_use is not None:
        files_to_use = read_lines(files_to_use)
    data_pkls = "/data/HisClima/DatosHisclima/graphs/graphs_preprocessed/graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5"
    # data_pkls = "/data/HisClima/DatosHisclima/graphs/graphs_preprocessed/graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_headers"
    
  
    # NAF
    ##Cols
    # if type_ == "col":
    #     results = f"{data_path}/results_cols.txt"
    # elif type_ == "row":
    #     results = f"{data_path}/results_rows.txt"
    # elif type_ == "span":
    #     results = f"{data_path}/results_span.txt"
    results = f"{data_path}/results.txt"


    dir_img = "/data/HisClima/DatosHisclima/data/GT-Corregido"
    # dir_img = "/data2/jose/corpus/tablas_DU/icdar19_abp_small/"
    file_list = get_all(data_pkls)
    # print(file_list)
    print_bbs = True    
    if results == "cols":
        dir_to_save = os.path.join(data_path, dir_to_save_local)
    elif results != "no":
        results = os.path.abspath(results)
        dir_to_save = "/".join(results.split("/")[:-1])
        dir_to_save = os.path.join(dir_to_save, dir_to_save_local)
        results = read_results(results)
        print("A total of {} lines classifieds in results".format(len(results)))
    else:
        dir_to_save = os.path.join(results, dir_to_save_local)
    #TODO borrar
    # dir_to_save = os.path.join(data_path, "results_postprocess")
    # End
    # dir_to_save = "."
    create_dir(dir_to_save)
    create_dir(path_save)
    print("Saving data on : {}".format(dir_to_save))
    # data_path = "/data/READ_ABP_TABLE/dataset111/graphs_preprocesseds/graphs_structure/graphs_structure_PI_fasttext_0.05_radius0.02"
    file_list = tqdm(file_list, desc="Files")
    nOk, nErr, nMiss = 0,0,0
    fP, fR, fF = 0,0,0
    min_cc = 10000000
    # file_search = "vol003_201_0" # 84
    file_search = "vol003_008_0" # 100
    file_names_ = [x.split("-edge")[0] for x in results.keys()]
    
    for raw_path in file_list:
        file_name_p = raw_path.split("/")[-1].split(".")[0]
        # raw_path_orig = os.path.join(data_path_orig, file_name_p+".pkl")
        
        if files_to_use is not None and file_name_p not in files_to_use:
            continue
        if file_name_p not in file_names_:
            continue
        # if file_search not in raw_path:
        #     continue
        page_path_xml = os.path.join(path_PAGE, f'{file_name_p}.xml')
        page_path_xml_dest = os.path.join(path_save, f'{file_name_p}.xml')
        copyfile(page_path_xml, page_path_xml_dest)
        page = TablePAGE(page_path_xml_dest)
        # Read data from `raw_path`.
        # print("File: {}".format(raw_path))
        f = open(raw_path, "rb")
        data_load = pickle.load(f)
        f.close()
        # f = open(raw_path_orig, "rb")
        # data_load_orig = pickle.load(f)
        # f.close()
        ids = data_load['ids']
        nodes = data_load['nodes']
        # nodes_orig = data_load_orig['nodes']
        edges = data_load['edges']
        labels = data_load['labels']
        ids_tl = idtl_to_list(data_load['ids_tl'])
        # edges_orig = data_load_orig['edges']
        # labels_orig = data_load_orig['labels']
        # edge_features_orig = data_load_orig['edge_features']
        edge_features = data_load['edge_features']
        # print("A total of {} nodes and {} edges for {}".format(len(nodes), len(edges), raw_path))

        if conjugate:
            # ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, \
            # new_edge_feats, ant_nodes, ant_edges = conjugate_nx(ids,
            #         nodes, edges, labels, edge_features, idx=aux1, list_idx=[525,626,161,549])
            if spans:
                ids, new_nodes, new_edges, new_labels, new_edge_feats, ant_nodes, ant_edges = data_load["conjugate"]
                gts_span = data_load["gts_span"]
                gts_span_dict = create_groups_span(gts_span)
            else:
                ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, new_edge_feats, ant_nodes, ant_edges = data_load["conjugate"]
            if len(edges) != len(new_nodes):
                print("Problem with {} {} edges and {} new_nodes".format(raw_path, len(edges), len(new_nodes)))
                continue
        else:
            ant_nodes = edges
        file_name = raw_path.split("/")[-1].split(".")[0]

        G = nx.Graph()
        G_gt = nx.Graph()
        out_graph = []
        weighted_edges = []
        weighted_edges_GT = []
        weighted_edges_dict_gt = {}
        for i, node in enumerate(nodes):
            # G.add_node(i, attr=node)
            # G.add_node(i)
            # G_gt.add_node(i)
            if type_ == "span":
                # ctype = labels[i][type_]
                r, c = labels[i]['row'], labels[i]['col']
                if r == -1 or c == -1:
                    ctype = -1
                else:
                    ctype = gts_span_dict.get((r,c))
            else:
                ctype = labels[i][type_]
            # if ctype == 9:
            #     print(node)
            if ctype != -1:
                G.add_node(i)
                G_gt.add_node(i)
                cc_type = weighted_edges_dict_gt.get(ctype, set())
                cc_type.add(i)
                weighted_edges_dict_gt[ctype] = cc_type
            else:
                out_graph.append(i)
        aux_dict = {}
        errors = []
        count_errors, fp, fn = 0, 0, 0
        if all_edges:
            
            for i, _ in enumerate(nodes):
                for j, _ in enumerate(nodes):
                    key = "{}_edge{}-{}".format(file_name, i, j)
                    try:
                        gt, hyp_prob = results.get(key)
                    except Exception as e:
                        print("Problem with key {} (no conjugation - all_edges)".format(key))
                        # continue
                        raise e
                    if ORACLE:
                        if i in out_graph or j in out_graph: continue
                        if gt:
                            hyp_prob = 1
                            weighted_edges.append((i,j,hyp_prob))
                        # else:
                        #     hyp_prob = 0
                    else:
                        added = False
                        if hyp_prob > min_w:
                            added = True
                            weighted_edges.append((i,j,hyp_prob))
                            if not gt:
                                count_errors += 1

        else:
            for idx, (i, j) in enumerate(ant_nodes):
                aux_dict[(i, j)] = idx
            for count, (i, j) in enumerate(edges):
                # if i == 394 or j == 394:
                #     print("------> ", i, labels[i][type_],j, labels[j][type_], i in out_graph, j in out_graph)
                idx_edge = aux_dict.get((i, j), aux_dict.get((j, i)))
                key = "{}-edge{}_{}".format(file_name, i,j)
                key2 = "{}-edge{}_{}".format(file_name, j, i)
                gt, hyp_prob = results.get(key, results.get(key2, [0,0]))
                if ORACLE:
                    if i in out_graph or j in out_graph: continue
                    if gt:
                    
                        hyp_prob = 1
                        weighted_edges.append((i,j,hyp_prob))
                        weighted_edges.append((j,i,hyp_prob))
                else:
                    added = False
                    if hyp_prob > min_w:
                        added = True
                        weighted_edges.append((i,j,hyp_prob))
                        if not gt:
                            count_errors += 1
                    if added and not gt: # Falso positivo
                        errors.append((i,j, True))
                        fp += 1
                    elif not added and gt: # Falso negativo
                        errors.append((i,j, False))
                        fn += 1
                if gt:
                    weighted_edges_GT.append((i,j,1))


        G.add_weighted_edges_from(weighted_edges)
        cc = nx.connected_component_subgraphs(G)
        cc = [sorted(list(c)) for c in cc]
        # exit()

        cc_gt = [ sorted(list(ccs)) for ctype,ccs in weighted_edges_dict_gt.items()]
        dict_node_group = get_order_ccs(cc, nodes)
        # print(ids_tl, f' -> {len(ids_tl)}')
        # print(dict_node_group, f' -> {len(dict_node_group)}')
        dict_line_group = idtl_to_dict(ids_tl, dict_node_group)
        # for k,v in dict_line_group.items():
        #     print(k,v)
        # exit()
        
        page.add_col_to_cell(dict_line_group)
        page.save_changes()

        cc_gt.sort()
        cc.sort()

        _nOk, _nErr, _nMiss = evalHungarian(cc, cc_gt, 1.0, jaccard_distance)
        _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)

        str_ = "A total of {} / {} ({}%) misclassified - {} fp {} fn".format(count_errors, len(nodes)*len(nodes),(count_errors/(len(nodes)*len(nodes))*100), fp, fn )
        print("{}, _nOk {}, _nErr {}, _nMiss {}, P: {} R: {} F1: {} - \n {}".format(raw_path, _nOk, _nErr, _nMiss, _fP, _fR, _fF, str_))
        min_cc = min(min_cc, min([len(x) for x in cc_gt]))
        nOk += _nOk
        nErr += _nErr
        nMiss += _nMiss
        fP += _fP
        fR += _fR
        fF += _fF
        # exit()
      
        # exit()
    fP, fR, fF = fP/len(file_list), fR/len(file_list), fF/len(file_list)
    # print("P: {} R: {} F1: {}".format( fP, fR, fF))
    fP, fR, fF = computePRF(nOk, nErr, nMiss)
    print("P: {} R: {} F1: {}".format( fP, fR, fF))
    print("Min group of cc: ", min_cc)
    # exit()


if __name__ == "__main__":
    main()
    # if len(sys.argv) > 1 and sys.argv[1] != "-h":
    #     main()
    # else:
    #     print("Usage: python {} <dir with GT pkl> <file with results (txt)> <dir with the REAL images to load>".format(sys.argv[0]))
