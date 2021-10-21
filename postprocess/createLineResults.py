import sys, pickle
import numpy as np
import os,sys,inspect
import glob
from tqdm import tqdm
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
    lines = [x.strip().split(".")[0] for x in lines]
    return lines

def get_all(path, ext="pkl"):
    file_names = glob.glob(os.path.join(path, "*.{}".format(ext)))
    return file_names

def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def read_results(fname, prod=False):
    results = {}
    if type(fname) == str: 
        f = open(fname, "r")
        lines = f.readlines()
        f.close()
        ls = []
        text = ""
        for line in lines[1:]:
            line = line.rstrip()
            if line.startswith("/") :
                ls.append(text)
                text = line
                    
            else:
                text += " " + line
        ls.append(text)
        ls = ls[1:]
        res = []
        for line in ls:
            if not prod:
                id_line, label, *prediction = line.split(" ")
            else:
                id_line, *prediction = line.split(" ")
            id_line = id_line.split("/")[-1]#.split(".")[0]
            prediction = [x.replace("[", "").replace("]", "") for x in prediction if x]
            prediction = [x for x in prediction if x != ""]
            p = [float(x) for x in " ".join(prediction).rstrip().split(" ")]
            p = np.exp(p)
            p = np.argmax(p)  
            if not prod: 
                res.append(p == int(label))         
                results[id_line] = (int(label), p )
            else:
                results[id_line] = p
    else:
        # print(fname)
        if not prod:
            for id_line, label, prediction in fname:
                id_line = id_line.split("/")[-1].split(".")[0]
                p = np.exp(float(prediction))
                results[id_line] = int(label), p
        else:
            for id_line, prediction in fname:
                id_line = id_line.split("/")[-1].split(".")[0]
                p = np.exp(float(prediction))
                results[id_line] = p
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
    with_id = True ## 
    min_w = float(sys.argv[4])
    type_ = sys.argv[2]
    spans = len(sys.argv) > 5 and sys.argv[5].lower() in ["true", "si"]
    prod = len(sys.argv) > 6 and sys.argv[6].lower() in ["true", "si"]
    if spans:
        type_ = "span"
    if not prod:
        dir_to_save_local = "results_line_{}".format(type_)
    else:
        dir_to_save_local = "results_line_{}_prod".format(type_)
  
    data_path = sys.argv[1]
    path_save = os.path.join(data_path, f'page_{type_}_hyp')
    files_to_use = "/data/HisClima/DatosHisclima/test.lst"
    if files_to_use is not None:
        files_to_use = read_lines(files_to_use)
    # data_pkls = "/data/HisClima/DatosHisclima/graphs/graphs_preprocessed/graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5"
    # data_pkls = "/data/HisClima/DatosHisclima/graphs/graphs_preprocessed/graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_headers"
    data_pkls = sys.argv[3]
  
    # NAF
    ##Cols
    # if type_ == "col":
    #     results = f"{data_path}/results_cols.txt"
    # elif type_ == "row":
    #     results = f"{data_path}/results_rows.txt"
    # elif type_ == "span":
    #     results = f"{data_path}/results_span.txt"
    if not prod:
        results = f"{data_path}/results.txt"
    else:
        results = f"{data_path}/results_prod.txt"

    # dir_img = "/data2/jose/corpus/tablas_DU/icdar19_abp_small/"
    file_list = get_all(data_pkls)
    # print(file_list)
    if results == "cols":
        dir_to_save = os.path.join(data_path, dir_to_save_local)
    elif results != "no":
        results = os.path.abspath(results)
        dir_to_save = "/".join(results.split("/")[:-1])
        dir_to_save = os.path.join(dir_to_save, dir_to_save_local)
        results = read_results(results, prod=prod)
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
    # print(results)
    file_names_ = [x.split(".")[0] for x in results.keys()]
    set_keys = set()
    for raw_path in file_list:
        file_name_p = raw_path.split("/")[-1].split(".")[0]
        # raw_path_orig = os.path.join(data_path_orig, file_name_p+".pkl")
        if not prod:
            if files_to_use is not None and file_name_p not in files_to_use:
                continue
            if file_name_p not in file_names_:
                continue
        # if file_search not in raw_path:
        #     continue
        print(raw_path)
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
        if not prod:
            labels = data_load['labels']
        else:
            labels = []
        ids_tl = idtl_to_list(data_load['ids_tl'])
        print(data_load.keys())
        print(data_load['ids_tl'])
        exit()

        # edges_orig = data_load_orig['edges']
        # labels_orig = data_load_orig['labels']
        # edge_features_orig = data_load_orig['edge_features']
        edge_features = data_load['edge_features']
        # print("A total of {} nodes and {} edges for {}".format(len(nodes), len(edges), raw_path))

        cc_dict, cc_gt_dict = {}, {}
        for i, node in enumerate(nodes):            
            key = "{}.xml_{}".format(file_name_p, i)
            set_keys.add(key)
            if not prod:
                try:
                    gt, hyp = results.get(key)
                except Exception as e:
                    print(key)
                    raise e
            else:
                try:
                    hyp = results.get(key)
                except Exception as e:
                    print(key)
                    raise e
            # print(f'{key} {gt} {hyp}')
            # exit()
            arr_cc_hyp = cc_dict.get(hyp, [])
            arr_cc_hyp.append(i)
            cc_dict[hyp] = arr_cc_hyp

            if not prod:
                arr_cc_gt = cc_gt_dict.get(gt, [])
                arr_cc_gt.append(i)
                cc_gt_dict[gt] = arr_cc_gt
                # if "vol003_040_0" in raw_path:
                #     print(gt)
        keys = list(sorted(cc_dict.keys()))
        cc, cc_gt = [], []
        for k in keys:
            cc.append(cc_dict.get(k, []))
            if not prod:
                group_cc = cc_gt_dict.get(k, [])
                cc_gt.append(group_cc)    

        # cc = nx.connected_component_subgraphs(G)
        # cc = [sorted(list(c)) for c in cc]
        # exit()

        # cc_gt = [ sorted(list(ccs)) for ctype,ccs in weighted_edges_dict_gt.items()]
        
        dict_node_group = {}
        dict_node_group_gt = {}
        for group_, arr in enumerate(cc):
            for n in arr:
                dict_node_group[n] = group_
        if not prod:
            for group_, arr in enumerate(cc_gt):
                for n in arr:
                    dict_node_group_gt[n] = group_

        # print(ids_tl, f' -> {len(ids_tl)}')
        # print(dict_node_group, f' -> {len(dict_node_group)}')

        dict_line_group = idtl_to_dict(ids_tl, dict_node_group)
        if not prod:
            dict_line_group_gt = idtl_to_dict(ids_tl, dict_node_group_gt)
        path_save_file = os.path.join(dir_to_save, file_name_p)

        f = open(path_save_file, "w")
        # print(path_save_file)
        for k,v in dict_line_group.items():
            if not prod:
                v_gt = dict_line_group_gt.get(k, -1)
                # if v_gt == -1:
                #     continue #TODO eliminar
                f.write(f'{k} {v} {v_gt}\n')
            else:
                f.write(f'{k} {v}\n')
        f.close()
        # exit()
        
        if not prod:
            cc_gt.sort()
            cc.sort()

            _nOk, _nErr, _nMiss = evalHungarian(cc, cc_gt, 1.0, jaccard_distance)
            _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)

            # str_ = "A total of {} / {} ({}%) misclassified - {} fp {} fn".format(count_errors, len(nodes)*len(nodes),(count_errors/(len(nodes)*len(nodes))*100), fp, fn )
            # print("{}, _nOk {}, _nErr {}, _nMiss {}, P: {} R: {} F1: {}".format(raw_path, _nOk, _nErr, _nMiss, _fP, _fR, _fF))
            min_cc = min(min_cc, min([len(x) for x in cc_gt]))
            nOk += _nOk
            nErr += _nErr
            nMiss += _nMiss
            fP += _fP
            fR += _fR
            fF += _fF
            # exit()
    
    if not prod:
        fP, fR, fF = fP/len(file_list), fR/len(file_list), fF/len(file_list)
        # print("P: {} R: {} F1: {}".format( fP, fR, fF))
        fP, fR, fF = computePRF(nOk, nErr, nMiss)
        print("P: {} R: {} F1: {}".format( fP, fR, fF))
        print("Min group of cc: ", min_cc)
        # exit()
    print(f'Proccessed {len(set_keys)} lines')
if __name__ == "__main__":
    main()
    # if len(sys.argv) > 1 and sys.argv[1] != "-h":
    #     main()
    # else:
    #     print("Usage: python {} <dir with GT pkl> <file with results (txt)> <dir with the REAL images to load>".format(sys.argv[0]))