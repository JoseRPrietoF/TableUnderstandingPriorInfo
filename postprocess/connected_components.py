from __future__ import print_function
from builtins import range
import sys, pickle
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

kelly_colors = [
    (0,0,255),
    (0,255,0),
    (255,0,0),
    (0,50,120),
    (75,50,120),
    (250,50,120),
    (250,250,120),
    (250,250,0),
    (250,90,0),
    (80,90,80),
    (80,0,80),
]
dict_clases = {
            'CH': 0,
            'O': 1,
            'D': 1,
        }

color_classes = {
            'CH':  (70, 1, 155),
            'O': (0, 126, 254),
            'D': (0, 126, 254),
        }
dict_classes_inv = {v: k for k, v in dict_clases.items()}

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

def load_image(path):
    if os.path.exists(path+".jpg"):
        p = path+".jpg"
    elif os.path.exists(path+".JPG"):
        p = path+".JPG"
    elif os.path.exists(path+".png"):
        p = path+".png"
    elif os.path.exists(path+".PNG"):
        p = path+".PNG"

    image = cv2.imread(p)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # image = resize(image, size=size)
    # image = image.astype(np.float32)
    # image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    return image

def resize(img, size=(1024,512)):
    return cv2.resize(img.astype('float32'), size).astype('int32')

def show_cells(drawing, title, dir_to_save, fname=""):
    """
    Show the image
    :return:
    """
    plt.title(title)
    plt.imshow(drawing)
    # plt.show()
    plt.savefig("{}/{}.jpg".format(dir_to_save, fname), format='jpg', dpi=800)
    # plt.savefig("{}/{}.jpg".format(dir_to_save, title))
    plt.close()



def read_results(fname, gt=False):
    results = {}
    # gt = True
    if type(fname) == str: 
        f = open(fname, "r")
        lines = f.readlines()
        f.close()
        cont = False
        id_line, label, *prediction = "", "", ""
        for line in lines[1:]:
            if not gt:
                if not cont:
                    id_line, label, *prediction = line.split(" ")
                    if "]" not in prediction:
                        cont = True
                        continue
                else:
                    prediction.extend(line.strip().split(" "))
                    # print(prediction)
                    if any("]" in x for x in line):
                        cont = False
                    else:
                        continue
                prediction = [x.replace("[", "").replace("]", "") for x in prediction if x]
                prediction = [x for x in prediction if x != ""]
                p = [float(x) for x in " ".join(prediction).rstrip().split(" ")]
                p = np.exp(p)
                p = np.argmax(p) 
            else:
                id_line, label, prediction = line.split(" ")
                p = np.exp(float(prediction))
            
            fname, num_line = id_line.split("/")[-1].split(".")[0], id_line.split("/")[-1].split("_")[-1]
            
            # results[id_line] = (int(label), p )
            nums = results.get(fname, {})
            nums[num_line] = (int(label), p )
            results[fname] = nums
    else:
        # print(fname)
        for id_line, label, prediction in fname:
            id_line = id_line.split("/")[-1].split(".")[0]
            results[id_line] = int(label), np.exp(float(prediction))
    return results

def show_all_imgs(drawing, title="",path="", redim=None):
    """
    Show the image
    :return:
    """
    plt.title(title)
    fig, ax = plt.subplots(1, len(drawing))
    for i in range(len(drawing)):
        if redim is not None:
            a = resize(drawing[i], redim)
            # print(a.shape)
            ax[i].imshow(a)
        else:
            ax[i].imshow(drawing[i])
    plt.savefig("{}/{}.png".format(path, title), format='png', dpi=800)
    plt.show( dpi=900)
    plt.close()

def slope(x1, y1, x2, y2):
    try:
        return (y2-y1)/(x2-x1)
    except:
        return 0

def get_color_cc(cc):
    """
    Return a dict
    :param cc:
    :return:
    """
    res = {}
    for i,_ in enumerate(cc):
        res[i] = kelly_colors[i%len(kelly_colors)]
    res[-1] = [0,0,0]
    return res

def search_idx_in_cc(idx,cc):
    for i, c in enumerate(cc):
        if idx in c:
            return i
    return -1

def search_for(nodes, nodes_orig, edges_orig, labels_orig, type="col", edge_features_orig=[], labels=[]):
    data2 = [[z[0], z[1]] for z in nodes]
    nodes = []
    # print(data2)
    groups = []
    for i, data in enumerate(nodes_orig):
        x,y = data[0], data[1]
        if [x,y] not in data2:
            # print([x,y])
            type_id = labels_orig[i][type]
            nodes.append([x,y])
            groups.append(type_id)
    return nodes, groups

def print_BBs(nodes, labels, cc,
              results_gt, file_name, dir_to_save, drawing, _fF=None, gt=False, errors=None, nodes_orig=[], edges_orig=[], labels_orig=[], type="col", edge_features_orig=[] ):
    radius = 10
    color = (255, 0, 0)
    shape = drawing.shape
    # drawing_hyp = np.copy(drawing)
    height, width = shape[0], shape[1]
    # file_name = file_name.
    count_fail = 0
    words_failed = []
    # print(len(nodes))
    # print(len(labels))
    # exit()
    THICKNESS = 4
    # nodes_deleted, groups = search_for(nodes, nodes_orig, edges_orig, labels_orig, type, edge_features_orig, labels)

    # for group in groups:
    #     for count, data_node in enumerate(nodes):
    #         label_node = labels[count][type]
    #         if label_node == group:
    #             for c in cc:
    #                 if label_node
    # print(cc)
    # print(ids)
    # print(nodes_deleted, len(nodes_deleted))

    colors_cc = get_color_cc(cc)
    size_text = 1
    tick_text = 2
    bb_per_group = {}
    for count, data_node in enumerate(nodes):
        label_node = labels[count][type]
        
        x, y, w, h, prob_node = data_node[:5]
        x = int(x*width)
        y = int(y*height)
        w = int(w*width)
        h = int(h*height)
        bb = [
            x-(w//2), y-(h//2),x+(w//2), y+(h//2)
        ]
        x_min, y_min, x_max, y_max = map(int, bb)
        c_idx = search_idx_in_cc(count, cc)
        # if c_idx == -1:
            # print("cc", cc)
            # print("problem with img {} - node {}".format(file_name, count))
            # exit()
            # continue
        
        l_group = bb_per_group.get(c_idx, [])
        l_group.append(bb)
        bb_per_group[c_idx] = l_group

        color_rect = colors_cc[c_idx]

        cv2.rectangle(drawing, (x_min, y_min), (x_max, y_max),
                      # color=(50, 128, 5),
                      # color=results.get(count, (0,0,0)),
                      color=color_rect,
                      thickness=3,
                      )
        cv2.putText(drawing, str(count), (x, y), cv2.FONT_HERSHEY_COMPLEX, size_text, (0, 0, 0), 1)
        # cv2.putText(drawing, str(c_idx), (x, y), cv2.FONT_HERSHEY_COMPLEX, size_text, (50, 0, 0), tick_text)
        # cv2.putText(drawing, " " + str(c_idx), (x+10, y+10), cv2.FONT_HERSHEY_COMPLEX, size_text, (50, 0, 0), tick_text)
    file_name = file_name.split("/")[-1]
    if _fF is not None:
        title = file_name + "_{}_cols {} F1".format(len(cc), _fF)
    else:
        title = file_name + "_{}_cols".format(len(cc))
    if gt:
        file_name = file_name + "_gt"
    
    if errors is not None:
        color_fp, color_fn = (255,0,0), (0,255,0)
        done = set()
        for i,j,type_error in errors:
            if (i,j) not in done:
                done.add((i,j))
            else:
                continue
            data_node_i, data_node_j = nodes[i], nodes[j]
            xi, yi = data_node_i[:2]
            xi = int(xi*width)
            yi = int(yi*height)
            xj, yj = data_node_j[:2]
            xj = int(xj*width)
            yj = int(yj*height)
            if type_error: # fp
                cv2.line(drawing, (xi, yi), (xj, yj), color_fp, THICKNESS)
            # else:
            #     cv2.line(drawing, (xi, yi), (xj, yj), color_fn, THICKNESS//2)
          

    for group, l_group in bb_per_group.items():
        xs = np.array([[x[0], x[2]] for x in l_group]).flatten()
        ys = np.array([[x[1], x[3]] for x in l_group]).flatten()
        color_rect = colors_cc[group]
        x_max, x_min = max(xs), min(xs)
        y_max, y_min = max(ys), min(ys)
        cv2.rectangle(drawing, (x_min, y_min), (x_max, y_max),
                      # color=(50, 128, 5),
                      # color=results.get(count, (0,0,0)),
                      color=color_rect,
                      thickness=3,
        )
        # break

    show_cells(drawing,  title=title, fname=file_name, dir_to_save=dir_to_save)
    return count_fail, words_failed

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

def main():
    """
    Quick script to show mask images stored on pickle files
    """
    ORACLE = False
    conjugate = True
    all_edges = False
    min_w = 0.5
    type_ = sys.argv[2]
    spans = len(sys.argv) > 3 and sys.argv[3].lower() in ["true", "si"]
    if spans:
        type_ = "span"
    dir_to_save_local = "connected_components_{}".format(type_)
  
    data_path = sys.argv[1]
    files_to_use = "/data/HisClima/DatosHisclima/test.lst"
    if files_to_use is not None:
        files_to_use = read_lines(files_to_use)

    data_pkls = "/home/jose/projects/RPN_LSTM/works/work_2_TextLine/results/graphs_preprocessed/graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_textlines"
    # data_pkls = "/home/jose/projects/RPN_LSTM/works/work_2_TableCell/results/graphs_preprocessed/graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_tablecell"
    # data_pkls = "/data/HisClima/DatosHisclima/graphs/graphs_preprocessed/graph_k10_wh0ww0jh1jw1_min0_maxwidth0.5_tablecell" # COL
    # data_pkls = "/data/HisClima/DatosHisclima/graphs/graphs_preprocessed/graph_k10_wh0ww0jh10jw1_min0_maxwidth0.5"
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
    print("Saving data on : {}".format(dir_to_save))
    # data_path = "/data/READ_ABP_TABLE/dataset111/graphs_preprocesseds/graphs_structure/graphs_structure_PI_fasttext_0.05_radius0.02"
    file_list = tqdm(file_list, desc="Files")
    nOk, nErr, nMiss = 0,0,0
    fP, fR, fF = 0,0,0
    min_cc = 10000000
    file_search = "52709_004"
    file_names_ = [x.split("-edge")[0] for x in results.keys()]
    # print(results)
    
    for raw_path in file_list:
        file_name_p = raw_path.split("/")[-1].split(".")[0]
        # raw_path_orig = os.path.join(data_path_orig, file_name_p+".pkl")
        # print(file_name_p, file_names_)
        if files_to_use is not None and file_name_p not in files_to_use:
            continue
        # if file_name_p not in file_names_:
        #     continue
        # if file_search not in raw_path:
        #     continue

        # Read data from `raw_path`.
        print("File: {}".format(raw_path))
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
        # edges_orig = data_load_orig['edges']
        # labels_orig = data_load_orig['labels']
        # edge_features_orig = data_load_orig['edge_features']
        edge_features = data_load['edge_features']
        # print("A total of {} nodes and {} edges for {}".format(len(nodes), len(edges), raw_path))
        # print(len(data_load["conjugate"]))
       
        ant_nodes = edges
        file_name = raw_path.split("/")[-1].split(".")[0]
        img = load_image(os.path.join(dir_img, file_name))

        
        hyps = results.get(file_name_p)
        # print(hyps)
        # exit()
        cc, cc_gt = {}, {}
        for i, node in enumerate(nodes):
            # ctype = labels[i][type_]
            r, c = labels[i]['row'], labels[i]['col']
            if type_ == "row":
                l = r
            elif type_ == "col":
                l = c
            if r == -1 or c == -1:
                # ctype = -1
                pass
            hyp_node_i = hyps.get(str(i))

            cc_node = cc.get(hyp_node_i, [])
            cc_node.append(i)
            cc[hyp_node_i] = cc_node

            cc_node_gt = cc_gt.get(l, [])
            cc_node_gt.append(i)
            cc_gt[l] = cc_node_gt


        cc = [y for x,y in cc.items()]
        cc_gt = [y for x,y in cc_gt.items()]

        cc_gt.sort()
        cc.sort()
        
        _nOk, _nErr, _nMiss = evalHungarian(cc, cc_gt, 1.0, jaccard_distance)
        _fP, _fR, _fF = computePRF(_nOk, _nErr, _nMiss)
        # print(cc, len(cc))
        # print("  --   ")
        # print(cc_gt, len(cc_gt))
        # print(raw_path,  _nOk, _nErr, _nMiss, _fP, _fR, _fF)
        # print("{}, _nOk {}, _nErr {}, _nMiss {}, P: {} R: {} F1: {} - \n".format(raw_path, _nOk, _nErr, _nMiss, _fP, _fR, _fF))
        min_cc = min(min_cc, min([len(x) for x in cc_gt]))
        nOk += _nOk
        nErr += _nErr
        nMiss += _nMiss
        fP += _fP
        fR += _fR
        fF += _fF

        #     gt_edges_graph_dict[k] = v
        gt_edges_graph_dict = None
        if print_bbs:
            count_fail_, words_failed_ = print_BBs(nodes, labels, cc,
                                               gt_edges_graph_dict, file_name, dir_to_save, img, _fF, 
                                            #    nodes_orig=nodes_orig, edges_orig=edges_orig, labels_orig=labels_orig, type=type_, 
                                            #    edge_features_orig=edge_features_orig
                                               )
            # count_fail_, words_failed_ = print_BBs(nodes, labels, cc_gt,
            #                                    gt_edges_graph_dict, file_name, dir_to_save, img, _fF, gt=True)

            # count_fail_, words_failed_ = print_BBs(nodes, labels, cc,
            #                                    gt_edges_graph_dict, file_name, dir_to_save, img, _fF, gt=True, errors=errors)
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
