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

def get_all(path, ext="pkl"):
    file_names = glob.glob("{}*.{}".format(path,ext))
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
    plt.savefig("{}/{}.png".format(dir_to_save, fname), format='png', dpi=800)
    # plt.savefig("{}/{}.jpg".format(dir_to_save, title))
    plt.close()



def read_results(fname):
    f = open(fname, "r")
    lines = f.readlines()
    f.close()
    results = {}
    for line in lines[1:]:
        id_line, label, prediction = line.split(" ")
        id_line = id_line.split("/")[-1].split(".")[0]
        results[id_line] = (int(label), np.exp(float(prediction.rstrip())) )

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

def print_BBs(nodes, edges, edge_features, labels, results,
              results_gt, file_name, dir_to_save, drawing, ant_nodes):
    radius = 10
    color = (255, 0, 0)
    shape = drawing.shape
    drawing_hyp = np.copy(drawing)
    height, width = shape[0], shape[1]
    # file_name = file_name.
    count_fail = 0
    words_failed = []
    # print(len(nodes))
    # print(len(labels))
    # exit()
    THICKNESS = 10
    for count, data_node in enumerate(nodes):
        label_node = labels[count]['col']
        x, y, w, h, prob_node = data_node[:5]
        x = int(x*width)
        y = int(y*height)
        w = int(w*width)
        h = int(h*height)
        bb = [
            x-(w//2), y-(h//2),x+(w//2), y+(h//2)
        ]
        x_min, y_min, x_max, y_max = map(int, bb)
        # print(bb[0])
        # print(bb[1])
        # print(results[count])
        color_rect = (0,80,0)
        cv2.rectangle(drawing, (x_min, y_min), (x_max, y_max),
                      # color=(50, 128, 5),
                      # color=results.get(count, (0,0,0)),
                      color=color_rect,
                      thickness=-1,
                      )
        cv2.rectangle(drawing_hyp, (x_min, y_min), (x_max, y_max),
                      # color=(50, 128, 5),
                      # color=results.get(count, (0,0,0)),
                      color=color_rect,
                      thickness=-1,
                      )
        # cv2.putText(drawing, str(count), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(drawing, str(label_node), (x+10, y+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 0, 0), 1)

    aux_dict = {}
    for idx, (i,j) in enumerate(ant_nodes):
        aux_dict[(i,j)] = idx
    total_erros = []
    total_hyp = []
    for count, (i,j) in enumerate(edges):
        idx_edge = aux_dict.get((i,j), aux_dict.get((j,i)))
        key = "{}_edge{}".format(file_name, idx_edge)
        data_edge = edge_features[count]
        gt, hyp_prob = results[key]
        hyp = 0

        label_edge = results_gt.get((i,j), results_gt.get((j,i)))
        # gt == label_edge -> True
        x, y = data_edge[6], data_edge[7]
        orig_x, orig_y = data_edge[2], data_edge[3]
        dest_x, dest_y = data_edge[4], data_edge[5]

        # print("{} {} -> {} {}".format(orig_x, orig_y, dest_x, dest_y))
        orig_x = int(orig_x * width)
        x = int(x * width)
        orig_y = int(orig_y * height)
        dest_x = int(dest_x * width)
        dest_y = int(dest_y * height)
        y = int(y * height)

        # slope_value = abs(slope(orig_x, orig_y, dest_x, dest_y))
        if hyp_prob > 0.95:
            hyp = 1
        else:
            hyp = 0
        label_edge = hyp == gt
        total_hyp.append(hyp)
        total_erros.append(label_edge)
        if hyp_prob < 0.1:
            hyp_prob = '0'
        else:
            hyp_prob = str(hyp_prob)[:4]
        if hyp:
            cv2.line(drawing_hyp, (orig_x, orig_y), (dest_x, dest_y), (0, 0, 255), THICKNESS // 2)
            cv2.putText(drawing_hyp, hyp_prob, (x+10, y+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        if label_edge:
            # pass
            cv2.line(drawing, (orig_x, orig_y), (dest_x, dest_y), (0, 0, 255), THICKNESS // 2)
        else:
            # cv2.circle(drawing, (x, y), radius, (0, 0, 255), 100, )
            cv2.line(drawing, (orig_x, orig_y), (dest_x, dest_y), (255, 0, 0), THICKNESS // 2)
        cv2.putText(drawing, hyp_prob, (x+10, y+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0), 1)
    # print(file_name)
    # print(np.array(total_hyp).sum())
    # print(len(edges))
    # exit()
    acc = np.mean(total_erros)
    # print("Acc for {} - {}".format(file_name, acc))
    show_cells(drawing_hyp,  title=file_name+" {}".format(acc), fname=file_name+"_hyp", dir_to_save=dir_to_save)
    show_cells(drawing,  title=file_name+" {}".format(acc), fname=file_name, dir_to_save=dir_to_save)
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

def main():
    """
    Quick script to show mask images stored on pickle files
    """
    # data_path = sys.argv[1]
    # results = sys.argv[2]
    # dir_img = sys.argv[3]
    data_path = "/data/READ_ABP_TABLE/dataset111/graphs_structure/graphs_structure_PI_0.15_ZDEPTH/"
    results = "/home/jprieto/projects/TableUnderstanding/works/works_conjugate_CELL/work_graph_CELL_PI_0.15_4000epochs_16,16,16ngfs_base_1_model_EdgeFeatsConv_root/results.txt"
    # data_path = "/data/READ_ABP_TABLE/dataset111/graphs_structure/graphs_structure_BL/"
    # results = "/home/jprieto/projects/TableUnderstanding/work_prueba_BL2/results.txt"
    dir_img = "/data/READ_ABP_TABLE/dataset111/img/"
    dir_save_to = "results_postprocess"
    # dir_save_to = "all_graph_gt_pruned"
    # data_path = "/data/HisClima/graphs_structure/"
    # results = "/home/jprieto/projects/TableUnderstanding/works/works_conjugate_HisClima_COL/work_graph_COL_400epochs_16,16,16ngfs_base_1/results.txt"
    # dir_img = "/data/HisClima/labeled_by_Vero/"

    file_list = get_all(data_path)
    if results == "cols":
        dir_to_save = os.path.join(data_path,dir_save_to )
    elif results != "no":
        results = os.path.abspath(results)
        dir_to_save = "/".join(results.split("/")[:-1])
        dir_to_save = os.path.join(dir_to_save, dir_save_to)
        results = read_results(results)
        print("A total of {} lines classifieds in results".format(len(results)))
    else:
        dir_to_save = os.path.join(results, dir_save_to)
    #TODO borrar
    # dir_to_save = os.path.join(data_path, "results_postprocess")
    # End
    create_dir(dir_to_save)
    print("Saving data on : {}".format(dir_to_save))
    # data_path = "/data/READ_ABP_TABLE/dataset111/graphs_preprocesseds/graphs_structure/graphs_structure_PI_fasttext_0.05_radius0.02"
    file_list = tqdm(file_list, desc="Files")
    for raw_path in file_list:
        # if "0011_S_Thyrnau_005_0108" not in raw_path:
        #     continue

        # Read data from `raw_path`.
        # print("File: {}".format(raw_path))
        f = open(raw_path, "rb")
        data_load = pickle.load(f)
        f.close()
        ids = data_load['ids']
        nodes = data_load['nodes']
        edges = data_load['edges']
        labels = data_load['labels']
        edge_features = data_load['edge_features']
        # print("A total of {} nodes and {} edges for {}".format(len(nodes), len(edges), raw_path))
        ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, \
        new_edge_feats, ant_nodes, ant_edges = conjugate_nx(ids,
                nodes, edges, labels, edge_features, idx=aux1, list_idx=[525,626,161,549])

        file_name = raw_path.split("/")[-1].split(".")[0]
        img = load_image(os.path.join(dir_img, file_name))

        # if len(new_nodes) != len(edges):
        #     print("--------")
        #     print("Error: len nodes {} and len edges {} on {}".format(len(new_nodes), len(edges), file_name))
        #     exit()


        gt_edges_graph = list(zip(ant_nodes, new_labels_cols))

        gt_edges_graph_dict = {}
        for k,v in gt_edges_graph:
            gt_edges_graph_dict[k] = v

        # G = nx.Graph()
        # # G.add_nodes_from(nodes)
        # for i, node in enumerate(nodes):
        #     G.add_node(i, attr=node)
        # weighted_edges_all = [(i,j,results.get("{}_edge{}".format(file_name, count))[1]) for count, (i,j) in enumerate(edges)]
        # weighted_edges = [(i, j, w) for i, j, w in weighted_edges_all if w > 0.5]
        # print(len(weighted_edges))
        # G.add_weighted_edges_from(weighted_edges)
        # print(len(G.edges()))
        # cc = nx.connected_components(G)
        # cc = [list(c) for c in cc]
        # print(len(cc))
        # exit()

        count_fail_, words_failed_ = print_BBs(nodes, edges, edge_features, labels, results,
                                               gt_edges_graph_dict, file_name, dir_to_save, img, ant_nodes)
        # exit()


if __name__ == "__main__":
    main()
    # if len(sys.argv) > 1 and sys.argv[1] != "-h":
    #     main()
    # else:
    #     print("Usage: python {} <dir with GT pkl> <file with results (txt)> <dir with the REAL images to load>".format(sys.argv[0]))
