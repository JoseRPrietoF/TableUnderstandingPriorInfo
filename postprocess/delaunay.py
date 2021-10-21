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
from scipy.spatial import Delaunay
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
    plt.savefig("{}/{}.jpg".format(dir_to_save, fname), format='jpg', dpi=800)
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

def get_color_cc(cc):
    """
    Return a dict
    :param cc:
    :return:
    """
    res = {}
    for i,_ in enumerate(cc):
        res[i] = kelly_colors[i%len(kelly_colors)]
    return res

def search_idx_in_cc(idx,cc):
    for i, c in enumerate(cc):
        if idx in c:
            return i
    return -1

def print_BBs(mst,
              file_name, dir_to_save, drawing):
    nodes = mst.nodes(data=True)
    edges = mst.edges(data=True)
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
    THICKNESS = 10
    for count, (node, data_node) in enumerate(nodes):
        data_node = data_node['attr']

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

        # cv2.putText(drawing, str(count), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 0), 1)
        # cv2.putText(drawing, str(label_node), (x+10, y+10), cv2.FONT_HERSHEY_COMPLEX, 0.5, (50, 0, 0), 1)


    for count, (i,j, attr) in enumerate(edges):

        data_i = nodes[i]['attr']
        data_j = nodes[j]['attr']
        orig_x, orig_y = data_i[0], data_i[1]
        dest_x, dest_y = data_j[0], data_j[1]


        # print("{} {} -> {} {}".format(orig_x, orig_y, dest_x, dest_y))
        orig_x = int(orig_x * width)
        x = int(x * width)
        orig_y = int(orig_y * height)
        dest_x = int(dest_x * width)
        dest_y = int(dest_y * height)
        y = int(y * height)

        # slope_value = abs(slope(orig_x, orig_y, dest_x, dest_y))
        cv2.line(drawing, (orig_x, orig_y), (dest_x, dest_y), (0, 0, 255), THICKNESS // 2)

        # cv2.putText(drawing, str(count), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)

        # cv2.line(drawing, (orig_x, orig_y), (dest_x, dest_y), (0, 0, 255), THICKNESS // 2)
        #
        # if label_edge:
        #     # pass
        #     cv2.line(drawing, (orig_x, orig_y), (dest_x, dest_y), (0, 0, 255), THICKNESS // 2)
        # else:
        #     # cv2.putText(drawing, str(count), (x, y), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255), 1)
        #     # cv2.circle(drawing, (x, y), radius, (0, 0, 255), 100, )
        #     cv2.line(drawing, (orig_x, orig_y), (dest_x, dest_y), (255, 0, 0), THICKNESS // 2)


    show_cells(drawing,  title=file_name, fname=file_name, dir_to_save=dir_to_save)
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

def triangle_to_graph(nodes):
    res = []
    for n1,n2,n3 in nodes:
        res.append([n1,n2])
        res.append([n2,n3])
        res.append([n3,n1])
    return res

def main():
    """
    Quick script to show mask images stored on pickle files
    """
    # data_path = sys.argv[1]
    # results = sys.argv[2]
    # dir_img = sys.argv[3]
    dir_res = "Delaunay"
    data_path = "/data/READ_ABP_TABLE/dataset111/graphs_structure/graphs_structure_PI_0.15_ZDEPTH/"
    results = "/home/jprieto/projects/TableUnderstanding/works/Delaunay/"
    dir_img = "/data/READ_ABP_TABLE/dataset111/img/"
    file_list = get_all(data_path)

    dir_to_save = os.path.join(results, dir_res)

    create_dir(dir_to_save)
    print("Saving data on : {}".format(dir_to_save))
    # data_path = "/data/READ_ABP_TABLE/dataset111/graphs_preprocesseds/graphs_structure/graphs_structure_PI_fasttext_0.05_radius0.02"
    file_list = tqdm(file_list, desc="Files")
    for raw_path in file_list:
        # if "0077_T" not in raw_path:
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

        points = []
        G = nx.Graph()
        for i, node in enumerate(nodes):
            G.add_node(i, attr=node)
            points.append([node[0], node[1]])
        delaunay = Delaunay(points)
        # delaunay = Delaunay(points, qhull_options="QJ")
        simplices = delaunay.simplices
        # s = np.array(simplices)
        edges_delaunay = triangle_to_graph(simplices)
        G.add_edges_from(edges_delaunay)

        # # print("A total of {} nodes and {} edges for {}".format(len(nodes), len(edges), raw_path))
        # ids, new_nodes, new_edges, new_labels_cells, new_labels_cols, new_labels_rows, \
        # new_edge_feats, ant_nodes, ant_edges = conjugate_nx(ids,
        #         nodes, edges, labels, edge_features, idx=aux1, list_idx=[525,626,161,549])

        file_name = raw_path.split("/")[-1].split(".")[0]
        img = load_image(os.path.join(dir_img, file_name))




        count_fail_, words_failed_ = print_BBs(G,
                                               file_name, dir_to_save, img)
        exit()


if __name__ == "__main__":
    main()
    # if len(sys.argv) > 1 and sys.argv[1] != "-h":
    #     main()
    # else:
    #     print("Usage: python {} <dir with GT pkl> <file with results (txt)> <dir with the REAL images to load>".format(sys.argv[0]))
