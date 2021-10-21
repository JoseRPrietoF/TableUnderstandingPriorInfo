import os.path as osp
import os
import glob, pickle
import torch
from torch_geometric.data import Dataset, Data, DataLoader
from torch_geometric.utils import degree
import numpy as np
# from torch_geometric.utils import grid
import networkx as nx
import logging
import sys, re
from gensim.models import FastText as ft
sys.path.append('../utils')
try:
    from conjugate import conjugate_nx
except:
    from data.conjugate import conjugate_nx
try:
    from utils.optparse_graph import Arguments as arguments
except:
    from optparse_graph import Arguments as arguments

class ABPDataset_BIESO(Dataset):
    """
    In order to create a torch_geometric.data.InMemoryDataset, you need to implement four fundamental methods:

    torch_geometric.data.InMemoryDataset.raw_file_names():
    A list of files in the raw_dir which needs to be found in order to skip the download.

    torch_geometric.data.InMemoryDataset.processed_file_names():
    A list of files in the processed_dir which needs to be found in order to skip the processing.

    torch_geometric.data.InMemoryDataset.download():
    Downloads raw data into raw_dir.

    torch_geometric.data.InMemoryDataset.process():
    Processes raw data and saves it into the processed_dir.
    """
    def __init__(self, root, split, flist, opts=None, transform=None, pre_transform=None, onehot = False, dict_layouts = {}):
        # super(ABPDataset_BIESO, self).__init__(root, transform, pre_transform, None)
        self.root = None
        self.transform = transform
        self.pre_transform = pre_transform
        self.pre_filter = None
        self._indices = None

        self.opts = opts
        self.onehot = onehot
        self.split = split
        self.processed_dir_ = os.path.join(opts.work_dir, split)
        self.multihead = opts.multihead

        if not os.path.exists(self.processed_dir_):
            os.mkdir(self.processed_dir_)
        self.flist = flist
        self.flist_processed = []
        self.len_ = 0
        # self.processed_dir_ = self.path
        self.transform = transform
    
        if opts is not None:
            self.conjugate = self.opts.conjugate
        else:
            self.conjugate = "COL"
        
        # if self.opts.conjugate == "COL":
        #     self.num_classes = 17  -> HisClima
        # elif self.opts.conjugate == "ROW":
        #     self.num_classes = 28 # TODO  -> HisClima
        self.num_classes = opts.num_classes
        self.ids = []
        self.labels = []
        self.last_layout = 0
        self.dict_layouts = {}
        if self.multihead and split == "train":

            self.process_layouts()
        elif self.multihead:
            self.dict_layouts = dict_layouts

    def get_prob_class(self):
        # print(self.labels)
        _, counts = np.unique(self.labels, return_counts=True)
        print(np.unique(self.labels, return_counts=True))
        # _, counts = np.unique(np.argmax(self.labels, axis=1), return_counts=True)
        total = counts.sum()
        class_weights = counts / total   
        if len(class_weights) != self.num_classes:
            class_weights = np.append(class_weights, [0])
        return class_weights

    @property
    def raw_file_names(self):
        # return self.flist
        return []
    @property
    def processed_file_names(self):
        # return self.flist
        return []

    def len(self):
        """
        Returns the number of examples in your dataset.
        :return:
        """
        # print(self.processed_file_names)
        return self.len_

    def download(self):
        # Download to `self.raw_dir`.
        # self.raw_paths = []
        print("Download?")

    def add_label_to(self, edges, labels):
        # print(len(edges), len(labels))
        # print(edges)
        for i, info in enumerate(edges):
            x = list(edges[i])
            l = np.zeros(17)
            l[labels[i]] = 1
            # print(x)
            # x.extend(l)
            x = l
            # print(x)
            # exit()
            
            edges[i] = x
        return edges

    def process(self):
        self._process()
    
    def read_results(self, fname):
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

    def get_basename(self, raw_path):
        file_name = raw_path.split("/")[-1].split(".")[0]
        string = ""
        for i in re.split('_|-|\n',file_name):
            string += i + "_"
            if i.isnumeric():
                string = string[:-1]
                return string

    def process_layouts(self):
        s = set()
        for idx_flist, raw_path in enumerate(self.flist):
            string = self.get_basename(raw_path)
            s.add(string)
            
        for i, si in enumerate(s):
            self.dict_layouts[si] = i
        self.last_layout = i
            
    def _process(self):
        i = 0
        node_numeration = 0
        # if not os.path.exists(osp.join(self.root, 'processed')):
        #     os.mkdir(osp.join(self.root, 'processed'))
        self.labels = []
        # pna
        # self.deg = torch.zeros(90, dtype=torch.long)
        min_w = 0
        type_table = 0
        if self.opts.do_prune:
            if self.split == "train":
                path_prune = self.opts.results_prune_tr
            elif self.split == "test":
                path_prune = self.opts.results_prune_te
            results = self.read_results(path_prune)
            min_w = self.opts.min_w
        for idx_flist, raw_path in enumerate(self.flist):
            
            # Read data from `raw_path`.
            file_name = raw_path.split("/")[-1].split(".")[0]
            basename = self.get_basename(file_name)
            if self.multihead:
                type_table = self.dict_layouts.get(basename, self.last_layout+1)
            f = open(raw_path, "rb")
            data_load = pickle.load(f)
            f.close()
            ids = data_load['ids']
            edge_index = np.array(data_load['edges']).T
            # if self.split == "prod":
                # print(raw_path)
                # print(edge_index)
                # if len(edge_index) == 0:
                #     exit()
            info_edges = data_load['edge_features']
            nodes = data_load['nodes']
            if len(nodes) == 0:
                continue
            self.len_ += 1
            labels = []
            if self.split != "prod":
                for label in data_load['labels']:
                    l = label[self.opts.conjugate.lower()]
                    if l > 27:
                        print(raw_path)
                    if l == -1:
                        l = self.num_classes - 1 
                        # print(raw_path)
                    labels.append(l)
                
                self.labels.extend(labels)
                # print(raw_path, np.unique(labels), len(np.unique(labels)))
                # if len(np.unique(labels)) > 28:
                #     exit()
                
                if len(nodes) != len(labels):
                    print("Problem with nodes and labels")
                    exit()
           

            self.ids.extend(ids)
            positions = self.get_positions(nodes)
         
            x = torch.tensor(nodes, dtype=torch.float)
            
            node_num = np.array(range(len(labels) + node_numeration))
            node_numeration += len(labels)
            labels_tensor = torch.tensor(labels, dtype=torch.long)
            edge_index = torch.tensor(edge_index, dtype=torch.long)
            info_edges = torch.tensor(info_edges, dtype=torch.float)
            positions = torch.tensor(positions, dtype=torch.float)
            node_num = torch.tensor(node_num, dtype=torch.long)
            


            data = Data(x=x,
                        edge_index=edge_index,
                        y=labels_tensor,
                        edge_attr=info_edges,
                        pos=positions,
                        node_num=node_num,
                        type_table=type_table
                        # label_fixed=label_fixed,
                        )
            # d = degree(data.edge_index[1], num_nodes=data.num_nodes, dtype=torch.long)
            # self.deg += torch.bincount(d, minlength=self.deg.numel())

            self.labels.extend(labels)
            if self.pre_filter is not None and not self.pre_filter(data):
                continue

            if self.pre_transform is not None:
                data = self.pre_transform(data)

            torch.save(data, os.path.join(self.processed_dir_, 'data_{}.pt'.format(i)))
            self.flist_processed.append(os.path.join(self.processed_dir_, 'data_{}.pt'.format(i)))
            i += 1
        print(np.unique(self.labels))
        if self.split != "prod":
            self.prob_class = self.get_prob_class()
            # print("Deg: ", self.deg)
            # exit()

    def get_positions(self, nodes):
        res = []
        for node in nodes:
            res.append([node[6], node[7]]) # mid point
        return res

    def get(self, idx):
        """Implements the logic to load a single graph.
        Internally, torch_geometric.data.Dataset.__getitem__() gets data objects
        from torch_geometric.data.Dataset.get() and optionally transforms them according to transform.
        """
        data_save = torch.load(osp.join(self.processed_dir_, 'data_{}.pt'.format(idx)))
        if self.transform:
            data_save = self.transform(data_save)
        return data_save


def get_all(path, ext="pkl"):
    if path[-1] != "/":
        path = path+"/"
    file_names = glob.glob("{}*.{}".format(path,ext))
    return file_names

def tensor_to_numpy(tensor):
    return tensor.cpu().detach().numpy()

if __name__ == "__main__":
    def prepare():
        """
        Logging and arguments
        :return:
        """

        # Logger
        logger = logging.getLogger(__name__)
        logger.setLevel(logging.INFO)
        ch = logging.StreamHandler()
        # --- keep this logger at DEBUG level, until aguments are processed
        ch.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(module)s - %(levelname)s - %(message)s"
        )
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # --- Get Input Arguments
        in_args = arguments(logger)
        opts = in_args.parse()


        fh = logging.FileHandler(opts.log_file, mode="a")
        fh.setLevel(logging.INFO)
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        # --- restore ch logger to INFO
        ch.setLevel(logging.INFO)

        return logger, opts

    logger, opts = prepare()
    opts.tr_data = "/data/READ_ABP_TABLE/dataset111/graphs_preprocesseds/graphs_structure/graphs_structure_PI_0.15_lentext"
    opts.fix_class_imbalance = True
    opts.conjugate = "ALL"
    opts.classify = "NO"
    flist = get_all(opts.tr_data)
    dataset_tr = ABPDataset_BIESO(root=opts.tr_data, split="dev", flist=get_all(opts.tr_data), transform=None, opts=opts)
    dataloader = DataLoader(
        dataset_tr,
        batch_size=opts.batch_size,
        shuffle=False,
        # num_workers=opts.num_workers,
        # pin_memory=opts.pin_memory,
    )
    total_nodes = 0
    total_edges = 0
    total_nodes_conjugation = 0
    total_edges_conjugation = 0
    debug = False
  
    dataset_tr.process()
    dataset_tr.get_prob_class()
    res = []
    res_gt = []
    for v_batch, v_sample in enumerate(dataloader):
        # f_names = v_sample.fname
        y_gt = tensor_to_numpy(v_sample.y)
        print(y_gt)
        res_gt.append(y_gt)

    labels = np.hstack(res_gt)
    results_test = list(zip(dataset_tr.ids, labels))
    for i in results_test:
        print(i)
    # print("A total of {} nodes and {} edges".format(total_nodes, total_edges))
    # print("A total of {} nodes and {} edges  in conjugation".format(total_nodes_conjugation, total_edges_conjugation))
    # print("Prob class weight: {}".format(dataset_tr.prob_class))
    # print("A total of {} num nodes to classify".format(len(dataset_tr.labels)))
