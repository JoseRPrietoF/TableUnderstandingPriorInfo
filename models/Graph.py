import torch.nn.functional as F
import torch_geometric.nn as geo_nn
import torch
from torch import nn
import torch_geometric as tg
from torch_geometric.data import Data
# from torch_geometric.nn.conv import MessagePassing
from torch.nn import Parameter
from torch.nn import Sequential as Seq, Linear, ReLU
from torch_geometric.utils import to_undirected
from models.pna import PNAConvSimple
from torch_geometric.utils import degree
import numpy as np
try:
    from models.nnconv import EdgeFeatsConv, EdgeFeatsConvMult, ECNConv, EdgeConv2
except:
    from .nnconv import EdgeFeatsConv, EdgeFeatsConvMult, ECNConv, EdgeConv2
from models.operations import Mish, Mish_aux

# act_func = nn.ReLU
act_func = Mish

class myNNConv(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(myNNConv, self).__init__()
        mlp = Seq(Linear(2 * in_c, out_c),
                  act_func(),
                  Linear(out_c, out_c))
        self.k = opts.knn
        self.dynamic = self.k > 0
        self.NNConv = geo_nn.EdgeConv(mlp)
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dynamic:
            print("dynamic, ", self.k)
            edge_index = geo_nn.knn_graph(x, self.k, 
            # batch=batch, 
            loop=False, cosine=False)
            edge_index = to_undirected(edge_index)
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
        x = self.NNConv(x, edge_index,)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class myNNConv2(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(myNNConv2, self).__init__()
        mlp = Seq(Linear(2 * in_c, out_c),
                  act_func(),
                  Linear(out_c, out_c))
        self.k = opts.knn
        self.dynamic = self.k > 0
        self.NNConv = EdgeConv2(mlp)
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dynamic:
            print("dynamic, ", self.k)
            edge_index = geo_nn.knn_graph(x, self.k, 
            # batch=batch, 
            loop=False, cosine=False)
            edge_index = to_undirected(edge_index)
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
        x = self.NNConv(x, edge_index,)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class EdgeFeatsConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(EdgeFeatsConvNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        mlp = Seq(Linear((2 * in_c) + num_edge_features, out_c),
                  act_func(),
                  Linear(out_c, out_c))
        print(mlp)
        self.NNConv = EdgeFeatsConv(in_c=in_c,
                                    out_c=out_c,
                                    nn=mlp,
                                    root_weight=root_weight
                                    )
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
            # x = F.dropout(x, p=0.1)
        x = self.NNConv(x, edge_index, edge_attr)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class PNANN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=False, dropout=False, dataset=None, opts={}):
        super(PNANN, self).__init__()
        
        self.NNConv = PNAConvSimple(in_c,
                                    out_c,
                                    dataset,
                                    )
        # self.NNConv = EdgeFeatsConv(in_c, out_c, mlp)
        # self.NNConv = geo_nn.GraphUNet(in_c, 4, out_c,depth=2, pool_ratios=0.5)
        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
            # x = F.dropout(x, p=0.1)
        x = self.NNConv(x, edge_index, edge_attr)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class EdgeFeatsConvMultNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(EdgeFeatsConvMultNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        # num_node_features = dataset.num_node_features
        mlp_nodes = Seq(Linear(2 * in_c, out_c),
                  act_func(),
                  Linear(out_c, out_c))
        mlp_edges = nn.Sequential(
            nn.Linear(num_edge_features + in_c, out_c),
            act_func(),
            # nn.Linear(num_node_features,num_node_features)
        )

        print(mlp_nodes)
        print(mlp_edges)
        self.NNConv = EdgeFeatsConvMult(
            in_c=in_c,
            out_c=out_c,
            nn_nodes=mlp_nodes,
            nn_edges=mlp_edges,
            root_weight=root_weight
        )

        self.bn = bn
        self.dropout = dropout
        if bn:
            self.batchNorm = nn.BatchNorm1d(out_c)
    def forward(self, data):
        x, edge_index, edge_attr = data
        if self.dropout:
            edge_index, edge_attr = tg.utils.dropout_adj(edge_index, edge_attr,
                                                     p=0.1, force_undirected=True,
                                                     )
            # x = F.dropout(x, p=0.1)
        x = self.NNConv(x, edge_index, edge_attr)
        if self.bn:
            x = self.batchNorm(x)
        return x, edge_index, edge_attr

class GatConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(GatConvNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features

        self.NNConv = tg.nn.GATConv(in_channels =in_c,
                                    out_channels =out_c,
                                    heads=4,
                                    concat=False
                                    )
    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.NNConv(x, edge_index)
        return x, edge_index, edge_attr

class ECNConvNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(ECNConvNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        self.NNConv = ECNConv(in_channels =in_c,
                                    out_channels =out_c,
                                    num_edge_features=num_edge_features,
                                    # concat=False
                                    )
    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.NNConv(x, edge_index, edge_attr)
        return x, edge_index, edge_attr

class TransformerNN(torch.nn.Module):
    def __init__(self, in_c, out_c, bn=True, dropout=True, dataset=None, opts={}):
        super(TransformerNN, self).__init__()
        root_weight = opts.root_weight
        num_edge_features = dataset.num_edge_features
        self.NNConv = geo_nn.TransformerConv(in_channels =in_c,
                                    out_channels =out_c,
                                    heads=4,
                                    edge_dim = num_edge_features,
                                    concat=False,
                                    dropout=0.3
                                    )
    def forward(self, data):
        x, edge_index, edge_attr = data
        x = self.NNConv(x, edge_index, edge_attr)
        return x, edge_index, edge_attr

class Net(torch.nn.Module):

    def __init__(self, dataset, opts):
        """
        model default: EdgeConv
        """
        super(Net, self).__init__()
        self.opts = opts
        layers = opts.layers
        model = opts.model
        model = model.lower()
        self.multihead = opts.multihead
        # layer_used = "edgeconv"
        # model = "ecn"
        # layer_used = opts.model
        # model = "pna"
        if model == "edgeconv":
            layer_used = myNNConv
        elif model == "edgeconv2":
            layer_used = myNNConv2
        elif model == "edgefeatsconv":
            layer_used = EdgeFeatsConvNN
        elif model == "edgefeatsconvmult":
            layer_used = EdgeFeatsConvMultNN
        elif model == "pna":
            layer_used = PNANN
        elif model == "gat":
            layer_used = GatConvNN
        elif model == "ecn":
            layer_used = ECNConvNN
        elif model == "transformer":
            layer_used = TransformerNN
        else:
            print("Model {} not implemented".format(model))
            exit()

        print("Using {} layers".format(layer_used))
        # model = [
        #     EdgeFeatsConvNN(dataset.num_node_features, layers[0], dataset=dataset, opts=opts,),
        #     myNNConv(layers[0], layers[1], dataset=dataset, opts=opts,),
        #          ]
        # EdgeFeatsConvNN(dataset.num_node_features, layers[0], dataset=dataset, opts=opts,),
        model = [
            layer_used(dataset.num_node_features, layers[0], dataset=dataset, opts=opts,),
            Mish_aux(),
                 ]

        if len(layers) > 1 and not self.opts.multihead:
            for i in range(1, len(layers)):

                model = model + [
                    layer_used(layers[i - 1], layers[i], dataset=dataset, opts=opts,),
                    Mish_aux(),
                ]
        
        if self.opts.classify != "EDGES" and not self.opts.multihead:
            model = model + [
                layer_used(layers[-1], dataset.num_classes, bn=False, dropout=False, dataset=dataset, opts=opts,),
                
            ]
        elif self.opts.classify == "EDGES":
            #+ dataset.num_edge_features
            self.mlp_edges = nn.Sequential(nn.Linear(layers[-1] + dataset.num_edge_features , layers[-1]), nn.ReLU(True), nn.Linear(layers[-1], 2))
        
        if self.opts.multihead and len(layers) > 1:
            self.heads_aux = [None] * (dataset.last_layout+1)
            for i in range(1, len(layers)-opts.numConvsHead):
                model = model + [
                    layer_used(layers[i - 1], layers[i], dataset=dataset, opts=opts,),
                    Mish_aux(),
                ]
            for num_h in range(dataset.last_layout+1):
                model_h = []
                for i in range(len(layers)-opts.numConvsHead, len(layers)):
                    model_h += [
                            layer_used(layers[i - 1], layers[i], dataset=dataset, opts=opts,),
                            Mish_aux(),
                        ]
                model_h = model_h + [
                    layer_used(layers[-1], dataset.num_classes, bn=False, dropout=False, dataset=dataset, opts=opts,),
                ]
                self.heads_aux[num_h] = model_h
            for k,v in enumerate(self.heads_aux):
                self.heads_aux[k] = nn.Sequential(*v)
            self.heads = nn.ModuleList(self.heads_aux)
        self.model = nn.Sequential(*model)
        if self.opts.multihead:
            for k,v in enumerate(self.heads):
                self.heads[k] = nn.Sequential(*v)
                # print( self.heads[k] )
        self.num_params = 0
        for param in self.parameters():
            self.num_params += param.numel()

    def get_batch(self, x, edge_index_orig, edge_attr_orig, data):
        data_res = []
        batch = data.batch
        batch_sizes = []
        x_list, edge_index_list, edge_attr_list = [], [], []
        edge_index_orig_i, edge_index_orig_j = edge_index_orig[0], edge_index_orig[1]
        x_list_aux, edge_index_list_aux_i, edge_index_list_aux_j, edge_attr_list_aux, ys = [], [], [], [], []
        last_batch = 0
        last_index = 0
        last_index_edges = 0
        all_ys = []
        # print(edge_index_orig_i)
        # print(edge_index_orig_j)
        # print(x.shape, edge_index_orig.shape)
        
        for i, x_i in enumerate(x):
            num_batch = batch[i]
            added = False
            # print(i)
            if (num_batch != last_batch and i>0) or (i == len(x)-1 and i != 0):
                # print("i, ",i)
                
                if i == len(x)-1:
                    x_list_aux.append(x_i.detach().cpu().numpy())
                    data_y = data.y[i].detach().cpu().item()
                    all_ys.append(data_y)
                    ys.append(data_y)
                    added = True
                batch_sizes.append(i)
                # print(last_index_edges)
                # print(range(last_index_edges, len(edge_index_orig_i)))
                for j in range(last_index_edges, len(edge_index_orig_i)):
                    if (edge_index_orig_i[j] >= last_index and edge_index_orig_i[j] <= i \
                        and edge_index_orig_j[j] >= last_index and edge_index_orig_j[j] <= i):
                        ii = edge_index_orig_i[j].item()-last_index
                        jj = edge_index_orig_j[j].item()-last_index
                        edge_index_list_aux_i.append(ii)
                        edge_index_list_aux_j.append(jj)
                        edge_attr_list_aux.append(edge_attr_orig[j].detach().cpu().numpy())
                    else:
                        # print( edge_index_orig_i[j] >= last_index, edge_index_orig_i[j] < i, edge_index_orig_j[j] >= last_index, edge_index_orig_j[j] < i)
                        # print(f'edge_index_orig_j[j] {edge_index_orig_j[j]} i {i}')
                        last_index_edges = j
                        break
                
                last_index = i
                last_batch = num_batch
                edge_index = np.array([edge_index_list_aux_i, edge_index_list_aux_j])
                min_node, max_node = 0, len(x_list_aux)
                # print(i, len(x_list_aux), max(edge_index_list_aux_i), min(edge_index_list_aux_i), max(edge_index_list_aux_j), min(edge_index_list_aux_j) )
                # print(min(min(edge_index_list_aux_i), min(edge_index_list_aux_j)))
                # print(max(max(edge_index_list_aux_i), max(edge_index_list_aux_j)))
                for jjj, iii in zip(edge_index_list_aux_i, edge_index_list_aux_j):
                    if min_node < jjj >= max_node or min_node < iii >= max_node:
                        print(f'batch {num_batch} - num_nodes : {len(x_list_aux)}    - {min_node}, {max_node} - , {jjj}, {iii}')
                        exit()
                # print("---")
                if not edge_index_list_aux_i:
                    print(i, num_batch, )
                    print(f'len(x_list_aux) -> {len(x_list_aux)} ')
                    print(f'edge_index_list_aux_i {edge_index_list_aux_i}')
                    raise
                edge_index = torch.tensor(edge_index, dtype=torch.long)
                labels_tensor = torch.tensor(ys, dtype=torch.long)
                info_edges = torch.tensor(edge_attr_list_aux, dtype=torch.float)
                x_tensor = torch.tensor(x_list_aux, dtype=torch.float)
                # print(f'len(x_list_aux) -> {x_tensor.shape} ')
                data_aux = Data(x=x_tensor,
                        edge_index=edge_index,
                        y=labels_tensor,
                        edge_attr=info_edges,
                        ).cuda()
                data_res.append(data_aux)
                x_list_aux, edge_index_list_aux_i, edge_index_list_aux_j, edge_attr_list_aux, ys = [], [], [], [], []
            x_list_aux.append(x_i.detach().cpu().numpy())
            data_y = data.y[i].detach().cpu().item()
            all_ys.append(data_y)
            ys.append(data_y)
        all_ys = torch.tensor(all_ys, dtype=torch.long)
        # print(i, len(data_res), data_res)
        # print(edge_index_orig_i[0], edge_index_orig_j[0])
        return data_res, all_ys

    def forward(self, data):
        x, edge_index_orig = data.x, data.edge_index
        
        edge_attr_orig = data.edge_attr
        x, edge_index, edge_attr = self.model([x, edge_index_orig, edge_attr_orig])
        if self.multihead:
            type_table = data.type_table

            data_unbatched, all_ys = self.get_batch(x, edge_index_orig, edge_attr, data)
 
            x_list = []
            for i, d in enumerate(data_unbatched):
                type_table_i = type_table[i]
                x_, edge_index_, edge_attr_ = d.x, d.edge_index, d.edge_attr
                # print("---> ", x_.shape)
                # print(edge_index_)
                # max_nodes = x_.shape[0]
                # ii, jj = edge_index_[0], edge_index_[1]
                # max_ = max(max(ii), max(jj))
                # print(ii)
                # print(jj)
                # print(max_nodes, max_)
                # if max_ >= max_nodes:
                #     exit()
                # exit()

                # try:
                # print(f'type_table_i {type_table_i}')
                x1, _, _ =  self.heads[type_table_i]([x_, edge_index_, edge_attr_])
                # except Exception as e:
                #     # print( self.heads[type_table_i])
                #     print("----------------")
                #     raise e
                x_list.append(x1)
            x = torch.cat(x_list)
            # print(x.shape)
            
        return F.log_softmax(x, dim=1)


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        try:
            nn.init.xavier_normal_(m.weight.data)
        except:
            pass
            # print("object has no attribute 'weight'")
        # init.constant(m.bias.data, 0.0)
