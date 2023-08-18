#
# Created on March 2022
#
# Copyright (c) 2022 Meitar Ronen
#

import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import TensorDataset
from sklearn.preprocessing import MinMaxScaler, StandardScaler, Normalizer
from ogb.nodeproppred import DglNodePropPredDataset
import dgl
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
from dgl.nn.pytorch import SGConv
import numpy as np
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
from tqdm import tqdm
import dgl.function as fn
from sklearn import preprocessing as sk_prep

def transform_embeddings(transform_type, data):
    if transform_type == "normalize":
        return torch.Tensor(Normalizer().fit_transform(data))
    elif transform_type == "min_max":
        return torch.Tensor(MinMaxScaler().fit_transform(data))
    elif transform_type == "standard":
        return torch.Tensor(StandardScaler().fit_transform(data))
    else:
        raise NotImplementedError()

class MyDataset:
    @property
    def data_dim(self):
        """
        The dimension of the loaded data
        """
        return self._data_dim

    def __init__(self, args):
        self.args = args
        self.data_dir = args.dir

    def get_train_data(self):
        raise NotImplementedError()

    def get_test_data(self):
        raise NotImplementedError()

    def get_train_loader(self):
        train_loader = torch.utils.data.DataLoader(
            self.get_train_data(),
            batch_size=self.args.batch_size,
            shuffle=True,
            num_workers=0,
        )
        return train_loader

    def get_test_loader(self):
        test_data = self.get_test_data()
        if len(test_data) > 0:     
            return torch.utils.data.DataLoader(test_data, batch_size=self.args.batch_size, shuffle=False, num_workers=0)
        else:
            return None

    def get_loaders(self):
        return self.get_train_loader(), self.get_test_loader()


class GCN(nn.Module):
    def __init__(self,
                 g,
                 in_feats,
                 n_hidden,
                 n_classes,
                 n_layers,
                 activation,
                 dropout,
                 bias = True,
                 weight=True):
        super(GCN, self).__init__()
        self.g = g
        self.layers = nn.ModuleList()
        # input layer
        self.bns = torch.nn.ModuleList()
        self.res_linears = nn.ModuleList()
        self.layers.append(GraphConv(in_feats, n_hidden, weight = weight, bias = bias, activation=activation))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden, momentum = 0.01))
        self.res_linears.append(torch.nn.Linear(in_feats, n_hidden))
        # hidden layers
        for i in range(1, n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, weight=weight, bias=bias, activation=activation))
            # if i != (n_layers - 2):
            self.bns.append(torch.nn.BatchNorm1d(n_hidden, momentum = 0.01))
            self.res_linears.append(torch.nn.Linear(n_hidden, n_hidden))
        # output layer
        self.layers.append(GraphConv(n_hidden, n_classes))
        # self.bns.append(torch.nn.BatchNorm1d(n_hidden, momentum=0.01))
        self.res_linears.append(torch.nn.Identity())
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, blocks):
        collect = []
        h = blocks[0].srcdata['feat']
        h = self.dropout(h)
        num_output_nodes = blocks[-1].num_dst_nodes()
        collect.append(h[:num_output_nodes])
        for l, (layer, block) in enumerate(zip(self.layers, blocks)):
            h_res = h[:block.num_dst_nodes()]
            h = layer(block, h)
            # if l < len(self.layers) - 1:
            #     h = self.bns[l](h)
            h = self.dropout(h)
            collect.append(h[:num_output_nodes])
            h += self.res_linears[l](h_res)
        # return torch.cat(collect, -1)
        return collect[-1]



class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, k = 1):
        super(Encoder, self).__init__()
        self.g = g
        self.gnn_encoder = gnn_encoder
        if gnn_encoder == 'gcn':
            self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)
        elif gnn_encoder == 'sgc':
            self.conv = SGConv(in_feats, n_hidden, k=10, cached=True)

    def forward(self, blocks, corrupt=False):
        if corrupt:
            for block in blocks:
                block.ndata['feat']['_N'] = block.ndata['feat']['_N'][torch.randperm(block.num_src_nodes())]
        if self.gnn_encoder == 'gcn':
            features = self.conv(blocks)
        elif self.gnn_encoder == 'sgc':
            features = self.conv(self.g, blocks)
        return features

class GGD(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, proj_layers, gnn_encoder, num_hop):
        super(GGD, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, num_hop)
        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(n_hidden, n_hidden))
        self.loss = nn.BCEWithLogitsLoss()
        self.graphconv = GraphConv(in_feats, n_hidden, weight=False, bias=False, activation=None)

    def forward(self, features, labels, loss_func):
        h_1 = self.encoder(features, corrupt=False)
        h_2 = self.encoder(features, corrupt=True)

        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)

        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)

        lbl_1 = torch.ones(1, sc_1.shape[1])
        lbl_2 = torch.zeros(1, sc_1.shape[1])
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

        logits = torch.cat((sc_1, sc_2), 1)

        loss = loss_func(logits, lbl)

        return loss

    def embed(self, blocks):
        h_1 = self.encoder(blocks, corrupt=False)

        return h_1.detach()


class TensorDatasetWrapper(TensorDataset):
    def __init__(self, data, labels):
        super().__init__(data, labels)
        self.data = data
        self.targets = labels

def graph_power(embed, g):
    feat = embed.squeeze(0)

    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5)
    norm = norm.to(feat.device).unsqueeze(1)
    for _ in range(10):
        feat = feat * norm
        g.ndata['h2'] = feat
        g.update_all(fn.copy_u('h2', 'm'),
                     fn.sum('m', 'h2'))
        feat = g.ndata.pop('h2')
        feat = feat * norm

    return feat

class NodeSet(Dataset):
    def __init__(self, node_list, labels):
        super(NodeSet, self).__init__()
        self.node_list = node_list
        self.labels = labels
        assert len(self.node_list) == len(self.labels)

    def __len__(self):
        return len(self.node_list)

    def __getitem__(self, idx):
        return self.node_list[idx], self.labels[idx]

class NbrSampleCollater(object):
    def __init__(self, graph: dgl.DGLHeteroGraph,
                 block_sampler: dgl.dataloading.BlockSampler):
        self.graph = graph
        self.block_sampler = block_sampler

    def collate(self, batch):
        batch = torch.tensor(batch)
        nodes = batch[:, 0].long()
        labels = batch[:, 1]
        blocks = self.block_sampler.sample_blocks(self.graph, nodes)
        return blocks, labels

class Embedding_Dataset():
    def __init__(self, args):
        self.args = args
        self.device = args.encoder_device
    def preprocess(self, graph):
        global n_node_feats

        # make bidirected
        feat = graph.ndata["feat"]
        graph.ndata["feat"] = feat

        # add self-loop
        print(f"Total edges before adding self-loop {graph.number_of_edges()}")
        graph = graph.remove_self_loop().add_self_loop()
        print(f"Total edges after adding self-loop {graph.number_of_edges()}")

        graph.create_formats_()

        return graph

        # our implementation

    def train_embeding(self, g):
        in_feats = g.ndata['feat'].shape[1]
        all_labels = g.ndata['label']
        fanouts_train = self.args.fanouts_train
        # fanouts_test = [12,12]

        train_collater = NbrSampleCollater(
            g, dgl.dataloading.MultiLayerNeighborSampler(fanouts=fanouts_train, replace=False))

        train_node_set = NodeSet(torch.LongTensor(np.arange(g.num_nodes())).tolist(), all_labels.tolist())
        self.train_node_loader = DataLoader(dataset=train_node_set, batch_size=2048,
                                            shuffle=True, num_workers=0, pin_memory=True,
                                            collate_fn=train_collater.collate, drop_last=False)
        ggd = GGD(g,
                  in_feats=in_feats,
                  n_hidden=self.args.encoder_hidden_dim,
                  n_layers=self.args.encoder_layers,
                  activation=nn.PReLU(self.args.encoder_hidden_dim),
                  dropout=0.1,
                  proj_layers=self.args.proj_layers,
                  gnn_encoder=self.args.gnn_encoder,
                  num_hop=1)
        self.g = g
        ggd.cuda()

        ggd_optimizer = torch.optim.AdamW(ggd.parameters(),
                                          lr=0.001,
                                          weight_decay=0)
        best = 1e9
        b_xent = nn.BCEWithLogitsLoss()

        for epoch in range(self.args.embedding_epochs):

            ggd.train()

            loss = 0
            for n_iter, (blocks, labels) in enumerate(tqdm(self.train_node_loader, desc=f'train epoch {epoch}')):
                blocks = [block.to(self.device) for block in blocks[-1]]
                labels = labels.to(self.device)
                loss = ggd(blocks, labels, b_xent)
                ggd_optimizer.zero_grad()
                loss.backward()
                ggd_optimizer.step()

            if loss < best:
                best = loss
                best_t = epoch
                cnt_wait = 0
                self.embeding_model = ggd

            else:
                cnt_wait += 1

            if cnt_wait == 50:
                print('Early stopping!')
                break

            print("Epoch {:05d} | Loss {:.4f} | ".format(epoch, loss.item()))

    def get_embedding(self):
        embeds = []
        labels = []

        for n_iter, (blocks, label) in enumerate(
                tqdm(self.train_node_loader, desc=f'loading embedding for evaluation')):
            blocks = [block.to(self.device) for block in blocks[-1]]
            label = label.to(self.device)
            embed = self.embeding_model.embed(blocks)
            embeds.append(embed.cpu())
            labels.append(label.cpu())
        l_embeds = torch.cat(embeds, dim=0)
        g_embeds = graph_power(l_embeds, self.g)
        embeds = l_embeds + g_embeds
        embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")
        embeds = torch.FloatTensor(embeds).cuda()
        l_labels = torch.cat(labels, dim=0)
        return embeds, l_labels

    def run_embedding(self):
        train_codes, adj, train_labels, n, k, d = load_data_ogb(self.args)
        adj.ndata['label'] = torch.tensor(train_labels).long()
        adj = self.preprocess(adj)
        self.train_embeding(adj)




class CustomDataset(MyDataset):
    def __init__(self, args,train_codes, train_labels):
        super().__init__(args)
        self.transformer = transforms.Compose([transforms.ToTensor()])
        self._data_dim = 0
        self.train_codes, self.train_labels = train_codes, train_labels

    # our implementation
    def get_train_data(self):
        if self.args.dataset != "ogbn-arxiv":
            train_codes = torch.Tensor(torch.load(os.path.join(self.data_dir, "train_data.pt")))
            if self.args.transform_input_data:
                train_codes = transform_embeddings(self.args.transform_input_data, train_codes)
            if self.args.use_labels_for_eval:
                train_labels = torch.load(os.path.join(self.data_dir, "train_labels.pt"))
            else:
                train_labels = torch.zeros((train_codes.size()[0]))
        else:
            train_codes, train_labels = self.train_codes, self.train_labels
        self._data_dim = train_codes.shape[1]
        train_labels =torch.tensor(train_labels)
        train_set = TensorDatasetWrapper(train_codes, train_labels)
        del train_codes
        del train_labels
        return train_set
    #
    def get_test_data(self):
        if self.args.dataset != "ogbn-arxiv":
            try:
                test_codes = torch.load(os.path.join(self.data_dir, "test_data.pt"))
                if self.args.use_labels_for_eval:
                    test_labels = torch.load(os.path.join(self.data_dir, "test_labels.pt"))
                else:
                    test_labels = torch.zeros((test_codes.size()[0]))
            except FileNotFoundError:
                print("Test data not found! running only with train data")
                return TensorDatasetWrapper(torch.empty(0), torch.empty(0))

            if self.args.transform_input_data:
                test_codes = transform_embeddings(self.args.transform_input_data, test_codes)
        else:
            print("Loading test data from OGB")
            test_codes, test_labels = self.train_codes, self.train_labels
        test_set = TensorDatasetWrapper(test_codes, test_labels)
        del test_codes
        del test_labels
        return test_set


def merge_datasets(set_1, set_2):
    """
    Merged two TensorDatasets into one
    """
    merged = torch.utils.data.ConcatDataset([set_1, set_2])
    return merged


def generate_mock_dataset(dim, len=3, dtype=torch.float32):
    """Generates a mock TensorDataset

    Args:
        dim (tuple): shape of the sample
        len (int): number of samples. Defaults to 10.
    """
    # Make sure train and test set are of the same type
    if type(dim) == int:
        data = torch.rand((len, dim))
    else:
        data = torch.rand((len, *dim))
    data = torch.tensor(data.clone().detach(), dtype=dtype)
    return TensorDataset(data, torch.zeros(len))


def load_data_ogb(args):
    """
    Load data for ogbn-arxiv dataset.
    args:
        args: parameters
    returns:
        features: node attributes
        g: dgl graph
        labels: node labels
        n: number of nodes
        k: number of clusters
        d: dimension number of node attributes
    """

    data = DglNodePropPredDataset(name=args.dataset.replace('_', '-'), root=args.dir)
    g, labels = data[0]
    # g = dgl.to_homogeneous(g)


    g = dgl.add_self_loop(g)
    features = g.ndata['feat']
    labels = labels.reshape(-1, ).numpy()

    n = features.shape[0]
    k = labels.max() + 1
    d = features.shape[-1]

    return features, g, labels, n, k, d
