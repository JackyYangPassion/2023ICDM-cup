import torch
import torch.nn as nn
# from gcn import GCN
import dgl.function as fn
import torch
import torch.nn as nn
from dgl.nn.pytorch import GraphConv
import copy
import numpy as np
import torch.nn.functional as F

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
        self.layers.append(GraphConv(in_feats, n_hidden, weight = weight, bias = bias, activation=activation))
        # self.layers.append(SAGEConv(in_feats, n_hidden, bias=bias,aggregator_type='mean', activation=activation))
        self.bns.append(torch.nn.BatchNorm1d(n_hidden, momentum = 0.01))
        # hidden layers
        for i in range(n_layers - 1):
            self.layers.append(GraphConv(n_hidden, n_hidden, weight=weight, bias=bias, activation=activation))
            # self.layers.append(SAGEConv(n_hidden, n_hidden, bias=bias, aggregator_type='mean', activation=activation))
            # if i != (n_layers - 2):
            self.bns.append(torch.nn.BatchNorm1d(n_hidden, momentum = 0.01))
        # output layer
        # self.layers.append(GraphConv(n_hidden, n_classes))
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, features):
        h = features
        for i, layer in enumerate(self.layers):
            if i != 0:
                h = self.dropout(h)
            h = layer(self.g, h)
            # h = self.bns[i](h)
        return h

class Encoder(nn.Module):
    def __init__(self, g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, k = 1):
        super(Encoder, self).__init__()
        self.g = g
        self.gnn_encoder = gnn_encoder
        if gnn_encoder == 'gcn':
            self.conv = GCN(g, in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, features, corrupt=False):
        if corrupt:
            perm = torch.randperm(self.g.number_of_nodes())
            features = features[perm]
        if self.gnn_encoder == 'gcn':
            features = self.conv(features)

        return features

class GGD(nn.Module):
    def __init__(self, g, in_feats, n_hidden, out_put, n_layers, activation, dropout, proj_layers, gnn_encoder, num_hop, n_cluster, tradeoff, power):
        super(GGD, self).__init__()
        self.encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, num_hop)
        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(n_hidden, out_put))
        self.loss = nn.BCEWithLogitsLoss()
        self.power = power
        self.cluster_center = torch.nn.Parameter(torch.Tensor(n_cluster, n_hidden))

        self.tradeoff = tradeoff

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

        logits = torch.cat((sc_1, sc_2), 1)

        loss = loss_func(logits, labels)

        return loss

    def embed(self, features, g):
        h_1 = self.encoder(features, corrupt=False)

        feat = h_1.clone().squeeze(0)

        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(h_1.device).unsqueeze(1)
        for _ in range(self.power):
            feat = feat * norm
            g.ndata['h2'] = feat
            g.update_all(fn.copy_u('h2', 'm'),
                             fn.sum('m', 'h2'))
            feat = g.ndata.pop('h2')
            feat = feat * norm

        h_2 = feat.unsqueeze(0)

        return h_1.detach(), h_2.detach()
    
    @staticmethod
    def dis_fun(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        distance = xx_cc - 2 * xc
        return distance

    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def clustering_loss(self, h):
        # print(h.shape)
        sample_center_distance = self.dis_fun(h, self.cluster_center)
        center_distance = self.dis_fun(self.cluster_center, self.cluster_center)
        self.no_diag(center_distance, self.cluster_center.shape[0])
        clustering_loss = sample_center_distance.mean() - center_distance.mean()

        return clustering_loss, sample_center_distance

    def clustering(self, h):
        sample_center_distance = self.dis_fun(h, self.cluster_center)
        cluster_results = torch.argmin(sample_center_distance, dim=-1)
        return cluster_results.cpu().detach().numpy()
    
class Classifier(nn.Module):
    def __init__(self, n_hidden, n_classes):
        super(Classifier, self).__init__()
        self.fc = nn.Linear(n_hidden, n_classes)
        self.reset_parameters()

    def reset_parameters(self):
        self.fc.reset_parameters()

    def forward(self, features):
        features = self.fc(features)
        return torch.log_softmax(features, dim=-1)

class EMA:
    def __init__(self, beta, epochs):
        super().__init__()
        self.beta = beta
        self.step = 0
        self.total_steps = epochs

    def update_average(self, old, new):
        if old is None:
            return new
        beta = 1 - (1 - self.beta) * (np.cos(np.pi * self.step / self.total_steps) + 1) / 2.0
        self.step += 1
        return old * beta + (1 - beta) * new

def set_requires_grad(model, val):
    for p in model.parameters():
        p.requires_grad = val

def update_moving_average(ema_updater, ma_model, current_model):
    for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
        old_weight, up_weight = ma_params.data, current_params.data
        ma_params.data = ema_updater.update_average(old_weight, up_weight)

def loss_fn2(x, y):
    x = F.normalize(x, dim=-1, p=2)
    y = F.normalize(y, dim=-1, p=2)
    return 2 - 2 * (x * y).sum(dim=-1)


moving_average_decay=0.99
epochs=1000
class GGD_BGRL(nn.Module):
    def __init__(self, g, in_feats, n_hidden, out_put, n_layers, activation, dropout, proj_layers, gnn_encoder, num_hop,
                 n_cluster, tradeoff, power, cluster_loss_weight):
        super(GGD_BGRL, self).__init__()
        self.student_encoder = Encoder(g, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, num_hop)
        self.teacher_encoder = copy.deepcopy(self.student_encoder)
        set_requires_grad(self.teacher_encoder, False)

        self.teacher_ema_updater = EMA(moving_average_decay, epochs)

        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(n_hidden, out_put))
        self.loss = nn.BCEWithLogitsLoss()
        self.power = power
        self.cluster_loss_weight = cluster_loss_weight
        self.cluster_center = torch.nn.Parameter(torch.Tensor(n_cluster, n_hidden))

        self.tradeoff = tradeoff
    def reset_moving_average(self):
        del self.teacher_encoder
        self.teacher_encoder = None
    def update_moving_average(self):
        assert self.teacher_encoder is not None, 'teacher encoder has not been created yet'
        update_moving_average(self.teacher_ema_updater, self.teacher_encoder, self.student_encoder)

    def ggd_forward(self, features, labels, loss_func):
        h_1 = self.student_encoder(features, corrupt=False)
        h_2 = self.student_encoder(features, corrupt=True)

        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)

        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)

        logits = torch.cat((sc_1, sc_2), 1)

        loss = loss_func(logits, labels)

        return loss
    
    def forward(self, features, labels, loss_func):
        h_1 = self.student_encoder(features, corrupt=False)
        h_2 = self.student_encoder(features, corrupt=True)

        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)

        with torch.no_grad():
            v1_teacher = self.teacher_encoder(features, corrupt=False)
            v2_teacher = self.teacher_encoder(features, corrupt=False)

        loss1 = loss_fn2(h_1, v2_teacher.detach())
        loss2 = loss_fn2(h_2, v1_teacher.detach())

        loss_bgrl = loss1 + loss2
        loss_bgrl = loss_bgrl.mean()

        # sc_1 = sc_1.sum(1).unsqueeze(0)
        # sc_2 = sc_2.sum(1).unsqueeze(0)

        # logits = torch.cat((sc_1, sc_2), 1)
        #
        # loss = loss_func(logits, labels)

        return loss_bgrl

    def embed(self, features, g):
        h_1 = self.student_encoder(features, corrupt=False)

        feat = h_1.clone().squeeze(0)

        degs = g.in_degrees().float().clamp(min=1)
        norm = torch.pow(degs, -0.5)
        norm = norm.to(h_1.device).unsqueeze(1)
        for _ in range(self.power):
            feat = feat * norm
            g.ndata['h2'] = feat
            g.update_all(fn.copy_u('h2', 'm'),
                         fn.sum('m', 'h2'))
            feat = g.ndata.pop('h2')
            feat = feat * norm

        h_2 = feat.unsqueeze(0)

        return h_1.detach(), h_2.detach()

    @staticmethod
    def dis_fun(x, c):
        xx = (x * x).sum(-1).reshape(-1, 1).repeat(1, c.shape[0])
        cc = (c * c).sum(-1).reshape(1, -1).repeat(x.shape[0], 1)
        xx_cc = xx + cc
        xc = x @ c.T
        distance = xx_cc - 2 * xc
        return distance

    @staticmethod
    def no_diag(x, n):
        x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

    def clustering_loss(self, h):
        # print(h.shape)
        sample_center_distance = self.dis_fun(h, self.cluster_center)
        center_distance = self.dis_fun(self.cluster_center, self.cluster_center)
        self.no_diag(center_distance, self.cluster_center.shape[0])
        clustering_loss = self.cluster_loss_weight * sample_center_distance.mean() - center_distance.mean()

        return clustering_loss, sample_center_distance

    def clustering(self, h):
        sample_center_distance = self.dis_fun(h, self.cluster_center)
        cluster_results = torch.argmin(sample_center_distance, dim=-1)
        return cluster_results.cpu().detach().numpy()