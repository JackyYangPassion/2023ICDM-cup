import torch
import torch.nn as nn
from gcn import GCN

class Encoder(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, k = 1):
        super(Encoder, self).__init__()
        self.gnn_encoder = gnn_encoder
        if gnn_encoder == 'gcn':
            self.conv = GCN(in_feats, n_hidden, n_hidden, n_layers, activation, dropout)

    def forward(self, msfgs, features):
        output = self.conv(msfgs, features)
        return output

class GGD(nn.Module):
    def __init__(self, in_feats, n_hidden, n_layers, activation, dropout, proj_layers, gnn_encoder, num_hop):
        super(GGD, self).__init__()
        self.encoder = Encoder(in_feats, n_hidden, n_layers, activation, dropout, gnn_encoder, num_hop)
        self.mlp = torch.nn.ModuleList()
        for i in range(proj_layers):
            self.mlp.append(nn.Linear(n_hidden, n_hidden))
        self.loss = nn.BCEWithLogitsLoss()

    def forward(self, mfgs, features, permuted_feature, loss_func):

        h_1 = self.encoder(mfgs, features)
        h_2 = self.encoder(mfgs, permuted_feature)

        sc_1 = h_1.squeeze(0)
        sc_2 = h_2.squeeze(0)
        for i, lin in enumerate(self.mlp):
            sc_1 = lin(sc_1)
            sc_2 = lin(sc_2)

        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)

        sc_1 = sc_1.sum(1).unsqueeze(0)
        sc_2 = sc_2.sum(1).unsqueeze(0)

        lbl_1 = torch.ones(1, sc_1.shape[1])
        lbl_2 = torch.zeros(1, sc_1.shape[1])
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

        logits = torch.cat((sc_1, sc_2), 1)

        loss = loss_func(logits, lbl)

        return loss

    def embed(self, mfgs, features):
        h_1 = self.encoder(mfgs, features)
        return h_1
        # degs = g.in_degrees().float().clamp(min=1)
        # norm = torch.pow(degs, -0.5)
        # norm = norm.to(h_1.device).unsqueeze(1)
        # for _ in range(10):
        #     feat = feat * norm
        #     g.ndata['h2'] = feat
        #     g.update_all(fn.copy_u('h2', 'm'),
        #                      fn.sum('m', 'h2'))
        #     feat = g.ndata.pop('h2')
        #     feat = feat * norm

        # h_2 = feat.unsqueeze(0)

        # return h_1.detach(), h_2.detach()

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