import os
import wandb
import argparse
from utils import *
from tqdm import tqdm
from model import DinkNet, DinkNet_dgl
from kmeans import kmeans as GPU_KMeans
import torch.nn as nn
# from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans
import pandas as pd

def train(args=None):
    # setup random seed
    setup_seed(args.seed)

    # load graph data
    if args.dataset in ["cora", "citeseer"]:
        x, adj, y, n, k, d = load_data(args)
    elif args.dataset in ["amazon_photo"]:
        x, adj, y, n, k, d = load_amazon_photo()
    elif args.dataset in ["ogbn_arxiv", "ogbn_mag", "ogbn_products"]:
        x, adj, y, n, k, d = load_data_ogb(args)
    elif args.dataset in ["icdm"]:
        x, adj, y, n, k, d = load_icdm_data()
        k = args.k
        
    # label of discriminative task
    disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0)

    # model
    if args.dataset in ["cora", "citeseer"]:
        model = DinkNet(n_in=d, n_h=args.hid_units, n_cluster=k, tradeoff=args.tradeoff, activation=args.activate)
    elif args.dataset in ["amazon_photo", "ogbn_arxiv", "ogbn_mag", "ogbn_products", 'icdm']:
        model = DinkNet_dgl(g=adj, n_in=d, n_h=args.hid_units, n_cluster=k,
                            tradeoff=args.tradeoff, encoder_layers=args.encoder_layer,
                            activation=args.activate, projector_layers=args.projector_layer)

    # to device
    x, adj, disc_y, model = map(lambda tmp: tmp.to(args.device), [x, adj, disc_y, model])

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0

    loss_func = nn.BCEWithLogitsLoss()


    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()

        lbl_1 = torch.ones(1, adj.num_nodes())
        lbl_2 = torch.zeros(1, adj.num_nodes())
        lbl = torch.cat((lbl_1, lbl_2), 1).to(args.device)

        x_aug = aug_feature_dropout(x).squeeze(0)
        logit = model(x_aug.cuda())
        loss = loss_func(logit.unsqueeze(0), lbl)
        loss.backward()
        optimizer.step()
        print("Epoch: {:03d} | Loss: {:.4f}".format(epoch, loss.item()))

        # discrimination loss
        # logit = model.embed(x, adj)
        # loss_disc = loss_func(logit, disc_y)
        # if (epoch + 1) % args.eval_inter == 0:
        #     tqdm.write("epoch {:03d} ｜ loss:{:.2f}".format(epoch, loss_disc))
        #
        # loss_disc.backward()
        # optimizer.step()

        # h = model.embed(x, adj)


        # _, kmeans_mus = GPU_KMeans(X=h.detach(), num_clusters=k, device=torch.device('cpu'), tol=0.1)
        # cluster = KMeans(n_clusters=k, verbose=1).fit(h.cpu().detach().numpy())
        # centroids = cluster.cluster_centers_
        # assignments = cluster.labels_

        # model.cluster_center.data = torch.Tensor(centroids).to(args.device)

        # acc, nmi, ari, f1 = evaluation(y, assignments)
        # if acc > best_acc:
        #     best_acc = acc
        #     best_model = copy.deepcopy(dgi)
        # print("{:.2f} {:.2f} {:.2f} {:.2f}".format(acc, nmi, ari, f1))

    model.eval()
    h = model.embed(x, adj)
    print(f"Init Kmeans center, k = {k}")
    _, kmeans_mus = GPU_KMeans(X=h.detach(), num_clusters=k, device=torch.device(args.device), tol=0.1)
    # cluster_ids_x, cluster_centers = kmeans(
    # X=h, num_clusters=k, distance='euclidean', device=torch.device(args.device))
    # cluster = KMeans(n_clusters=k).fit(h.cpu().detach().numpy())
    # centroids = cluster_centers.cpu().numpy()
    # assignments = cluster_ids_x.cpu().numpy()
    centroids = kmeans_mus.cpu().numpy()
    model.cluster_center.data = torch.Tensor(kmeans_mus).to(args.device)
    # print(centroids.shape)
    torch.save(model.state_dict(), "./pretrain_models/DinkNet_" + args.dataset + f"_{k}" + ".pt")


if __name__ == '__main__':
    # cluster_center = torch.nn.Parameter(torch.Tensor(np.ones([40,1500])))
    # print(cluster_center)
    # hyper-parameter settings
    parser = argparse.ArgumentParser("DinkNet")

    # data
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="training device")
    parser.add_argument("--dataset", type=str, default="icdm", help="dataset name")
    parser.add_argument("--dataset_dir", type=str, default="./data", help="dataset root path")

    # model
    parser.add_argument("--tradeoff", type=float, default=1e-10, help="tradeoff parameter")
    parser.add_argument("--activate", type=str, default="prelu", help="activation function")
    parser.add_argument("--hid_units", type=int, default=1500, help="number of hidden units")
    parser.add_argument("--encoder_layer", type=int, default=1, help="number of encoder layers")
    parser.add_argument("--projector_layer", type=int, default=1, help="number of projector layers")

    # training
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wandb", action='store_true', default=False, help="enable wandb")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--eval_inter", type=int, default=10, help="interval of evaluation")
    parser.add_argument("--k", "-k", type=int, default=10, help="clusther_num")
    args = parser.parse_args()

    train(args=args)
