import argparse, time
import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
import dgl
from dgl import DGLGraph
from dgl.data import register_data_args, load_data
from dgl.nn import EdgeWeightNorm
import random
import copy
import dgl.function as fn

from tqdm import tqdm
from ggd import GGD, Classifier
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import os
from sklearn import preprocessing as sk_prep
from utils import evaluation
from dgl.dataloading import GraphDataLoader, MultiLayerFullNeighborSampler


def aug_feature_dropout(input_feat, drop_percent=0.2):
    # aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def evaluate(model, features, labels, mask):
    model.eval()
    with torch.no_grad():
        logits = model(features)
        logits = logits[mask]
        labels = labels[mask]
        _, indices = torch.max(logits, dim=1)
        correct = torch.sum(indices == labels)
        return correct.item() * 1.0 / len(labels)

def load_data_ogb(dataset, args):
    global n_node_feats, n_classes

    if args.data_root_dir == 'default':
        data = DglNodePropPredDataset(name=dataset)
    else:
        data = DglNodePropPredDataset(name=dataset, root=args.data_root_dir)

    evaluator = Evaluator(name=dataset)

    splitted_idx = data.get_idx_split()
    train_idx, val_idx, test_idx = splitted_idx["train"], splitted_idx["valid"], splitted_idx["test"]
    graph, labels = data[0]

    n_node_feats = graph.ndata["feat"].shape[1]
    n_classes = (labels.max() + 1).item()

    return graph, labels, train_idx, val_idx, test_idx, evaluator

def preprocess(graph):
    global n_node_feats

    # make bidirected
    feat = graph.ndata["feat"]
    graph = dgl.to_bidirected(graph)
    graph.ndata["feat"] = feat

    # add self-loop
    print(f"Total edges before adding self-loop {graph.number_of_edges()}")
    graph = graph.remove_self_loop().add_self_loop()
    print(f"Total edges after adding self-loop {graph.number_of_edges()}")

    graph.create_formats_()

    return graph


def main(args):
    cuda = True
    free_gpu_id = int(args.gpu)
    torch.cuda.set_device(args.gpu)
    # load and preprocess dataset
    if 'ogbn' not in args.dataset_name:
        data = load_data(args)
        features = torch.FloatTensor(data.features)
        labels = torch.LongTensor(data.labels)
        if hasattr(torch, 'BoolTensor'):
            train_mask = torch.BoolTensor(data.train_mask)
            val_mask = torch.BoolTensor(data.val_mask)
            test_mask = torch.BoolTensor(data.test_mask)
        else:
            train_mask = torch.ByteTensor(data.train_mask)
            val_mask = torch.ByteTensor(data.val_mask)
            test_mask = torch.ByteTensor(data.test_mask)
        in_feats = features.shape[1]
        n_classes = data.num_labels
        n_edges = data.graph.number_of_edges()
        g = data.graph
        features = features.cuda()
        labels = labels.cuda()
        train_mask = train_mask.cuda()
        val_mask = val_mask.cuda()
        test_mask = test_mask.cuda()
        if args.self_loop:
            g.remove_edges_from(nx.selfloop_edges(g))
            g.add_edges_from(zip(g.nodes(), g.nodes()))
        g = DGLGraph(g)
    else:
        g, labels, train_mask, val_mask, test_mask, evaluator = load_data_ogb(args.dataset_name, args)
        g = preprocess(g)

        features = g.ndata['feat']
        labels = labels.T.squeeze(0)

        g, labels, train_idx, val_idx, test_idx, features = map(
            lambda x: x.to(free_gpu_id), (g, labels, train_mask, val_mask, test_mask, features)
        )

        in_feats = g.ndata['feat'].shape[1]
        n_classes = labels.T.max().item() + 1
        n_edges = g.num_edges()

    g = g.to(free_gpu_id)
    # create GGD model
    ggd = GGD(in_feats,
              args.n_hidden,
              args.n_layers,
              nn.PReLU(args.n_hidden),
              args.dropout,
              args.proj_layers,
              args.gnn_encoder,
              args.num_hop)

    if cuda:
        ggd.cuda()

    ggd_optimizer = torch.optim.AdamW(ggd.parameters(),
                                     lr=args.ggd_lr,
                                     weight_decay=args.weight_decay)

    b_xent = nn.BCEWithLogitsLoss()

    g = g.to(torch.device('cuda:0'))
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(args.n_layers)
    train_data_loader = dgl.dataloading.DataLoader(
                # The following arguments are specific to DGL's DataLoader.
                g,  # The graph
                torch.LongTensor(np.arange(g.num_nodes())).cuda(),  # The node IDs to iterate over in minibatches
                sampler,  # The neighbor sampler
                device=torch.device('cuda:0'),  # Put the sampled MFGs on CPU or GPU
                # The following arguments are inherited from PyTorch DataLoader.
                batch_size=1024,  # Batch size
                shuffle=True,  # Whether to shuffle the nodes for every epoch
                drop_last=False,  # Whether to drop the last incomplete batch
                num_workers=0,  # Number of sampler processes
            )
    
    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    counts = 0
    dur = []

    tag = str(int(np.random.random() * 10000000000))

    for epoch in range(args.n_ggd_epochs):
        ggd.train()
        if epoch >= 3:
            t0 = time.time()
        
        aug_feat = aug_feature_dropout(features, args.drop_feat)
        aug_feat = aug_feat.cuda()
        total_loss = 0
        for step, (input_nodes, output_nodes, mfgs) in  enumerate(tqdm(train_data_loader, desc=f'train epoch {epoch}')):
            perm = torch.randperm(g.number_of_nodes())
            permuted_feat = aug_feat[perm]
            input_feat = aug_feat[mfgs[0].srcdata[dgl.NID]]
            permuted_feat = permuted_feat[mfgs[0].srcdata[dgl.NID]]
            
            loss = ggd(mfgs, input_feat.cuda(), permuted_feat.cuda(), b_xent)
            
            ggd_optimizer.zero_grad()
            loss.backward()
            ggd_optimizer.step()
            total_loss += loss.item()
        avg_loss = total_loss / (step + 1)
        if avg_loss < best:
            best = avg_loss
            best_t = epoch
            cnt_wait = 0
            torch.save(ggd.state_dict(), 'models/best_ggd' + tag + '.pkl')
        else:
            cnt_wait += 1

        if cnt_wait == args.patience:
            print('Early stopping!')
            break

        if epoch >= 3:
            dur.append(time.time() - t0)

        print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
              "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), avg_loss,
                                            n_edges / np.mean(dur) / 1000))

        counts += 1

    print('Training Completed.')

    # create classifier model
    classifier = Classifier(args.n_hidden, n_classes)
    if cuda:
        classifier.cuda()

    classifier_optimizer = torch.optim.AdamW(classifier.parameters(),
                                            lr=args.classifier_lr,
                                            weight_decay=args.weight_decay)

    # train classifier
    print('Loading {}th epoch'.format(best_t))

    ggd.load_state_dict(torch.load('models/best_ggd' + tag + '.pkl'))
    ggd.eval()
    test_data_loader = dgl.dataloading.DataLoader(
                # The following arguments are specific to DGL's DataLoader.
                g,  # The graph
                torch.LongTensor(np.arange(g.num_nodes())).cuda(),  # The node IDs to iterate over in minibatches
                sampler,  # The neighbor sampler
                device=torch.device('cuda:0'),  # Put the sampled MFGs on CPU or GPU
                # The following arguments are inherited from PyTorch DataLoader.
                batch_size=1024,  # Batch size
                shuffle=False,  # Whether to shuffle the nodes for every epoch
                drop_last=False,  # Whether to drop the last incomplete batch
                num_workers=0,  # Number of sampler processes
            )
    embeds = []
    for step, (input_nodes, output_nodes, mfgs) in  enumerate(tqdm(test_data_loader)):
        input_feat = features[mfgs[0].srcdata[dgl.NID]]
        embed = ggd.embed(mfgs, input_feat.cuda())
        embeds.append(embed)
    
    l_embeds = torch.cat(embeds, dim=0)
    feat = l_embeds.clone().squeeze(0)
    degs = g.in_degrees().float().clamp(min=1)
    norm = torch.pow(degs, -0.5)
    norm = norm.to(l_embeds).unsqueeze(1)
    for _ in range(10):
        feat = feat * norm
        g.ndata['h2'] = feat
        g.update_all(fn.copy_u('h2', 'm'),
                            fn.sum('m', 'h2'))
        feat = g.ndata.pop('h2')
        feat = feat * norm

    g_embeds = feat.unsqueeze(0)

    embeds = (l_embeds + g_embeds).squeeze(0)

    embeds = sk_prep.normalize(X=embeds.detach().cpu().numpy(), norm="l2")

    embeds = torch.FloatTensor(embeds).cuda()

    dur = []
    best_acc, best_val_acc = 0, 0
    print('Testing Phase ==== Please Wait.')
    # for epoch in range(args.n_classifier_epochs):
    #     classifier.train()
    #     if epoch >= 3:
    #         t0 = time.time()

    #     classifier_optimizer.zero_grad()
    #     preds = classifier(embeds)
    #     loss = F.nll_loss(preds[train_mask], labels[train_mask])
    #     loss.backward()
    #     classifier_optimizer.step()

    #     if epoch >= 3:
    #         dur.append(time.time() - t0)

    #     val_acc = evaluate(classifier, embeds, labels, val_mask)
    #     if epoch > 1000:
    #         if val_acc > best_val_acc:
    #             best_val_acc = val_acc
    #             test_acc = evaluate(classifier, embeds, labels, test_mask)
    #             if test_acc > best_acc:
    #                 best_acc = test_acc
    #     # print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | Accuracy {:.4f} | "
    #     #       "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
    #     #                                     val_acc, n_edges / np.mean(dur) / 1000))
    # print("Valid Accuracy {:.4f}".format(best_val_acc))

    # # best_acc = evaluate(classifier, embeds, labels, test_mask)
    # print("Test Accuracy {:.4f}".format(best_acc))
    # return
    
    from kmeans_pytorch import kmeans
    cluster_ids_x, cluster_centers = kmeans(
    X=embeds, num_clusters=10, distance='euclidean', device=torch.device('cuda:0'))

    acc, nmi, ari, f1 = evaluation(labels.cpu().numpy(), cluster_ids_x.cpu().numpy())
    print("test      ｜ acc:{:.2f} ｜ nmi:{:.2f} ｜ ari:{:.2f} ｜ f1:{:.2f}".format(acc, nmi, ari, f1))

def get_free_gpu():
    os.system('nvidia-smi -q -d Memory |grep -A4 GPU|grep Free >tmp')
    memory_available = [int(x.split()[2]) for x in open('tmp', 'r').readlines()]
    return np.argmax(memory_available)

if __name__ == '__main__':
    import warnings

    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(description='GGD')
    register_data_args(parser)
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--ggd-lr", type=float, default=0.001,
                        help="ggd learning rate")
    parser.add_argument("--drop_feat", type=float, default=0.1,
                        help="feature dropout rate")
    parser.add_argument("--classifier-lr", type=float, default=0.05,
                        help="classifier learning rate")
    parser.add_argument("--n-ggd-epochs", type=int, default=500,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=6000,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=512,
                        help="number of hidden gcn units")
    parser.add_argument("--proj_layers", type=int, default=1,
                        help="number of project linear layers")
    parser.add_argument("--n-layers", type=int, default=1,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=500,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--n_trails", type=int, default=5,
                        help="number of trails")
    parser.add_argument("--gnn_encoder", type=str, default='gcn',
                        help="choice of gnn encoder")
    parser.add_argument("--num_hop", type=int, default=10,
                        help="number of k for sgc")
    parser.add_argument('--data_root_dir', type=str, default='default',
                           help="dir_path for saving graph data. Note that this model use DGL loader so do not mix up with the dir_path for the Pyg one. Use 'default' to save datasets at current folder.")
    parser.add_argument('--dataset_name', type=str, default='cora',
                        help='Dataset name: cora, citeseer, pubmed, cs, phy')
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    main(args)
    # accs = []
    # for i in range(args.n_trails):
    #     accs.append(main(args))
    # mean_acc = str(np.array(accs).mean())
    # print('mean accuracy:' + mean_acc)

    # file_name = str(args.dataset_name)
    # f = open('result/' + 'result_' + file_name + '.txt', 'a')
    # f.write(str(args) + '\n')
    # f.write(mean_acc + '\n')