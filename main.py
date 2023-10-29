import argparse, time
from datetime import datetime
import json
import numpy as np
import torch
import torch.nn as nn
import dgl
from dgl.data import register_data_args
import random
import copy
from ggd import GGD
from ogb.nodeproppred import DglNodePropPredDataset, Evaluator
import os
from sklearn import preprocessing as sk_prep
from kmeans_pytorch import kmeans
from sklearn.cluster import KMeans
from tqdm import tqdm
import wandb
import distutils.util
import pickle
import dgl.function as fn
from kmodes.kmodes import KModes

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1

def setup_seed(seed):
    """
    fix the random seed.
    args:
        seed: the random seed
    returns:
        none
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    return None

def aug_feature_dropout(input_feat, drop_percent=0.2):
    # aug_input_feat = copy.deepcopy((input_feat.squeeze(0)))
    aug_input_feat = copy.deepcopy(input_feat)
    drop_feat_num = int(aug_input_feat.shape[1] * drop_percent)
    drop_idx = random.sample([i for i in range(aug_input_feat.shape[1])], drop_feat_num)
    aug_input_feat[:, drop_idx] = 0

    return aug_input_feat

def kmodes_decoder(embed, num_kmeans=50,num_cluster_k_means=10):
    all_labels = []
    for i in range(num_kmeans):
        cluster_ids_x, cluster_centers = kmeans(
            X=embed, num_clusters=num_cluster_k_means, distance='euclidean', device=torch.device('cuda:0'))
        all_labels.append(cluster_ids_x)
    all = torch.stack(all_labels).t()
    km = KModes(n_clusters=num_cluster_k_means, n_jobs=N_CPUS)
    cluster_ids_x = km.fit_predict(all.cpu().numpy(), max_iter=300, random_state=42)

    return cluster_ids_x, km.cluster_centroids_

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
    free_gpu_id = f"cuda:{int(args.gpu)}"
    torch.cuda.set_device(args.gpu)
    # load and preprocess dataset
    if args.dataset == 'icdm':
        args.dataset_name = "icdm2023_session1_test"
        if os.path.exists(args.data_root_dir+'/'+ args.dataset_name+'/'+ 'graph.bin'):
            g,_ = dgl.load_graphs(args.data_root_dir+'/'+ args.dataset_name+'/'+ 'graph.bin')
            g = g[0]

        else:
            with open(args.data_root_dir+'/'+ args.dataset_name +'/'+ args.dataset_name+ '_edge.txt', 'r') as f:
                edges = [tuple(map(int, line.strip().split(','))) for line in f.readlines()]
            features = []
            with open(args.data_root_dir+'/'+ args.dataset_name +'/'+ args.dataset_name+ '_node_feat.txt', 'r') as f:
                for i in f:
                    j = i.split(',')
                    j = np.array(j).astype('float64')
                    features.append(j)
            features = np.vstack(features)
            features = torch.FloatTensor(features)
            # 从边列表创建DGL图
            src, dst = zip(*edges)
            g = dgl.graph((src, dst))
            dgl.save_graphs(args.data_root_dir+'/'+ args.dataset_name+'/'+ 'graph.bin', g)
        g = preprocess(g)
        features = g.ndata['feat']
        in_feats = g.ndata['feat'].shape[1]
        labels = torch.ones(features.shape[0]).T.squeeze(0)
        n_edges = g.num_edges()
        n_classes = args.K
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
    
    if args.activation == "relu":
        activation = nn.ReLU()
    elif args.activation == "prelu":
        activation = nn.PReLU(args.n_hidden)
    elif args.activation == "leakyrelu":
        activation = nn.LeakyReLU()
    else:
        activation = None
    
    ggd = GGD(g,
              in_feats,
              args.n_hidden,
              args.output_dim,
              args.n_layers,
              activation,
              args.dropout,
              args.proj_layers,
              args.gnn_encoder,
              args.num_hop, n_classes, args.tradeoff, args.power, args.cluster_loss_weight)

    if cuda:
        ggd.cuda()

    ggd_optimizer = torch.optim.AdamW(ggd.parameters(),
                                     lr=args.ggd_lr,
                                     weight_decay=args.weight_decay)

    b_xent = nn.BCEWithLogitsLoss()

    # train deep graph infomax
    cnt_wait = 0
    best = 1e9
    best_t = 0
    counts = 0
    dur = []
    
    start_time = datetime.now().strftime("%d_%m_%Y_%H_%M_%S")
    tag = f"{args.dataset}_{args.K}_{args.gnn_encoder}_{args.num_hop}_{args.n_hidden}_{args.output_dim}_{args.n_layers}_{args.dropout}_{args.proj_layers}_{args.ggd_lr}_{args.weight_decay}_{args.drop_feat}_{args.tradeoff}_{args.classifier_lr}_{args.n_ggd_epochs}_{args.n_classifier_epochs}_{args.power}_{args.kmeans}_{args.postfix}_{start_time}"

    save_path = os.path.join("./results", tag)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    # Save args
    with open(f"{save_path}/args.txt", "w") as file:
        json.dump(args.__dict__, file, indent=4)
        
    model_path = f"{save_path}/pretrain_model.pt"
    
    if args.wandb:
        wandb.init(config=args,
                   project="ICML23_DDG",
                   name=tag)
    
    if args.pretrain:
        # Pre-train stage
        for epoch in range(args.n_ggd_epochs):
            ggd.train()
            if epoch >= 3:
                t0 = time.time()

            ggd_optimizer.zero_grad()

            lbl_1 = torch.ones(1, g.num_nodes())
            lbl_2 = torch.zeros(1, g.num_nodes())
            lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

            aug_feat = aug_feature_dropout(features, args.drop_feat)
            loss = ggd(aug_feat.cuda(), lbl, b_xent)
            loss.backward()
            ggd_optimizer.step()

            # if loss < best:
            #     best = loss
            #     best_t = epoch
            #     cnt_wait = 0
            #     torch.save(ggd.state_dict(), model_path)
            # else:
            #     cnt_wait += 1

            # if cnt_wait == args.patience:
            #     print('Early stopping!')
            #     break

            if epoch >= 3:
                dur.append(time.time() - t0)
            if args.wandb:
                wandb.log({"pretrain_loss": loss.item()}, step=epoch)
            print("Epoch {:05d} | Time(s) {:.4f} | Loss {:.4f} | "
                "ETputs(KTEPS) {:.2f}".format(epoch, np.mean(dur), loss.item(),
                                                n_edges / np.mean(dur) / 1000))

            counts += 1

        print('Training Completed.')
        # Train classifier
        print('Loading {}th epoch'.format(best_t))

        # ggd.load_state_dict(torch.load(model_path))

        # graph power embedding reinforcement
        l_embeds, g_embeds= ggd.embed(features.cuda(), g)

        embeds = (l_embeds + g_embeds).squeeze(0)

        embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")

        embeds = torch.FloatTensor(embeds).cuda()
        torch.save(embeds, f"{save_path}/pretrain_embs.pt")
            
        print(f"Init Kmeans center, k = {n_classes}")

        # CPU kmeans
        if args.kmeans == "cpu":
            cluster = KMeans(n_clusters=n_classes, verbose=1, n_init="auto").fit(embeds.cpu().detach().numpy())
            cluster_centers = cluster.cluster_centers_
            results = cluster.labels_
        elif args.kmeans == "kmodes":
            cluster_ids_x, cluster_centers = kmodes_decoder(embeds, num_kmeans=args.num_kmeans,num_cluster_k_means=n_classes)
            results = cluster_ids_x
        else:
            # GPU kemans
            cluster_ids_x, cluster_centers = kmeans(
            X=embeds, num_clusters=n_classes, distance='euclidean', device=torch.device('cuda:0'))
            results = cluster_ids_x.cpu().numpy()
            
        with open(f"{save_path}/pretrain_kmeans_results_{tag}.txt", "w") as file:
            for item in results:
                file.write("%s\n" % item)
        
        ggd.cluster_center.data = torch.Tensor(cluster_centers).cuda()
        torch.save(ggd.state_dict(), model_path)
    elif args.pre_train_weight_path != "":
        ggd.load_state_dict(torch.load(args.pre_train_weight_path))
    else:
        ggd.load_state_dict(torch.load(model_path))
    
    if args.pretrain_only:
        return
    
    # Finetuning
    print("Finetuning.....")
    optimizer = torch.optim.Adam(ggd.parameters(), lr=args.classifier_lr)
    cnt_wait = 0
    best = 1e9
    best_t = 0
    counts = 0
    model_path = f"{save_path}/finetune_model.pt"
    for epoch in tqdm(range(args.n_classifier_epochs)):
        ggd.train()
        optimizer.zero_grad()
        
        lbl_1 = torch.ones(1, g.num_nodes())
        lbl_2 = torch.zeros(1, g.num_nodes())
        lbl = torch.cat((lbl_1, lbl_2), 1).cuda()

        aug_feat = aug_feature_dropout(features, args.drop_feat)
        dis_loss = ggd(aug_feat.cuda(), lbl, b_xent)
        
        # graph power embedding reinforcement
        l_embeds, g_embeds= ggd.embed(features.cuda(), g)

        embeds = (l_embeds + g_embeds).squeeze(0)

        embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")

        embeds = torch.FloatTensor(embeds).cuda()
        clustering_loss, _ = ggd.clustering_loss(embeds)
        
        loss = args.tradeoff * dis_loss + clustering_loss
        loss.backward()
        optimizer.step()
        print("Epoch: {:03d} | Loss: {:.4f}".format(epoch, loss.item()))
        if args.wandb:
            wandb.log({"finetune_loss": loss.item()}, step=epoch)
        # if loss < best:
        #     best = loss
        #     best_t = epoch
        #     cnt_wait = 0
        #     torch.save(ggd.state_dict(), model_path)
        # else:
        #     cnt_wait += 1

        # if cnt_wait == args.patience:
        #     print('Early stopping!')
        #     break
    
    torch.save(ggd.state_dict(), model_path)
    ggd.eval()
    print("Testing on {} dataset".format(args.dataset))
    # graph power embedding reinforcement
    l_embeds, g_embeds= ggd.embed(features.cuda(), g)

    embeds = (l_embeds + g_embeds).squeeze(0)

    embeds = sk_prep.normalize(X=embeds.cpu().numpy(), norm="l2")

    embeds = torch.FloatTensor(embeds).cuda()
    y_hat = ggd.clustering(embeds)
    with open(f"{save_path}/final_results_{tag}.txt", "w") as file:
        for item in y_hat:
            file.write("%s\n" % item)
    print("Results saved at: ", save_path)
    if args.wandb:
        wandb.save(f"{save_path}/final_results_{tag}.txt")
        wandb.save(f"{save_path}/pretrain_kmeans_results_{tag}.txt")
    
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
    parser.add_argument("--K", type=int, default=10,
                        help="num K")
    parser.add_argument("--gpu", type=int, default=0,
                        help="gpu")
    parser.add_argument("--ggd-lr", type=float, default=0.001,
                        help="ggd learning rate")
    parser.add_argument("--drop_feat", type=float, default=0.1,
                        help="feature dropout rate")
    parser.add_argument("--classifier-lr", type=float, default=0.05,
                        help="classifier learning rate")
    parser.add_argument("--n-ggd-epochs", type=int, default=1,
                        help="number of training epochs")
    parser.add_argument("--n-classifier-epochs", type=int, default=6000,
                        help="number of training epochs")
    parser.add_argument("--n-hidden", type=int, default=256,
                        help="number of hidden gcn units")
    parser.add_argument("--output_dim", type=int, default=10,
                        help="number of hidden gcn units")
    parser.add_argument("--proj_layers", type=int, default=1,
                        help="number of project linear layers")
    parser.add_argument("--n-layers", type=int, default=3,
                        help="number of hidden gcn layers")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=100,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.add_argument("--n_trails", type=int, default=5,
                        help="number of trails")
    parser.add_argument("--gnn_encoder", type=str, default='gcn',
                        help="choice of gnn encoder")
    parser.add_argument("--num_hop", type=int, default=10,
                        help="number of k for sgc")
    parser.add_argument("--tradeoff", type=float, default=1e-10, help="tradeoff parameter")
    parser.add_argument("--power", type=int, default=10)
    parser.add_argument("--num_kmeans", type=int, default=2)
    parser.add_argument("--activation", type=str, choices=["relu", "prelu", "leakyrelu", "none"], default="prelu")
    parser.add_argument("--cluster_loss_weight", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument('--data_root_dir', type=str, default='./data',
                           help="dir_path for saving graph data. Note that this model use DGL loader so do not mix up with the dir_path for the Pyg one. Use 'default' to save datasets at current folder.")
    parser.add_argument('--dataset_name', type=str, default='icdm2023_session1_test',
                        help='icdm2023_session1_test,ogbn-arxiv')
    parser.add_argument("--kmeans", default="gpu", choices=["gpu", "cpu", "kmodes"])
    parser.add_argument("--pretrain", default=True, type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument("--pretrain_only", default=False, type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument("--wandb", default=False, type=lambda x:bool(distutils.util.strtobool(x)), help="enable wandb")
    parser.add_argument("--save_pretrain_graph", default=False, type=lambda x:bool(distutils.util.strtobool(x)))
    parser.add_argument("--pre_train_weight_path", default="", type=str, help="pretrain checkpoint path")
    parser.add_argument("--postfix", default="", type=str)
    parser.set_defaults(self_loop=False)
    args = parser.parse_args()
    print(args)
    # setup random seed
    setup_seed(args.seed)
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