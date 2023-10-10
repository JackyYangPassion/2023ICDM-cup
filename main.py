import os
import wandb
import argparse
from utils import *
from tqdm import tqdm
from model import DinkNet, DinkNet_dgl
import uuid

def compute_data_covs_hard_assignment(labels, codes, K, mus, prior):
    covs = []
    for k in range(K):
        codes_k = codes[labels == k]
        N_k = float(len(codes_k))
        if N_k > 0:
            cov_k = torch.matmul(
                (codes_k - mus[k].cpu().repeat(len(codes_k), 1)).T,
                (codes_k - mus[k].cpu().repeat(len(codes_k), 1)),
            )
            cov_k = cov_k / N_k
        else:
            if prior:
                _, cov_k = prior.init_priors()
            else:
                cov_k = torch.eye(codes.shape[1]) * 0.0005
        covs.append(cov_k)
    return torch.stack(covs)

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
        # print(x.shape)
        
    # label of discriminative task
    disc_y = torch.cat((torch.ones(n), torch.zeros(n)), 0)

    # model
    if args.dataset in ["cora", "citeseer"]:
        model = DinkNet(n_in=d, n_h=args.hid_units, n_cluster=k, tradeoff=args.tradeoff, activation=args.activate)
    elif args.dataset in ["amazon_photo", "ogbn_arxiv", "ogbn_mag", "ogbn_products", "icdm"]:
        model = DinkNet_dgl(g=adj, n_in=d, n_h=args.hid_units, n_cluster=k,
                            tradeoff=args.tradeoff, encoder_layers=args.encoder_layer, 
                            activation=args.activate, projector_layers=args.projector_layer)

    # to device
    x, adj, disc_y, model = map(lambda tmp: tmp.to(args.device), [x, adj, disc_y, model])

    final_postfix = f"pretrain_config-{args.postfix}_finetuning_config-{args.lr}_{args.epochs}_{args.k}"
    
    # load pre-trained model parameter
    model.load_state_dict(torch.load("./pretrain_models/DinkNet_{}.pt".format(args.postfix)))

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_loss = 99999
    patience = 10
    count = 0
    # uid = str(uuid.uuid1()).split("-")[0]
    # check_path = f"./check_points/{uid}"
    # if not os.path.exists(check_path):
    #     os.mkdir(check_path)
    #     check_path = f"{check_path}/model.pt"
    # training
    if args.wandb:
        if not os.path.exists("./wandb/"):
            os.makedirs("./wandb")

        wandb.init(config=args,
                   project="ICML23_DinkNet",
                   name="baseline_{}".format(args.dataset),
                   dir="./wandb/",
                   job_type="training",
                   reinit=True)

    for epoch in tqdm(range(args.epochs)):
        model.train()
        optimizer.zero_grad()

        loss, sample_center_distance = model.cal_loss(x, adj, disc_y)

        loss.backward()
        optimizer.step()
        print("Epoch: {:03d} | Loss: {:.4f}".format(epoch, loss.item()))
        # if loss.item() < best_loss:
        #     best_loss = loss.item()
        #     torch.save(model.state_dict(), check_path)
        # else:
        #     count += 1
        #     print(f"count: {count} out of {patience}")
        #     if count > patience:
        #         break

    # testing
    # model.load_state_dict(torch.load(check_path))
    torch.save(model.state_dict(), "./final_models/DinkNet_" + f"_{final_postfix}.pt")
    model.eval()
    print("Testing on {} dataset".format(args.dataset))
    y_hat = model.clustering(x, adj)
    with open("results/" + f"DinkNet_{final_postfix}_results.txt", "w") as file:
        for item in y_hat:
            file.write("%s\n" % item)
    # acc, nmi, ari, f1 = evaluation(y, y_hat)

    # # logging
    # tqdm.write("test      ｜ acc:{:.2f} ｜ nmi:{:.2f} ｜ ari:{:.2f} ｜ f1:{:.2f}".format(acc, nmi, ari, f1))


if __name__ == '__main__':

    # hyper-parameter settings
    parser = argparse.ArgumentParser("DinkNet")

    # data
    parser.add_argument("--seed", type=int, default=2023, help="random seed")
    parser.add_argument("--device", type=str, default="cpu", help="training device")
    parser.add_argument("--dataset", type=str, default="citeseer", help="dataset name")
    parser.add_argument("--dataset_dir", type=str, default="./data", help="dataset root path")

    # model
    parser.add_argument("--tradeoff", type=float, default=1e-10, help="tradeoff parameter")
    parser.add_argument("--activate", type=str, default="prelu", help="activation function")
    parser.add_argument("--hid_units", type=int, default=1536, help="number of hidden units")
    parser.add_argument("--encoder_layer", type=int, default=1, help="number of encoder layers")
    parser.add_argument("--projector_layer", type=int, default=1, help="number of projector layers")

    # training
    parser.add_argument("--lr", type=float, default=1e-2, help="learning rate")
    parser.add_argument("--wandb", action='store_true', default=False, help="enable wandb")
    parser.add_argument("--epochs", type=int, default=200, help="number of epochs")
    parser.add_argument("--eval_inter", type=int, default=10, help="interval of evaluation")
    parser.add_argument("--k", "-k", type=int, default=10, help="clusther_num")
    parser.add_argument("--postfix", "-postfix", type=str, default="", help="postfix")
    args = parser.parse_args()

    train(args=args)
