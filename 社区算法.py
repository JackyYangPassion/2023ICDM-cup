import argparse

from utils import *

import infomap
from infomap import Infomap
from sklearn.cluster import DBSCAN
import networkx as nx
from networkx.algorithms.community import kernighan_lin_bisection, louvain_communities, greedy_modularity_communities


im = Infomap("--two-level --directed")
parser = argparse.ArgumentParser("DinkNet")
parser.add_argument("--dataset_dir", type=str, default="./data", help="dataset root path")
parser.add_argument("--dataset", type=str, default="amazon_photo", help="dataset name")
args = parser.parse_args()
G1 = nx.karate_club_graph()
if args.dataset in ["cora", "citeseer"]:
    x, adj, y, n, k, d = load_data(args)
    x = x.numpy()[0]
    adj = adj.to_dense().numpy()

elif args.dataset in ["amazon_photo"]:
    x, adj, y, n, k, d = load_amazon_photo()
elif args.dataset in ["ogbn_arxiv", "ogbn_mag", "ogbn_products"]:
    x, adj, y, n, k, d = load_data_ogb(args)


print("实际标签数", k)
# # infoMap 算法
if args.dataset in ["cora", "citeseer"]:
    for i in range(n):
        for j in range(n):
            if adj[i, j] == 1:
                    im.add_link(i, j)
                    G1.add_edge(i, j)
else:
    srt, dst = map(lambda x : x.numpy(), adj.edges())
    for i in range(srt.shape[0]):
        im.add_link(srt[i], dst[i])
        G1.add_edge(srt[i], dst[i])

im.run()
print(f"Found {im.num_top_modules} modules with codelength: {im.codelength}")
#
# DBSCAN算法
clustering = DBSCAN().fit(x)
print(clustering.labels_)
print("DBSCAN算法数量", len(set(clustering.labels_)))

#
com = list(kernighan_lin_bisection(G1))
print('KL算法社区数量', len(com))

com = list(greedy_modularity_communities(G1))
print('贪心模块度社区算法社区数量', len(com))


com = list(louvain_communities(G1))
print('louvain算法社区数量', len(com))