import torch
import numpy as np
import wandb
import os
from kmodes.kmodes import KModes
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sweep_id', type=str, default="38kajskc")
parser.add_argument('--boost_num', type=int, default=-1)
parser.add_argument('--k', type=int, default=10)
parser.add_argument('--save_path', type=str, default="./results/boost")
args = parser.parse_args()

N_CPUS = int(os.environ['SLURM_CPUS_PER_TASK']) if 'SLURM_CPUS_PER_TASK' in os.environ else 1
api = wandb.Api()

sweep_id = args.sweep_id
# sweep_id = "test"
boost_num = args.boost_num
save_path = f"{args.save_path}/{sweep_id}_{boost_num}"

def get_result_list(id):
    path_list = []
    sweep = api.sweep(f"rmanluo/sweep_parameters_icdm/{id}")
    for r in sweep.runs:
        print(f"results/{r.name}/final_results_{r.name}.txt")
        path_list.append(f"results/{r.name}/final_results_{r.name}.txt")
    return path_list

def load_results(path):
    preds = []
    with open(path, 'r') as f:
        for line in f:
            preds.append(int(line.strip()))
    return preds

def new_decoder(pred, num_clusters_kmodes):
    km = KModes(n_clusters=num_clusters_kmodes, n_jobs=N_CPUS, verbose=1)
    cluster_ids_x = km.fit_predict(pred, max_iter=300, random_state=42)
    return cluster_ids_x

# result_list = ["results/icdm_10_gcn_10_64_10_4_0.0_1_0.0001_0.0_0.1_1e-10_0.0005_1_300_15_gpu_2_27_10_2023_19_42_17/final_results_icdm_10_gcn_10_64_10_4_0.0_1_0.0001_0.0_0.1_1e-10_0.0005_1_300_15_gpu_2_27_10_2023_19_42_17.txt", "./results/icdm_10_gcn_10_64_10_4_0.0_1_0.0001_0.0_0.1_1e-10_0.0005_1_300_15_gpu_2_27_10_2023_19_42_09/final_results_icdm_10_gcn_10_64_10_4_0.0_1_0.0001_0.0_0.1_1e-10_0.0005_1_300_15_gpu_2_27_10_2023_19_42_09.txt"]

result_list = get_result_list(sweep_id)
if boost_num != -1:
    result_list = result_list[:min(boost_num, len(result_list))]
    
all_results = []
for path in result_list:
    preds = load_results(path)
    preds = np.array(preds)
    all_results.append(preds)

all_results = np.stack(all_results, axis=1)
print(all_results.shape)

preds = new_decoder(all_results, num_clusters_kmodes=args.k)

if not os.path.exists(save_path):
    os.makedirs(save_path)
    
with open(f"{save_path}/boost_results.txt", "w") as file:
    for item in preds:
        file.write("%s\n" % item)

with open(f"{save_path}/used_preds.txt", "w") as file:
    for p in result_list:
        file.write("%s\n" % p)
        
print("Results saved at: ", save_path)