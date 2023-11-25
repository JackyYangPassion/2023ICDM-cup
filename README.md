# ICDM2023-Fast团队

## Reproduce results
`python main.py --K=10 --classifier-lr=0.0005 --dataset=icdm --ggd-lr=0.0001 --gnn_encoder=gcn --n-classifier-epochs=300 --n-ggd-epochs=1 --n-hidden=64 --n-layers=4 --power=15 --proj_layers=1 --seed=2023`

## Requirements
GPU A100 80G

## Procedure

### Step 1. GGD pre-training
main.py L216-L256

1. Pretrain the model the with the GGD loss in 1 epoch.
2. Generate the node embeddings with the trained model.

### Step 2. Initialize the community detection classifier with k-means
main.py L256-L300

### Step 3. Finetune the model with the community detection loss
main.py L303-L367

1. Minimize distance between the node embeddings and the cluster centroids.
2. Maximize distance between the cluster centroids.
