# 1 epoch
python3 ggd_test/train_arxiv.py --dataset_name 'ogbn-arxiv' --dataset=ogbn-arxiv --ggd-lr 0.0001 --n-hidden 256 --n-layers 3 --proj_layers 1 --gnn_encoder 'gcn' --n-ggd-epochs 1

# 100 epochs
python3 ggd_test/train_arxiv.py --dataset_name 'ogbn-arxiv' --dataset=ogbn-arxiv --ggd-lr 0.0001 --n-hidden 256 --n-layers 3 --proj_layers 1 --gnn_encoder 'gcn' --n-ggd-epochs 100