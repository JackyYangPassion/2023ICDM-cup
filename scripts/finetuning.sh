python3 main.py --dataset "icdm" --ggd-lr 0.0001 --n-hidden 256 --n-layers 3 --proj_layers 1 --gnn_encoder 'gcn' --n-ggd-epochs 1 --n-classifier-epochs 10 --classifier-lr 0.0001

# srun --job-name "InteractiveJob" --cpus-per-task 4 --mem 32G  --gres=gpu:1 --partition A100 --time=22-00:00:00 --pty bash