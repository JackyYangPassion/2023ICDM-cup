JOBNAME="NONE_RELU"
sbatch --job-name ${JOBNAME} --cpus-per-task 5 submit_1-A100_job.sh \
"conda activate py310" \
"python3 main.py --K=10 --classifier-lr=0.0005 --dataset=icdm --ggd-lr=0.0001 --gnn_encoder=gcn --n-classifier-epochs=300 --n-ggd-epochs=1 --n-hidden=64 --n-layers=4 --power=15 --proj_layers=1 --seed=2023 --wandb=True --activation none --postfix ${JOBNAME}"

JOBNAME="REWEIGHT_CLUSTING"
sbatch --job-name ${JOBNAME} --cpus-per-task 5 submit_1-A100_job.sh \
"conda activate py310" \
"python3 main.py --K=10 --classifier-lr=0.0005 --dataset=icdm --ggd-lr=0.0001 --gnn_encoder=gcn --n-classifier-epochs=300 --n-ggd-epochs=1 --n-hidden=64 --n-layers=4 --power=15 --proj_layers=1 --seed=2023 --wandb=True --cluster_loss_weight 2 --postfix ${JOBNAME}"

JOBNAME="NONE_RELU+REWEIGHT_CLUSTING"
sbatch --job-name ${JOBNAME} --cpus-per-task 5 submit_1-A100_job.sh \
"conda activate py310" \
"python3 main.py --K=10 --classifier-lr=0.0005 --dataset=icdm --ggd-lr=0.0001 --gnn_encoder=gcn --n-classifier-epochs=300 --n-ggd-epochs=1 --n-hidden=64 --n-layers=4 --power=15 --proj_layers=1 --seed=2023 --wandb=True --cluster_loss_weight 2 --activation none --postfix ${JOBNAME}"