K_LIST="10 25" # 5 10 15 20 25
N_HID_LIST="256" # 64 128 256 512 1024
N_LAYER_LIST="3" # 1 2 3 4
LR_1_LIST="0.0001" #  0.001 0.01
LR_2_LIST="0.0001" #  0.001 0.01
EPOCH_1_LIST="1" # 1 10 100
EPOCH_2_LIST="10 100 200" #  200
CPU_KMEANS=False
PRETRAIN=False
PRETRAIN_ONLY=False

# N_HID=256
# N_LAYER=3
# LR_1=0.0001
# LR_2=0.01
# EPOCH_1=100
# EPOCH_2=100

DATA_LIST="icdm" # ogbn_arxiv ogbn_mag ogbn_products icdm
for DATA in $DATA_LIST; do
    for K in $K_LIST; do
        for N_HID in $N_HID_LIST; do
            for N_LAYER in $N_LAYER_LIST; do
                for LR_1 in $LR_1_LIST; do
                    for LR_2 in $LR_2_LIST; do
                        for EPOCH_1 in $EPOCH_1_LIST; do
                            for EPOCH_2 in $EPOCH_2_LIST; do
                                POSTFIX="${K}_${DATA}_${N_HID}_${N_LAYER}_${LR_1}_${EPOCH_1}_${LR_2}_${EPOCH_2}"
                                sbatch --job-name ${POSTFIX}_${LR_2}_${EPOCH_2}_${K} --cpus-per-task 5 submit_1-A100_job.sh \
                                    "conda activate py310" \
                                    "python3 main.py --dataset ${DATA} --ggd-lr ${LR_1} --n-hidden ${N_HID}  --n-layers ${N_LAYER} --proj_layers 1 --gnn_encoder 'gcn' --n-ggd-epochs ${EPOCH_1} --n-classifier-epochs ${EPOCH_2} --classifier-lr ${LR_2} --K ${K} --cpu_kmeans ${CPU_KMEANS} --pretrain ${PRETRAIN} --pretrain_only ${PRETRAIN_ONLY}"
                            done
                        done
                    done
                done
            done
        done
    done
done
