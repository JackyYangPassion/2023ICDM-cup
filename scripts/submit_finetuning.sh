K_LIST="5 10 15 20 25" # "1 2 3 4 5"
# K_LIST="25"
# N_HID_LIST="64 128 256 512"
# N_LAYER_LIST="1 2 3"
# LR_1_LIST="0.0001 0.001 0.01"
# LR_2_LIST="0.0001 0.001 0.01"
# EPOCH_1_LIST="1 100"
# EPOCH_2_LIST="100 200"

N_HID=256
N_LAYER=3
LR_1=0.0001
LR_2=0.01
EPOCH_1=100
EPOCH_2=100

DATA_LIST="icdm" # ogbn_arxiv ogbn_mag ogbn_products icdm
for DATA in $DATA_LIST; do
    for K in $K_LIST; do
        POSTFIX="${DATA}_${N_HID}_${N_LAYER}_${LR_1}_${EPOCH_1}_${K}"
        sbatch --job-name ${POSTFIX}_${LR_2}_${EPOCH_2}_${K} --cpus-per-task 5 submit_1-A100_job.sh \
            "conda activate py310" \
            "python pretrain_icdm.py --device cuda:0 --dataset ${DATA} --hid_units ${N_HID} --encoder_layer ${N_LAYER} --lr ${LR_1} --epochs ${EPOCH_1} -k ${K}" \
            "python main.py --device cuda:0 --dataset ${DATA} --hid_units ${N_HID} --encoder_layer ${N_LAYER} --lr ${LR_2} --epochs ${EPOCH_2} -k ${K} --postfix ${POSTFIX}"
    done
done