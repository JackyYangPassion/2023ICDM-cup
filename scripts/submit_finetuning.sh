K_LIST="2 5 10 15" # "1 2 3 4 5"
K_LIST="5 10"
N_HID=256
N_LAYER=3
LR_1=0.0001
LR_2=0.00001
EPOCH_1=1
EPOCH_2=100
for K in $K_LIST; do
    sbatch --job-name ${K}-ICDM --cpus-per-task 1 submit_1-A100_job.sh \
        "conda activate py310" \
        "python pretrain_icdm.py --device cuda:0 --dataset icdm --hid_units ${N_HID} --encoder_layer ${N_LAYER} --lr ${LR_1} --epochs ${EPOCH_1} -k ${K}" \
        "python main.py --device cuda:0 --dataset icdm --hid_units ${N_HID} --encoder_layer ${N_LAYER} --lr ${LR_2} --epochs ${EPOCH_2} -k ${K}"
done
