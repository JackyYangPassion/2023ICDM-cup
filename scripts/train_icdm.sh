python pretrain_icdm.py --hid_units 256 --encoder_layer 3 --lr 1e-4 --epochs 100

# Pretrain
python pretrain_icdm.py --device cuda:0 --dataset icdm --hid_units 256 --encoder_layer 3 --lr 0.0001 --epochs 1 -k 10

# Inference
python main.py --device cuda:0 --dataset icdm --hid_units 256 --encoder_layer 3 --lr 0.00001 --epochs 100 -k 10