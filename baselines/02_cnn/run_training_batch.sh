CUDA_VISIBLE_DEVICES=0 python dnn_training.py --bagging 5 --config configs/config_salami_short.yml
CUDA_VISIBLE_DEVICES=0 python dnn_training.py --bagging 5 --config configs/config_salami_long.yml
CUDA_VISIBLE_DEVICES=0 python dnn_training.py --bagging 5 --config configs/config_jsd_short.yml
CUDA_VISIBLE_DEVICES=0 python dnn_training.py --bagging 5 --config configs/config_jsd_long.yml
CUDA_VISIBLE_DEVICES=0 python dnn_training.py --bagging 5 --config configs/config_salami-jsd_short.yml
CUDA_VISIBLE_DEVICES=0 python dnn_training.py --bagging 5 --config configs/config_salami-jsd_long.yml
