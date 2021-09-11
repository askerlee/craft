#!/usr/bin/env bash
env CUDA_VISIBLE_DEVICES=0,3 python train.py --name setrans-chairs --stage chairs --validation chairs --output results/chairs/setrans --num_steps 120000 --lr 0.00025 --image_size 368 496 --wdecay 0.0001 --gpus 0 1 --batch_size 8 --val_freq 10000 --print_freq 100 --mixed_precision --setrans --rafter --corrnorm global --posr 7
env CUDA_VISIBLE_DEVICES=0,3 python train.py --name setrans-things --stage things --validation sintel --output results/things/setrans --restore_ckpt results/chairs/setrans/setrans-chairs.pth --num_steps 140000 --lr 0.000125 --image_size 400 720 --wdecay 0.0001 --gpus 0 1 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision --setrans --rafter --corrnorm global --posr 7
env CUDA_VISIBLE_DEVICES=0,3 python train.py --name setrans-sintel --stage sintel --validation sintel --output results/sintel/setrans --restore_ckpt results/things/setrans/130000_setrans-things.pth --num_steps 140000 --lr 0.000125 --image_size 368 768 --wdecay 0.00001 --gamma 0.85 --gpus 0 1 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision --setrans --rafter --corrnorm global --posr 7
env CUDA_VISIBLE_DEVICES=0,3 python train.py --name setrans-kitti --stage kitti --validation kitti --output results/kitti/setrans --restore_ckpt results/sintel/setrans/setrans-sintel.pth --num_steps 60000 --lr 0.000125 --image_size 288 960 --wdecay 0.00001 --gamma 0.85 --gpus 0 1 --batch_size 6 --val_freq 10000 --print_freq 100 --mixed_precision --setrans --rafter --corrnorm global --posr 7
