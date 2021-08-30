#!/usr/bin/env bash
python evaluate.py --dataset chairs --model checkpoints/gma-chairs.pth
python evaluate.py --dataset sintel --model checkpoints/gma-things.pth
python evaluate.py --dataset sintel --model checkpoints/gma-sintel.pth
python evaluate.py --dataset kitti --model checkpoints/gma-kitti.pth