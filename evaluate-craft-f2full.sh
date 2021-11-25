#!/usr/bin/env bash
python evaluate.py --dataset chairs --model checkpoints/craft-chairs.pth --craft --f2 full --setrans
python evaluate.py --dataset sintel --model checkpoints/craft-things.pth --craft --f2 full --setrans
python evaluate.py --dataset sintel --model checkpoints/craft-sintel.pth --craft --f2 full --setrans
python evaluate.py --dataset kitti --model checkpoints/craft-kitti.pth --craft --f2 full --setrans
