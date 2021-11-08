python3 evaluate.py --model checkpoints/craft-sintel.pth --craft --corrnorm global --f2 full --setrans --dataset slowflow --xshifts 100,120,140,160,180,200,220,240,260,280,300 --yshifts 50,60,70,80,90,100,110,120,130,140,150 --slowset 100,3
env CUDA_VISIBLE_DEVICES=1 python3 evaluate.py --model results/sintel/craft-f2full-gma/craft-sintel.pth --craft --corrnorm global --f2 full --dataset slowflow --xshifts 100,120,140,160,180,200,220,240,260,280,300 --yshifts 50,60,70,80,90,100,110,120,130,140,150 --slowset 100,3

python3 evaluate.py --model checkpoints/gma-sintel.pth --dataset slowflow --xshifts 100,120,140,160,180,200,220,240,260,280,300 --yshifts 50,60,70,80,90,100,110,120,130,140,150 --slowset 100,3
