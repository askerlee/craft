export SET=clean # final
env SAVECORR=raftcorr-sintel-$SET.pth python3 evaluate.py --model checkpoints/raft-sintel.pth --raft --img1 datasets/Sintel/test/$SET/market_4/frame_0034.png --img2 datasets/Sintel/test/$SET/market_4/frame_0035.png
env SAVECORR=gmacorr-sintel-$SET.pth python3 evaluate.py --model checkpoints/gma-sintel.pth --img1 datasets/Sintel/test/$SET/market_4/frame_0034.png --img2 datasets/Sintel/test/$SET/market_4/frame_0035.png
env SAVECORR=craftcorr-sintel-$SET.pth python3 evaluate.py --model checkpoints/craft-sintel.pth --craft --setrans --f2 full --img1 datasets/Sintel/test/$SET/market_4/frame_0034.png --img2 datasets/Sintel/test/$SET/market_4/frame_0035.png

python3 attvis.py --model raft --img1 datasets/Sintel/test/$SET/market_4/frame_0034.png --img2 datasets/Sintel/test/$SET/market_4/frame_0035.png --points 16,160.32,280 --att raftcorr-sintel-$SET.pth --savedir attvis/$SET
python3 attvis.py --model gma --img1 datasets/Sintel/test/$SET/market_4/frame_0034.png --img2 datasets/Sintel/test/$SET/market_4/frame_0035.png --points 16,160.32,280 --att gmacorr-sintel-$SET.pth --savedir attvis/$SET
python3 attvis.py --model craft --img1 datasets/Sintel/test/$SET/market_4/frame_0034.png --img2 datasets/Sintel/test/$SET/market_4/frame_0035.png --points 16,160.32,280 --att craftcorr-sintel-$SET.pth --savedir attvis/$SET
