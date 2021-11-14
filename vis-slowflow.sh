env SAVECORR=raftcorr-slowflow.pth python3 evaluate.py --model checkpoints/raft-sintel.pth --raft --img1 datasets/slowflow/100/sequence_R03/Animals/seq14_0000000.png --img2 datasets/slowflow/100/sequence_R03/Animals/seq14_0000001.png --flow datasets/slowflow/100/flow/Animals/seq14_0000000.flo --xshifts 220 --yshifts 110 --scale 0.5
env SAVECORR=gmacorr-slowflow.pth python3 evaluate.py --model checkpoints/gma-sintel.pth --img1 datasets/slowflow/100/sequence_R03/Animals/seq14_0000000.png --img2 datasets/slowflow/100/sequence_R03/Animals/seq14_0000001.png --flow datasets/slowflow/100/flow/Animals/seq14_0000000.flo --xshifts 220 --yshifts 110 --scale 0.5
env SAVECORR=craftcorr-slowflow.pth python3 evaluate.py --model checkpoints/craft-sintel.pth --craft --f2 full --setrans --img1 datasets/slowflow/100/sequence_R03/Animals/seq14_0000000.png --img2 datasets/slowflow/100/sequence_R03/Animals/seq14_0000001.png --flow datasets/slowflow/100/flow/Animals/seq14_0000000.flo --xshifts 220 --yshifts 110 --scale 0.5
env SAVECORR=nof2corr-slowflow.pth python3 evaluate.py --model checkpoints/craft-sintel.pth --craft --f2 none --setrans --img1 datasets/slowflow/100/sequence_R03/Animals/seq14_0000000.png --img2 datasets/slowflow/100/sequence_R03/Animals/seq14_0000001.png --flow datasets/slowflow/100/flow/Animals/seq14_0000000.flo --xshifts 220 --yshifts 110 --scale 0.5

python3 attvis.py --model raft --img1 datasets/slowflow/100/sequence_R03/Animals/seq14_0000000.png --img2 datasets/slowflow/100/sequence_R03/Animals/seq14_0000001.png --points 552,256 --att raftcorr-slowflow.pth --scale 0.5
python3 attvis.py --model gma --img1 datasets/slowflow/100/sequence_R03/Animals/seq14_0000000.png --img2 datasets/slowflow/100/sequence_R03/Animals/seq14_0000001.png --points 552,256 --att gmacorr-slowflow.pth --scale 0.5
python3 attvis.py --model craft --img1 datasets/slowflow/100/sequence_R03/Animals/seq14_0000000.png --img2 datasets/slowflow/100/sequence_R03/Animals/seq14_0000001.png --points 552,256 --att craftcorr-slowflow.pth --scale 0.5
python3 attvis.py --model nof2 --img1 datasets/slowflow/100/sequence_R03/Animals/seq14_0000000.png --img2 datasets/slowflow/100/sequence_R03/Animals/seq14_0000001.png --points 552,256 --att nof2corr-slowflow.pth --scale 0.5
