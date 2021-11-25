# CRAFT: Cross-Attentional Flow Transformers for Optical Flow Estimation
This repository contains the source code for our optical flow estimation method:

[CRAFT: Cross-Attentional Flow Transformers for Robust Optical Flow](https://arxiv.org/abs/xxxx)<br/>
Anonymous

## Environments
You will have to choose cudatoolkit version to match your compute environment. 
The code is tested on PyTorch 1.8.0 but other versions may also work. 
```Shell
conda create --name craft python==3.8
conda activate craft
conda install pytorch=1.8.0 torchvision=0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install matplotlib imageio einops scipy opencv-python
```

## Train
```Shell
sh train-craft-f2full.sh
```
## Evaluate
```Shell
sh evaluate-craft-f2full.sh
```
## License
WTFPL. See [LICENSE](LICENSE) file. 

## Acknowledgement
The overall code framework is adapted from [GMA](https://github.com/zacjiang/GMA/). We thank the authors for their contributions.
