# CRAFT: Cross-Attentional Flow Transformers for Optical Flow Estimation
This repository contains the source code for our optical flow estimation method:

[CRAFT: Cross-Attentional Flow Transformers for Optical Flow Estimation](https://arxiv.org/abs/xxxx)<br/>
**Xiuchao Sui**, Shaohua Li, Yan Wu, Xinxing Xu, Kenneth Kwok<br/>
IHPC & I2R, A*STAR, Singapore<br/>

## Environments
You will have to choose cudatoolkit version to match your compute environment. 
The code is tested on PyTorch 1.8.0 but other versions may also work. 
```Shell
conda create --name craft python==3.8
conda activate craft
conda install pytorch=1.8.0 torchvision=0.9.0 cudatoolkit=11.1 -c pytorch -c conda-forge
pip install matplotlib imageio einops scipy opencv-python
```
## Demo
```Shell
sh demo.sh
```
## Train
```Shell
sh train.sh
```
## Evaluate
```Shell
sh evaluate.sh
```
## License
WTFPL. See [LICENSE](LICENSE) file. 

## Acknowledgement
The overall code framework is adapted from [GMA](https://github.com/zacjiang/GMA/). We thank the authors for their contributions.
