# CRAFT: Cross-Attentional Flow Transformers for Robust Optical Flow
This repository contains the source code for our optical flow estimation method:

[CRAFT: Cross-Attentional Flow Transformers for Robust Optical Flow](https://arxiv.org/abs/xxxx)<br/>
Xiuchao Sui, Shaohua Li, Xue Geng, Yan Wu, Xinxing Xu, Yong Liu, Rick Goh, Hongyuan Zhu

## Environments
You will have to choose cudatoolkit version to match your compute environment. 
The code is tested on PyTorch 1.8.0 but other versions may also work. 
```Shell
pip install -r requirements.txt
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

## Citation
@InProceedings{craft,
author="Sui, Xiuchao and Li, Shaohua and Geng, Xue and Wu, Yan and Xu, Xinxing and Liu, Yong and Goh, Rick Siow Mong and Zhu, Hongyuan",  
title="CRAFT: Cross-Attentional Flow Transformers for Robust Optical Flow",  
booktitle="CVPR",  
year="2022"}
    
## Acknowledgement
The overall code framework is adapted from [GMA](https://github.com/zacjiang/GMA/). We thank the authors for their contributions.
