import torch
import sys

def pth_grep(pth_filepath, keys):
    ckpt = torch.load(pth_filepath, map_location='cpu')
    for var_name in ckpt.keys():
        for key in keys:
            if key in var_name:
                print("{} ({}) => {}".format(var_name, key, ckpt[var_name]))
                
def pth_cmp(pth_filepath1, pth_filepath2, keys):
    ckpt1 = torch.load(pth_filepath1, map_location='cpu')
    ckpt2 = torch.load(pth_filepath2, map_location='cpu')
    
    for var_name in ckpt1.keys():
        for key in keys:
            if key in var_name:
                v1 = ckpt1[var_name]
                v2 = ckpt2[var_name]
                v1_mean = v1.abs().mean()
                v1_std  = v1.std()
                v2_mean = v2.abs().mean()
                v2_std  = v2.std()
                diff_mean = (v1 - v2).abs().mean()
                diff_std  = (v1 - v2).abs().std()
                print("{} ({}):".format(var_name, key))
                print("v1 mean/std:   {:.3f}, {:.3f}".format(v1_mean,   v1_std))
                print("v2 mean/std:   {:.3f}, {:.3f}".format(v2_mean,   v2_std))
                print("diff mean/std: {:.3f}, {:.3f}".format(diff_mean, diff_std))
                

command = sys.argv[1]        
if command == 'grep':
    pth_filepath = sys.argv[2]
    keys = sys.argv[3].split(",")
    pthgrep(pth_filepath, keys)
    
if command == 'cmp':
    pth_filepath1 = sys.argv[2]
    pth_filepath2 = sys.argv[3]
    keys = sys.argv[4].split(",")
    pth_cmp(pth_filepath1, pth_filepath2, keys)
    