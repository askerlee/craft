import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt

from network import RAFTER
from setrans import gen_all_indices

import datasets
from utils import flow_viz
from utils import frame_utils
import torchvision.transforms as transforms

def save_corr(filename, corr):
    # corr = F.avg_pool2d(corr, 4, stride=4).squeeze(1).squeeze(0)
    print("{}: {}. mean/std: {:.5f}, {:.5f}".format(filename, list(corr.shape), 
           corr.abs().mean(), corr.std()))
    plt.imshow(corr.data.numpy())
    plt.colorbar()
    plt.savefig(filename, dpi=1200)
    plt.clf()

    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', help="restore checkpoint")
    parser.add_argument('--dataset', help="dataset for evaluation")
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--mixed_precision', type=bool, default=True, help='use mixed precision')
    parser.add_argument('--model_name')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')
    parser.add_argument('--radius', dest='corr_radius', type=int, default=4)    

    parser.add_argument('--pos', dest='pos_embed_type', type=str, 
                        choices=['lsinu', 'hwadd'], default='lsinu')
    parser.add_argument('--corrnorm', dest='corr_norm_type', type=str, 
                        choices=['none', 'local', 'global'], default='none')
    parser.add_argument('--setrans', dest='setrans', action='store_true', 
                        help='use setrans (Squeeze-Expansion Transformer)')
    parser.add_argument('--intermodes', dest='inter_num_modes', type=int, default=1, 
                        help='Number of modes in inter-frame attention')
    parser.add_argument('--intramodes', dest='intra_num_modes', type=int, default=4, 
                        help='Number of modes in intra-frame attention')
    parser.add_argument('--rafter', dest='rafter', action='store_true', 
                        help='use rafter (Recurrent All-Pairs Field Transformer)')
    # In inter-frame attention, having QK biases performs slightly better.
    parser.add_argument('--interqknobias', dest='inter_qk_have_bias', action='store_false', 
                        help='Do not use biases in the QK projections in the inter-frame attention')
    parser.add_argument('--interpos', dest='inter_pos_embed_weight', type=float, default=0.5)
    parser.add_argument('--intrapos', dest='intra_pos_embed_weight', type=float, default=1.0)
    parser.add_argument('--perturbpew', dest='perturb_pew_range', type=float, default=0.,
                        help='The range of added random noise to pos_embed_weight during training')
    
    args = parser.parse_args()

    print("Args:\n{}".format(args))
    
    model = torch.nn.DataParallel(RAFTER(args))
    checkpoint = torch.load(args.model, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Load old checkpoint.
        model.load_state_dict(checkpoint)
            
    model.eval()
    model = model.module()
    
    model_sig = args.model.split("/")[-1].split(".")[0]
    
    vis_whole_pos = False
    
    if vis_whole_pos:
        N = 64
        if args.rafter:
            fmap1 = fmap2 = torch.zeros(1, 256, N, N, device='cpu')
            coords1 = gen_all_indices(fmap1.shape[2:], device='cpu')
            coords1 = coords1.unsqueeze(0).repeat(fmap1.shape[0], 1, 1, 1)
            model.corr_fn.update(fmap1, fmap2, coords1)
            # corr: [N*N, 1, N, N]
            inter_corr = model.corr_fn.corr_pyramid[0]
            # corr: [4096, 4096]
            inter_corr = inter_corr.reshape(N*N, N*N)
            save_corr("{}-inter-pos-attn.pdf".format(model_sig), inter_corr)
        
        if args.setrans:
            inp_feat = torch.zeros(1, 128, N, N, device='cpu')
        else:
            # GMA cannot take zero visual features (it will output all-zero attention).
            inp_feat = torch.randn(1, 128, N, N, device='cpu').abs()
            
        intra_corr = model.att(inp_feat)
        
        nhead = intra_corr.shape[1]
        for i in range(nhead):
            save_corr("{}-intra-pos-attn-{}.pdf".format(model_sig, i), intra_corr[0, i])
    
    else:
        # (436, 1024, 3)
        image1 = Image.open("visualization/ambush_2/frame_0001.png")
        image2 = Image.open("visualization/ambush_2/frame_0002.png")
        
        data_transforms = transforms.Compose([
                                    transforms.Resize((368, 768)),
                                    transforms.ToTensor(),
                                ])
        # image: [3, 368, 768], within [0, 1]
        image1 = data_transforms(image1)
        image1 = 2 * image1 - 1.0
        image2 = data_transforms(image2)
        image2 = 2 * image2 - 1.0
        image1 = image1.unsqueeze(0)
        image2 = image2.unsqueeze(0)
        with torch.no_grad():
            fmap1, fmap2 = model.fnet([image1, image2])
            coords0, coords1 = model.initialize_flow(image1)
            model.corr_fn.update(fmap1, fmap2, coords1, coords2=None)
            corr = self.corr_fn(coords1)
        breakpoint()
                