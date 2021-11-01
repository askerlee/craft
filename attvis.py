import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
from os.path import join
import numpy as np
import torch
import torch.nn.functional as F
import imageio
import matplotlib.pyplot as plt

from network import CRAFT
from setrans import gen_all_indices
from corr import CorrBlock

import datasets
from utils import flow_viz
from utils import frame_utils
import torchvision.transforms as transforms
import cv2
torch.set_printoptions(sci_mode=False, precision=4)
np.set_printoptions(suppress=True, precision=4)

def save_matrix(filename, mat, print_stats=False):
    # corr = F.avg_pool2d(corr, 4, stride=4).squeeze(1).squeeze(0)
    if print_stats:
        print("{}: {}. mean/std: {:.5f}, {:.5f}".format(filename, list(mat.shape), 
               np.abs(mat).mean(), mat.std()))
               
    plt.imshow(mat)
    plt.colorbar()
    plt.savefig(filename) # dpi=1200
    plt.clf()
    print(f"Saved '{filename}'")
    
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
    parser.add_argument('--craft', dest='craft', action='store_true', 
                        help='use craft (Cross-Attentional Flow Transformer)')
    # In inter-frame attention, having QK biases performs slightly better.
    parser.add_argument('--interqknobias', dest='inter_qk_have_bias', action='store_false', 
                        help='Do not use biases in the QK projections in the inter-frame attention')
    parser.add_argument('--interpos', dest='inter_pos_embed_weight', type=float, default=0.5)
    parser.add_argument('--intrapos', dest='intra_pos_embed_weight', type=float, default=1.0)
    
    parser.add_argument('--savecorr', default=False, action='store_true')
    parser.add_argument('--corrsource', type=str, default=None)
    
    args = parser.parse_args()

    print("Args:\n{}".format(args))
    
    model = torch.nn.DataParallel(CRAFT(args))
    checkpoint = torch.load(args.model, map_location='cpu')
    if 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        # Load old checkpoint.
        model.load_state_dict(checkpoint)
            
    model.eval()
    model = model.module
    
    model_sig = args.model.split("/")[-1].split(".")[0]
    
    vis_whole_pos = False
    
    if vis_whole_pos:
        N = 64
        if args.craft:
            fmap1 = fmap2 = torch.zeros(1, 256, N, N, device='cpu')
            coords1 = gen_all_indices(fmap1.shape[2:], device='cpu')
            coords1 = coords1.unsqueeze(0).repeat(fmap1.shape[0], 1, 1, 1)
            model.corr_fn.update(fmap1, fmap2, coords1)
            # inter_corr: [N*N, 1, N, N]
            inter_corr = model.corr_fn.corr_pyramid[0]
            # inter_corr: [4096, 4096]
            inter_corr = inter_corr.reshape(N*N, N*N)
            save_matrix("{}-inter-pos-attn.pdf".format(model_sig), 
                        inter_corr.data.numpy(), print_stats=True)
        
        if args.setrans:
            inp_feat = torch.zeros(1, 128, N, N, device='cpu')
        else:
            # GMA cannot take zero visual features (it will output all-zero attention).
            inp_feat = torch.randn(1, 128, N, N, device='cpu').abs()
            
        intra_corr = model.att(inp_feat)
        
        nhead = intra_corr.shape[1]
        for i in range(nhead):
            save_matrix("{}-intra-pos-attn-{}.pdf".format(model_sig, i), 
                        intra_corr[0, i].data.numpy(), print_stats=True)
    
    else:
        examples = [ { 
                        'name':        'ambush2',
                        'folder':      'visualization/ambush_2', 
                        'image1':      'frame_0001.png', 
                        'image2':      'frame_0002.png',
                        'orig_size':   [436, 1024],
                        'input_size':  [368, 768],                          
                        'points':      [ [56, 440], [192, 640], [32, 816] ] 
                     }
                   ]

        model_name = f'craft' if args.craft else 'gma'
        
        for example in examples:
            name = example['name']
            orig_size, input_size = example['orig_size'], example['input_size']
            # (436, 1024, 3)
            image1_obj = Image.open(join(example['folder'], example['image1']))
            image2_obj = Image.open(join(example['folder'], example['image2']))
            
            image1a = transforms.Resize(input_size)(image1_obj)
            image2a = transforms.Resize(input_size)(image2_obj)
            # image1_np, image2_np: [368, 768, 3]
            image1_np = np.array(image1a)
            image2_np = np.array(image2a)
            H0, W0    = image1_np.shape[:2]
            H1, W1    = H0 // 8, W0 // 8
            
            if args.corrsource == 'model':
                data_transforms = transforms.Compose([
                                            transforms.Resize(input_size),
                                            transforms.ToTensor(),
                                        ])

                # image1, image2: [3, 368, 768], values within [0, 1]
                image1 = data_transforms(image1_obj)
                image2 = data_transforms(image2_obj)
                image1 = 2 * image1 - 1.0
                image2 = 2 * image2 - 1.0
                # image1, image2: [1, 3, 368, 768]
                image1 = image1.unsqueeze(0)
                image2 = image2.unsqueeze(0)
                
                with torch.no_grad():
                    # fmap1, fmap2: [1, 256, 46, 96].
                    fmap1, fmap2 = model.fnet([image1, image2])
                    H, W = fmap1.shape[2:]
                    assert H == H1 and W == W1
                    # coords0 == coords2. [1, 2, 46, 96]
                    coords0, coords1 = model.initialize_flow(image1)
                    if args.craft:
                        corr_fn = model.corr_fn
                        corr_fn.update(fmap1, fmap2, coords1, coords2=None)
                    else:
                        corr_fn = CorrBlock(fmap1, fmap2, radius=args.corr_radius)
                        
                    # inter_corr: [H*W, 1, H, W] = [4416, 1, 46, 96]
                    inter_corr = corr_fn.corr_pyramid[0]
                    # inter_corr: [46, 96, 46, 96]
                    inter_corr = inter_corr.reshape(H, W, H, W)
                
                sig = "pew" + str(args.inter_pos_embed_weight)
            
            elif args.corrsource == 'posonly':
                assert args.inter_pos_embed_weight > 0
                fmap1 = fmap2 = torch.zeros(1, 256, H1, W1, device='cpu')
                coords1 = gen_all_indices(fmap1.shape[2:], device='cpu')
                coords1 = coords1.unsqueeze(0).repeat(fmap1.shape[0], 1, 1, 1)
                with torch.no_grad():
                    if args.craft:
                        corr_fn = model.corr_fn
                        corr_fn.update(fmap1, fmap2, coords1, coords2=None)
                        vispos1 = corr_fn.vispos_encoder(fmap1, coords1)
                        torch.save(vispos1, f"{name}-{model_name}-posfeat.pth")
                    else:
                        corr_fn = CorrBlock(fmap1, fmap2, radius=args.corr_radius)
                    
                # inter_corr: [H*W, 1, H, W] = [4416, 1, 46, 96]
                inter_corr = corr_fn.corr_pyramid[0]
                # inter_corr: [46, 96, 46, 96]
                inter_corr = inter_corr.reshape(H1, W1, H1, W1)
                sig = 'pos'
            else:
                corr_filenames = args.corrsource.split(",")
                if len(corr_filenames) == 1:
                    inter_corr = torch.load(corr_filenames, map_location='cpu')
                elif len(corr_filenames) == 2:
                    inter_corr1 = torch.load(corr_filenames[0], map_location='cpu')
                    print(f"Loaded correlation matrix from '{corr_filenames[0]}'")
                    inter_corr2 = torch.load(corr_filenames[1], map_location='cpu')
                    print(f"Loaded correlation matrix from '{corr_filenames[1]}'")
                    # Compare two inter_corr matrices.
                    inter_corr  = inter_corr2 - inter_corr1
                    sig = 'diff'
                else:
                    breakpoint()

            if args.savecorr:
                corr_filename = f"{name}-{model_name}-{sig}.pth"
                torch.save(inter_corr, corr_filename)
                print(f"Saved correlation matrix into '{corr_filename}'")
                    
            samp_points = example['points']
            for pi, point in enumerate(samp_points):
                h, w = point
                h_input = int(h * input_size[0] / orig_size[0])
                w_input = int(w * input_size[1] / orig_size[1])
                h = h_input // 8
                w = w_input // 8
                print(f"{point[0]},{point[1]} => {h_input},{w_input} => {h},{w}")
                
                corr = inter_corr[h, w].numpy()
                save_matrix(f"{name}-{point[0]},{point[1]}-{model_name}-{sig}-mat.png", corr, print_stats=True)
                corr = cv2.resize(corr, (W0, H0))
                corr = corr - corr.min()
                corr = (255 * corr / corr.max()).astype(np.uint8)
                # heatmap: [368, 768, 3]
                heatmap = cv2.applyColorMap(corr, cv2.COLORMAP_JET)
                overlaid_img = image2_np * 0.6 + heatmap * 0.3
                overlaid_img = overlaid_img.astype(np.uint8)
                overlaid_img_obj = Image.fromarray(overlaid_img)
                filename = f"{name}-{point[0]},{point[1]}-{model_name}-{sig}.png"
                overlaid_img_obj.save(filename)
                print(f"Saved '{filename}'")
                