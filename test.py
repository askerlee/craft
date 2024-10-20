import torch
import torch.nn.functional as F
import cv2
import numpy as np
from core.network import CRAFT
from core.utils.utils import load_checkpoint
import argparse

def backward_warp_by_flow(image2, flow1to2):
    H, W, _ = image2.shape
    flow1to2 = flow1to2.copy()
    flow1to2[:, :, 0] += np.arange(W)  # Adjust x-coordinates
    flow1to2[:, :, 1] += np.arange(H)[:, None]  # Adjust y-coordinates
    image1_recovered = cv2.remap(image2, flow1to2, None, cv2.INTER_LINEAR)
    return image1_recovered

#model = raft_large(pretrained=True, progress=False).to('cuda')
#model = model.eval()
craft_config = { 'mixed_precision': True, 'craft': True, 'setrans': True, 
                 'f1trans': 'none', 'f2trans': 'full', 'inter_num_modes': 4, 'intra_pos_code_type': 'bias',
                 'intra_pos_code_weight': 1., 'pos_bias_radius': 7, 'f2_attn_mask_radius': -1, 
               }
gma_config = { 'mixed_precision': True, 'craft': False, 'setrans': False,
               'f1trans': 'none', 'f2trans': 'none', 'inter_num_modes': 4, 
               'intra_pos_code_weight': 1., 'pos_bias_radius': 7, 'f2_attn_mask_radius': -1,
               'position_only': False, 'position_and_content': False  }

parser = argparse.ArgumentParser()
parser.add_argument('--model', type=str, default='gma', help='craft or gma')
parser.add_argument('--subj', dest='subj_name', type=str, default='xxr', help='subject name')
args = parser.parse_args()
if args.model == 'craft':
    model = CRAFT(craft_config).to('cuda')
    flow_model_ckpt_path = "checkpoints/craft-sintel.pth"
else:
    model = CRAFT(gma_config).to('cuda')
    flow_model_ckpt_path = "checkpoints/gma-sintel.pth"

load_checkpoint(model, flow_model_ckpt_path)

img1_path = f'examples/{args.subj_name}.png'
img2_path = f'examples/{args.subj_name}-adaface.png'

img1 = cv2.imread(img1_path)
img2 = cv2.imread(img2_path)
print(f"Input images: {img1_path}, {img2_path}")
img1_batch = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to('cuda')
img2_batch = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to('cuda')
S = 1 / 4
img1_batch = F.interpolate(img1_batch, scale_factor=S, mode='bilinear', align_corners=False)
img2_batch = F.interpolate(img2_batch, scale_factor=S, mode='bilinear', align_corners=False)

_, flow_predictions = model(img1_batch, img2_batch, num_iters=12, test_mode=1)
flow = flow_predictions[-1].unsqueeze(0)
flow = F.interpolate(flow, scale_factor=1 / S, mode='bilinear', align_corners=False) / S
flow = flow.permute(0, 2, 3, 1)
img1_recovered = backward_warp_by_flow(img2, flow[0].detach().cpu().numpy())
cv2.imwrite(f'examples/{args.subj_name}_recovered.png', img1_recovered)

