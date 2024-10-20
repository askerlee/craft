import torch
import cv2
import numpy as np
from core.network import CRAFT
from core.craft_nogma import CRAFT_nogma
from core.utils.utils import load_checkpoint

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
                  'nogma': False }
gma_config = { 'mixed_precision': True, 'craft': False, 'setrans': False,
               'f1trans': 'none', 'f2trans': 'none', 'inter_num_modes': 4, 
               'intra_pos_code_weight': 1., 'pos_bias_radius': 7, 'f2_attn_mask_radius': -1,
               'position_only': False, 'position_and_content': False  }

if craft_config.get('nogma', False):
    # CRAFT_nogma needs different checkpoint weights, which are not available.
    model = CRAFT_nogma(craft_config).to('cuda')
else:
    model = CRAFT(craft_config).to('cuda')

flow_model_ckpt_path = "checkpoints/craft-sintel.pth"
load_checkpoint(model, flow_model_ckpt_path)

img1 = cv2.imread('examples/xxr.png')
img2 = cv2.imread('examples/xxr-superman.png')
img1_batch = torch.from_numpy(img1).permute(2, 0, 1).unsqueeze(0).float().to('cuda')
img2_batch = torch.from_numpy(img2).permute(2, 0, 1).unsqueeze(0).float().to('cuda')
#img1_batch = F.interpolate(img1_batch, scale_factor=2, mode='bilinear', align_corners=False)
#img2_batch = F.interpolate(img2_batch, scale_factor=2, mode='bilinear', align_corners=False)

_, flow_predictions = model(img1_batch, img2_batch, num_iters=12, test_mode=1)
flow = flow_predictions[-1].unsqueeze(0)
#flow = F.interpolate(flow_predictions[-1].unsqueeze(0), scale_factor=0.5, mode='bilinear', align_corners=False) * 0.5
flow = flow.permute(0, 2, 3, 1)
img1_recovered = backward_warp_by_flow(img2, flow[0].detach().cpu().numpy())
cv2.imwrite('examples/xxr_recovered.png', img1_recovered)

