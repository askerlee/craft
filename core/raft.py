import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import CorrBlock
from .utils.utils import coords_grid, upflow8


class RAFT(nn.Module):
    def __init__(self, config):
        super(RAFT, self).__init__()
        self.config = config
        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        config.corr_levels = 4

        print("RAFT lookup radius: %d" %config.corr_radius)
        if 'dropout' not in self.config:
            self.config.dropout = 0

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance', dropout=config.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=config.dropout)
        self.update_block = BasicUpdateBlock(self.config, hidden_dim=hdim)

        test_rand_proj = False
        test_zero_proj = True
        if test_rand_proj:
            self.rand_proj = nn.Conv2d(128, 128, 1, padding=0)
        elif test_zero_proj:
            self.rand_proj = torch.zeros_like
        else:
            self.rand_proj = lambda x: x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H//8, W//8).to(img.device)
        coords1 = coords_grid(N, H//8, W//8).to(img.device)

        # optical flow computed as difference: flow = coords1 - coords0
        return coords0, coords1

    def upsample_flow(self, flow, mask):
        """ Upsample flow field [H/8, W/8, 2] -> [H, W, 2] using convex combination """
        N, _, H, W = flow.shape
        mask = mask.view(N, 1, 9, 8, 8, H, W)
        mask = torch.softmax(mask, dim=2)

        up_flow = F.unfold(8 * flow, [3,3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=0):
        """ Estimate optical flow between pair of frames """

        # image1, image2: [1, 3, 440, 1024]
        # image1 mean: [-0.1528, -0.2493, -0.3334]
        image1 = 2 * (image1 / 255.0) - 1.0
        image2 = 2 * (image2 / 255.0) - 1.0

        image1 = image1.contiguous()
        image2 = image2.contiguous()

        hdim = self.hidden_dim
        cdim = self.context_dim

        # run the feature network
        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])        
        
        # fmap1, fmap2: [1, 256, 55, 128]. 1/8 size of the original image.
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
        self.corr_fn = CorrBlock(fmap1, fmap2, radius=self.config.corr_radius)

        # run the context network
        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            # cnet: context network to extract features from image1 only.
            # cnet arch is the same as fnet. 
            # fnet extracts features specifically for correlation computation.
            # cnet_feat: extracted features focus on semantics of image1? 
            # (semantics of each pixel, used to guess its motion?)
            cnet_feat = self.cnet(image1)
            # net_feat, inp_feat: [1, 128, 55, 128]
            net_feat, inp_feat = torch.split(cnet_feat, [hdim, cdim], dim=1)
            net_feat = torch.tanh(net_feat)
            inp_feat = torch.relu(inp_feat)

            net_feat = self.rand_proj(net_feat)
            inp_feat = self.rand_proj(inp_feat)

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            corr = self.corr_fn(coords1) # index correlation volume

            flow = coords1 - coords0
                
            with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
                # net_feat: hidden features of ConvGRU. 
                # inp_feat: input  features to ConvGRU.
                # up_mask is scaled to 0.25 of original values.
                # update_block: BasicUpdateBlock
                # In the first few iterations, delta_flow.abs().max() could be 1.3 or 0.8. Later it becomes 0.2~0.3.
                net_feat, up_mask, delta_flow = self.update_block(net_feat, inp_feat, corr, flow)


            # F(t+1) = F(t) + \Delta(t)
            coords1 = coords1 + delta_flow
                
            # upsample predictions
            if up_mask is None:
                # coords0 is fixed as original coords.
                # upflow8: upsize to 8 * height, 8 * width. 
                # flow value also *8 (scale the offsets proportionally to the resolution).
                flow_up = upflow8(coords1 - coords0)
            else:
                # The final high resolution flow field is found 
                # by using the mask to take a weighted combination over the neighborhood.
                flow_up = self.upsample_flow(coords1 - coords0, up_mask)
            flow_predictions.append(flow_up)

        if test_mode == 1:
            return coords1 - coords0, flow_up
        if test_mode == 2:
            return coords1 - coords0, flow_predictions
            
        return flow_predictions
