import torch
import torch.nn as nn
import torch.nn.functional as F

from .update import BasicUpdateBlock
from .extractor import BasicEncoder
from .corr import TransCorrBlock
from .utils.utils import coords_grid, upflow8
from .setrans import SETransConfig, SelfAttVisPosTrans
from .utils.utils import print0
from easydict import EasyDict as edict

class CRAFT_nogma(nn.Module):
    def __init__(self, config):
        super(CRAFT_nogma, self).__init__()
        if config is None:
            self.config = edict()
        else:
            self.config = edict(config)

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128

        # corr_levels determines the shape of the model params. 
        # So it cannot be changed arbitrarily.
        if not hasattr(self.config, 'corr_levels'):
            self.config.corr_levels = 4
        if not hasattr(self.config, 'corr_radius'):
            self.config.corr_radius = 4
        if not hasattr(self.config, 'dropout'):
            self.config.dropout = 0
        if not hasattr(self.config, 'mixed_precision'):
            self.config.mixed_precision = True
        if not hasattr(self.config, 'num_heads'):
            self.config.num_heads = 1

        print0(f"corr_levels: {self.config.corr_levels}, corr_radius: {self.config.corr_radius}")

        self.inter_trans_config = SETransConfig()
        self.inter_trans_config.update_config(config)
        self.inter_trans_config.in_feat_dim = 256
        self.inter_trans_config.feat_dim    = 256
        self.inter_trans_config.max_pos_size     = 160
        self.inter_trans_config.out_attn_scores_only    = True
        self.inter_trans_config.attn_diag_cycles = 1000
        self.inter_trans_config.num_modes       = config.get('inter_num_modes', 4)          # default: 4
        self.inter_trans_config.qk_have_bias    = config.get('inter_qk_have_bias', True)    # default: True
        self.inter_trans_config.pos_code_type   = config.get('inter_pos_code_type', 'bias') # default: bias
        self.inter_trans_config.pos_code_weight = config.get('inter_pos_code_weight', 0.5)  # default: 0.5
        self.config.inter_trans_config = self.inter_trans_config
        print0("Inter-frame trans config:\n{}".format(self.inter_trans_config.__dict__))
            
        self.corr_fn = TransCorrBlock(self.inter_trans_config, radius=self.config.corr_radius,
                                      do_corr_global_norm=True)

        if self.config.f2trans != 'none':
            # f2_trans has the same configuration as GMA att, 
            # except that the feature dimension is doubled, and not out_attn_probs_only.
            self.f2_trans_config = SETransConfig()
            self.f2_trans_config.update_config(config)
            self.f2_trans_config.do_half_attn = (self.config.f2trans == 'half')
            self.f2_trans_config.in_feat_dim = 256
            self.f2_trans_config.feat_dim  = 256
            # if do_half_attn, has_input_skip will be changed to False within SelfAttVisPosTrans.__init__().
            self.f2_trans_config.has_input_skip = True
            # No FFN. f2trans simply aggregates similar features.
            self.f2_trans_config.has_FFN = False
            
            # Not tying QK performs slightly better.
            self.f2_trans_config.tie_qk_scheme = None
            self.f2_trans_config.qk_have_bias  = False
            self.f2_trans_config.out_attn_probs_only    = False
            self.f2_trans_config.attn_diag_cycles   = 1000
            self.f2_trans_config.num_modes          = config.get('f2_num_modes', 4)             # default: 4
            self.f2_trans_config.pos_code_type      = config.get('intra_pos_code_type', 'bias') # default: bias
            self.f2_trans_config.pos_code_weight    = config.get('f2_pos_code_weight', 0.5)     # default: 0.5
            self.f2_trans = SelfAttVisPosTrans(self.f2_trans_config, "F2 transformer")
            print0("F2-trans config:\n{}".format(self.f2_trans_config.__dict__))
            self.config.f2_trans_config = self.f2_trans_config
                    
        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256, norm_fn='instance',    dropout=self.config.dropout)        
        self.cnet = BasicEncoder(output_dim=hdim+cdim, norm_fn='batch', dropout=self.config.dropout)
        self.update_block = BasicUpdateBlock(self.config, hidden_dim=hdim)

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

        up_flow = F.unfold(8 * flow, [3, 3], padding=1)
        up_flow = up_flow.view(N, 2, 9, 1, 1, H, W)

        up_flow = torch.sum(mask * up_flow, dim=2)
        up_flow = up_flow.permute(0, 1, 4, 2, 5, 3)
        return up_flow.reshape(N, 2, 8*H, 8*W)


    def forward(self, image1, image2, num_iters=12, flow_init=None, upsample=True, test_mode=0):
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
            if self.config.f1trans != 'none':
                fmap12 = torch.cat([fmap1, fmap2], dim=0)
                fmap12  = self.f2_trans(fmap12)
                fmap1, fmap2 = torch.split(fmap12, [fmap1.shape[0], fmap2.shape[0]])

            elif self.config.f2trans != 'none':
                fmap2  = self.f2_trans(fmap2)
                
        # fmap1, fmap2: [1, 256, 55, 128]. 1/8 size of the original image.
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()

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

        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        with torch.amp.autocast('cuda', enabled=self.config.mixed_precision):
            self.corr_fn.update(fmap1, fmap2, coords1)

        flow_predictions = []
        for itr in range(num_iters):
            coords1 = coords1.detach()
            # corr: [6, 324, 50, 90]. 324: neighbors. 
            # radius = 4 -> neighbor points = (4*2+1)^2 = 81. Upsize x4 -> 324.            
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
