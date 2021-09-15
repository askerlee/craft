import torch
import torch.nn as nn
import torch.nn.functional as F

from update import GMAUpdateBlock
from extractor import BasicEncoder
from corr import CorrBlock, TransCorrBlock
from utils.utils import bilinear_sampler, coords_grid, upflow8
from gma import Attention, Aggregate
from setrans import SETransConfig, SelfAttVisPosTrans
import copy

try:
    autocast = torch.cuda.amp.autocast
except:
    # dummy autocast for PyTorch < 1.6
    class autocast:
        def __init__(self, enabled):
            pass

        def __enter__(self):
            pass

        def __exit__(self, *args):
            pass

class CRAFT(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args

        self.hidden_dim = hdim = 128
        self.context_dim = cdim = 128
        args.corr_levels = 4

        if 'dropout' not in self.args:
            self.args.dropout = 0

        # default CRAFT corr_radius: 4
        if args.corr_radius == -1:
            args.corr_radius = 4
        print("Lookup radius: %d" %args.corr_radius)
        
        self.do_corr_global_norm = (args.corr_norm_type == 'global')
        
        if args.craft:
            self.inter_trans_config = SETransConfig()
            self.inter_trans_config.update_config(args)
            self.inter_trans_config.in_feat_dim = 256
            self.inter_trans_config.feat_dim    = 256
            self.inter_trans_config.max_pos_size     = 160
            self.inter_trans_config.out_attn_scores_only    = True
            self.inter_trans_config.attn_diag_cycles = 1000
            self.inter_trans_config.num_modes        = args.inter_num_modes
            self.inter_trans_config.qk_have_bias     = args.inter_qk_have_bias
            self.inter_trans_config.pos_code_type   = args.inter_pos_code_type
            self.inter_trans_config.pos_code_weight = args.inter_pos_code_weight
            self.inter_trans_config.perturb_posw_range  = args.perturb_inter_posw_range
            self.args.inter_trans_config = self.inter_trans_config
            print("Inter-frame trans config:\n{}".format(self.inter_trans_config.__dict__))
            
            self.corr_fn = TransCorrBlock(self.inter_trans_config, radius=self.args.corr_radius,
                                          do_corr_global_norm=self.do_corr_global_norm)
        
        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=256,         norm_fn='instance', dropout=args.dropout)
        self.cnet = BasicEncoder(output_dim=hdim + cdim, norm_fn='batch',    dropout=args.dropout)
       
        if args.setrans:
            self.intra_trans_config = SETransConfig()
            self.intra_trans_config.update_config(args)
            self.intra_trans_config.in_feat_dim = 128
            self.intra_trans_config.feat_dim  = 128
            # Having FFN reduces performance.
            # has_FFN & has_input_skip are for GMAUpdateBlock.aggregator.
            self.intra_trans_config.has_FFN = False
            self.intra_trans_config.has_input_skip = True
            # Not tying QK performs slightly better.
            self.intra_trans_config.tie_qk_scheme = None
            self.intra_trans_config.qk_have_bias  = False
            self.intra_trans_config.out_attn_probs_only    = True
            self.intra_trans_config.attn_diag_cycles = 1000
            self.intra_trans_config.num_modes           = args.intra_num_modes
            self.intra_trans_config.pos_code_type       = args.intra_pos_code_type
            self.intra_trans_config.pos_code_weight     = args.intra_pos_code_weight
            self.intra_trans_config.perturb_posw_range  = args.perturb_intra_posw_range
            self.att = SelfAttVisPosTrans(self.intra_trans_config, "Intra-frame attention")
            self.args.intra_trans_config = self.intra_trans_config
            print("Intra-frame trans config:\n{}".format(self.intra_trans_config.__dict__))
        else:
            self.att = Attention(args=self.args, dim=cdim, heads=self.args.num_heads, max_pos_size=160, dim_head=cdim)

        # if args.setrans, initialization of GMAUpdateBlock.aggregator needs to access self.args.intra_trans_config.
        # So GMAUpdateBlock() construction has to be done after initializing intra_trans_config.
        self.update_block = GMAUpdateBlock(self.args, hidden_dim=hdim)
 
    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eval()

    def initialize_flow(self, img):
        """ Flow is represented as difference between two coordinate grids flow = coords1 - coords0"""
        N, C, H, W = img.shape
        coords0 = coords_grid(N, H // 8, W // 8).to(img.device)
        coords1 = coords_grid(N, H // 8, W // 8).to(img.device)

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
        return up_flow.reshape(N, 2, 8 * H, 8 * W)

    def forward(self, image1, image2, iters=12, flow_init=None, upsample=True, test_mode=False):
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
        with autocast(enabled=self.args.mixed_precision):
            fmap1, fmap2 = self.fnet([image1, image2])

        # fmap1, fmap2: [1, 256, 55, 128]. 1/8 size of the original image.
        # correlation matrix: 7040*7040 (55*128=7040).
        fmap1 = fmap1.float()
        fmap2 = fmap2.float()
            
        if not self.args.craft:
            self.corr_fn = CorrBlock(fmap1, fmap2, radius=self.args.corr_radius)

        with autocast(enabled=self.args.mixed_precision):
            # run the context network
            # cnet: context network to extract features from image1 only.
            # cnet arch is the same as fnet. 
            # fnet extracts features specifically for correlation computation.
            # cnet_feat: extracted features focus on semantics of image1? 
            # (semantics of each pixel, used to guess its motion?)
            cnet_feat = self.cnet(image1)
            
            # Both fnet and cnet are BasicEncoder. output is from conv (no activation function yet).
            # net_feat, inp_feat: [1, 128, 55, 128]
            net_feat, inp_feat = torch.split(cnet_feat, [hdim, cdim], dim=1)
            net_feat = torch.tanh(net_feat)
            inp_feat = torch.relu(inp_feat)
            # attention, att_c, att_p = self.att(inp_feat)
            attention = self.att(inp_feat)
                
        # coords0 is always fixed as original coords.
        # coords1 is iteratively updated as coords0 + current estimated flow.
        # At this moment coords0 == coords1.
        coords0, coords1 = self.initialize_flow(image1)

        if flow_init is not None:
            coords1 = coords1 + flow_init

        if self.args.craft:
            with autocast(enabled=self.args.mixed_precision):
                # only update() once, instead of dynamically updating coords1.
                self.corr_fn.update(fmap1, fmap2, coords1, coords2=None)
            
        flow_predictions = []
        for itr in range(iters):
            coords1 = coords1.detach()
            # corr: [6, 324, 50, 90]. 324: neighbors. 
            # radius = 4 -> neighbor points = (4*2+1)^2 = 81. Upsize x4 -> 324.
            corr = self.corr_fn(coords1)  # index correlation volume
            
            flow = coords1 - coords0
            with autocast(enabled=self.args.mixed_precision):
                # net_feat: hidden features of ConvGRU. 
                # inp_feat: input  features to ConvGRU.
                # up_mask is scaled to 0.25 of original values.
                # update_block: GMAUpdateBlock
                # In the first few iterations, delta_flow.abs().max() could be 1.3 or 0.8. Later it becomes 0.2~0.3.
                net_feat, up_mask, delta_flow = self.update_block(net_feat, inp_feat, corr, flow, attention)

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

        if test_mode:
            return coords1 - coords0, flow_up

        return flow_predictions
