import math
import numpy as np
import copy
import random

import torch
import torch.nn as nn
from torch.nn import Parameter
import torch.nn.functional as F
from torch import einsum
from einops import rearrange
from setrans_ablation import RandPosEmbedder, SinuPosEmbedder, ZeroEmbedder, MultiHeadFeatTrans
torch.set_printoptions(sci_mode=False)

bb2_stage_dims = {  'raft-small':   [32, 32,  64,  96,   128],   
                    'raft-basic':   [64, 64,  96,  128,  256],   
                    'resnet34':     [64, 64,  128, 256,  512],
                    'resnet50':     [64, 256, 512, 1024, 2048],
                    'resnet101':    [64, 256, 512, 1024, 2048],
                    'resibn101':    [64, 256, 512, 1024, 2048],   # resibn: resnet + IBN layers
                    'eff-b0':       [16, 24,  40,  112,  1280],   # input: 224
                    'eff-b1':       [16, 24,  40,  112,  1280],   # input: 240
                    'eff-b2':       [16, 24,  48,  120,  1408],   # input: 260
                    'eff-b3':       [24, 32,  48,  136,  1536],   # input: 300
                    'eff-b4':       [24, 32,  56,  160,  1792],   # input: 380
                    'i3d':          [64, 192, 480, 832,  1024]    # input: 224
                 }

# Can also be implemented using torch.meshgrid().
def gen_all_indices(shape, device):
    indices = torch.arange(shape.numel(), device=device).view(shape)

    out = []
    for dim_size in reversed(shape):
        out.append(indices % dim_size)
        indices = torch.div(indices, dim_size, rounding_mode='trunc')
    return torch.stack(tuple(reversed(out)), len(shape))


class SETransConfig(object):
    def __init__(self):
        self.feat_dim       = -1
        self.in_feat_dim    = -1
        # self.backbone_type  = 'eff-b4'          # resnet50, resnet101, resibn101, eff-b1~b4
        # self.bb_stage_idx   = 4                 # Last stage of the five stages. Index starts from 0.
        # self.set_backbone_type(self.backbone_type)
        # self.use_pretrained = True        

        # Positional encoding settings.
        self.pos_dim            = 2
        self.pos_embed_weight   = 1
        # If perturb_pew_range > 0, add random noise to pos_embed_weight during training.
        # perturb_pew_range: the scale of the added random noise (relative to pos_embed_weight)
        self.perturb_pew_range  = 0
        
        # Architecture settings
        # Number of modes in the expansion attention block.
        # When doing ablation study of multi-head, num_modes means num_heads, 
        # to avoid introducing extra config parameters.
        self.num_modes = 4
        self.tie_qk_scheme = 'shared'           # shared, loose, or none.
        self.mid_type      = 'shared'           # shared, private, or none.
        self.trans_output_type  = 'private'     # shared or private.
        self.act_fun = F.gelu
        
        self.attn_clip = 100
        self.attn_diag_cycles = 800
        self.base_initializer_range = 0.02
        
        self.qk_have_bias = False
        # Without the bias term, V projection often performs better.
        self.v_has_bias = False
        # Add an identity matrix (*0.02*query_idbias_scale) to query/key weights
        # to make a bias towards identity mapping.
        # Set to 0 to disable the identity bias.
        self.query_idbias_scale = 10
        self.feattrans_lin1_idbias_scale = 10

        # Pooling settings
        self.pool_modes_feat  = 'softmax'   # softmax, max, mean, or none. With [] means keepdim=True.

        # Randomness settings
        self.hidden_dropout_prob = 0.1
        self.attention_probs_dropout_prob = 0.2
        self.pos_embed_type  = 'lsinu'
        self.ablate_multihead       = False
        self.out_attn_probs_only    = False
        # When out_attn_scores_only, dropout is not applied to attention scores.
        self.out_attn_scores_only   = False
        
    def set_backbone_type(self, args):
        if self.try_assign(args, 'backbone_type'):
            self.bb_stage_dims  = bb2_stage_dims[self.backbone_type]
            self.in_feat_dim    = self.bb_stage_dims[-1]
    
    # return True if any parameter is successfully set, and False if none is set.
    def try_assign(self, args, *keys):
        is_successful = False
        
        for key in keys:
            if key in args:
                if isinstance(args, dict):
                    self.__dict__[key] = args[key]
                else:
                    self.__dict__[key] = args.__dict__[key]
                is_successful = True
                
        return is_successful

    def update_config(self, args):
        self.set_backbone_type(args)
        self.try_assign(args, 'use_pretrained', 'apply_attn_stage', 'num_modes', 
                              'trans_output_type', 'mid_type', 'base_initializer_range', 
                              'pos_embed_type', 'ablate_multihead', 'attn_clip', 'attn_diag_cycles', 
                              'tie_qk_scheme', 'feattrans_lin1_idbias_scale', 'qk_have_bias', 'v_has_bias',
                              # out_attn_probs_only/out_attn_scores_only are only True for the optical flow correlation block.
                              'out_attn_probs_only', 'out_attn_scores_only',
                              'in_feat_dim', 'perturb_pew_range', 'pos_bias_radius')
        
        if self.try_assign(args, 'out_feat_dim'):
            self.feat_dim   = self.out_feat_dim
        else:
            self.feat_dim   = self.in_feat_dim
            
        if 'dropout_prob' in args and args.dropout_prob >= 0:
            self.hidden_dropout_prob          = args.dropout_prob
            self.attention_probs_dropout_prob = args.dropout_prob
            print("Dropout prob: %.2f" %(args.dropout_prob))
            
CONFIG = SETransConfig()


# =================================== SETrans Initialization ====================================#
class SETransInitWeights(nn.Module):
    """ An abstract class to handle weights initialization """
    def __init__(self, config, *inputs, **kwargs):
        super(SETransInitWeights, self).__init__()
        self.config = config

    def init_weights(self, module):
        """ Initialize the weights.
            type(module.weight)      # <class 'torch.nn.parameter.Parameter'>
            type(module.weight.data) # <class 'torch.Tensor'>
        """
        if isinstance(module, (nn.Linear, nn.Embedding)):
            base_initializer_range  = self.config.base_initializer_range
            module.weight.data.normal_(mean=0.0, std=base_initializer_range)
            # Slightly different from the TF version which uses truncated_normal
            # for initialization cf https://github.com/pytorch/pytorch/pull/5617
        if isinstance(module, nn.Linear) and module.bias is not None:
            module.bias.data.zero_()

def tie_qk(module):
    if isinstance(module, CrossAttFeatTrans) and module.tie_qk_scheme != 'none':
        module.tie_qk()

def add_identity_bias(module):
    if isinstance(module, CrossAttFeatTrans) or isinstance(module, ExpandedFeatTrans):
        module.add_identity_bias()
                        
#====================================== SETrans Shared Modules ========================================#

class MMPrivateMid(nn.Module):
    def __init__(self, config):
        super(MMPrivateMid, self).__init__()
        # Use 1x1 convolution as a group linear layer.
        # Equivalent to each group going through a respective nn.Linear().
        self.num_modes      = config.num_modes
        self.feat_dim       = config.feat_dim
        feat_dim_allmode    = self.feat_dim * self.num_modes
        self.group_linear   = nn.Conv1d(feat_dim_allmode, feat_dim_allmode, 1, groups=self.num_modes)
        self.mid_act_fn     = config.act_fun
        # This dropout is not presented in huggingface transformers.
        # Added to conform with lucidrains and rwightman's implementations.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, x):
        x_trans = self.group_linear(x)      # [B0, 1792*4, U] -> [B0, 1792*4, U]
        x_act   = self.mid_act_fn(x_trans)  # [B0, 1792*4, U]
        x_drop  = self.dropout(x_act)
        return x_drop

class MMSharedMid(nn.Module):
    def __init__(self, config):
        super(MMSharedMid, self).__init__()
        self.num_modes      = config.num_modes
        self.feat_dim       = config.feat_dim
        feat_dim_allmode    = self.feat_dim * self.num_modes
        self.shared_linear  = nn.Linear(self.feat_dim, self.feat_dim)
        self.mid_act_fn     = config.act_fun
        # This dropout is not presented in huggingface transformers.
        # Added to conform with lucidrains and rwightman's implementations.
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    # x: [B0, 1792*4, U] or [B0, 4, U, 1792]
    def forward(self, x):
        if len(x.shape) == 3:
            # shape_4d: [B0, 4, 1792, U].
            shape_4d    = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
            # x_4d: [B0, 4, U, 1792].
            x_4d        = x.view(shape_4d).permute([0, 1, 3, 2])
            reshaped    = True
        else:
            x_4d        = x
            reshaped    = False

        x_trans         = self.shared_linear(x_4d)
        x_act           = self.mid_act_fn(x_trans)
        x_drop          = self.dropout(x_act)
        
        if reshaped:
            # restore the original shape
            x_drop      = x_drop.permute([0, 1, 3, 2]).reshape(x.shape)

        return x_drop

# MMPrivateOutput/MMSharedOutput <- MMandedFeatTrans <- CrossAttFeatTrans
# MM***Output has a shortcut (residual) connection.
class MMPrivateOutput(nn.Module):
    def __init__(self, config):
        super(MMPrivateOutput, self).__init__()
        self.num_modes  = config.num_modes
        self.feat_dim   = config.feat_dim
        feat_dim_allmode = self.feat_dim * self.num_modes
        self.group_linear = nn.Conv1d(feat_dim_allmode, feat_dim_allmode, 1, groups=self.num_modes)
        self.resout_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # x, shortcut: [B0, 1792*4, U]
    def forward(self, x, shortcut):
        x        = self.group_linear(x)
        # x_comb: [B0, 1792*4, U]. Residual connection.
        x_comb   = x + shortcut
        shape_4d = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
        # x_comb_4d, x_drop_4d: [B0, 4, U, 1792].
        x_comb_4d = x.view(shape_4d).permute([0, 1, 3, 2])
        x_drop_4d = self.dropout(x_comb_4d)
        x_normed  = self.resout_norm_layer(x_drop_4d)
        return x_normed

# MMPrivateOutput/MMSharedOutput <- MMandedFeatTrans <- CrossAttFeatTrans
# MM***Output has a shortcut (residual) connection.
class MMSharedOutput(nn.Module):
    # feat_dim_allmode is not used. Just to keep the ctor arguments the same as MMPrivateOutput.
    def __init__(self, config):
        super(MMSharedOutput, self).__init__()
        self.num_modes = config.num_modes
        self.feat_dim  = config.feat_dim
        self.shared_linear = nn.Linear(self.feat_dim, self.feat_dim)
        self.resout_norm_layer = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=True)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    # x, shortcut: [B0, 1792*4, U] or [B0, 4, U, 1792]
    def forward(self, x, shortcut):
        # shape_4d: [B0, 4, 1792, U].
        shape_4d    = ( x.shape[0], self.num_modes, self.feat_dim, x.shape[2] )
        if len(x.shape) == 3:
            x_4d    = x.view(shape_4d).permute([0, 1, 3, 2])
        else:
            x_4d    = x
        if len(shortcut.shape) == 3:
            shortcut_4d = shortcut.view(shape_4d).permute([0, 1, 3, 2])
        else:
            shortcut_4d = shortcut

        # x_4d, shortcut_4d: [B0, 4, U, 1792].
        x_trans     = self.shared_linear(x_4d)
        # x_4d, x_comb: [B0, 4, U, 1792]. Residual connection.
        x_comb      = x_trans + shortcut_4d
        x_drop      = self.dropout(x_comb)
        x_normed    = self.resout_norm_layer(x_drop)
        return x_normed

# group_dim: the tensor dimension that corresponds to the multiple groups.
class LearnedSoftAggregate(nn.Module):
    def __init__(self, num_feat, group_dim, keepdim=False):
        super(LearnedSoftAggregate, self).__init__()
        self.group_dim  = group_dim
        self.num_feat   = num_feat
        self.feat2score = nn.Linear(num_feat, 1)
        self.keepdim    = keepdim

    def forward(self, x, score_basis=None):
        # Assume the last dim of x is the feature dim.
        if score_basis is None:
            score_basis = x
        
        if self.num_feat == 1:
            mode_scores = self.feat2score(score_basis.unsqueeze(-1)).squeeze(-1)
        else:
            mode_scores = self.feat2score(score_basis)
        attn_probs  = mode_scores.softmax(dim=self.group_dim)
        x_aggr      = (x * attn_probs).sum(dim=self.group_dim, keepdim=self.keepdim)
        return x_aggr

# ExpandedFeatTrans <- CrossAttFeatTrans.
# ExpandedFeatTrans has a residual connection.
class ExpandedFeatTrans(nn.Module):
    def __init__(self, config, name):
        super(ExpandedFeatTrans, self).__init__()
        self.config = config
        self.name = name
        self.in_feat_dim = config.in_feat_dim
        self.feat_dim = config.feat_dim
        self.num_modes = config.num_modes
        self.feat_dim_allmode = self.feat_dim * self.num_modes
        # first_linear is the value projection in other transformer implementations.
        # The output of first_linear will be divided into num_modes groups.
        # first_linear is always 'private' for each group, i.e.,
        # parameters are not shared (parameter sharing makes no sense).
        self.first_linear = nn.Linear(self.in_feat_dim, self.feat_dim_allmode, bias=config.v_has_bias)
            
        self.base_initializer_range = config.base_initializer_range
        self.has_FFN        = getattr(config, 'has_FFN', True)
        self.has_input_skip = getattr(config, 'has_input_skip', False)

        print("{}: v_has_bias: {}, has_FFN: {}, has_input_skip: {}".format(
              self.name, config.v_has_bias, self.has_FFN, self.has_input_skip))
              
        if config.pool_modes_feat[0] == '[':
            self.pool_modes_keepdim = True
            self.pool_modes_feat = config.pool_modes_feat[1:-1]     # remove '[' and ']'
        else:
            self.pool_modes_keepdim = False
            self.pool_modes_feat = config.pool_modes_feat

        if self.pool_modes_feat == 'softmax':
            agg_basis_feat_dim = self.feat_dim

            # group_dim = 1, i.e., features will be aggregated across the modes.
            self.feat_softaggr = LearnedSoftAggregate(agg_basis_feat_dim, group_dim=1,
                                                      keepdim=self.pool_modes_keepdim)

        if self.has_FFN:
            self.mid_type = config.mid_type
            if self.mid_type == 'shared':
                self.intermediate = MMSharedMid(self.config)
            elif self.mid_type == 'private':
                self.intermediate = MMPrivateMid(self.config)
            else:
                self.intermediate = config.act_fun

            if config.trans_output_type == 'shared':
                self.output = MMSharedOutput(config)
            elif config.trans_output_type == 'private':
                self.output = MMPrivateOutput(config)

        # Have to ensure U1 == U2.
        if self.has_input_skip:
            self.input_skip_coeff = Parameter(torch.ones(1))
            self.skip_layer_norm  = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)
            
    def add_identity_bias(self):
        if self.config.feattrans_lin1_idbias_scale > 0:
            # first_linear dimension is num_modes * feat_dim.
            # If in_feat_dim == feat_dim, only add identity bias to the first mode.
            # If in_feat_dim > feat_dim, expand to more modes until all in_feat_dim dimensions are covered.
            identity_weight = torch.diag(torch.ones(self.feat_dim)) * self.base_initializer_range \
                              * self.config.feattrans_lin1_idbias_scale
            # Only bias the weight of the first mode.
            # The total initial "weight mass" in each row is reduced by 1792*0.02*0.5.
            self.first_linear.weight.data[:self.feat_dim, :self.feat_dim] = \
                self.first_linear.weight.data[:self.feat_dim, :self.feat_dim] * 0.5 + identity_weight

    # input_feat: [3, 4416, 128]. attention_probs: [3, 4, 4416, 4416]. 
    def forward(self, input_feat, attention_probs):
        # input_feat: [B, U2, 1792], mm_first_feat: [B, Units, 1792*4]
        # B: batch size, U2: number of the 2nd group of units,
        # IF: in_feat_dim, could be different from feat_dim, due to layer compression
        # (different from squeezed attention).
        B, U2, IF = input_feat.shape
        U1 = attention_probs.shape[2]
        F = self.feat_dim
        M = self.num_modes
        mm_first_feat = self.first_linear(input_feat)
        # mm_first_feat after transpose: [B, 1792*4, U2]
        mm_first_feat = mm_first_feat.transpose(1, 2)

        # mm_first_feat_4d: [B, 4, U2, 1792]
        mm_first_feat_4d = mm_first_feat.view(B, M, F, U2).transpose(2, 3)

        # attention_probs: [B, Modes, U1, U2]
        # mm_first_feat_fusion: [B, 4, U1, 1792]
        mm_first_feat_fusion = torch.matmul(attention_probs, mm_first_feat_4d)
        mm_first_feat_fusion_3d = mm_first_feat_fusion.transpose(2, 3).reshape(B, M*F, U1)
        mm_first_feat = mm_first_feat_fusion_3d

        if self.has_FFN:
            # mm_mid_feat:   [B, 1792*4, U1]. Group linear & gelu of mm_first_feat.
            mm_mid_feat   = self.intermediate(mm_first_feat)
            # mm_last_feat:  [B, 4, U1, 1792]. Group/shared linear & residual & Layernorm
            mm_last_feat = self.output(mm_mid_feat, mm_first_feat)
            mm_trans_feat = mm_last_feat
        else:
            mm_trans_feat = mm_first_feat_fusion
            
        if self.pool_modes_feat == 'softmax':
            trans_feat = self.feat_softaggr(mm_trans_feat)
        elif self.pool_modes_feat == 'max':
            trans_feat = mm_trans_feat.max(dim=1)[0]
        elif self.pool_modes_feat == 'mean':
            trans_feat = mm_trans_feat.mean(dim=1)
        elif self.pool_modes_feat == 'none':
            trans_feat = mm_trans_feat

        # Have to ensure U1 == U2.
        if self.has_input_skip:
            trans_feat = trans_feat + self.input_skip_coeff * input_feat
            trans_feat = self.skip_layer_norm(trans_feat)
            
        # trans_feat: [B, U1, 1792]
        return trans_feat

class CrossAttFeatTrans(SETransInitWeights):
    def __init__(self, config, name):
        super(CrossAttFeatTrans, self).__init__(config)
        self.config     = config
        self.name       = name
        self.num_modes  = config.num_modes
        self.in_feat_dim    = config.in_feat_dim
        self.feat_dim       = config.feat_dim
        self.attention_mode_dim = self.in_feat_dim // self.num_modes   # 448
        # att_size_allmode: 512 * modes
        self.att_size_allmode = self.num_modes * self.attention_mode_dim
        self.query = nn.Linear(self.in_feat_dim, self.att_size_allmode, bias=config.qk_have_bias)
        self.key   = nn.Linear(self.in_feat_dim, self.att_size_allmode, bias=config.qk_have_bias)
        self.base_initializer_range = config.base_initializer_range

        self.out_attn_scores_only   = config.out_attn_scores_only
        self.out_attn_probs_only    = config.out_attn_probs_only
        self.ablate_multihead   = config.ablate_multihead

        if self.out_attn_scores_only or self.out_attn_probs_only:
            self.out_trans  = None
            if self.num_modes > 1:
                # Each attention value is a scalar. So num_feat = 1.
                self.attn_softaggr = LearnedSoftAggregate(1, group_dim=1, keepdim=True)
            
        elif self.ablate_multihead:
            self.out_trans  = MultiHeadFeatTrans(config, name + "-out_trans")
        else:
            self.out_trans  = ExpandedFeatTrans(config,  name + "-out_trans")

        self.att_dropout = nn.Dropout(config.attention_probs_dropout_prob)

        self.tie_qk_scheme    = config.tie_qk_scheme
        print("{}: in_feat_dim: {}, feat_dim: {}, modes: {}, qk_have_bias: {}".format(
              self.name, self.in_feat_dim, self.feat_dim, self.num_modes, config.qk_have_bias))

        # if using ShiftedPosBias, then add positional embeddings here.
        if config.pos_embed_type == 'bias':
            self.pos_biases_weight = config.pos_embed_weight
            # args.perturb_pb_range is the relative ratio. Get the absolute range here.
            self.perturb_pb_range  = self.pos_biases_weight * config.perturb_pew_range
            print("Positional biases weight perturbation: {:.3}".format(self.perturb_pb_range))
        else:
            self.pos_biases_weight = 1
                          
        self.attn_clip    = config.attn_clip
        if 'attn_diag_cycles' in config.__dict__:
            self.attn_diag_cycles   = config.attn_diag_cycles
        else:
            self.attn_diag_cycles   = 500
        self.max_attn    = 0
        self.clamp_count = 0
        self.call_count  = 0
        self.apply(self.init_weights)
        self.apply(tie_qk)
        # tie_qk() has to be executed after weight initialization.
        self.apply(add_identity_bias)
        
    # if tie_qk_scheme is not None, it overrides the initialized self.tie_qk_scheme
    def tie_qk(self, tie_qk_scheme=None):
        # override config.tie_qk_scheme
        if tie_qk_scheme is not None:
            self.tie_qk_scheme = tie_qk_scheme

        if self.tie_qk_scheme == 'shared':
            self.key.weight = self.query.weight
            if self.key.bias is not None:
                self.key.bias = self.query.bias

        elif self.tie_qk_scheme == 'loose':
            self.key.weight.data.copy_(self.query.weight)
            if self.key.bias is not None:
                self.key.bias.data.copy_(self.query.bias)

    def add_identity_bias(self):
        identity_weight = torch.diag(torch.ones(self.attention_mode_dim)) * self.base_initializer_range \
                          * self.config.query_idbias_scale
        repeat_count = self.in_feat_dim // self.attention_mode_dim
        identity_weight = identity_weight.repeat([1, repeat_count])
        # only bias the weight of the first mode
        # The total initial "weight mass" in each row is reduced by 1792*0.02*0.5.
        self.key.weight.data[:self.attention_mode_dim] = \
            self.key.weight.data[:self.attention_mode_dim] * 0.5 + identity_weight

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_modes, -1)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    # pos_biases: [1, 1, U1, U2].
    # attention_mask is seldom used. Abandon it.
    def forward(self, query_feat, key_feat=None, pos_biases=None):
        # query_feat: [B, U1, 1792]
        # if key_feat == None: self attention.
        if key_feat is None:
            key_feat = query_feat
        # mixed_query_layer, mixed_key_layer: [B, U1, 1792], [B, U2, 1792]
        mixed_query_layer = self.query(query_feat)
        mixed_key_layer   = self.key(key_feat)
        # query_layer, key_layer: [B, 4, U1, 448], [B, 4, U2, 448]
        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer   = self.transpose_for_scores(mixed_key_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2)) # [B0, 4, U1, 448] [B0, 4, 448, U2]
        attention_scores = attention_scores / math.sqrt(self.attention_mode_dim)  # [B0, 4, U1, U2]

        with torch.no_grad():
            curr_max_attn = attention_scores.max().item()
            pos_count     = (attention_scores > 0).sum()
            curr_avg_attn = attention_scores.sum() / pos_count
            curr_avg_attn = curr_avg_attn.item()

        if curr_max_attn > self.max_attn:
            self.max_attn = curr_max_attn

        if curr_max_attn > self.attn_clip:
            attention_scores = torch.clamp(attention_scores, -self.attn_clip, self.attn_clip)
            self.clamp_count += 1
        
        if self.training:
            self.call_count += 1
            if self.call_count % self.attn_diag_cycles == 0:
                print("max-attn: {:.2f}, avg-attn: {:.2f}, clamp-count: {}".format(self.max_attn, curr_avg_attn, self.clamp_count))
                self.max_attn    = 0
                self.clamp_count = 0

        # Apply the positional biases
        if pos_biases is not None:
            if self.perturb_pb_range > 0 and self.training:
                pew_noise = random.uniform(-self.perturb_pb_range, 
                                            self.perturb_pb_range)
            else:
                pew_noise = 0
                        
            #[B0, 8, U1, U2] = [B0, 8, U1, U2]  + [1, 1, U1, U2].
            attention_scores = attention_scores + (self.pos_biases_weight + pew_noise) * pos_biases

        # Normalize the attention scores to probabilities.
        attention_probs = nn.functional.softmax(attention_scores, dim=-1)

        # lucidrains doesn't have this dropout but rwightman has. Will keep it.
        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.att_dropout(attention_probs)     #[B0, 4, U1, U2]

        if self.out_attn_probs_only:
            # [6, 4, 4500, 4500]
            return attention_probs

        # When out_attn_scores_only, dropout is not applied to attention scores.
        elif self.out_attn_scores_only:
            if self.num_modes > 1:
                if self.num_modes == 2:
                    attention_scores = attention_scores.mean(dim=1, keepdim=True)
                else:
                    # [3, num_modes=4, 4500, 4500] => [3, 1, 4500, 4500]
                    attention_scores = self.attn_softaggr(attention_scores)
            # attention_scores = self.att_dropout(attention_scores)
            return attention_scores

        else:
            # out_feat: [B0, U1, 1792], in the same size as query_feat.
            out_feat      = self.out_trans(key_feat, attention_probs)
            return out_feat

class SelfAttVisPosTrans(nn.Module):
    def __init__(self, config, name):
        nn.Module.__init__(self)
        self.config = config
        self.setrans = CrossAttFeatTrans(self.config, name)
        self.vispos_encoder = SETransInputFeatEncoder(self.config)
        self.out_attn_only = config.out_attn_scores_only or config.out_attn_probs_only
        
    def forward(self, x):
        coords = gen_all_indices(x.shape[2:], device=x.device)
        coords = coords.unsqueeze(0).repeat(x.shape[0], 1, 1, 1)
        
        x_vispos, pos_biases = self.vispos_encoder(x, coords)
        # if out_attn_scores_only/out_attn_probs_only, 
        # x_trans is an attention matrix in the shape of (query unit number, key unit number)
        # otherwise, output features are in the same shape as the query features.
        # key features are recombined to get new query features by matmul(attention_probs, V(key features))
        #             frame1 frame2
        # corr: [1, 1, 7040, 7040]
        # x_vispos: [B0, num_voxels, 256]
        x_trans = self.setrans(x_vispos, pos_biases=pos_biases)
        if not self.out_attn_only:
            x_trans = x_trans.permute(0, 2, 1).reshape(x.shape)
            
        return x_trans

# =================================== SETrans BackBone Components ==============================#

class LearnSinuPosEmbedder(nn.Module):
    def __init__(self, pos_dim, pos_embed_dim, omega=1, affine=True):
        super().__init__()
        self.pos_dim = pos_dim
        self.pos_embed_dim = pos_embed_dim
        self.pos_fc = nn.Linear(self.pos_dim, self.pos_embed_dim, bias=True)
        self.pos_mix_norm_layer = nn.LayerNorm(self.pos_embed_dim, eps=1e-12, elementwise_affine=affine)
        self.omega = omega
        print("Learnable Sinusoidal positional encoding")
        
    def forward(self, pos_normed):
        pos_embed_sum = 0
        pos_embed0 = self.pos_fc(pos_normed)
        pos_embed_sin = torch.sin(self.omega * pos_embed0[:, :, 0::2])
        pos_embed_cos = torch.cos(self.omega * pos_embed0[:, :, 1::2])
        # Interlace pos_embed_sin and pos_embed_cos.
        pos_embed_mix = torch.stack((pos_embed_sin, pos_embed_cos), dim=3).view(pos_embed0.shape)
        pos_embed_out = self.pos_mix_norm_layer(pos_embed_mix)

        return pos_embed_out

class ShiftedPosBias(nn.Module):
    def __init__(self, pos_dim=2, pos_bias_radius=8, max_pos_shape=(200, 200)):
        super().__init__()
        self.pos_dim = pos_dim
        self.R = R = pos_bias_radius
        # biases: [17, 17]
        pos_bias_shape = [ pos_bias_radius * 2 + 1 for i in range(pos_dim) ]
        self.biases = Parameter(torch.zeros(pos_bias_shape))
        # Currently only feature maps with a 2D spatial shape (i.e., 2D images) are supported.
        if self.pos_dim == 2:
            all_h1s, all_w1s, all_h2s, all_w2s = [], [], [], []
            for i in range(max_pos_shape[0]):
                i_h1s, i_w1s, i_h2s, i_w2s = [], [], [], []
                for j in range(max_pos_shape[1]):
                    h1s, w1s, h2s, w2s = torch.meshgrid(torch.tensor(i), torch.tensor(j), 
                                                        torch.arange(i, i+2*R+1), torch.arange(j, j+2*R+1))
                    i_h1s.append(h1s)
                    i_w1s.append(w1s)
                    i_h2s.append(h2s)
                    i_w2s.append(w2s)
                                                  
                i_h1s = torch.cat(i_h1s, dim=1)
                i_w1s = torch.cat(i_w1s, dim=1)
                i_h2s = torch.cat(i_h2s, dim=1)
                i_w2s = torch.cat(i_w2s, dim=1)
                all_h1s.append(i_h1s)
                all_w1s.append(i_w1s)
                all_h2s.append(i_h2s)
                all_w2s.append(i_w2s)
            
            all_h1s = torch.cat(all_h1s, dim=0)
            all_w1s = torch.cat(all_w1s, dim=0)
            all_h2s = torch.cat(all_h2s, dim=0)
            all_w2s = torch.cat(all_w2s, dim=0)
            
        self.all_h1s = all_h1s
        self.all_w1s = all_w1s
        self.all_h2s = all_h2s
        self.all_w2s = all_w2s
        
    def forward(self, feat):
        feat_shape = feat.shape
        R = self.R
        spatial_shape = feat_shape[-self.pos_dim:]
        padded_pos_shape  = list(spatial_shape) + [ 2*R + spatial_shape[i] for i in range(self.pos_dim) ]
        padded_pos_biases = torch.zeros(padded_pos_shape, device=feat.device)
        
        if self.pos_dim == 2:
            H, W = spatial_shape
            all_h1s = self.all_h1s[:H, :W]
            all_w1s = self.all_w1s[:H, :W]
            all_h2s = self.all_h2s[:H, :W]
            all_w2s = self.all_w2s[:H, :W]
            padded_pos_biases[(all_h1s, all_w1s, all_h2s, all_w2s)] = self.biases
                
        # Remove padding.
        pos_biases = padded_pos_biases[:, :, R:-R, R:-R]
        
        # [H, W, H, W] => [1, 1, H, W, H, W]
        for i in range(len(feat_shape) - self.pos_dim):
            pos_biases = pos_biases.unsqueeze(0)
            
        return pos_biases
        
class SETransInputFeatEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.feat_dim         = config.in_feat_dim  # 256
        self.pos_embed_dim    = self.feat_dim
        self.dropout          = nn.Dropout(config.hidden_dropout_prob)
        self.comb_norm_layer  = nn.LayerNorm(self.feat_dim, eps=1e-12, elementwise_affine=False)
        self.pos_embed_type   = config.pos_embed_type
        
        # if using ShiftedPosBias, do not add positional embeddings here.
        if config.pos_embed_type != 'bias':
            self.pos_embed_weight = config.pos_embed_weight
            # args.perturb_pew_range is the relative ratio. Get the absolute range here.
            self.perturb_pew_range  = self.pos_embed_weight * config.perturb_pew_range
            print("Positional embedding weight perturbation: {:.3}".format(self.perturb_pew_range))
        else:
            self.pos_embed_weight = 0
        
        # Box position encoding. no affine, but could have bias.
        # 2 channels => 1792 channels
        if config.pos_embed_type == 'lsinu':
            self.pos_embedder = LearnSinuPosEmbedder(config.pos_dim, self.pos_embed_dim, omega=1, affine=False)
        elif config.pos_embed_type == 'rand':
            self.pos_embedder = RandPosEmbedder(config.pos_dim, self.pos_embed_dim, shape=(36, 36), affine=False)
        elif config.pos_embed_type == 'sinu':
            self.pos_embedder = SinuPosEmbedder(config.pos_dim, self.pos_embed_dim, shape=(36, 36), affine=False)
        elif config.pos_embed_type == 'zero':
            self.pos_embedder = ZeroEmbedder(self.pos_embed_dim)
        elif config.pos_embed_type == 'bias':
            self.pos_embedder = ShiftedPosBias(config.pos_dim, config.pos_bias_radius)
            
    # return: [B0, num_voxels, 256]
    def forward(self, vis_feat, voxels_pos, get_pos_biases=True):
        # vis_feat:  [8, 256, 46, 62]
        batch, dim, ht, wd  = vis_feat.shape

        if self.pos_embed_type != 'bias':
            # voxels_pos: [8, 46, 62, 2]
            voxels_pos_normed = voxels_pos / voxels_pos.max()
            # voxels_pos_normed: [B0, num_voxels, 2]
            # pos_embed:         [B0, num_voxels, 256]
            voxels_pos_normed   = voxels_pos_normed.view(batch, ht * wd, -1)
            pos_embed   = self.pos_embedder(voxels_pos_normed)
            pos_biases  = None
        else:
            pos_embed   = 0
            # ShiftedPosBias() may be a bit slow. So only generate when necessary.
            if get_pos_biases:
                # pos_biases: [1, 1, H, W, H, W]
                pos_biases  = self.pos_embedder(vis_feat)
                # pos_biases: [1, 1, H*W, H*W]
                pos_biases  = pos_biases.reshape(1, 1, ht*wd, ht*wd)
                   
        vis_feat    = vis_feat.view(batch, dim, ht * wd).transpose(1, 2)
        
        if self.perturb_pew_range > 0 and self.training:
            pew_noise = random.uniform(-self.perturb_pew_range, 
                                        self.perturb_pew_range)
        else:
            pew_noise = 0
            
        feat_comb           = vis_feat + (self.pos_embed_weight + pew_noise) * pos_embed
            
        feat_normed         = self.comb_norm_layer(feat_comb)
        feat_normed         = self.dropout(feat_normed)
                  
        if get_pos_biases:
            return feat_normed, pos_biases
        else:
            return feat_normed
