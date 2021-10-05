import sys
sys.path.append('core')

from PIL import Image
import argparse
import os
import numpy as np
import torch
import imageio

from network import CRAFT
import network
# back-compatible with older checkpoints.
network.RAFTER = CRAFT
from torchvision import transforms

import datasets
from utils import flow_viz
from utils import frame_utils

from utils.utils import InputPadder, forward_interpolate

# Just an empty Logger definition to satisfy torch.load().
class Logger:
    def __init__(self):
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}
        
@torch.no_grad()
def create_sintel_submission(model, warm_start=False, output_path='sintel_submission', test_mode=1):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=32, flow_init=flow_prev, test_mode=test_mode)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence

    print("Created sintel submission.")

@torch.no_grad()
def create_sintel_submission_vis(model, warm_start=False, output_path='sintel_submission', test_mode=1):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split='test', aug_params=None, dstype=dstype)

        flow_prev, sequence_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (sequence, frame) = test_dataset[test_id]
            if sequence != sequence_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=32, flow_init=flow_prev, test_mode=test_mode)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            # Visualizations
            flow_img = flow_viz.flow_to_image(flow)
            image = Image.fromarray(flow_img)
            if not os.path.exists(f'vis_test/RAFT/{dstype}/'):
                os.makedirs(f'vis_test/RAFT/{dstype}/flow')

            if not os.path.exists(f'vis_test/ours/{dstype}/'):
                os.makedirs(f'vis_test/ours/{dstype}/flow')

            if not os.path.exists(f'vis_test/gt/{dstype}/'):
                os.makedirs(f'vis_test/gt/{dstype}/image')

            # image.save(f'vis_test/ours/{dstype}/flow/{test_id}.png')
            image.save(f'vis_test/RAFT/{dstype}/flow/{test_id}.png')
            imageio.imwrite(f'vis_test/gt/{dstype}/image/{test_id}.png', image1[0].cpu().permute(1, 2, 0).numpy())
            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, sequence)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            sequence_prev = sequence

    print("Created sintel submission.")

@torch.no_grad()
def create_kitti_submission(model, output_path='kitti_submission', test_mode=1):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=test_mode)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

    print("Created KITTI submission.")

@torch.no_grad()
def create_kitti_submission_vis(model, output_path='kitti_submission', test_mode=1):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None)

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=test_mode)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        # Visualizations
        flow_img = flow_viz.flow_to_image(flow)
        image = Image.fromarray(flow_img)
        if not os.path.exists(f'vis_kitti'):
            os.makedirs(f'vis_kitti/flow')
            os.makedirs(f'vis_kitti/image')

        image.save(f'vis_kitti/flow/{test_id}.png')
        imageio.imwrite(f'vis_kitti/image/{test_id}_0.png', image1[0].cpu().permute(1, 2, 0).numpy())
        imageio.imwrite(f'vis_kitti/image/{test_id}_1.png', image2[0].cpu().permute(1, 2, 0).numpy())

    print("Created KITTI submission.")

@torch.no_grad()
def validate_chairs(model, iters=6, test_mode=1):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    val_dataset = datasets.FlyingChairs(split='validation')
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        _, flow_pr = model(image1, image2, iters=iters, test_mode=test_mode)
        epe = torch.sum((flow_pr[0].cpu() - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs_epe': epe}


@torch.no_grad()
def validate_things(model, iters=6, test_mode=1):
    """ Perform evaluation on the FlyingThings (test) split """
    model.eval()
    results = {}

    for dstype in ['frames_cleanpass', 'frames_finalpass']:
        epe_list = {}
        val_dataset = datasets.FlyingThings3D(dstype=dstype, split='validation')
        print(f'Dataset length {len(val_dataset)}')
        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_prs = model(image1, image2, iters=iters, test_mode=test_mode)
            # if test_mode == 2: list of 12 tensors, each is [1, 2, 544, 960].
            # if test_mode == 1: a tensor of [1, 2, 544, 960].
            if test_mode == 1:
                flow_prs = [ flow_prs ]
            
            for it, flow_pr in enumerate(flow_prs):
                flow = padder.unpad(flow_pr[0]).cpu()
                epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
                epe_list.setdefault(it, [])
                epe_list[it].append(epe.view(-1).numpy())

        for it in range(iters):
            epe_all = np.concatenate(epe_list[it])

            epe = np.mean(epe_all)
            px1 = np.mean(epe_all < 1)
            px3 = np.mean(epe_all < 3)
            px5 = np.mean(epe_all < 5)

            print("Iter %d, Valid (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (it, dstype, epe, px1, px3, px5))
            
        results[dstype] = epe

    return results


@torch.no_grad()
def validate_sintel(model, iters=6, test_mode=1):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    
    for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype)
        epe_list = {}

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_prs = model(image1, image2, iters=iters, test_mode=test_mode)
            # if test_mode == 2: list of 12 tensors, each is [1, 2, H, W].
            # if test_mode == 1: a tensor of [1, 2, H, W].
            if test_mode == 1:
                flow_prs = [ flow_prs ]

            for it, flow_pr in enumerate(flow_prs):
                flow = padder.unpad(flow_pr[0]).cpu()
                epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
                epe_list.setdefault(it, [])
                epe_list[it].append(epe.view(-1).numpy())
                                            
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

        for it in range(iters):
            epe_all = np.concatenate(epe_list[it])
            
            epe = np.mean(epe_all)
            px1 = np.mean(epe_all<1)
            px3 = np.mean(epe_all<3)
            px5 = np.mean(epe_all<5)

            print("Iter %d, Valid (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (it, dstype, epe, px1, px3, px5))

        results[dstype] = epe

    return results

@torch.no_grad()
def validate_hd1k(model, iters=6, test_mode=1):
    """ Peform validation using the HD1k data """
    model.eval()
    results = {}
    val_dataset = datasets.HD1K()
    epe_list = []
    
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _ = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        halfsize = transforms.Resize((360, 853))
        image1   = halfsize(image1)
        image2   = halfsize(image2)
        flow_gt  = halfsize(flow_gt)
            
        padder = InputPadder(image1.shape)
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=test_mode)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        epe_list.append(epe.view(-1).numpy())

        if val_id % 100 == 0:
            print(f"{val_id}/{len(val_dataset)}")
            epe_all = np.concatenate(epe_list)
            epe = np.mean(epe_all)
            px1 = np.mean(epe_all<1)
            px3 = np.mean(epe_all<3)
            px5 = np.mean(epe_all<5)
            print("EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (epe, px1, px3, px5))

    print(f"{val_id}/{len(val_dataset)}")
    epe_all = np.concatenate(epe_list)
    epe = np.mean(epe_all)
    px1 = np.mean(epe_all<1)
    px3 = np.mean(epe_all<3)
    px5 = np.mean(epe_all<5)
    print("EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (epe, px1, px3, px5))
                    
    results = np.mean(epe_list)

    return results

@torch.no_grad()
def validate_sintel_occ(model, iters=6, test_mode=1):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['albedo', 'clean', 'final']:
    # for dstype in ['clean', 'final']:
        val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
        epe_list = []
        epe_occ_list = []
        epe_noc_list = []

        for val_id in range(len(val_dataset)):
            image1, image2, flow_gt, _, occ, _ = val_dataset[val_id]
            image1 = image1[None].cuda()
            image2 = image2[None].cuda()

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_pr = model(image1, image2, iters=iters, test_mode=test_mode)
            flow = padder.unpad(flow_pr[0]).cpu()

            epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
            epe_list.append(epe.view(-1).numpy())

            epe_noc_list.append(epe[~occ].numpy())
            epe_occ_list.append(epe[occ].numpy())

        epe_all = np.concatenate(epe_list)

        epe_noc = np.concatenate(epe_noc_list)
        epe_occ = np.concatenate(epe_occ_list)

        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        epe_occ_mean = np.mean(epe_occ)
        epe_noc_mean = np.mean(epe_noc)

        print("Validation (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (dstype, epe, px1, px3, px5))
        print("Occ epe: %f, Noc epe: %f" % (epe_occ_mean, epe_noc_mean))
        results[dstype] = np.mean(epe_list)

    return results


@torch.no_grad()
def separate_inout_sintel_occ():
    """ Peform validation using the Sintel (train) split """
    dstype = 'clean'
    val_dataset = datasets.MpiSintel(split='training', dstype=dstype, occlusion=True)
    # coords = torch.meshgrid(torch.arange(ht), torch.arange(wd))
    # coords = torch.stack(coords[::-1], dim=0).float()
    # return coords[None].expand(batch, -1, -1, -1)

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, occ, occ_path = val_dataset[val_id]
        _, h, w = image1.size()
        coords = torch.meshgrid(torch.arange(h), torch.arange(w))
        coords = torch.stack(coords[::-1], dim=0).float()

        coords_img_2 = coords + flow_gt
        out_of_frame = (coords_img_2[0] < 0) | (coords_img_2[0] > w) | (coords_img_2[1] < 0) | (coords_img_2[1] > h)
        occ_union = out_of_frame | occ
        in_frame = occ_union ^ out_of_frame

        # Generate union of occlusions and out of frame
        # path_list = occ_path.split('/')
        # path_list[-3] = 'occ_plus_out'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, occ_union.int().numpy() * 255)

        # Generate out-of-frame
        # path_list = occ_path.split('/')
        # path_list[-3] = 'out_of_frame'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, out_of_frame.int().numpy() * 255)

        # # Generate in-frame occlusions
        # path_list = occ_path.split('/')
        # path_list[-3] = 'in_frame_occ'
        # dir_path = os.path.join('/', *path_list[:-1])
        # img_path = os.path.join('/', *path_list)
        # if not os.path.exists(dir_path):
        #     os.makedirs(dir_path)
        #
        # imageio.imwrite(img_path, in_frame.int().numpy() * 255)



@torch.no_grad()
def validate_kitti(model, iters=6, test_mode=1):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    val_dataset = datasets.KITTI(split='training')

    out_list, epe_list = [], []
    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, valid_gt = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=test_mode)
        flow = padder.unpad(flow_pr[0]).cpu()

        epe = torch.sum((flow - flow_gt)**2, dim=0).sqrt()
        mag = torch.sum(flow_gt**2, dim=0).sqrt()

        epe = epe.view(-1)
        mag = mag.view(-1)
        val = valid_gt.view(-1) >= 0.5

        out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
        epe_list.append(epe[val].mean().item())
        out_list.append(out[val].cpu().numpy())

    epe_list = np.array(epe_list)
    out_list = np.concatenate(out_list)

    epe = np.mean(epe_list)
    f1 = 100 * np.mean(out_list)

    print("Validation KITTI: %f, %f" % (epe, f1))
    return {'kitti_epe': epe, 'kitti_f1': f1}

def save_checkpoint(cp_path, model, optimizer_state, lr_scheduler_state, logger):
    save_state = { 'model':        model.state_dict(),
                   'optimizer':    optimizer_state,
                   'lr_scheduler': lr_scheduler_state,
                   'logger':       logger.__dict__
                 }

    torch.save(save_state, cp_path)
    print(f"{cp_path} saved")

def fix_checkpoint(args, model):
    checkpoint = torch.load(args.model, map_location='cuda')

    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    else:
        # Load old checkpoint.
        msg = model.load_state_dict(checkpoint, strict=False)

    print(f"Model checkpoint loaded from {args.model}: {msg}.")

    logger = Logger()
    if 'logger' in checkpoint:
        if type(checkpoint['logger']) == dict:
            logger_dict = checkpoint['logger']
        else:
            logger_dict = checkpoint['logger'].__dict__
        
        # The scheduler will be updated by checkpoint['lr_scheduler'], no need to update here.
        for key in ('args', 'scheduler', 'model'):
            if key in logger_dict:
                del logger_dict[key]
        logger.__dict__.update(logger_dict)
        print("Logger loaded.")
    else:
        print("Logger NOT loaded.")
    
    optimizer_state     = checkpoint['optimizer']    if 'optimizer'    in checkpoint else None
    lr_scheduler_state  = checkpoint['lr_scheduler'] if 'lr_scheduler' in checkpoint else None
    
    save_checkpoint(args.model + "2", model, optimizer_state, lr_scheduler_state, logger)
    
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
    parser.add_argument('--fix', action='store_true', help='Fix loaded checkpoint')
    parser.add_argument('--submit', action='store_true', help='Make a sintel/kitti submission')
    parser.add_argument('--test_mode', default=1, type=int, 
                        help='Test mode (1: normal, 2: evaluate performance of every iteration)')

    # Ablations
    parser.add_argument('--replace', default=False, action='store_true',
                        help='Replace local motion feature with aggregated motion features')
    parser.add_argument('--no_alpha', default=False, action='store_true',
                        help='Remove learned alpha, set it to 1')
    parser.add_argument('--no_residual', default=False, action='store_true',
                        help='Remove residual connection. Do not add local features with the aggregated features.')
    parser.add_argument('--radius', dest='corr_radius', type=int, default=4)    

    parser.add_argument('--corrnorm', dest='corr_norm_type', type=str, 
                        choices=['none', 'local', 'global'], default='none')
    parser.add_argument('--posr', dest='pos_bias_radius', type=int, default=7, 
                        help='The radius of positional biases')
                        
    parser.add_argument('--f2trans', dest='f2trans', action='store_true', 
                        help='Use transformer on frame 2 features')
    parser.add_argument('--f2posw', dest='f2_pos_code_weight', type=float, default=0.5)
                            
    parser.add_argument('--setrans', dest='setrans', action='store_true', 
                        help='use setrans (Squeeze-Expansion Transformer)')
    parser.add_argument('--intermodes', dest='inter_num_modes', type=int, default=4, 
                        help='Number of modes in inter-frame attention')
    parser.add_argument('--intramodes', dest='intra_num_modes', type=int, default=4, 
                        help='Number of modes in intra-frame attention')
    parser.add_argument('--craft', dest='craft', action='store_true', 
                        help='use craft (Cross-Attentional Flow Transformer)')
    # In inter-frame attention, having QK biases performs slightly better.
    parser.add_argument('--interqknobias', dest='inter_qk_have_bias', action='store_false', 
                        help='Do not use biases in the QK projections in the inter-frame attention')
                        
    parser.add_argument('--interpos', dest='inter_pos_code_type', type=str, 
                        choices=['lsinu', 'bias'], default='bias')
    parser.add_argument('--interposw', dest='inter_pos_code_weight', type=float, default=0.5)
    parser.add_argument('--perturbinterposw', dest='perturb_inter_posw_range', type=float, default=0.,
                        help='The range of added random noise to pos_embed_weight during training')
    parser.add_argument('--intrapos', dest='intra_pos_code_type', type=str, 
                        choices=['lsinu', 'bias'], default='bias')
    parser.add_argument('--intraposw', dest='intra_pos_code_weight', type=float, default=1.0)
    parser.add_argument('--perturbintraposw', dest='perturb_intra_posw_range', type=float, default=0.,
                        help='The range of added random noise to pos_embed_weight during training')
    
    args = parser.parse_args()

    print("Args:\n{}".format(args))
    
    if args.dataset == 'separate':
        separate_inout_sintel_occ()
        sys.exit()

    model = torch.nn.DataParallel(CRAFT(args))
    
    if args.fix:
        fix_checkpoint(args, model)
        exit()
        
    checkpoint = torch.load(args.model)
    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    else:
        # Load old checkpoint.
        msg = model.load_state_dict(checkpoint, strict=False)
    
    print(f"Model checkpoint loaded from {args.model}: {msg}.")
        
    model.cuda()
    model.eval()

    if args.dataset == 'sintel' and args.submit:
        create_sintel_submission(model, warm_start=True)
        exit(0)
        # create_sintel_submission_vis(model, warm_start=True)
        
    if args.dataset == 'kitti' and args.submit:
        create_kitti_submission(model)
        exit(0)
    # create_kitti_submission_vis(model)

    with torch.no_grad():
        if args.dataset == 'chairs':
            validate_chairs(model.module, iters=args.iters, test_mode=args.test_mode)

        elif args.dataset == 'things':
            validate_things(model.module, iters=args.iters, test_mode=args.test_mode)

        elif args.dataset == 'sintel':
            validate_sintel(model.module, iters=args.iters, test_mode=args.test_mode)

        elif args.dataset == 'sintel_occ':
            validate_sintel_occ(model.module, iters=args.iters, test_mode=args.test_mode)

        elif args.dataset == 'kitti':
            validate_kitti(model.module, iters=args.iters, test_mode=args.test_mode)

        elif args.dataset == 'hd1k':
            validate_hd1k(model.module, iters=args.iters, test_mode=args.test_mode)
