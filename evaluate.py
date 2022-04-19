import sys
sys.path.append("core")

from PIL import Image
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio
import cv2

from network import CRAFT
from raft import RAFT
from craft_nogma import CRAFT_nogma
import network
# back-compatible with older checkpoints.
network.RAFTER = CRAFT
from torchvision import transforms
import torch.utils.data as data

import datasets
from utils import flow_viz
from utils import frame_utils

from utils.utils import InputPadder, forward_interpolate
from fvcore.nn import FlopCountAnalysis
# np.seterr(all='raise')

# Just an empty Logger definition to satisfy torch.load().
class Logger:
    def __init__(self):
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

# img and flow are 3D or 4D tensors.
# mask: a 2D tensor of H*W. 
# The flow values at mask==True are valid and will be used to compute EPE.
def shift_pixels(img, flow, xy_shift):
    if xy_shift is not None:
        x_shift, y_shift = xy_shift
        # the format of flow is u, v, i.e., x, y, not y, x.
        offset_tensor = torch.tensor([x_shift, y_shift], dtype=torch.float32, device=flow.device)
        if flow.ndim == 4:
            offset_tensor = offset_tensor.reshape([1, 2, 1, 1])
        else:
            offset_tensor = offset_tensor.reshape([2, 1, 1])

    if xy_shift is None or (x_shift == 0 and y_shift == 0):
        mask = torch.ones(img.shape[-2:], dtype=bool, device=img.device)
        return img, flow, mask

    img2 = torch.zeros_like(img)
    if flow is not None:
        flow2 = torch.zeros_like(flow)
    else:
        flow2 = None

    mask = torch.zeros(img.shape[-2:], dtype=bool, device=img.device)

    if x_shift > 0 and y_shift > 0:
        img2[..., y_shift:, x_shift:] = img[..., :-y_shift, :-x_shift]
        mask[y_shift:, x_shift:] = True
        if flow is not None:
            flow2[..., y_shift:, x_shift:] = flow[..., :-y_shift, :-x_shift]
    if x_shift > 0 and y_shift < 0:
        img2[..., :y_shift, x_shift:] = img[..., -y_shift:, :-x_shift]
        mask[:y_shift, x_shift:] = True
        if flow is not None:
            flow2[..., :y_shift, x_shift:] = flow[..., -y_shift:, :-x_shift]
    if x_shift < 0 and y_shift > 0:
        img2[..., y_shift:, :x_shift] = img[..., :-y_shift, -x_shift:]
        mask[y_shift:, :x_shift] = True
        if flow is not None:
            flow2[..., y_shift:, :x_shift] = flow[..., :-y_shift, -x_shift:]
    if x_shift < 0 and y_shift < 0:
        img2[..., :y_shift, :x_shift] = img[..., -y_shift:, -x_shift:]
        mask[:y_shift, :x_shift] = True
        if flow is not None:
            flow2[..., :y_shift, :x_shift] = flow[..., -y_shift:, -x_shift:]

    if flow2 is not None:
        flow2 -= offset_tensor
    return img2, flow2, mask

def shift_flow(flow, xy_shift):
    x_shift, y_shift = xy_shift
    if xy_shift is None:
        return flow
    flow2 = np.zeros_like(flow)
    if x_shift > 0 and y_shift > 0:
        flow2[y_shift:, x_shift:] = flow[:-y_shift, :-x_shift]
    if x_shift > 0 and y_shift < 0:
        flow2[:y_shift, x_shift:] = flow[-y_shift:, :-x_shift]
    if x_shift < 0 and y_shift > 0:
        flow2[y_shift:, :x_shift] = flow[:-y_shift, -x_shift:]
    if x_shift < 0 and y_shift < 0:
        flow2[:y_shift, :x_shift] = flow[-y_shift:, -x_shift:]
    return flow2

@torch.no_grad()
def create_sintel_submission_vis(model_name, model, warm_start=False, output_path='sintel_submission',
                                 test_mode=1, do_vis=False, split='test'):
    """ Create submission for the Sintel leaderboard """
    model.eval()
    for dstype in ['clean', 'final']:
        test_dataset = datasets.MpiSintel(split=split, aug_params=None, dstype=dstype, debug=True)
        # If split==training, we manually set test_dataset to test mode.
        test_dataset.is_test = True

        flow_prev, scene_prev = None, None
        for test_id in range(len(test_dataset)):
            image1, image2, (scene, frame_id) = test_dataset[test_id]
            if scene != scene_prev:
                flow_prev = None

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

            flow_low, flow_pr = model.module(image1, image2, iters=32, flow_init=flow_prev, test_mode=test_mode)
            flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

            # Visualizations
            if do_vis:
                flow_img = flow_viz.flow_to_image(flow)
                flow_image = Image.fromarray(flow_img)
                if not os.path.exists(f'vis_sintel/{split}/{model_name}/{dstype}/{scene}'):
                    os.makedirs(f'vis_sintel/{split}/{model_name}/{dstype}/{scene}')

                #if not os.path.exists(f'vis_test/gt/{dstype}/{scene}'):
                #    os.makedirs(f'vis_test/gt/{dstype}/{scene}')

                # image.save(f'vis_test/ours/{dstype}/flow/{test_id}.png')
                flow_image.save(f'vis_sintel/{split}/{model_name}/{dstype}/{scene}/frame_{frame_id+1:04d}.png')
                #imageio.imwrite(f'vis_test/gt/{dstype}/{scene}/{frame_id+1}.png', image1[0].cpu().permute(1, 2, 0).numpy())

            if warm_start:
                flow_prev = forward_interpolate(flow_low[0])[None].cuda()

            output_dir = os.path.join(output_path, dstype, scene)
            output_file = os.path.join(output_dir, 'frame%04d.flo' % (frame_id+1))

            if not os.path.exists(output_dir):
                os.makedirs(output_dir)

            frame_utils.writeFlow(output_file, flow)
            scene_prev = scene

    if do_vis:
        print("Created sintel visualization.")
    else:
        print("Created sintel submission.")

@torch.no_grad()
def create_kitti_submission_vis(model_name, model, output_path='kitti_submission', 
                                test_mode=1, do_vis=False):
    """ Create submission for the KITTI leaderboard """
    model.eval()
    test_dataset = datasets.KITTI(split='testing', aug_params=None, debug=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(f'vis_kitti/{model_name}'):
        os.makedirs(f'vis_kitti/{model_name}')

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id,) = test_dataset[test_id]
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=test_mode)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()

        output_filename = os.path.join(output_path, frame_id)
        frame_utils.writeFlowKITTI(output_filename, flow)

        # Do visualizations
        if do_vis:
            flow_img = flow_viz.flow_to_image(flow)
            flow_image = Image.fromarray(flow_img)
            # frame_id: '000100_10.png'
            flow_image.save(f'vis_kitti/{model_name}/{frame_id}')
            # imageio.imwrite(f'vis_kitti/{model_name}/{frame_id}.png', image1[0].cpu().permute(1, 2, 0).numpy())

    if do_vis:
        print("Created KITTI visualization.")
    else:
        print("Created KITTI submission.")

@torch.no_grad()
def create_viper_submission_vis(model_name, model, output_path='viper_submission', 
                                test_mode=1, do_vis=False):
    """ Create submission for the viper leaderboard """
    model.eval()
    test_dataset = datasets.VIPER(split='test', aug_params=None, debug=True)

    if not os.path.exists(output_path):
        os.makedirs(output_path)
    if not os.path.exists(f'vis_viper/{model_name}'):
        os.makedirs(f'vis_viper/{model_name}')

    scale = 0.5
    inv_scale = 1.0 / scale

    for test_id in range(len(test_dataset)):
        image1, image2, (frame_id, ) = test_dataset[test_id]
        image1 = image1[None].to(f'cuda:{model.device_ids[0]}')
        image2 = image2[None].to(f'cuda:{model.device_ids[0]}')
        # To reduce RAM use, scale images to half size.
        image1 = F.interpolate(image1, scale_factor=scale, mode='bilinear', align_corners=False)
        image2 = F.interpolate(image2, scale_factor=scale, mode='bilinear', align_corners=False)

        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_pr = model.module(image1, image2, iters=24, test_mode=test_mode)
        flow = padder.unpad(flow_pr[0]).permute(1, 2, 0).cpu().numpy()
        # Scale flow back to original size.
        flow = cv2.resize(flow, None, fx=inv_scale, fy=inv_scale, interpolation=cv2.INTER_LINEAR)
        flow = flow * [inv_scale, inv_scale]

        # Do visualizations
        if do_vis:
            flow_img = flow_viz.flow_to_image(flow)
            flow_image = Image.fromarray(flow_img)
            # frame_id: "{scene}_{img0_idx}", without suffix.
            flow_image.save(f'vis_viper/{model_name}/{frame_id}.png')
            # imageio.imwrite(f'vis_viper/{model_name}/{frame_id}.png', image1[0].cpu().permute(1, 2, 0).numpy())

        output_filename = os.path.join(output_path, frame_id + ".flo")
        frame_utils.writeFlow(output_filename, flow)

    if do_vis:
        print("Created VIPER visualization.")
    else:
        print("Created VIPER submission.")

@torch.no_grad()
def validate_chairs(model, iters=6, test_mode=1, xy_shift=None, batch_size=1):
    """ Perform evaluation on the FlyingChairs (test) split """
    model.eval()
    epe_list = []

    if xy_shift is not None:
        x_shift, y_shift = xy_shift
        print(f"Apply x,y shift {x_shift},{y_shift}")
        offset_tensor = torch.tensor([x_shift, y_shift], dtype=torch.float32)
    else:
        offset_tensor = torch.tensor([0, 0], dtype=torch.float32)
    offset_tensor = offset_tensor.reshape([1, 2, 1, 1])

    val_dataset = datasets.FlyingChairs(split='validation')
    val_loader  = data.DataLoader(val_dataset, batch_size=batch_size,
                                  pin_memory=False, shuffle=False, num_workers=4, drop_last=False)

    for data_blob in iter(val_loader):  
        image1, image2, flow_gt, _, _ = data_blob
        image1 = image1.cuda()
        image2 = image2.cuda()

        image1, flow_gt, val_mask = shift_pixels(image1, flow_gt, xy_shift)
        val_mask = val_mask.unsqueeze(0).expand(image1.shape[0], -1, -1)

        _, flow_pr = model(image1, image2, iters=iters, test_mode=test_mode)
        flow = flow_pr.cpu()
        epe = torch.sum((flow - flow_gt)**2, dim=1)[val_mask].sqrt()
        epe_list.append(epe.view(-1).numpy())

    epe = np.mean(np.concatenate(epe_list))
    print("Validation Chairs EPE: %f" % epe)
    return {'chairs_epe': epe}


@torch.no_grad()
def validate_things(model, iters=6, test_mode=1, xy_shift=None, batch_size=1, max_val_count=-1, 
                    verbose=False, seg_interval=-1, dstype='both'):
    """ Perform evaluation on the FlyingThings (test) split """
    model.eval()
    results = {}
    if seg_interval == -1:
        seg_interval = 100
    mag_endpoints = [1, 10, 20, 30, np.inf]
    if xy_shift is not None:
        x_shift, y_shift = xy_shift
        print(f"Apply x,y shift {x_shift},{y_shift}")
        offset_tensor = torch.tensor([x_shift, y_shift], dtype=torch.float32)
    else:
        offset_tensor = torch.tensor([0, 0], dtype=torch.float32)
    offset_tensor = offset_tensor.reshape([1, 2, 1, 1])

    if dstype == 'both':
        dstypes = ['frames_cleanpass', 'frames_finalpass']
    if dstype == 'clean':
        dstypes = ['frames_cleanpass']
    if dstype == 'final':
        dstypes = ['frames_finalpass']

    for dstype in dstypes:
        epe_list = {}
        segs_len  = []
        epe_seg  = []
        mags_seg = {}
        mags_err = {}
        mags_segs_err = {}
        mags_segs_len = {}

        # test_mode == 1, a list of iters=6 flows.
        if test_mode == 1:
            its = [0]
        # test_mode == 2, the final flow only.
        elif test_mode == 2:
            its = range(iters)
        elif test_mode == 0:
            breakpoint()
                       
        val_dataset = datasets.FlyingThings3D(dstype=dstype, aug_params=None, split='validation')
        # Use multiple workers to push GPU utility to near 100%.
        val_loader  = data.DataLoader(val_dataset, batch_size=batch_size,
                                      pin_memory=False, shuffle=False, num_workers=4, drop_last=False)


        print(f'Dataset length {len(val_dataset)}')
        val_count = 0
        if max_val_count == -1:
            max_val_count = len(val_dataset)

        # if blur_params is not None:
        #     GaussianBlur = transforms.GaussianBlur(blur_params['kernel'], blur_params['sigma'])
        # else:
        #     GaussianBlur = None

        for data_blob in iter(val_loader):  
            image1, image2, flow_gt, _, _ = data_blob
            image1 = image1.cuda()
            image2 = image2.cuda()

            image1, flow_gt, val_mask = shift_pixels(image1, flow_gt, xy_shift)
            val_mask = val_mask.unsqueeze(0).expand(image1.shape[0], -1, -1)

            # if GaussianBlur is not None:
            #     image1 = GaussianBlur(image1)
            #     image2 = GaussianBlur(image2)

            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_prs = model(image1, image2, iters=iters, test_mode=test_mode)
            # if test_mode == 2: list of iters=6 tensors, each is [1, 2, 544, 960].
            # if test_mode == 1: a tensor of [1, 2, 544, 960].
            if test_mode == 1:
                flow_prs = [ flow_prs ]
            
            for it, flow_pr in enumerate(flow_prs):
                flow = padder.unpad(flow_pr).cpu()
                epe = torch.sum((flow - flow_gt)**2, dim=1)[val_mask].sqrt()
                epe_list.setdefault(it, [])
                epe_list[it].append(epe.view(-1).numpy())

            epe_seg.append(epe.view(-1).numpy())
            orig_flow_gt = flow_gt.cpu() + offset_tensor
            mag = torch.sum(orig_flow_gt**2, dim=1)[val_mask].sqrt()

            prev_mag_endpoint = 0
            for mag_endpoint in mag_endpoints:
                mags_seg.setdefault(mag_endpoint, [])
                mag_in_range = (mag >= prev_mag_endpoint) & (mag < mag_endpoint)
                mags_seg[mag_endpoint].append(mag_in_range.view(-1).numpy())
                prev_mag_endpoint = mag_endpoint

            val_count += len(image1)
            segs_len.append(len(image1))

            if (seg_interval > 0 and val_count % seg_interval == 0) or val_count >= max_val_count:
                epe_seg     = np.concatenate(epe_seg)
                mean_epe    = np.mean(epe_seg)
                px1 = np.mean(epe_seg < 1)
                px3 = np.mean(epe_seg < 3)
                px5 = np.mean(epe_seg < 5)

                for mag_endpoint in mag_endpoints:
                    mag_seg = np.concatenate(mags_seg[mag_endpoint])
                    if mag_seg.sum() == 0:
                        mag_err = 0
                    else:
                        mag_err = np.mean(epe_seg[mag_seg])
                    mags_err[mag_endpoint] = mag_err
                    mags_segs_err.setdefault(mag_endpoint, [])
                    mags_segs_err[mag_endpoint].append(mags_err[mag_endpoint])
                    mags_segs_len.setdefault(mag_endpoint, [])
                    mags_segs_len[mag_endpoint].append(mag_seg.sum().item())

                if verbose:
                    print(f"{val_count}/{max_val_count}: EPE {mean_epe:.4f}, "
                        f"1px {px1:.4f}, 3px {px3:.4f}, 5px {px5:.4f}", end='')
                    prev_mag_endpoint = 0
                    for mag_endpoint in mag_endpoints:
                        print(f", {prev_mag_endpoint}-{mag_endpoint} {mags_err[mag_endpoint]:.2f}", end='')
                        prev_mag_endpoint = mag_endpoint
                    print()

                epe_seg = []
                mags_seg = {}

            # The validation data is too big. Just evaluate some.
            if val_count >= max_val_count:
                break

        for it in its:
            epe_all = np.concatenate(epe_list[it])

            epe = np.mean(epe_all)
            px1 = np.mean(epe_all < 1)
            px3 = np.mean(epe_all < 3)
            px5 = np.mean(epe_all < 5)

            print(f"Iter {it}, Valid ({dstype}) EPE: {epe:.4f}, 1px: {px1:.4f}, 3px: {px3:.4f}, 5px: {px5:.4f}", end='')

        for mag_endpoint in mag_endpoints:
            mag_errs = np.array(mags_segs_err[mag_endpoint])
            mag_lens = np.array(mags_segs_len[mag_endpoint])
            mag_err = np.sum(mag_errs * mag_lens) / np.sum(mag_lens)
            mags_err[mag_endpoint] = mag_err

        prev_mag_endpoint = 0
        for mag_endpoint in mag_endpoints:
            print(f", {prev_mag_endpoint}-{mag_endpoint} {mags_err[mag_endpoint]:.2f}", end='')
            prev_mag_endpoint = mag_endpoint
        print()

        results[dstype] = epe

    return results


@torch.no_grad()
def validate_sintel(model, iters=6, test_mode=1, xy_shift=None, batch_size=1, max_val_count=-1, 
                    verbose=False, seg_interval=-1, dstype='both'):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    if seg_interval == -1:
        seg_interval = 100
    mag_endpoints = [1, 10, 20, 30, np.inf]
    if xy_shift is not None:
        x_shift, y_shift = xy_shift
        print(f"Apply x,y shift {x_shift},{y_shift}")
        offset_tensor = torch.tensor([x_shift, y_shift], dtype=torch.float32)
    else:
        offset_tensor = torch.tensor([0, 0], dtype=torch.float32)
    offset_tensor = offset_tensor.reshape([1, 2, 1, 1])

    if dstype == 'both':
        dstypes = ['clean', 'final']
    if dstype == 'clean':
        dstypes = ['clean']
    if dstype == 'final':
        dstypes = ['final']

    for dstype in dstypes:
        val_dataset = datasets.MpiSintel(split='training', aug_params=None, dstype=dstype)
        # Use multiple workers to push GPU utility to near 100%.
        val_loader  = data.DataLoader(val_dataset, batch_size=batch_size,
                                      pin_memory=False, shuffle=False, num_workers=4, drop_last=False)

        print(f'Dataset length {len(val_dataset)}')
        val_count = 0
        if max_val_count == -1:
            max_val_count = len(val_dataset)

        epe_list = {}
        segs_len = []
        epe_seg  = []
        mags_seg = {}
        mags_err = {}
        mags_segs_err = {}
        mags_segs_len = {}

        if test_mode == 1:
            its = [0]
        elif test_mode == 2:
            its = range(iters)
        elif test_mode == 0:
            breakpoint()
        
        val_count = 0
        # if blur_params is not None:
        #     GaussianBlur = transforms.GaussianBlur(blur_params['kernel'], blur_params['sigma'])
        # else:
        #     GaussianBlur = None

        for data_blob in iter(val_loader):  
            image1, image2, flow_gt, _, _ = data_blob
            image1 = image1.cuda()
            image2 = image2.cuda()

            # if GaussianBlur is not None:
            #     #image1 = GaussianBlur(image1)
            #     image2 = GaussianBlur(image2)

            # Shift image1 pixels along x and y axes.
            image1, flow_gt, val_mask = shift_pixels(image1, flow_gt, xy_shift)
            val_mask = val_mask.unsqueeze(0).expand(image1.shape[0], -1, -1)
            
            padder = InputPadder(image1.shape)
            image1, image2 = padder.pad(image1, image2)

            _, flow_prs = model(image1, image2, iters=iters, test_mode=test_mode)
            # if test_mode == 2: list of 12 tensors, each is [1, 2, H, W].
            # if test_mode == 1: a tensor of [1, 2, H, W].
            if test_mode == 1:
                flow_prs = [ flow_prs ]

            for it, flow_pr in enumerate(flow_prs):
                flow = padder.unpad(flow_pr).cpu()
                epe = torch.sum((flow - flow_gt)**2, dim=1)[val_mask].sqrt()
                epe_list.setdefault(it, [])
                epe_list[it].append(epe.view(-1).numpy())

            epe_seg.append(epe.view(-1).numpy())
            orig_flow_gt = flow_gt.cpu() + offset_tensor
            mag = torch.sum(orig_flow_gt**2, dim=1)[val_mask].sqrt()

            prev_mag_endpoint = 0
            for mag_endpoint in mag_endpoints:
                mags_seg.setdefault(mag_endpoint, [])
                mag_in_range = (mag >= prev_mag_endpoint) & (mag < mag_endpoint)
                mags_seg[mag_endpoint].append(mag_in_range.view(-1).numpy())
                prev_mag_endpoint = mag_endpoint

            val_count += len(image1)
            segs_len.append(len(image1))

            if (seg_interval > 0 and val_count % seg_interval == 0) or val_count >= max_val_count:
                epe_seg     = np.concatenate(epe_seg)
                mean_epe    = np.mean(epe_seg)
                px1 = np.mean(epe_seg < 1)
                px3 = np.mean(epe_seg < 3)
                px5 = np.mean(epe_seg < 5)

                for mag_endpoint in mag_endpoints:
                    mag_seg = np.concatenate(mags_seg[mag_endpoint])
                    if mag_seg.sum() == 0:
                        mag_err = 0
                    else:
                        mag_err = np.mean(epe_seg[mag_seg])
                    mags_err[mag_endpoint] = mag_err
                    mags_segs_err.setdefault(mag_endpoint, [])
                    mags_segs_err[mag_endpoint].append(mags_err[mag_endpoint])
                    mags_segs_len.setdefault(mag_endpoint, [])
                    mags_segs_len[mag_endpoint].append(mag_seg.sum().item())

                if verbose:
                    print(f"{val_count}/{max_val_count}: EPE {mean_epe:.4f}, "
                        f"1px {px1:.4f}, 3px {px3:.4f}, 5px {px5:.4f}", end='')
                    prev_mag_endpoint = 0
                    for mag_endpoint in mag_endpoints:
                        print(f", {prev_mag_endpoint}-{mag_endpoint} {mags_err[mag_endpoint]:.2f}", end='')
                        prev_mag_endpoint = mag_endpoint
                    print()

                epe_seg = []
                mags_seg = {}

        for it in its:
            epe_all = np.concatenate(epe_list[it])
            
            epe = np.mean(epe_all)
            px1 = np.mean(epe_all<1)
            px3 = np.mean(epe_all<3)
            px5 = np.mean(epe_all<5)

            print("Iter %d, Valid (%s) EPE: %f, 1px: %f, 3px: %f, 5px: %f" % (it, dstype, epe, px1, px3, px5), end='')

        for mag_endpoint in mag_endpoints:
            mag_errs = np.array(mags_segs_err[mag_endpoint])
            mag_lens = np.array(mags_segs_len[mag_endpoint])
            mag_err = np.sum(mag_errs * mag_lens) / np.sum(mag_lens)
            mags_err[mag_endpoint] = mag_err

        prev_mag_endpoint = 0
        for mag_endpoint in mag_endpoints:
            print(f", {prev_mag_endpoint}-{mag_endpoint} {mags_err[mag_endpoint]:.2f}", end='')
            prev_mag_endpoint = mag_endpoint
        print()

        results[dstype] = epe

    return results

@torch.no_grad()
def validate_sintel_occ(model, iters=6, test_mode=1):
    """ Peform validation using the Sintel (train) split """
    model.eval()
    results = {}
    for dstype in ['clean', 'final', 'albedo']:
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
def validate_hd1k(model, iters=6, test_mode=1, seg_interval=-1):
    """ Peform validation using the HD1k data """
    model.eval()
    results = {}
    val_dataset = datasets.HD1K()

    if seg_interval == -1:
        seg_interval = 100    
    epe_list = []
    val_count = 0

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, _ = val_dataset[val_id]
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
        val_count += len(image1)

        if seg_interval > 0 and val_count % seg_interval == 0:
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
def validate_kitti(model, iters=6, test_mode=1, xy_shift=None, batch_size=1, max_val_count=-1, use_kitti_train=False,
                   verbose=False, seg_interval=100):
    """ Peform validation using the KITTI-2015 (train) split """
    model.eval()
    # use_kitti_train: use the test split within the official training split. 
    # Otherwise evaluation is done on the training data.
    if use_kitti_train:
        val_dataset = datasets.KITTITrain(split='testing')
    else:    
        # Evaluate on training data, as ground truth is only available in training data.
        val_dataset = datasets.KITTI(split='training')

    if xy_shift is not None:
        x_shift, y_shift = xy_shift
        print(f"Apply x,y shift {x_shift},{y_shift}")
        offset_tensor = torch.tensor([x_shift, y_shift], dtype=torch.float32)
    else:
        offset_tensor = torch.tensor([0, 0], dtype=torch.float32)
    offset_tensor = offset_tensor.reshape([1, 2, 1, 1])

    val_loader  = data.DataLoader(val_dataset, batch_size=batch_size,
                                  pin_memory=False, shuffle=False, num_workers=4, drop_last=False)

    out_list, epe_list = {}, {}
    out_seg,  epe_seg  = [], []
    if seg_interval == -1:
        seg_interval = 100    
    mag_endpoints = [1, 10, 20, 30, np.inf]
    segs_len = []
    mags_seg = {}
    mags_err = {}
    mags_segs_err = {}
    mags_segs_len = {}

    if test_mode == 1:
        its = [0]
    elif test_mode == 2:
        its = range(iters)
    elif test_mode == 0:
        breakpoint()
    
    val_count = 0
    if max_val_count == -1:
        max_val_count = len(val_dataset)
    else:
        max_val_count = min(max_val_count, len(val_dataset))
        if max_val_count < len(val_dataset):
            print(f"Evaluate first {max_val_count} of {len(val_dataset)}")

    for data_blob in iter(val_loader):
        image1, image2, flow_gt, valid_gt, _ = data_blob
        image1 = image1.cuda()
        image2 = image2.cuda()

        # Shift image1 pixels along x and y axes.
        image1, flow_gt, val_mask = shift_pixels(image1, flow_gt, xy_shift)
        val_mask = val_mask.unsqueeze(0).expand(image1.shape[0], -1, -1)
        valid_gt[~val_mask] = 0

        # (540, 960) => (544, 960), to be divided by 8.
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_prs = model(image1, image2, iters=iters, test_mode=test_mode)
        
        # if test_mode == 2: list of 12 tensors, each is [1, 2, H, W].
        # if test_mode == 1: a tensor of [1, 2, H, W].
        if test_mode == 1:
            flow_prs = [ flow_prs ]

            
        for it, flow_pr in enumerate(flow_prs):
            flow = padder.unpad(flow_pr).cpu()
            epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
            epe = epe.view(-1)
            val = valid_gt.view(-1) >= 0.5

            orig_flow_gt = flow_gt.cpu() + offset_tensor
            mag = torch.sum(orig_flow_gt**2, dim=1).sqrt()
            mag = mag.view(-1)
            out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
                        
            epe_list.setdefault(it, [])
            out_list.setdefault(it, [])
            epe_list[it].append(epe[val].view(-1).numpy())
            # epe_list[it].append(epe[val].mean().item())
            out_list[it].append(out[val].view(-1).numpy())

        # epe, out of the last iteration.
        epe_seg.append(epe[val].view(-1).numpy())
        out_seg.append(out[val].view(-1).numpy())

        prev_mag_endpoint = 0
        for mag_endpoint in mag_endpoints:
            mags_seg.setdefault(mag_endpoint, [])
            # prev_mag_endpoint: mag of the last iteration.
            mag_in_range = (mag >= prev_mag_endpoint) & (mag < mag_endpoint)
            mags_seg[mag_endpoint].append(mag_in_range.view(-1)[val].numpy())
            prev_mag_endpoint = mag_endpoint

        val_count += len(image1)
        segs_len.append(len(image1))

        if (seg_interval > 0 and val_count % seg_interval == 0) or val_count >= max_val_count:
            epe_seg     = np.concatenate(epe_seg)
            out_seg     = np.concatenate(out_seg)
            mean_epe    = np.mean(epe_seg)
            px1 = np.mean(epe_seg < 1)
            px3 = np.mean(epe_seg < 3)
            px5 = np.mean(epe_seg < 5)
                    
            for mag_endpoint in mag_endpoints:
                mag_seg = np.concatenate(mags_seg[mag_endpoint])
                if mag_seg.sum() == 0:
                    mag_err = 0
                else:
                    mag_err = np.mean(epe_seg[mag_seg])
                mags_err[mag_endpoint] = mag_err
                mags_segs_err.setdefault(mag_endpoint, [])
                mags_segs_err[mag_endpoint].append(mags_err[mag_endpoint])
                mags_segs_len.setdefault(mag_endpoint, [])
                mags_segs_len[mag_endpoint].append(mag_seg.sum().item())

            if verbose:
                print(f"{val_count}/{max_val_count}: EPE {mean_epe:.4f}, "
                        f"1px {px1:.4f}, 3px {px3:.4f}, 5px {px5:.4f}", end='')
                prev_mag_endpoint = 0
                for mag_endpoint in mag_endpoints:
                    print(f", {prev_mag_endpoint}-{mag_endpoint} {mags_err[mag_endpoint]:.2f}", end='')
                    prev_mag_endpoint = mag_endpoint
                print()

            epe_seg = []
            out_seg = []
            mags_seg = {}

        # The validation data is too big. Just evaluate some.
        if val_count >= max_val_count:
            break
                                                       
    for it in its:
        # epe_all = np.array(epe_list[it])
        epe_all = np.concatenate(epe_list[it])
        out_all = np.concatenate(out_list[it])
        
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        f1 = 100 * np.mean(out_all)
        print("Iter %d, Valid EPE: %.4f, F1: %.4f, 1px: %.4f, 3px: %.4f, 5px: %.4f" % (it, epe, f1, px1, px3, px5), end='')

    for mag_endpoint in mag_endpoints:
        mag_errs = np.array(mags_segs_err[mag_endpoint])
        mag_lens = np.array(mags_segs_len[mag_endpoint])
        # Average EPE in this magnitude range.
        mag_err = np.sum(mag_errs * mag_lens) / np.sum(mag_lens)
        mags_err[mag_endpoint] = mag_err

    prev_mag_endpoint = 0
    for mag_endpoint in mag_endpoints:
        print(f", {prev_mag_endpoint}-{mag_endpoint} {mags_err[mag_endpoint]:.2f}", end='')
        prev_mag_endpoint = mag_endpoint
    print()

    return {'epe': epe, 'f1': f1}


# Set max_val_count=-1 to evaluate the whole dataset.
@torch.no_grad()
def validate_viper(model, iters=6, test_mode=1, batch_size=2, max_val_count=500, 
                   verbose=False, seg_interval=100):
    """ Peform validation using the VIPER validation split """
    model.eval()
    # original size: (1080, 1920).
    # real_scale = 2** scale = 0.5. scale: log(0.5) / log(2) = -1.
    val_dataset = datasets.VIPER(split='validation', 
                                 aug_params={'crop_size': (540, 960), 'min_scale': -1, 'max_scale': -1,
                                             'spatial_aug_prob': 1})

    val_loader  = data.DataLoader(val_dataset, batch_size=batch_size,
                                  pin_memory=False, shuffle=False, num_workers=4, drop_last=False)

                                   
    out_list, epe_list = {}, {}
    out_seg,  epe_seg  = [], []
    if seg_interval == -1:
        seg_interval = 100    
    mag_endpoints = [1, 10, 20, 30, np.inf]
    segs_len = []
    mags_seg = {}
    mags_err = {}
    mags_segs_err = {}
    mags_segs_len = {}

    if test_mode == 1:
        its = [0]
    elif test_mode == 2:
        its = range(iters)
    elif test_mode == 0:
        breakpoint()
    
    val_count = 0
    if max_val_count == -1:
        max_val_count = len(val_dataset)
    else:
        max_val_count = min(max_val_count, len(val_dataset))
        if max_val_count < len(val_dataset):
            print(f"Evaluate first {max_val_count} of {len(val_dataset)}")
            
    for data_blob in iter(val_loader):
        image1, image2, flow_gt, valid_gt, _ = data_blob
        image1 = image1.cuda()
        image2 = image2.cuda()
        
        # (540, 960) => (544, 960), to be divided by 8.
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_prs = model(image1, image2, iters=iters, test_mode=test_mode)
        
        # if test_mode == 2: list of 12 tensors, each is [1, 2, H, W].
        # if test_mode == 1: a tensor of [1, 2, H, W].
        if test_mode == 1:
            flow_prs = [ flow_prs ]

        for it, flow_pr in enumerate(flow_prs):
            flow = padder.unpad(flow_pr).cpu()
            epe = torch.sum((flow - flow_gt)**2, dim=1).sqrt()
            epe = epe.view(-1)
            val = valid_gt.view(-1) >= 0.5

            mag = torch.sum(flow_gt**2, dim=1).sqrt()
            mag = mag.view(-1)
            out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
                        
            epe_list.setdefault(it, [])
            out_list.setdefault(it, [])
            epe_list[it].append(epe[val].view(-1).numpy())
            out_list[it].append(out[val].view(-1).numpy())

        epe_seg.append(epe[val].view(-1).numpy())
        out_seg.append(out[val].view(-1).numpy())

        prev_mag_endpoint = 0
        for mag_endpoint in mag_endpoints:
            mags_seg.setdefault(mag_endpoint, [])
            # Use mag of the last it.
            mag_in_range = (mag >= prev_mag_endpoint) & (mag < mag_endpoint)
            mags_seg[mag_endpoint].append(mag_in_range.view(-1)[val].numpy())
            prev_mag_endpoint = mag_endpoint

        val_count += len(image1)
        segs_len.append(len(image1))

        if (seg_interval > 0 and val_count % seg_interval == 0) or val_count >= max_val_count:
            epe_seg     = np.concatenate(epe_seg)
            out_seg     = np.concatenate(out_seg)
            mean_epe    = np.mean(epe_seg)
            px1 = np.mean(epe_seg < 1)
            px3 = np.mean(epe_seg < 3)
            px5 = np.mean(epe_seg < 5)
                    
            for mag_endpoint in mag_endpoints:
                mag_seg = np.concatenate(mags_seg[mag_endpoint])
                if mag_seg.sum() == 0:
                    mag_err = 0
                else:
                    mag_err = np.mean(epe_seg[mag_seg])
                mags_err[mag_endpoint] = mag_err
                mags_segs_err.setdefault(mag_endpoint, [])
                mags_segs_err[mag_endpoint].append(mags_err[mag_endpoint])
                mags_segs_len.setdefault(mag_endpoint, [])
                mags_segs_len[mag_endpoint].append(mag_seg.sum().item())

            if verbose:
                print(f"{val_count}/{max_val_count}: EPE {mean_epe:.4f}, "
                        f"1px {px1:.4f}, 3px {px3:.4f}, 5px {px5:.4f}", end='')
                prev_mag_endpoint = 0
                for mag_endpoint in mag_endpoints:
                    print(f", {prev_mag_endpoint}-{mag_endpoint} {mags_err[mag_endpoint]:.2f}", end='')
                    prev_mag_endpoint = mag_endpoint
                print()

            epe_seg = []
            out_seg = []
            mags_seg = {}

        # The validation data is too big. Just evaluate some.
        if val_count >= max_val_count:
            break
                                                       
    for it in its:
        epe_all = np.concatenate(epe_list[it])
        out_all = np.concatenate(out_list[it])
        
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        f1 = 100 * np.mean(out_all)
        print("Iter %d, Valid EPE: %.4f, F1: %.4f, 1px: %.4f, 3px: %.4f, 5px: %.4f" % (it, epe, f1, px1, px3, px5), end='')

    for mag_endpoint in mag_endpoints:
        mag_errs = np.array(mags_segs_err[mag_endpoint])
        mag_lens = np.array(mags_segs_len[mag_endpoint])
        mag_err = np.sum(mag_errs * mag_lens) / np.sum(mag_lens)
        mags_err[mag_endpoint] = mag_err

    prev_mag_endpoint = 0
    for mag_endpoint in mag_endpoints:
        print(f", {prev_mag_endpoint}-{mag_endpoint} {mags_err[mag_endpoint]:.2f}", end='')
        prev_mag_endpoint = mag_endpoint
    print()

    return {'epe': epe, 'f1': f1}

# Set max_val_count=-1 to evaluate the whole dataset.
@torch.no_grad()
def validate_slowflow(model, iters=6, test_mode=1, xy_shift=None, 
                      blur_set=(100, 0), verbose=False, seg_interval=-1):
    """ Peform validation using the VIPER validation split """
    model.eval()

    blur_magnitude, blur_num_frames = blur_set
    val_dataset = datasets.SlowFlow(split='test', aug_params=None,
                                    blur_mag=blur_magnitude, blur_num_frames=blur_num_frames)
    # slowflow images are 1024x768, or 1280x576. Scale them to half size to fit in RAM.
    scale = 0.5
    scale_tensor = torch.tensor([scale, scale]).reshape(2, 1, 1)

    out_list, epe_list = {}, {}
    out_seg,  epe_seg  = [], []  
    mag_endpoints = [1, 10, 20, 30, np.inf]
    segs_len = []
    mags_seg = {}
    mags_err = {}
    mags_segs_err = {}
    mags_segs_len = {}

    if test_mode == 1:
        its = [0]
    elif test_mode == 2:
        its = range(iters)
    elif test_mode == 0:
        breakpoint()
    
    val_count = 0
    max_val_count = len(val_dataset)
    prev_scene = None

    if xy_shift is not None:
        x_shift, y_shift = xy_shift
        print(f"Apply x,y shift {x_shift},{y_shift}")
        offset_tensor = torch.tensor([x_shift, y_shift], dtype=torch.float32)
    else:
        offset_tensor = torch.tensor([0, 0], dtype=torch.float32)

    for val_id in range(len(val_dataset)):
        image1, image2, flow_gt, _, (scene, img1_trunk) = val_dataset[val_id]
        image1 = image1[None].cuda()
        image2 = image2[None].cuda()
        
        # To reduce RAM use, scale images to half size.
        image1 = F.interpolate(image1, scale_factor=scale, mode='bilinear', align_corners=False)
        image2 = F.interpolate(image2, scale_factor=scale, mode='bilinear', align_corners=False)
        flow_gt = F.interpolate(flow_gt.unsqueeze(0), scale_factor=scale, mode='bilinear', align_corners=False)
        flow_gt = flow_gt.squeeze(0) * scale_tensor

        # Shift image1 and the groundtruth flow along x and y axes.
        image1, flow_gt, val_mask = shift_pixels(image1, flow_gt, xy_shift)

        # Make images divideable by 8.
        padder = InputPadder(image1.shape, mode='kitti')
        image1, image2 = padder.pad(image1, image2)

        _, flow_prs = model(image1, image2, iters=iters, test_mode=test_mode)
        
        # if test_mode == 2: list of 12 tensors, each is [1, 2, H, W].
        # if test_mode == 1: a tensor of [1, 2, H, W].
        if test_mode == 1:
            flow_prs = [ flow_prs ]

        for it, flow_pr in enumerate(flow_prs):
            flow = padder.unpad(flow_pr[0]).cpu()
            epe = torch.sum((flow - flow_gt)**2, dim=0)[val_mask].sqrt()
            epe = epe.view(-1)

            orig_flow_gt = flow_gt.cpu() + offset_tensor
            mag = torch.sum(orig_flow_gt**2, dim=0)[val_mask].sqrt()
            mag = mag.view(-1)
            out = ((epe > 3.0) & ((epe/mag) > 0.05)).float()
                        
            epe_list.setdefault(it, [])
            out_list.setdefault(it, [])
            epe_list[it].append(epe.view(-1).numpy())
            out_list[it].append(out.view(-1).numpy())

        epe_seg.append(epe.view(-1).numpy())
        out_seg.append(out.view(-1).numpy())

        prev_mag_endpoint = 0
        for mag_endpoint in mag_endpoints:
            mags_seg.setdefault(mag_endpoint, [])
            # Use mag of the last it.
            mag_in_range = (mag >= prev_mag_endpoint) & (mag < mag_endpoint)
            mags_seg[mag_endpoint].append(mag_in_range.view(-1).numpy())
            prev_mag_endpoint = mag_endpoint

        val_count += len(image1)
        segs_len.append(len(image1))

        if (prev_scene and scene != prev_scene) or (seg_interval > 0 and val_count % seg_interval == 0) \
          or val_count >= max_val_count:
            epe_seg     = np.concatenate(epe_seg)
            out_seg     = np.concatenate(out_seg)
            mean_epe    = np.mean(epe_seg)
            px1 = np.mean(epe_seg < 1)
            px3 = np.mean(epe_seg < 3)
            px5 = np.mean(epe_seg < 5)
                    
            for mag_endpoint in mag_endpoints:
                mag_seg = np.concatenate(mags_seg[mag_endpoint])
                if mag_seg.sum() == 0:
                    mag_err = 0
                else:
                    mag_err = np.mean(epe_seg[mag_seg])
                mags_err[mag_endpoint] = mag_err
                mags_segs_err.setdefault(mag_endpoint, [])
                mags_segs_err[mag_endpoint].append(mags_err[mag_endpoint])
                mags_segs_len.setdefault(mag_endpoint, [])
                mags_segs_len[mag_endpoint].append(mag_seg.sum().item())

            if verbose:
                seg_scene = prev_scene if (scene != prev_scene) else scene
                print(f"{seg_scene} {img1_trunk} {val_count}/{max_val_count}: EPE {mean_epe:.4f}, "
                    f"1px {px1:.4f}, 3px {px3:.4f}, 5px {px5:.4f}", end='')
                prev_mag_endpoint = 0
                for mag_endpoint in mag_endpoints:
                    print(f", {prev_mag_endpoint}-{mag_endpoint} {mags_err[mag_endpoint]:.2f}", end='')
                    prev_mag_endpoint = mag_endpoint
                print()

            epe_seg = []
            out_seg = []
            mags_seg = {}

        prev_scene = scene
        # The validation data is too big. Just evaluate some.
        if val_count >= max_val_count:
            break
                                                       
    for it in its:
        epe_all = np.concatenate(epe_list[it])
        out_all = np.concatenate(out_list[it])
        
        epe = np.mean(epe_all)
        px1 = np.mean(epe_all<1)
        px3 = np.mean(epe_all<3)
        px5 = np.mean(epe_all<5)

        f1 = 100 * np.mean(out_all)
        print("Iter %d, Valid EPE: %.4f, F1: %.4f, 1px: %.4f, 3px: %.4f, 5px: %.4f" % (it, epe, f1, px1, px3, px5), end='')

    for mag_endpoint in mag_endpoints:
        mag_errs = np.array(mags_segs_err[mag_endpoint])
        mag_lens = np.array(mags_segs_len[mag_endpoint])
        mag_err = np.sum(mag_errs * mag_lens) / np.sum(mag_lens)
        mags_err[mag_endpoint] = mag_err

    prev_mag_endpoint = 0
    for mag_endpoint in mag_endpoints:
        print(f", {prev_mag_endpoint}-{mag_endpoint} {mags_err[mag_endpoint]:.2f}", end='')
        prev_mag_endpoint = mag_endpoint
    print()

    return {'epe': epe, 'f1': f1}

def save_checkpoint(cp_path, model, optimizer_state, lr_scheduler_state, logger):
    save_state = { 'model':        model.state_dict(),
                   'optimizer':    optimizer_state,
                   'lr_scheduler': lr_scheduler_state,
                   'logger':       logger.__dict__
                 }

    torch.save(save_state, cp_path)
    print(f"{cp_path} saved")

@torch.no_grad()
def gen_flow(model, model_name, iters, image1_path, image2_path, flow_path=None, output_path='output', 
             test_mode=1, scale=1., xy_shift=None, calc_flop=False):
    """ Generate flow given two images """
    model.eval()

    if not os.path.exists(output_path):
        os.makedirs(output_path)

    if xy_shift is not None:
        x_shift, y_shift = xy_shift
        print(f"Apply x,y shift {x_shift},{y_shift}")
    else:
        x_shift, y_shift = (0, 0)
    # the format of flow is u, v, i.e., x, y, not y, x.
    offset_tensor = torch.tensor([x_shift, y_shift], dtype=torch.float32).reshape(2, 1, 1)

    # split image1_path into path and file name
    _, image1_name = os.path.split(image1_path)
    # split file name into file name and extension
    image1_name_noext, _ = os.path.splitext(image1_name)
    _, image2_name = os.path.split(image2_path)
    image2_name_noext, _ = os.path.splitext(image2_name)

    image1 = frame_utils.read_gen(image1_path)
    image2 = frame_utils.read_gen(image2_path)
    image1 = np.array(image1).astype(np.uint8)[..., :3]
    image2 = np.array(image2).astype(np.uint8)[..., :3]

    # grayscale images
    if len(image1.shape) == 2:
        image1 = np.tile(image1[...,None], (1, 1, 3))
        image2 = np.tile(image2[...,None], (1, 1, 3))

    image1 = torch.from_numpy(image1).permute(2, 0, 1).float()
    image2 = torch.from_numpy(image2).permute(2, 0, 1).float()

    if flow_path:
        flow_gt = frame_utils.read_gen(flow_path)
    else:
        flow_gt = None

    if scale < 1:
        # Scale down images to reduce RAM use.
        image1 = F.interpolate(image1[None], scale_factor=scale, mode='bilinear', align_corners=False)
        image2 = F.interpolate(image2[None], scale_factor=scale, mode='bilinear', align_corners=False)
        image1 = image1[0]
        image2 = image2[0]
        scale_image1_path = os.path.join(output_path, image1_name_noext + f'-{scale}.png')
        Image.fromarray(image1.permute(1, 2, 0).numpy().astype(np.uint8)).save(scale_image1_path)
        print(f"Save scaled image1 to {scale_image1_path}")
        scale_image2_path = os.path.join(output_path, image2_name_noext + f'-{scale}.png')
        Image.fromarray(image2.permute(1, 2, 0).numpy().astype(np.uint8)).save(scale_image2_path)
        print(f"Save scaled image2 to {scale_image2_path}")
        
        if flow_gt is not None:
            flow_gt = cv2.resize(flow_gt, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
            flow_gt = flow_gt * [scale, scale]        
            flow_gt_img = flow_viz.flow_to_image(flow_gt)
            scale_flow_path = os.path.join(output_path, image1_name_noext + f'-flow-{scale}.png')
            Image.fromarray(flow_gt_img).save(scale_flow_path)
            # cv2.imwrite(scale_flow_path, flow_gt_img[..., ::-1])
            print(f"Save scaled flow gt to {scale_flow_path}")

    # Shift image1 pixels along x and y axes.
    image1, _, val_mask = shift_pixels(image1, None, xy_shift)

    if xy_shift is not None:
        image1_np = image1.permute(1, 2, 0).numpy()
        shift_image1_path = os.path.join(output_path, image1_name_noext + f'-{x_shift},{y_shift}.png')
        Image.fromarray(image1_np.astype(np.uint8)).save(shift_image1_path)
        print(f"Save shifted image1 to {shift_image1_path}")

        if flow_gt is not None:
            shift_flow_gt = shift_flow(flow_gt, xy_shift)
            shift_flow_img = flow_viz.flow_to_image(shift_flow_gt)
            shift_flow_path = os.path.join(output_path, image1_name_noext + f'-{x_shift},{y_shift}-flow.png')
            Image.fromarray(shift_flow_img).save(shift_flow_path)
            print(f"Save shifted flow gt to {shift_flow_path}")
            flow_gt = shift_flow_gt

    padder = InputPadder(image1.shape) #, mode='kitti')
    image1, image2 = padder.pad(image1[None].to(f'cuda:{model.device_ids[0]}'), image2[None].to(f'cuda:{model.device_ids[0]}'))

    # https://github.com/facebookresearch/fvcore/blob/main/docs/flop_count.md
    if calc_flop:
        flops = FlopCountAnalysis(model, (image1, image2, iters))
        print(flops.by_module())
        exit()

    _, flow_prs = model.module(image1, image2, iters=iters, test_mode=test_mode)
    # if test_mode = 1, flow_pr: [1, 2, 512, 640]
    # if test_mode = 2, flow_pr is a list of flow fields, each field is [1, 2, 512, 640].
    if test_mode == 1:
        flow_prs = [ flow_prs ]

    for it, flow_pr in enumerate(flow_prs):
        flow = flow_pr[0].cpu() + offset_tensor
        flow = padder.unpad(flow).permute(1, 2, 0).numpy()
        flow[~val_mask] = 0

        if flow_gt is not None:
            epe = np.sqrt(np.sum((flow - flow_gt)**2, axis=2)[val_mask])
            epe = epe.mean()
            print(f"EPE: {epe:.4f}")

            gt_rad   = np.sqrt(np.square(flow_gt[..., 0]) + np.square(flow_gt[..., 1]))
            flow_rad = np.sqrt(np.square(flow[..., 0])    + np.square(flow[..., 1]))
            gt_max_rad = gt_rad.max()
            total_pixel_count = val_mask.sum()
            exceed_counts = (flow_rad > gt_max_rad).sum()
            exceed_ratio  = exceed_counts / total_pixel_count
            print(f"{exceed_counts}/{exceed_ratio*100:.1f}% pixels exceed max gt flow radius {gt_max_rad:.4f}, ", end="")
            # flow_to_image() normalizes the whole flow with the maximum radius. 
            # If the maximum radius of flow is different from the maximum radius of flow_gt, 
            # the flow image will look quite different from the flow_gt image, 
            # even if they are numerically very close (e.g. EPE 1.xx).
            # However, if too many pixels are beyond the max gt flow radius, 
            # then do not clip the radiuses of these pixels.
            # Otherwise, the flow image will become a monotonic blob and lose all details.
            if exceed_ratio > 0 and exceed_ratio <= 0.1:
                scales = np.ones_like(flow_rad)
                scales[flow_rad > gt_max_rad] = gt_max_rad / flow_rad[flow_rad > gt_max_rad]
                flow = flow * np.expand_dims(scales, 2)
                print("Clip these radiuses.")
            else:
                print("Do not clip radiuses.")

        flow_img = flow_viz.flow_to_image(flow)
        image = Image.fromarray(flow_img)
        flow_savepath = os.path.join(output_path, image1_name_noext + f"-{model_name}-{iters}-{it:02d}.png")
        image.save(flow_savepath)

        print(f"Generated flow {flow_savepath}.")

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
    parser.add_argument('--split', type=str, default="test", help="split of dataset for evaluation")
    parser.add_argument('--dstype', type=str, default="both", help="dstype (clean, final, or both) of dataset for evaluation")
    parser.add_argument('--slowset', type=str, help="slowflow set for evaluation, e.g. 300,3")
    parser.add_argument('--img1', type=str, default=None, help="first image for evaluation")
    parser.add_argument('--img2', type=str, default=None, help="second image for evaluation")
    parser.add_argument('--flow', type=str, default=None, help="ground truth flow")
    parser.add_argument('--output', type=str, default="output", help="output directory")
    parser.add_argument('--flop', dest='calc_flop', action='store_true', help="Compute model FLOPs")

    parser.add_argument('--verbose', action='store_true', help="print stats every 100 iterations")
    parser.add_argument('--seginterval', dest='seg_interval', 
                        type=int, default=-1, help="print stats every N iterations")

    parser.add_argument('--craft', dest='craft', action='store_true', 
                        help='use craft (Cross-Attentional Flow Transformer)')
    parser.add_argument('--setrans', dest='use_setrans', action='store_true', 
                        help='use setrans (Squeeze-Expansion Transformer)')
    parser.add_argument('--raft', action='store_true', 
                        help='use raft')
    parser.add_argument('--nogma', action='store_true', help='(ablation) Do not use GMA')

    parser.add_argument('--sofi', action='store_true', help='use sofi')
    
    parser.add_argument('--iters', type=int, default=12)
    parser.add_argument('--num_heads', default=1, type=int,
                        help='number of heads in attention and aggregation')
    parser.add_argument('--position_only', default=False, action='store_true',
                        help='only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='use position and content-wise attention')
    parser.add_argument('--fullprec', dest='mixed_precision',
                        action='store_false', help='use full precision (default: mixed precision)')
    parser.add_argument('--model_name')
    parser.add_argument('--fix', action='store_true', help='Fix loaded checkpoint')
    parser.add_argument('--submit', action='store_true', help='Make a sintel/kitti submission')
    parser.add_argument('--vis', action='store_true', help='Make a sintel/kitti visualizaton')
    parser.add_argument('--test_mode', default=1, type=int, 
                        help='Test mode (1: normal, 2: evaluate performance of every iteration)')
    parser.add_argument('--maxval', dest='max_val_count', default=-1, type=int, 
                        help='Maximum number of evaluated examples')
    parser.add_argument('--bs', dest='batch_size', default=1, type=int, 
                        help='Batch size')

    parser.add_argument('--scale', dest='scale', default=1, type=float)
    parser.add_argument('--xshifts', dest='x_shifts', default=None, type=str, 
                        help='Shift image1 pixels along x with these offsets')
    parser.add_argument('--yshifts', dest='y_shifts', default=None, type=str, 
                        help='Shift image1 pixels along y with these offsets')

    """ parser.add_argument('--blurk', dest='blur_kernel', default=5, type=int, 
                            help='Gaussian blur kernel size')
        # Only if blur_sigma > 0 (regardless of blur_kernel), Gaussian blur will be applied.
        parser.add_argument('--blurs', dest='blur_sigma', default=-1, type=int, 
                            help='Gaussian blur sigma') 
    """

    # Ablations
    parser.add_argument('--radius', dest='corr_radius', type=int, default=4)    
    parser.add_argument('--posr', dest='pos_bias_radius', type=int, default=7, 
                        help='The radius of positional biases')

    parser.add_argument('--f1', dest='f1trans', type=str, 
                        choices=['none', 'shared', 'private'], default='none',
                        help='Whether to use transformer on frame 1 features. '
                             'shared:  use the same self-attention as f2trans. '
                             'private: use a private self-attention.')
    parser.add_argument('--f2', dest='f2trans', type=str, 
                        choices=['none', 'full'], default='full',
                        help='Whether to use transformer on frame 2 features.')  

    parser.add_argument('--f2posw', dest='f2_pos_code_weight', type=float, default=0.5)
    parser.add_argument('--f2radius', dest='f2_attn_mask_radius', type=int, default=-1)
                            
    parser.add_argument('--intermodes', dest='inter_num_modes', type=int, default=4, 
                        help='Number of modes in inter-frame attention')
    parser.add_argument('--intramodes', dest='intra_num_modes', type=int, default=4, 
                        help='Number of modes in intra-frame attention')
    parser.add_argument('--f2modes', dest='f2_num_modes',       type=int, default=4, 
                        help='Number of modes in F2 Transformer')                        
   # In inter-frame attention, having QK biases performs slightly better.
    parser.add_argument('--interqknobias', dest='inter_qk_have_bias', action='store_false', 
                        help='Do not use biases in the QK projections in the inter-frame attention')
                        
    parser.add_argument('--interpos', dest='inter_pos_code_type', type=str, 
                        choices=['lsinu', 'bias'], default='bias')
    parser.add_argument('--interposw', dest='inter_pos_code_weight', type=float, default=0.5)
    parser.add_argument('--intrapos', dest='intra_pos_code_type', type=str, 
                        choices=['lsinu', 'bias'], default='bias')
    parser.add_argument('--intraposw', dest='intra_pos_code_weight', type=float, default=1.0)
    
    args = parser.parse_args()

    print("Args:\n{}".format(args))
    
    if args.dataset == 'separate':
        separate_inout_sintel_occ()
        sys.exit()

    if args.raft:
        model = nn.DataParallel(RAFT(args))
    elif args.nogma:
        model = nn.DataParallel(CRAFT_nogma(args))
    elif args.sofi:
        sys.path.append("../rift")
        from model.IFNet import IFNet
        from model.RIFT import SOFI_Wrapper
        flownet = IFNet(esti_sofi=True)
        model = nn.DataParallel(SOFI_Wrapper(flownet))
    else:    
        model = nn.DataParallel(CRAFT(args))
    
    if args.fix:
        fix_checkpoint(args, model)
        exit()
        
    checkpoint = torch.load(args.model)
    if 'model' in checkpoint:
        # Ablation study of the impact of positional biases.
        if args.craft and args.f2trans == 'full':
            pass
            #checkpoint['model']['module.corr_fn.vispos_encoder.pos_coder.biases'].zero_()
            #checkpoint['model']['module.f2_trans.vispos_encoder.pos_coder.biases'].zero_()
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    else:
        # Load old checkpoint.
        msg = model.load_state_dict(checkpoint, strict=False)

    print(f"Model checkpoint loaded from {args.model}: {msg}.")

    model.cuda()
    model.eval()

    model_name = os.path.split(args.model)[-1].split(".")[0]
    if 'craft' in model_name:
        if args.f1trans == 'full':
            model_name = model_name.replace("craft", "craft-f1f2")
        else:
            model_name = model_name.replace("craft", f"craft-f2{args.f2trans}")

    if args.seg_interval > 0:
        args.verbose = True

    if args.x_shifts and args.y_shifts:
        x_shifts = [int(x) for x in args.x_shifts.split(',')]
        y_shifts = [int(y) for y in args.y_shifts.split(',')]
        args.xy_shifts = list(zip(x_shifts, y_shifts))
    else:
        args.xy_shifts = [None]

    if args.img1 is not None:
        gen_flow(model, model_name, args.iters, args.img1, args.img2, args.flow, args.output, 
                 test_mode=args.test_mode, scale=args.scale, xy_shift=args.xy_shifts[0],
                 calc_flop=args.calc_flop)
        exit(0)

    if args.dataset == 'sintel' and (args.submit or args.vis):
        create_sintel_submission_vis(model_name, model, warm_start=True,
                                     do_vis=args.vis, split=args.split)
        exit(0)

    if args.dataset == 'kitti' and (args.submit or args.vis):
        create_kitti_submission_vis(model_name, model, do_vis=args.vis)
        exit(0)

    if args.dataset == 'viper' and (args.submit or args.vis):
        create_viper_submission_vis(model_name, model, do_vis=args.vis)
        exit(0)

    """     if args.blur_sigma > 0:
            print(f"Blur kernel size: {args.blur_kernel}, sigma: {args.blur_sigma}".format(args.blur_kernel))
            blur_params = { 'kernel': args.blur_kernel, 'sigma': args.blur_sigma }
        else:
            blur_params = None """

    for xy_shift in args.xy_shifts:
        with torch.no_grad():
            if args.dataset == 'chairs':
                validate_chairs(model.module, iters=args.iters, test_mode=args.test_mode, 
                                xy_shift=xy_shift, batch_size=args.batch_size)

            elif args.dataset == 'things':
                validate_things(model.module, iters=args.iters, test_mode=args.test_mode, 
                                xy_shift=xy_shift, batch_size=args.batch_size,
                                max_val_count=args.max_val_count, 
                                verbose=args.verbose, seg_interval=args.seg_interval,
                                dstype=args.dstype)

            elif args.dataset == 'sintel':
                validate_sintel(model.module, iters=args.iters, test_mode=args.test_mode, 
                                xy_shift=xy_shift, batch_size=args.batch_size,
                                max_val_count=args.max_val_count, 
                                verbose=args.verbose, seg_interval=args.seg_interval,
                                dstype=args.dstype)

            elif args.dataset == 'sintel_occ':
                validate_sintel_occ(model.module, iters=args.iters, test_mode=args.test_mode)

            elif args.dataset == 'kitti':
                validate_kitti(model.module, iters=args.iters, test_mode=args.test_mode,
                               xy_shift=xy_shift, max_val_count=args.max_val_count, batch_size=args.batch_size,
                               verbose=args.verbose, seg_interval=args.seg_interval)
            elif args.dataset == 'kittitrain':
                validate_kitti(model.module, iters=args.iters, test_mode=args.test_mode, 
                               xy_shift=xy_shift, max_val_count=args.max_val_count, batch_size=args.batch_size,
                               verbose=args.verbose, seg_interval=args.seg_interval, 
                               use_kitti_train=True)

            elif args.dataset == 'viper':
                validate_viper(model.module, iters=args.iters, test_mode=args.test_mode, 
                            max_val_count=args.max_val_count, batch_size=args.batch_size,
                            verbose=args.verbose, seg_interval=args.seg_interval)

            elif args.dataset == 'slowflow':
                sf_blur_mag, sf_blur_num_frames = args.slowset.split(",")
                sf_blur_mag = int(sf_blur_mag)
                sf_blur_num_frames = int(sf_blur_num_frames)
                validate_slowflow(model.module, iters=args.iters, test_mode=args.test_mode,
                                  xy_shift=xy_shift,
                                  blur_set=(sf_blur_mag, sf_blur_num_frames),
                                  verbose=args.verbose, seg_interval=args.seg_interval)

            elif args.dataset == 'hd1k':
                validate_hd1k(model.module, iters=args.iters, test_mode=args.test_mode, 
                              seg_interval=args.seg_interval)

