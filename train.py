from __future__ import print_function, division
import sys
sys.path.append('core')

import copy
from datetime import datetime
import argparse
import os
import time
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist

from network import CRAFT
from raft import RAFT
from craft_nogma import CRAFT_nogma
import datasets
import evaluate
from utils.utils import print0

from torch.cuda.amp import GradScaler

# exclude extremly large displacements
MAX_FLOW = 400

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def sequence_loss(flow_preds, flow_gt, valid, gamma):
    """ Loss function defined over sequence of flow predictions """

    # n_predictions = args.iters = 12
    n_predictions = len(flow_preds)    
    flow_loss = 0.0

    # exclude invalid pixels and extremely large displacements. 
    # MAX_FLOW = 400.
    valid = (valid >= 0.5) & ((flow_gt**2).sum(dim=1).sqrt() < MAX_FLOW)

    for i in range(n_predictions):
        # Exponentially increasing weights. (Eq.7 in RAFT paper)
        # As i increases, flow_preds[i] is expected to be more and more accurate, 
        # so we are less and less tolerant to errors through gradually increased i_weight.
        i_weight = gamma**(n_predictions - i - 1)
        i_loss = (flow_preds[i] - flow_gt).abs()
        flow_loss += i_weight * (valid[:, None] * i_loss).mean()

    epe = torch.sum((flow_preds[-1] - flow_gt)**2, dim=1).sqrt()
    epe = epe.view(-1)[valid.view(-1)]
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    if world_size > 1:
        flow_loss = reduce_tensor(flow_loss, world_size)
        epe = gather_tensor(epe, world_size)

    metrics = {
        'epe': epe.mean().item(),
        '1px': (epe < 1).float().mean().item(),
        '3px': (epe < 3).float().mean().item(),
        '5px': (epe < 5).float().mean().item(),
    }

    return flow_loss, metrics


def fetch_optimizer(args, model):
    """ Create the optimizer and learning rate scheduler """
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wdecay, eps=args.epsilon)

    pct_start = 0.05
        
    scheduler = optim.lr_scheduler.OneCycleLR(optimizer=optimizer, max_lr=args.lr, total_steps=args.num_steps+100,
                                              pct_start=pct_start, cycle_momentum=False, anneal_strategy='linear')

    return optimizer, scheduler

def reduce_tensor(tensor, world_size):
    rt = tensor.clone()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    rt /= world_size
    return rt

def gather_tensor(tensor, world_size):
    tensor_list = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(tensor_list, tensor)
    gathered_tensor = torch.cat(tensor_list, dim=0)
    return gathered_tensor

class Logger:
    def __init__(self, scheduler, args):
        self.args = args
        self.scheduler = scheduler
        self.total_steps = 0
        self.running_loss_dict = {}
        self.train_epe_list = []
        self.train_steps_list = []
        self.val_steps_list = []
        self.val_results_dict = {}

    def _print_training_status(self):
        metrics_data = [np.mean(self.running_loss_dict[k]) for k in sorted(self.running_loss_dict.keys())]
        training_str = "[{:6d}, {:10.7f}] ".format(self.total_steps+1, self.scheduler.get_lr()[0])
        # metrics_data[:-1]: '1px', '3px', '5px', 'epe'. metrics_data[-1] is 'time'.
        metrics_str = ("{:10.4f}, "*len(metrics_data[:-1])).format(*metrics_data[:-1])

        # Compute time left
        time_left_sec = (self.args.num_steps - (self.total_steps+1)) * metrics_data[-1]
        time_left_sec = time_left_sec.astype(int)
        time_left_hm = "{:02d}h{:02d}m".format(time_left_sec // 3600, time_left_sec % 3600 // 60)
        time_left_hm = f"{time_left_hm:>9}"
        # print the training status
        print0(training_str + metrics_str + time_left_hm)

        # logging running loss to total loss
        self.train_epe_list.append(np.mean(self.running_loss_dict['epe']))
        self.train_steps_list.append(self.total_steps)

        for key in self.running_loss_dict:
            self.running_loss_dict[key] = []

    def push(self, metrics):
        self.total_steps += 1
        for key in metrics:
            if key not in self.running_loss_dict:
                self.running_loss_dict[key] = []

            self.running_loss_dict[key].append(metrics[key])

        if self.total_steps % self.args.print_freq == self.args.print_freq-1:
            self._print_training_status()
            self.running_loss_dict = {}

def save_checkpoint(cp_path, model, optimizer, lr_scheduler, logger):
    logger_dict = copy.copy(logger.__dict__)
    for key in ('args', 'scheduler'):
        if key in logger_dict:
            del logger_dict[key]

    save_state = { 'model':        model.state_dict(),
                   'optimizer':    optimizer.state_dict(),
                   'lr_scheduler': lr_scheduler.state_dict(),
                   'logger':       logger_dict
                 }

    torch.save(save_state, cp_path)
    print0(f"{cp_path} saved")

def load_checkpoint(args, model, optimizer, lr_scheduler, logger):
    checkpoint = torch.load(args.restore_ckpt, map_location='cuda')

    if 'model' in checkpoint:
        msg = model.load_state_dict(checkpoint['model'], strict=False)
    else:
        # Load old checkpoint.
        msg = model.load_state_dict(checkpoint, strict=False)

    print0(f"Model checkpoint loaded from {args.restore_ckpt}: {msg}.")

    if args.load_optimizer_state and 'optimizer' in checkpoint:
        optimizer.load_state_dict(checkpoint['optimizer'])
        print0("Optimizer state loaded.")
    else:
        print0("Optimizer state NOT loaded.")
        
    if args.load_scheduler_state and 'lr_scheduler' in checkpoint:
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        print0("Scheduler state loaded.")
        if 'logger' in checkpoint:
            # https://stackoverflow.com/questions/243836/how-to-copy-all-properties-of-an-object-to-another-object-in-python
            logger.__dict__.update(checkpoint['logger'])
            print0("Logger loaded.")
        else:
            print0("Logger NOT loaded.")
    else:
        print0("Scheduler state NOT loaded.")
        print0("Logger NOT loaded.")
        
def main(args):
    torch.cuda.set_device(args.local_rank)
    if len(args.gpus) > 1 and torch.cuda.device_count() > 1:
        torch.distributed.init_process_group(backend='nccl',
                                            init_method='env://')

    if args.raft:
        model = RAFT(args)
    elif args.nogma:
        model = CRAFT_nogma(args)
    else:
        model = CRAFT(args)

    model.cuda()
    if len(args.gpus) > 1 and torch.cuda.device_count() > 1:
        model = nn.parallel.DistributedDataParallel(model, device_ids=[args.local_rank],
                                                    output_device=args.local_rank,
                                                    find_unused_parameters=True)

    print0(f"Parameter Count: {count_parameters(model)}")

    model.train()

    train_loader = datasets.fetch_dataloader(args)
    optimizer, scheduler = fetch_optimizer(args, model)

    logger = Logger(scheduler, args)

    if args.restore_ckpt is not None:
        load_checkpoint(args, model, optimizer, scheduler, logger)

    if args.freeze_bn and args.stage != 'chairs':
        model.module.freeze_bn()

    while logger.total_steps <= args.num_steps:
        train(model, train_loader, optimizer, scheduler, logger, args)
        if logger.total_steps >= args.num_steps:
            plot_train(logger, args)
            plot_val(logger, args)
            break

    PATH = args.output+f'/{args.name}.pth'
    if args.local_rank == 0:
        save_checkpoint(PATH, model, optimizer, scheduler, logger)
    return PATH


def train(model, train_loader, optimizer, scheduler, logger, args):
    # Recreate scaler every epoch.
    scaler = GradScaler(enabled=args.mixed_precision)

    for i_batch, data_blob in enumerate(train_loader):
        tic = time.time()
        # the last element in data_blob is extra_info, which is a list of strings.
        image1, image2, flow, valid = [x.cuda() for x in data_blob[:4]]

        if args.add_noise:
            stdv = np.random.uniform(0.0, 5.0)
            image1 = (image1 + stdv * torch.randn(*image1.shape).cuda()).clamp(0.0, 255.0)
            image2 = (image2 + stdv * torch.randn(*image2.shape).cuda()).clamp(0.0, 255.0)

        optimizer.zero_grad()

        flow_pred = model(image1, image2, iters=args.iters)

        loss, metrics = sequence_loss(flow_pred, flow, valid, args.gamma)
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)

        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        scaler.step(optimizer)
        if scheduler is not None:
            scheduler.step()
        scaler.update()
        toc = time.time()

        metrics['time'] = toc - tic
        # metrics is a dict, with keys: 'epe', '1px', '3px', '5px', 'time'.
        logger.push(metrics)

        # Validate
        if logger.total_steps % args.val_freq == args.val_freq - 1:
            PATH = args.output + f'/{logger.total_steps+1}_{args.name}.pth'
            if args.local_rank == 0:
                save_checkpoint(PATH, model, optimizer, scheduler, logger)
                validate(model, args, logger)
                plot_train(logger, args)
                plot_val(logger, args)

        if logger.total_steps >= args.num_steps:
            break


def validate(model, args, logger):
    model.eval()
    results = {}

    # Evaluate results
    for val_dataset in args.validation:
        if val_dataset == 'chairs':
            results.update(evaluate.validate_chairs(model.module, args.iters))
        elif val_dataset == 'sintel':
            results.update(evaluate.validate_sintel(model.module, args.iters))
        elif val_dataset == 'kitti':
            results.update(evaluate.validate_kitti(model.module, args.iters))
        elif val_dataset == 'viper':
            results.update(evaluate.validate_viper(model.module, args.iters))

    # Record results in logger
    for key in results.keys():
        if key not in logger.val_results_dict.keys():
            logger.val_results_dict[key] = []
        logger.val_results_dict[key].append(results[key])

    logger.val_steps_list.append(logger.total_steps)
    
    model.train()
    if args.freeze_bn and args.stage != 'chairs':
        model.module.freeze_bn()

def plot_val(logger, args):
    for key in logger.val_results_dict.keys():
        # plot validation curve
        plt.figure()
        plt.plot(logger.val_steps_list, logger.val_results_dict[key])
        plt.xlabel('x_steps')
        plt.ylabel(key)
        plt.title(f'Results for {key} for the validation set')
        plt.savefig(args.output+f"/{key}.png", bbox_inches='tight')
        plt.close()


def plot_train(logger, args):
    # plot training curve
    plt.figure()
    plt.plot(logger.train_steps_list, logger.train_epe_list)
    plt.xlabel('x_steps')
    plt.ylabel('EPE')
    plt.title('Running training error (EPE)')
    plt.savefig(args.output+"/train_epe.png", bbox_inches='tight')
    plt.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default='craft', help="name your experiment")
    parser.add_argument('--stage', help="determines which dataset to use for training")

    parser.add_argument('--craft', dest='craft', action='store_true', 
                        help='use craft (Cross-Attentional Flow Transformer)')
    parser.add_argument('--setrans', dest='setrans', action='store_true', 
                        help='use setrans (Squeeze-Expansion Transformer) as the intra-frame attention')
    parser.add_argument('--raft', action='store_true', help='use raft')
    parser.add_argument('--nogma', action='store_true', help='(ablation) Do not use GMA')

    parser.add_argument('--validation', type=str, nargs='+')
    parser.add_argument('--restore_ckpt', help="restore checkpoint")
    parser.add_argument('--loadopt',   dest='load_optimizer_state', action='store_true', 
                        help='Do not load optimizer state from checkpoint (default: not load)')
    parser.add_argument('--loadsched', dest='load_scheduler_state', action='store_true', 
                        help='Load scheduler state from checkpoint (default: not load)')
    
    parser.add_argument('--output', type=str, default='checkpoints', 
                        help='output directory to save checkpoints and plots')
    parser.add_argument('--radius', dest='corr_radius', type=int, default=4)    

    parser.add_argument('--lr', type=float, default=0.00002)
    parser.add_argument('--num_steps', type=int, default=100000)
    parser.add_argument('--batch_size', type=int, default=6)
    parser.add_argument('--workers', dest='num_workers', type=int, default=4)

    parser.add_argument('--image_size', type=int, nargs='+', default=[384, 512])
    parser.add_argument('--gpus', type=int, nargs='+', default=[0, 1])

    parser.add_argument('--mixed_precision', default=False, action='store_true', help='use mixed precision')
    parser.add_argument('--wdecay', type=float, default=.00005)
    parser.add_argument('--epsilon', type=float, default=1e-8)
    parser.add_argument('--clip', type=float, default=1.0)
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--upsample-learn', action='store_true', default=False,
                        help='If True, use learned upsampling, otherwise, use bilinear upsampling.')
    parser.add_argument('--gamma', type=float, default=0.8, help='exponential loss weighting of the sequential predictions')
    parser.add_argument('--add_noise', action='store_true')
    # default: not freeze bn.
    parser.add_argument('--freeze_bn', action='store_true')
    
    parser.add_argument('--iters', type=int, default=12)
    
    parser.add_argument('--val_freq', type=int, default=10000,
                        help='validation frequency')
    parser.add_argument('--print_freq', type=int, default=100,
                        help='printing frequency')

    parser.add_argument('--model_name', default='', help='specify model name')

    parser.add_argument('--position_only', default=False, action='store_true',
                        help='(GMA) only use position-wise attention')
    parser.add_argument('--position_and_content', default=False, action='store_true',
                        help='(GMA) use position and content-wise attention')
    parser.add_argument('--num_heads', default=1, type=int,
                        help='(GMA) number of heads in attention and aggregation')
    parser.add_argument('--posr', dest='pos_bias_radius', type=int, default=7, 
                        help='The radius of positional biases')

    # f1trans is for ablation only, not suggested.                    
    parser.add_argument('--f1', dest='f1trans', type=str, 
                        choices=['none', 'full', 'half'], default='none',
                        help='Whether to use transformer on frame 1 features. '
                             'Half: do self-attention only on half of the channels')                         
    parser.add_argument('--f2', dest='f2trans', type=str, 
                        choices=['none', 'full', 'half'], default='none',
                        help='Whether to use transformer on frame 2 features. '
                             'Half: do self-attention only on half of the channels')                        

    parser.add_argument('--f2posw', dest='f2_pos_code_weight', type=float, default=0.5)
    parser.add_argument('--f2radius', dest='f2_attn_mask_radius', type=int, default=-1)
 
    parser.add_argument('--intermodes', dest='inter_num_modes', type=int, default=4, 
                        help='Number of modes in inter-frame attention')
    parser.add_argument('--intramodes', dest='intra_num_modes', type=int, default=4, 
                        help='Number of modes in intra-frame attention')
    # In inter-frame attention, having QK biases performs slightly better.
    parser.add_argument('--interqknobias', dest='inter_qk_have_bias', action='store_false', 
                        help='Do not use biases in the QK projections in the inter-frame attention')
                        
    parser.add_argument('--interpos', dest='inter_pos_code_type', type=str, 
                        choices=['lsinu', 'bias'], default='bias')
    parser.add_argument('--interposw', dest='inter_pos_code_weight', type=float, default=0.5)
    parser.add_argument('--intrapos', dest='intra_pos_code_type', type=str, 
                        choices=['lsinu', 'bias'], default='bias')
    parser.add_argument('--intraposw', dest='intra_pos_code_weight', type=float, default=1.0)

    # args = parser.parse_args() # raise unrecognized argument 'setrans' error
    args, _ = parser.parse_known_args()

    torch.manual_seed(1234)
    np.random.seed(1234)

    args.local_rank = int(os.environ.get('LOCAL_RANK', 0))

    if args.local_rank == 0 and not os.path.isdir(args.output):
        os.makedirs(args.output)

    timestamp = datetime.now().strftime("%m%d%H%M")
    print0("Time: {}".format(timestamp))
    print0("Args:\n{}".format(args))
    main(args)
