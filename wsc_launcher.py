from pathlib import Path
import logging
import os
import uuid
import subprocess
import sys
import random
import time
import json
import math
import numpy as np
import argparse

from main import *

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
# from torch.optim.lr_scheduler import StepLR
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
# import torchvision.datasets as datasets
# import torchvision.models as modelds
# from torch.utils.data import Subset

import tensorboard_logger as tb_logger

from utils import gather_from_all
# from datasets import build_dataset
# from memory import build_mem
# from itertools import permutations

from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, WFDataset_lab
from models.model_GPT import GPTConfig, Multi_GPT, Projector
from ddp_utils import knn_monitor
from data_aug.wf_data_augs import Crop

logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
_logger = logging.getLogger('train')

parser = argparse.ArgumentParser(description='SimCLR Training')
parser.add_argument('--data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='imagenet', choices=['imagenet', 'cifar100'],
                    help='dataset (imagenet, cifar100)')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=800, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=4096, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--learning-rate', default=4.8, type=float, metavar='LR',
                    help='base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=10, type=int, metavar='N',
                    help='print frequency')
parser.add_argument('--save-freq', default=10, type=int, metavar='N',
                    help='save frequency')
parser.add_argument('--topk-path', type=str, default='./imagenet_resnet50_top10.pkl',
                    help='path to topk predictions from pre-trained classifier')
parser.add_argument('--checkpoint-dir', type=Path,
                    metavar='DIR', help='path to checkpoint directory')
parser.add_argument('--log-dir', type=Path,
                    metavar='LOGDIR', help='path to tensorboard log directory')
parser.add_argument('--rotation', default=0.0, type=float,
                    help="coefficient of rotation loss")
parser.add_argument('--scale', default='0.05,0.14', type=str)
parser.add_argument("--seed", default=42, type=int,
                    help="seed")

# Training / loss specific parameters
parser.add_argument('--temp', default=0.2, type=float,
                    help='Temperature for InfoNCE loss')
parser.add_argument('--mask-mode', type=str, default='',
                    help='Masking mode (masking out only positives, masking out all others than the topk classes',
                    choices=['pos', 'supcon', 'supcon_all', 'topk', 'topk_sum', 'topk_agg_sum', 'weight_anchor_logits', 'weight_class_logits'])
parser.add_argument('--topk', default=5, type=int, metavar='K',
                    help='Top k classes to use')
parser.add_argument('--topk-only-first', action='store_true', default=False,
                    help='Whether to only use the first block of anchors')
parser.add_argument('--memory-bank', action='store_true', default=False,
                    help='Whether to use memory bank')
parser.add_argument('--mem-size', default=100000, type=int,
                    help='Size of memory bank')
parser.add_argument('--opt-momentum', default=0.9, type=float,
                    help='Momentum for optimizer')
parser.add_argument('--optimizer', default='adam', type=str,
                    help='Optimizer', choices=['lars', 'sgd', 'adam'])

# Transform
parser.add_argument('--weak-aug', action='store_true', default=False,
                    help='Whether to use augmentation reguarlization (strong & weak augmentation)')

# Slurm setting
parser.add_argument('--ngpus-per-node', default=6, type=int, metavar='N',
                    help='number of gpus per node')
parser.add_argument('--nodes', default=5, type=int, metavar='N',
                    help='number of nodes')
parser.add_argument("--timeout", default=360, type=int,
                    help="Duration of the job")
parser.add_argument("--partition", default="el8", type=str,
                    help="Partition where to submit")

parser.add_argument("--exp", default="SimCLR", type=str,
                    help="Name of experiment")

# new params
parser.add_argument('--out_dim', default=5, type=int,
                        help='feature dimension (default: 5)')
parser.add_argument('--proj_dim', default=5, type=int,
                    help='projection dimension (default: 5)')

parser.add_argument('--use_gpt', action='store_true') # default = False
parser.add_argument('-ns', '--noise_scale', default=1.0,
                        help='how much to scale the noise augmentation (default: 1)')
   
# GPT args
parser.add_argument('--n_layer', default=20, type=int)
parser.add_argument('--n_head', default=4, type=int)
parser.add_argument('--n_embd', default=64, type=int)
parser.add_argument('--is_causal', action='store_true') # default = False
# parser.add_argument('--block_size', default=2678, type=int) # this is the max sequence length
parser.add_argument('--block_size', default=121, type=int) # this is the max sequence length
parser.add_argument('--pos_enc', default ='seq_11times', type=str)    
parser.add_argument('--multi_chan', default=False, action='store_true')

parser.add_argument('--dropout', default=0.2, type=float)
parser.add_argument('--bias', action='store_true') # default = False
parser.add_argument('--vocab_size', default=50304, type=int) # default to GPT-2 vocab size
parser.add_argument('--online_head', action='store_true') # default = False
parser.add_argument('--ddp', action='store_true', default=True) 
parser.add_argument('--num_extra_chans', default=0, type=int)
parser.add_argument('--knn-freq', default=100, type=int, metavar='N',
                        help='save frequency')
parser.add_argument('--add_train', action='store_true') # default = False
parser.add_argument('--use_chan_pos', action='store_true') # default = False
# wsc setting

parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://10.3.1.9:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')
parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")



best_acc = 0
best_vote_acc = 0

class Trainer(object):
    def __init__(self, args):
        self.args = args

    def __call__(self):
        import main
        self._setup_gpu_args()
        main.main_worker(self.args.gpu, self.args)

    def checkpoint(self):
        import os
        import submitit

        self.args.dist_url = get_init_file(self.args).as_uri()
        checkpoint_file = os.path.join(self.args.checkpoint_dir, "checkpoint.pth")
        if os.path.exists(checkpoint_file):
            self.args.resume = checkpoint_file
        print("Requeuing ", self.args)
        empty_trainer = type(self)(self.args)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        import submitit

        job_env = submitit.JobEnvironment()
        self.args.gpu = job_env.local_rank
        self.args.rank = job_env.global_rank
        self.args.world_size = job_env.num_tasks
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")


def get_init_file(args):
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(args.job_dir, exist_ok=True)
    init_file = args.job_dir / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file


def main_worker(gpu, ngpus_per_node, args):
    if args.seed is not None:
        fix_seed(args.seed)

    global best_acc
    global best_vote_acc

    print(args)
    print('main_worker')
    args.gpu = gpu


    if args.rank == 0:
        args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        stats_file = open(args.checkpoint_dir / 'stats.txt', 'a', buffering=1)
        print(' '.join(sys.argv))
        print(' '.join(sys.argv), file=stats_file)

        logger = tb_logger.Logger(logdir=args.log_dir, flush_secs=2)

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
            print(f'dist url: {args.dist_url}')
            print(f'rank: {args.rank}')
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True


    model = SimCLR(args).cuda(gpu)
    model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])

    if args.optimizer == 'lars':
        optimizer = LARS(model.parameters(), lr=0, weight_decay=args.weight_decay,
                        weight_decay_filter=exclude_bias_and_norm,
                        lars_adaptation_filter=exclude_bias_and_norm)
    elif args.optimizer == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=0,
                                    momentum=args.opt_momentum, weight_decay=args.weight_decay)
    elif args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.learning_rate, weight_decay=args.weight_decay)

    # build memory bank and its loss
    mem_bank = None

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])

        if args.memory_bank:
            mem_bank.load_state_dict(ckpt['mem_bank'])
    else:
        start_epoch = 0
    
    num_extra_chans = args.num_extra_chans if args.multi_chan else 0
    ds = ContrastiveLearningDataset(args.data, args.out_dim, multi_chan=args.multi_chan, use_chan_pos=args.use_chan_pos)
    dataset = ds.get_dataset('wfs', 2, args.noise_scale, num_extra_chans)
  
    sampler = torch.utils.data.distributed.DistributedSampler(dataset, drop_last=True)
    assert args.batch_size % args.world_size == 0
    per_device_batch_size = args.batch_size // args.world_size
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=per_device_batch_size, num_workers=args.workers,
        pin_memory=True, sampler=sampler)
    
    if args.rank == 0:
        if args.multi_chan:
            memory_dataset = WFDataset_lab(args.data, split='train', multi_chan=args.multi_chan, transform=Crop(prob=0.0, num_extra_chans=num_extra_chans, ignore_chan_num=True), use_chan_pos=args.use_chan_pos)
            memory_loader = torch.utils.data.DataLoader(
                memory_dataset, batch_size=128, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)
            test_dataset = WFDataset_lab(args.data, split='test', multi_chan=args.multi_chan, transform=Crop(prob=0.0, num_extra_chans=num_extra_chans, ignore_chan_num=True), use_chan_pos=args.use_chan_pos)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)
        else:
            memory_dataset = WFDataset_lab(args.data, split='train', multi_chan=False)
            memory_loader = torch.utils.data.DataLoader(
                memory_dataset, batch_size=128, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)
            test_dataset = WFDataset_lab(args.data, split='test', multi_chan=False)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)

    start_time = time.time()
    scaler = torch.cuda.amp.GradScaler()

    # test knn first
    if args.rank == 0:
        knn_score = knn_monitor(net=model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
        print(f"knn_acc:{knn_score}") 

    for epoch in range(start_epoch, args.epochs):
        sampler.set_epoch(epoch)
        model.train()

        for step, (wf, labels) in enumerate(loader, start=epoch * len(loader)):
            labels = labels[0].long()
            if args.use_chan_pos:
                y1 = wf[0][0].float()
                y2 = wf[1][0].float()
                chan_pos = wf[0][1].float()
                chan_pos2 = wf[1][1].float()
     
            else:
                y1 = wf[0].float()
                y2 = wf[1].float()
                chan_pos = None
                chan_pos2 = None
                
            if not args.multi_chan:
                y1, y2 = torch.squeeze(y1, dim=1), torch.squeeze(y2, dim=1)
                y1, y2 = torch.unsqueeze(y1, dim=-1), torch.unsqueeze(y2, dim=-1)
            else:
                y1, y2 = y1.view(-1, (args.num_extra_chans*2+1)*121), y2.view(-1, (args.num_extra_chans*2+1)*121)
                y1, y2 = torch.unsqueeze(y1, dim=-1), torch.unsqueeze(y2, dim=-1)
            y1 = y1.cuda(gpu, non_blocking=True)
            y2 = y2.cuda(gpu, non_blocking=True)
            if args.use_chan_pos:
                chan_pos = chan_pos.cuda(gpu, non_blocking=True)
                chan_pos2 = chan_pos2.cuda(gpu, non_blocking=True)

            labels = labels.cuda(gpu, non_blocking=True)
            if args.optimizer != 'adam':
                lr = adjust_learning_rate(args, optimizer, loader, step)
            else:
                lr = args.learning_rate
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast():
                loss, acc = model.forward(y1, y2, labels, chan_pos=chan_pos, chan_pos2=chan_pos2)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            if step % args.print_freq == 0:
                torch.distributed.reduce(acc.div_(args.world_size), 0)
                if args.rank == 0:
                    print(f'epoch={epoch}, step={step}, loss={loss.item()}, acc={acc.item()}', flush=True)
                    stats = dict(epoch=epoch, step=step, learning_rate=lr,
                                 loss=loss.item(), acc=acc.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats), file=stats_file)

        if args.rank == 0:
            # save checkpoint
            if args.memory_bank:
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                            optimizer=optimizer.state_dict(), mem_bank=mem_bank.state_dict())
            else:
                state = dict(epoch=epoch + 1, model=model.state_dict(),
                            optimizer=optimizer.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

            # save checkpoint to epoch
            if epoch % args.save_freq == 0 and epoch != 0:
                torch.save(state, args.checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch))

            # log to tensorboard
            logger.log_value('loss', loss.item(), epoch)
            logger.log_value('acc', acc.item(), epoch)
            logger.log_value('learning_rate', lr, epoch)

            if epoch % args.knn_freq == 0:
                knn_score = knn_monitor(net=model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
                print(f"Epoch {epoch}, my knn_acc:{knn_score}")  
                logger.log_value('knn_acc', knn_score, epoch)

    if args.rank == 0:
        # save final model
        torch.save(dict(backbone=model.module.backbone.state_dict(),
                        projector=model.module.projector.state_dict(),
                        head=model.module.online_head.state_dict()),
                args.checkpoint_dir / 'resnet50.pth')


def main():
    args = parser.parse_args()
    # args.scale = [float(x) for x in args.scale.split(',')]

    args.checkpoint_dir = args.checkpoint_dir / args.exp
    args.log_dir = args.log_dir / args.exp
    args.job_dir = args.checkpoint_dir

    args.checkpoint_dir.mkdir(parents=True, exist_ok=True)
    args.log_dir.mkdir(parents=True, exist_ok=True)

    get_init_file(args)

    num_gpus_per_node = args.ngpus_per_node
    nodes = args.nodes
    timeout_min = args.timeout
    partition = args.partition

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    if args.multiprocessing_distributed:
        # Since we have ngpus_per_node processes per node, the total world_size
        # needs to be adjusted accordingly
        args.world_size = ngpus_per_node * args.world_size
        # Use torch.multiprocessing.spawn to launch distributed processes: the
        # main_worker process function
        print('launching distributed processes')
        print(f'world_size: {args.world_size}')
        mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
    else:
        # Simply call main_worker function
        main_worker(args.gpu, ngpus_per_node, args)

    _logger.info("Submitted job_id:", job.job_id)


if __name__ == '__main__':
    main()