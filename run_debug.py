import argparse
import numpy as np
import os 
import sys
import subprocess
import random

import torch
import torch.backends.cudnn as cudnn

from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, WFDataset_lab
from models.model_simclr import ModelSimCLR, Projector, Projector2
from simclr_debug import SimCLR
from models.model_GPT import GPTConfig, Single_GPT
from torch.nn.parallel import DistributedDataParallel as DDP


# def main(args):
#     assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
#     # check if gpu training is available
#     args.ngpus_per_node = torch.cuda.device_count()
#     assert args.ngpus_per_node > 0, "Only GPU training is currently supported. Please run with at least 1 GPU."
#     if 'SLURM_JOB_ID' in os.environ:
#         cmd = 'scontrol show hostnames ' + os.getenv('SLURM_JOB_NODELIST')
#         stdout = subprocess.check_output(cmd.split())
#         host_name = stdout.decode().splitlines()[0]
#         args.rank = int(os.getenv('SLURM_NODEID')) * args.ngpus_per_node
#         args.world_size = int(os.getenv('SLURM_NNODES')) * args.ngpus_per_node
#         args.dist_url = f'tcp://{host_name}:58478'
#     elif not args.disable_cuda and args.multi_chan:
#         # single-node distributed training
#         args.rank = 0
#         args.world_size = args.ngpus_per_node
#         args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
#     # elif not args.disable_cuda and torch.cuda.is_available():
#         # args.device = torch.device('cuda')
#         cudnn.deterministic = True
#         cudnn.benchmark = True
#     # else:
#     #     args.device = torch.device('cpu')
#     #     args.gpu_index = -1
#     torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

# the above does not allow for one-node one-gpu training. (device_count would count all gpus on the node but doesnt mean these are allocated)
# consider reinstating the above if single-node multi-gpu training is needed
def main(args):
    print("starting main loop")
    args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp)
    args.log_dir = os.path.join(args.log_dir, args.exp)
    
    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    main_worker(0, args)
    
def main_worker(gpu, args):
    # args.rank += gpu
    
    if args.ddp:
        torch.distributed.init_process_group(
            backend='nccl', init_method=args.dist_url,
            world_size=args.world_size, rank=args.rank)
    
    # torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True
    # torch.backends.cudnn.deterministic = True

    # tr_dataset = WFDataset_lab(args.data, split='train')
    # te_dataset = WFDataset_lab(args.data, split='test')
    
    num_extra_chans = args.num_extra_chans if args.multi_chan else 0
    
    dataset = ContrastiveLearningDataset(args.data, args.out_dim, multi_chan=args.multi_chan)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, args.noise_scale, num_extra_chans)
    print("ddp:", args.ddp)
    
    if args.ddp:
        sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, drop_last=True)
        assert args.batch_size % args.world_size == 0
        per_device_batch_size = args.batch_size // args.world_size

        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=per_device_batch_size, drop_last=True,
            num_workers=args.workers, pin_memory=True, sampler=sampler)
    else:
        sampler = None
        
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)

    # define memory and test dataset for knn monitoring
    if not args.ddp:
        memory_dataset = WFDataset_lab(args.data, split='train', multi_chan=args.multi_chan)
        memory_loader = torch.utils.data.DataLoader(
            memory_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)
        
        test_dataset = WFDataset_lab(args.data, split='test', multi_chan=args.multi_chan)
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True, drop_last=False)
    else:
        memory_loader = None
        test_loader = None
           
    if args.use_gpt:
        model_args = dict(n_layer=args.n_layer, n_head=args.n_head, n_embd=args.n_embd, block_size=args.block_size,
                  bias=args.bias, vocab_size=args.vocab_size, dropout=args.dropout, out_dim=args.out_dim, is_causal=args.is_causal, 
                  proj_dim=args.proj_dim, pos=args.pos_enc, multi_chan=args.multi_chan) 
        gptconf = GPTConfig(**model_args)
        model = Single_GPT(gptconf).cuda(gpu)
    else:
        model = ModelSimCLR(base_model=args.arch, out_dim=args.out_dim, proj_dim=args.proj_dim, \
            fc_depth=args.fc_depth, expand_dim=args.expand_dim, multichan=args.multi_chan).cuda(gpu)

    if not args.no_proj:
        if args.arch == 'custom_encoder':
            proj = Projector(rep_dim=args.out_dim, proj_dim=args.proj_dim)
        elif args.arch == 'custom_encoder2':
            proj = Projector2(rep_dim=args.out_dim, proj_dim=args.proj_dim)
        elif args.arch == 'fc_encoder':
            proj = Projector(rep_dim=args.out_dim, proj_dim=args.proj_dim)
    else:
        proj = None
    
    # for n, p in model.named_parameters():
    #     print(n, p.numel())
    # print("number of encoder params: ", sum(p.numel() for p in self.backbone.parameters()))
    print("number of transfomer params: ", sum(p.numel() for n,p in model.named_parameters() if 'transformer' in n))
    print("number of fcpart params: ", sum(p.numel() for n,p in model.named_parameters() if ('lm_head' in n and 'proj' not in n)))
    print("number of Proj params: ", sum(p.numel() for n,p in model.named_parameters() if ('proj' in n)))
    print("number of online classifier params: ", sum(p.numel() for n,p in model.named_parameters() if 'online_head' in n))
    
    # moved this out from simclr.py since model needs to be pushed to gpu before defining optimizers else loading optimizer dict will be an issue
    if args.ddp:
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = DDP(model, device_ids=[gpu])
        
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = None
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs * len(train_loader)), eta_min=0,
                                                           last_epoch=-1)

    print("model and optimizer initialized!")
        # automatically resume from checkpoint if it exists
    if os.path.exists(os.path.join(args.checkpoint_dir, "checkpoint.pth")):
        print("loading from previous checkpoint: ", args.checkpoint_dir)
        ckpt = torch.load(os.path.join(args.checkpoint_dir, "checkpoint.pth"),
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])

    else:
        start_epoch = 0
        
    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    # with torch.cuda.device(args.gpu_index):
    print("starting SimCLR..")
    
    simclr = SimCLR(model=model, proj=proj, optimizer=optimizer, scheduler=scheduler, gpu=gpu, 
                    sampler=sampler, args=args, start_epoch=start_epoch)
    simclr.train(train_loader, memory_loader, test_loader)

def make_sh_and_submit(args):
    os.makedirs('./scripts/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    options = args.arg_str
    if '--data' in options:
        name = ''.join([opt1.replace("--","").replace("=","") for opt1 in options.split(" ")[:-2]])
        name = args.add_prefix + name
        if 'spike_data' in name:
            assert "data needs to be the last argument"
    else:
        name = ''.join([opt1.replace("--","").replace("=","") for opt1 in options.split(" ")])
        name = args.add_prefix + name
    import getpass
    username = getpass.getuser()
    preamble = (
        f'#!/bin/sh\n#SBATCH --cpus-per-task=20\n#SBATCH --gres=gpu:volta:1\n#SBATCH '
        f'-o ./logs/{name}.out\n#SBATCH '
        f'--job-name={name}\n#SBATCH '
        f'--open-mode=append\n\n'
    )
    with open(f'./scripts/{name}.sh', 'w') as file:
        file.write(preamble)
        file.write("echo \"current time: $(date)\";\n")
        file.write(
            f'python {sys.argv[0]} '
            f'{options} --exp={name} '
        )
        # if args.server == 'sc' or args.server == 'rumensc':
            # file.write(f'--data_root=/home/gridsan/{username}/MAML-Soljacic/cifar_stl_data/ ')
        # file.write(f'--data_root=/home/gridsan/groups/MAML-Soljacic/cifar_stl_data/ ')
    print('Submitting the job with options: ')
    print(options)

    os.system(f'sbatch ./scripts/{name}.sh')

        
    
if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='PyTorch SimCLR')
    parser.add_argument('--data', metavar='DIR', default='/home/gridsan/cloh/spike_contrastive/datasets/single_dy016_random_neurons_04_28_2023/',
                        help='path to dataset')
    parser.add_argument('-dataset-name', default='wfs',
                        help='dataset name', choices=['wfs', 'stl10', 'cifar10'])
    parser.add_argument('--optimizer', default='adam', choices = ['adam', 'sgd'],
                        help='optimizer')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='attention',
                        help='default: custom_encoder)')
    parser.add_argument('-ns', '--noise_scale', default=1.0,
                        help='how much to scale the noise augmentation (default: 1)')
    parser.add_argument('-j', '--workers', default=12, type=int, metavar='N',
                        help='number of data loading workers (default: 32)')
    parser.add_argument('--epochs', default=300, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=1024, type=int,
                        metavar='N',
                        help='mini-batch size (default: 256), this is the total '
                            'batch size of all GPUs on the current node when '
                            'using Data Parallel or Distributed Data Parallel')
    parser.add_argument('--lr', '--learning-rate', default=0.001, type=float,
                        metavar='LR', help='initial learning rate', dest='lr')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--seed', default=None, type=int,
                        help='seed for initializing training. ')
    parser.add_argument('--disable-cuda', action='store_true',
                        help='Disable CUDA')
    parser.add_argument('--fp16', action='store_true',
                        help='Whether or not to use 16-bit precision GPU training.')
    parser.add_argument('--out_dim', default=5, type=int,
                        help='feature dimension (default: 2)')
    parser.add_argument('--proj_dim', default=5, type=int,
                        help='projection dimension (default: 5)')
    parser.add_argument('--log-every-n-steps', default=100, type=int,
                        help='Log every n steps')
    parser.add_argument('--temperature', default=0.07, type=float,
                        help='softmax temperature (default: 0.07)')
    parser.add_argument('--n-views', default=2, type=int, metavar='N',
                        help='Number of views for contrastive learning training.')
    parser.add_argument('--gpu-index', default=0, type=int, help='Gpu index.')
    parser.add_argument('--exp', default='test', type=str)
    parser.add_argument('--fc_depth', default=2, type=int)
    parser.add_argument('--submit', action='store_true')
    parser.add_argument('--arg_str', default='--', type=str)
    parser.add_argument('--add_prefix', default='', type=str)
    parser.add_argument('--no_proj', default=True, action='store_true')
    parser.add_argument('--expand_dim', default=16, type=int)
    parser.add_argument('--multi_chan', default=False, action='store_true')
    parser.add_argument('--n_channels', default=11, type=int)

    parser.add_argument('--checkpoint-dir', default='./runs', type=str) # can define type as PATH as well
    parser.add_argument('--log-dir', default='./logs/', type=str) # can define type as PATH as well
    parser.add_argument('--ddp', action='store_true')
    parser.add_argument('--rank', default=0, type=int)
    parser.add_argument('--world_size', default=2, type=int)
    parser.add_argument('--eval_knn_every_n_epochs', default=1, type=int)
    
    parser.add_argument('--use_gpt', action='store_true') # default = False
    
    parser.add_argument('--n_layer', default=20, type=int)
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--n_embd', default=64, type=int)
    parser.add_argument('--is_causal', action='store_true') # default = False
    # parser.add_argument('--block_size', default=2678, type=int) # this is the max sequence length
    parser.add_argument('--block_size', default=121, type=int) # this is the max sequence length
    
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--bias', action='store_true') # default = False
    parser.add_argument('--vocab_size', default=50304, type=int) # default to GPT-2 vocab size
    parser.add_argument('--online_head', action='store_true') # default = False
    parser.add_argument('--pos_enc', default ='seq_11times', type=str)    
    parser.add_argument('--no_collide', action='store_true') # default = False
    parser.add_argument('--num_extra_chans', default=0, type=int)
    parser.add_argument('--add_train', action='store_true') # default = False
    parser.add_argument('--use_chan_pos', action='store_true') # default = False
  
    args = parser.parse_args()
    
    print("MULTICHAN", args.multi_chan)
    
    if args.submit:
        make_sh_and_submit(args)
    else:
        main(args)
