import logging
import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist
from ddp_utils import knn_monitor
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, WFDataset_lab
from main import SimCLR
import argparse
import torch.nn.functional as F
from data_aug.wf_data_augs import Crop

def main(args):
    print("starting knn eval")
    # args.checkpoint_dir = os.path.join(args.checkpoint_dir, args.exp)
    # args.log_dir = os.path.join(args.log_dir, args.exp)
    
    # os.makedirs(args.checkpoint_dir, exist_ok=True)
    # os.makedirs(args.log_dir, exist_ok=True)
    
    main_worker(0, args)
    
def main_worker(gpu, args):
    # args.rank += gpu
    
    if args.ddp:
        raise NotImplementedError("DDP not implemented")
    
    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True


    # dataset = ContrastiveLearningDataset(args.data, args.out_dim, multi_chan=args.multi_chan, no_collide=args.no_collide)
    # dataset = ContrastiveLearningDataset(args.data, args.out_dim, multi_chan=args.multi_chan)

    # train_dataset = dataset.get_dataset('wfs', 2, args.noise_scale)
    
    # sampler = None
    
    # train_loader = torch.utils.data.DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True,
    #     num_workers=args.workers, pin_memory=True, drop_last=True)

    # define memory and test dataset for knn monitoring
    if not args.ddp:
        if args.multi_chan:
            memory_dataset = WFDataset_lab(args.data, split='train', multi_chan=args.multi_chan, transform=Crop(prob=0.0, num_extra_chans=5, ignore_chan_num=True), use_chan_pos=args.use_chan_pos)
            memory_loader = torch.utils.data.DataLoader(
                memory_dataset, batch_size=128, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)
            test_dataset = WFDataset_lab(args.data, split='test', multi_chan=args.multi_chan, transform=Crop(prob=0.0, num_extra_chans=5, ignore_chan_num=True), use_chan_pos=args.use_chan_pos)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)
        else:
            memory_dataset = WFDataset_lab(args.data, split='train', multi_chan=False, use_chan_pos=args.use_chan_pos)
            memory_loader = torch.utils.data.DataLoader(
                memory_dataset, batch_size=128, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)
            test_dataset = WFDataset_lab(args.data, split='test', multi_chan=False, use_chan_pos=args.use_chan_pos)
            test_loader = torch.utils.data.DataLoader(
                test_dataset, batch_size=args.batch_size, shuffle=False,
                num_workers=args.workers, pin_memory=True, drop_last=False)
        
        # memory_dataset = WFDataset_lab(args.data, split='train', multi_chan=args.multi_chan, transform=Crop(prob=0.0, num_extra_chans=5, ignore_chan_num=True))
        # memory_loader = torch.utils.data.DataLoader(
        #     memory_dataset, batch_size=args.batch_size, shuffle=False,
        #     num_workers=args.workers, pin_memory=True, drop_last=False)
        
        # test_dataset = WFDataset_lab(args.data, split='test', multi_chan=args.multi_chan, transform=Crop(prob=0.0, num_extra_chans=5, ignore_chan_num=True))
        # test_loader = torch.utils.data.DataLoader(
        #     test_dataset, batch_size=args.batch_size, shuffle=False,
        #     num_workers=args.workers, pin_memory=True, drop_last=False)
    else:
        memory_loader = None
        test_loader = None
    
    model = SimCLR(args).cuda(gpu)
    # print("current model")
    # print(model.state_dict().keys())

    if not os.path.exists(args.checkpoint_dir):
        raise ValueError("Checkpoint not found")
    else:
        print("loading from previous checkpoint: ", args.checkpoint_dir)
        
        ckpt = torch.load(args.checkpoint_dir,
                            map_location='cpu')
        if args.multi_chan:
            start_epoch = ckpt['epoch']
            nonddp_state_dict = {k.replace('module.', ''): v for k, v in ckpt['model'].items()}
            model.load_state_dict(nonddp_state_dict)

        else:
            start_epoch = ckpt['epoch']
            
            nonddp_state_dict = {'backbone.'+k: v for k,v in ckpt['state_dict'].items() if 'projector' not in k}
            nonddp_state_dict.update({k: v for k,v in ckpt['state_dict'].items() if 'projector' in k})
            m, uek = model.load_state_dict(nonddp_state_dict, strict=False)
            
        # print("from file")
        # print(nonddp_state_dict.keys())
        

    knn_score = knn_monitor(net=model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=args)
    print(f"Epoch {start_epoch}, knn_acc:{knn_score}")
      
    # save representations
    ckpt_root_dir = '/'.join(args.checkpoint_dir.split('/')[:-1])
    # model_name = args.checkpoint_dir.split('/')[-2]
    model.eval()
    feature_bank = []
    with torch.no_grad():
        for data, target in test_loader:
            if not args.multi_chan:
                data = torch.squeeze(data, dim=1)
                data = torch.unsqueeze(data, dim=-1)
            else:
                if args.use_chan_pos:
                    data, chan_pos = data
                data = data.view(-1, 11*121)
                data = torch.unsqueeze(data, dim=-1)
            if args.use_chan_pos:
                feature = model(data.cuda(gpu, non_blocking=True), chan_pos=chan_pos.cuda(gpu, non_blocking=True))
            else:
                feature = model(data.cuda(gpu, non_blocking=True))
            feature_bank.append(feature)
            
        feature_bank = torch.cat(feature_bank, dim=0)
        print(feature_bank.shape)
        
        torch.save(feature_bank, os.path.join(ckpt_root_dir, 'test_reps2.pt'))
        print(f"saved test features to {ckpt_root_dir}/test_reps2.pt")
    
    feature_bank = []
    with torch.no_grad():
        for data, target in memory_loader:
            if not args.multi_chan:
                data = torch.squeeze(data, dim=1)
                data = torch.unsqueeze(data, dim=-1)
            else:
                if args.use_chan_pos:
                    data, chan_pos = data
                data = data.view(-1, 11*121)
                data = torch.unsqueeze(data, dim=-1)
            if args.use_chan_pos:
                feature = model(data.cuda(gpu, non_blocking=True), chan_pos=chan_pos.cuda(gpu, non_blocking=True))
            else:
                feature = model(data.cuda(gpu, non_blocking=True))
            feature_bank.append(feature)
            
        feature_bank = torch.cat(feature_bank, dim=0)
        print(feature_bank.shape)
        
        torch.save(feature_bank, os.path.join(ckpt_root_dir, 'train_reps2.pt'))
        print(f"saved train features to {ckpt_root_dir}/train_reps2.pt")

def make_sh_and_submit(args):
    os.makedirs('./scripts/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    options = args.arg_str
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
    parser.add_argument('--data', type=Path, metavar='DIR', default= '/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_data/multi_dy016_random_neurons_04_28_2023',
                        help='path to dataset')
    parser.add_argument('-dataset-name', default='wfs',
                        help='dataset name', choices=['wfs', 'stl10', 'cifar10'])
    parser.add_argument('--optimizer', default='adam', choices = ['adam', 'sgd'],
                        help='optimizer')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='custom_encoder',
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
    parser.add_argument('--eval_knn_every_n_epochs', default=1, type=int)
    
    parser.add_argument('--use_gpt', action='store_true') # default = False
    
    parser.add_argument('--n_layer', default=20, type=int)
    parser.add_argument('--n_head', default=4, type=int)
    parser.add_argument('--n_embd', default=64, type=int)
    parser.add_argument('--is_causal', action='store_true') # default = False
    # parser.add_argument('--block_size', default=2678, type=int) # this is the max sequence length
    parser.add_argument('--block_size', default=1331, type=int) # this is the max sequence length
    
    parser.add_argument('--dropout', default=0.2, type=float)
    parser.add_argument('--bias', action='store_true') # default = False
    parser.add_argument('--vocab_size', default=50304, type=int) # default to GPT-2 vocab size
    parser.add_argument('--online_head', action='store_true') # default = False
    parser.add_argument('--pos_enc', default ='conseq', type=str)    
    parser.add_argument('--no_collide', action='store_true') # default = False
    parser.add_argument('--ignore_proj', action='store_true') # default = False
    parser.add_argument('--use_chan_pos', action='store_true') # default = False
    parser.add_argument('--use_merge_layer', action='store_true') # default = False
    parser.add_argument('--add_layernorm', action='store_true') # default = False
    parser.add_argument('--num_extra_chans', default=0, type=int)

    parser.add_argument('--half_embed_each', action='store_true') # default = False
  
    args = parser.parse_args()
    
    if args.submit:
        make_sh_and_submit(args)
    else:
        main(args)

"""
python knn_eval.py --checkpoint-dir=/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_ddp/saved_models/mc_gpt_posseq_causal_nembd64_block1331_bs132_lr0.001/checkpoint_epoch20.pth --multi_chan --is_causal --batch-size=128
python knn_eval.py --checkpoint-dir=/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_ddp/saved_models/mc_gpt_posseq_causal_nembd32_block1331_bs132_lr0.001/checkpoint.pth --multi_chan --is_causal --batch-size=128 --n_embd=32
Epoch 800, my knn_acc:65.13333333333333
Epoch 791, my knn_acc:65.3
Epoch 701, my knn_acc:62.133333333333326


New runs:
python knn_eval.py --checkpoint-dir=/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_ddp/saved_models/0501_mc_gpt_conseq_causal_nembd64_block1331_bs128_extra5_lr0.0001/checkpoint.pth --multi_chan --is_causal --batch-size=128 --n_embd=64
python knn_eval.py --checkpoint-dir=/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_ddp/saved_models/0501_mc_gpt_conseq_causal_nembd64_block1331_bs128_extra5_lr0.001/checkpoint.pth --multi_chan --is_causal --batch-size=128 --n_embd=64
python knn_eval.py --checkpoint-dir=/gpfs/u/home/BNSS/BNSSlhch/scratch/spike_ddp/saved_models/0502_mc_gpt_conseq_causal_nembd64_block1331_bs128_extra5_lr0.0001_knn10_addtrain/checkpoint.pth --multi_chan --is_causal --batch-size=128 --n_embd=64

python knn_eval.py --checkpoint-dir=./ddp_models/0502_mc_gpt_conseq_causal_nembd64_block1331_bs128_extra5_lr0.0001_knn10_addtrain/checkpoint.pth --multi_chan --is_causal --batch-size=128 --n_embd=64 --pos_enc=conseq --data=/home/gridsan/cloh/spike_data/multi_dy016_random_neurons_04_28_2023

python knn_eval.py --checkpoint-dir=/gpfs/wscgpfs02/shivsr/cloh/spike_contrastive/saved_models/0510_mc_conseq_causal_n64_b1331_bs120_extra5_lr0.0005_poschan_mergelayer_layernorm/checkpoint.pth --multi_chan --is_causal --batch-size=128 --n_embd=64 --pos_enc=conseq --data=/home/gridsan/cloh/spike_data/multi_dy016_random_neurons_04_28_2023

python knn_eval.py --out_dim=128 --proj_dim=5 --batch-size=512 --lr=0.001 --epochs=800 --fp16 --use_gpt --is_causal --n_embd=32 --dropout=0.0 --data=/home/gridsan/evanv/charlotte/spike_data/single_mearec_random_neurons_05_10_2023 --checkpoint-dir=/home/gridsan/evanv/charlotte/spike_contrastive/runs/0511out_dim128proj_dim5batch-size512lr0.001epochs800fp16use_gptis_causaln_embd32add_traindropout0.0/checkpoint.pth 

"""