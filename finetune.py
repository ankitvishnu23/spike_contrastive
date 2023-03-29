import argparse
import torch
import torch.backends.cudnn as cudnn
from torchvision import models
from data_aug.contrastive_learning_dataset import ContrastiveLearningDataset, WFDataset_lab
from models.model_simclr import ModelSimCLR, Projector, Projector2
from simclr import SimCLR
import numpy as np
import os 
import sys
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils import save_checkpoint, AverageMeter, accuracy 

def validate(model, test_loader, writer, epoch, args, best_acc):

    # evaluate
    model.eval()
    top1 = AverageMeter('Acc@1')
    top5 = AverageMeter('Acc@5')

    num_classes = 10

    correct_per_class = torch.zeros(num_classes, device='cuda')
    total_per_class = torch.zeros(num_classes, device='cuda')

    print(total_per_class.shape)
    print(len(test_loader.dataset))
    # only in ensemble freeze eval mode or single model freeze eval mode, calculate ECE
   
    with torch.no_grad():
        for wf, target in tqdm(test_loader):
            target = target.cuda(non_blocking=True)
            output = model(wf.unsqueeze(dim=1).cuda(non_blocking=True))
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            top5.update(acc5[0].item(), wf.size(0))
            top1.update(acc1[0].item(), wf.size(0))

            # per class accuracy
            _, preds = output.max(1)
            correct_vec = (preds == target) # if each prediction is correct or not
            ind_per_class = (target.unsqueeze(1) == torch.arange(num_classes, device='cuda')) # indicator variable for each class
            correct_per_class += (correct_vec.unsqueeze(1) * ind_per_class).sum(0)
            total_per_class += ind_per_class.sum(0)

    # sanity check that the sum of total per class amounts to the whole dataset
    assert total_per_class.sum() == len(test_loader.dataset)
    acc_per_class = correct_per_class / total_per_class

    is_best = False
    if best_acc.top1 < top1.avg:
        best_acc.top1 = top1.avg
        best_acc.top5 = top5.avg
        is_best = True
        torch.save(acc_per_class.cpu(), os.path.join('./ft_runs', args.exp + 'best_acc_per_class.pth'))
        

    writer.add_scalar('Test Acc1', top1.avg, epoch)
    writer.add_scalar('Test Acc5', top5.avg, epoch)

    return acc_per_class, is_best

def main(args):
    assert args.n_views == 2, "Only two view training is supported. Please use --n-views 2."
    # check if gpu training is available
    if not args.disable_cuda and torch.cuda.is_available():
        args.device = torch.device('cuda')
        cudnn.deterministic = True
        cudnn.benchmark = True
    else:
        args.device = torch.device('cpu')
        args.gpu_index = -1

    tr_dataset = WFDataset_lab(args.data, split='train')
    te_dataset = WFDataset_lab(args.data, split='test')
    
    dataset = ContrastiveLearningDataset(args.data, args.out_dim)

    train_dataset = dataset.get_dataset(args.dataset_name, args.n_views, args.noise_scale)
    if args.finetune:
        train_loader = torch.utils.data.DataLoader(
            tr_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True, drop_last=True)
        

    # define memory and test dataset for knn monitoring
    memory_dataset = WFDataset_lab(args.data, split='train')
    memory_loader = torch.utils.data.DataLoader(
        memory_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    test_dataset = WFDataset_lab(args.data, split='test')
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    if args.finetune:
        model = ModelSimCLR(base_model=args.arch, out_dim=args.out_dim, proj_dim=args.proj_dim, \
            fc_depth=args.fc_depth, expand_dim=args.expand_dim, cls_head=args.cls_head)
        ckpt = torch.load('runs/'+args.pt_ckpt+'_checkpoint_0500.pth.tar', map_location="cpu")
        k1, k2 = model.load_state_dict(ckpt['state_dict'], strict=False)
        print(k1, k2)
        
        
    else:
        model = ModelSimCLR(base_model=args.arch, out_dim=args.out_dim, proj_dim=args.proj_dim, \
        fc_depth=args.fc_depth, expand_dim=args.expand_dim)

    if not args.no_proj:
        if args.arch == 'custom_encoder':
            proj = Projector(rep_dim=args.out_dim, proj_dim=args.proj_dim)
        elif args.arch == 'custom_encoder2':
            proj = Projector2(rep_dim=args.out_dim, proj_dim=args.proj_dim)
        elif args.arch == 'fc_encoder':
            proj = Projector(rep_dim=args.out_dim, proj_dim=args.proj_dim)
    else:
        proj = None
    if args.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = None
    else:
        optimizer = torch.optim.SGD(model.parameters(), args.lr, weight_decay=args.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=(args.epochs * len(train_loader)), eta_min=0,
                                                           last_epoch=-1)

    #  It’s a no-op if the 'gpu_index' argument is a negative integer or None.
    if args.finetune:
        os.makedirs('./ft_logs/', exist_ok=True)
        os.makedirs('./ft_runs/', exist_ok=True)
        
        writer = SummaryWriter('./ft_logs/'+args.exp)
        model.cuda()
        criterion = torch.nn.CrossEntropyLoss().cuda()
        best_acc = argparse.Namespace(top1=0, top5=0)
        
        for ep in range(args.epochs):
            print('Epoch {}'.format(ep))
            model.train()
            total_loss = 0.0
            for i, (wf, target) in enumerate(train_loader):
                wf, target = wf.cuda(non_blocking=True),  target.cuda(non_blocking=True)
                optimizer.zero_grad()
                output = model(wf.unsqueeze(dim=1))
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                if scheduler is not None:
                    scheduler.step()
                total_loss += loss.item()
            print(f'Loss at epoch {ep}: {total_loss/i}')
            writer.add_scalar('loss', total_loss/i, ep)
            curr_lr = optimizer.param_groups[0]['lr'] if scheduler == None else scheduler.get_lr()[0]
            writer.add_scalar('learning_rate', curr_lr, ep)

            # eval loop
            acc_per_class, is_best = validate(model, test_loader, writer, ep, args, best_acc)
            
        torch.save(acc_per_class.cpu(), os.path.join('./ft_runs', args.exp + 'last_acc_per_class.pth'))

        checkpoint_name = args.exp + 'latest.pth.tar'
        save_checkpoint({
            'epoch': args.epochs,
            'arch': args.arch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }, is_best=False, filename=os.path.join('./ft_runs', checkpoint_name))
            
    else:
        with torch.cuda.device(args.gpu_index):
            simclr = SimCLR(model=model, proj=proj, optimizer=optimizer, scheduler=scheduler, args=args)
            simclr.train(train_loader, memory_loader, test_loader)

def make_sh_and_submit(args):
    os.makedirs('./scripts/', exist_ok=True)
    os.makedirs('./logs/', exist_ok=True)
    options = args.arg_str
    name = ''.join([opt1.replace("--","").replace("=","") for opt1 in options.split(" ")])
    name = args.add_prefix + name
    
    import getpass
    username = getpass.getuser()
    preamble = (
        f'#!/bin/sh\n#SBATCH --gres=gpu:volta:1\n#SBATCH --cpus-per-task=20\n#SBATCH '
        f'-o ./logs/{name}.out\n#SBATCH '
        f'--job-name={name}\n#SBATCH '
        f'--open-mode=append\n\n'
    )
    with open(f'./scripts/{name}.sh', 'w') as file:
        file.write(preamble)
        file.write("echo \"current time: $(date)\";\n")
        # if args.finetune:
        #     file.write(
        #     f'python {sys.argv[0]} '
        #     f'{options} '
        #     )
        # else:
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
    parser.add_argument('--data', metavar='DIR', default='/home/gridsan/cloh/spike_contrastive/dy016/',
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
    parser.add_argument('--epochs', default=100, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-b', '--batch-size', default=256, type=int,
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
    parser.add_argument('--fp16-precision', action='store_true',
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
    parser.add_argument('--no_proj', default='True', action='store_true')
    parser.add_argument('--expand_dim', default=16, action='store_true')
    parser.add_argument('--finetune', default=True, action='store_true')
    parser.add_argument('--pt_ckpt', default='', type=str)
    parser.add_argument('--cls_head', default=None, type=str, choices=('linear', 'mlp2', 'mlp3'))

    
    args = parser.parse_args()
    
    if args.submit:
        make_sh_and_submit(args)
    else:
        main(args)
