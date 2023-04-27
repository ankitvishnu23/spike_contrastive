from pathlib import Path
import argparse
import json
import os
import random
import sys
import time
import urllib
import numpy as np

from torch import nn, optim
from torchvision import models, datasets, transforms
import torch
import torchvision
import tensorboard_logger as tb_logger

from utils import consume_prefix_in_state_dict_if_present

from classifier_heads import get_classifier

parser = argparse.ArgumentParser(description='Evaluate resnet50 features on ImageNet')
parser.add_argument('--data', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--pretrained', type=Path, metavar='FILE',
                    help='path to pretrained model')
parser.add_argument('--weights', default='freeze', type=str,
                    choices=('finetune', 'freeze'),
                    help='finetune or freeze resnet weights')
parser.add_argument('--train-percent', default=100, type=int,
                    choices=(100, 10, 1),
                    help='size of traing set in percent')
parser.add_argument('--workers', default=8, type=int, metavar='N',
                    help='number of data loader workers')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')
parser.add_argument('--lr-backbone', default=0.0, type=float, metavar='LR',
                    help='backbone base learning rate')
parser.add_argument('--lr-classifier', default=1.0, type=float, metavar='LR',
                    help='classifier base learning rate')
parser.add_argument('--weight-decay', default=1e-6, type=float, metavar='W',
                    help='weight decay')
parser.add_argument('--print-freq', default=100, type=int, metavar='N',
                    help='print frequency')

# add new args
parser.add_argument("--bayes", action="store_true")
parser.add_argument('--mlp_depth', default=1, type=int,
                    help='number of layers in classifier')
parser.add_argument('--mlp_hidden_dim', default=2048, type=int,
                    help='hidden dim in classifier')
parser.add_argument('--prior_mu', default=0., type=float)
parser.add_argument('--prior_sigma', default=1., type=float)
parser.add_argument('--use_our_sgd', action="store_true")
parser.add_argument('--kl', default=1., type=float)
parser.add_argument('--bayes_samples', default=50, type=int)

# def main():
#     args = parser.parse_args()
#     args.ngpus_per_node = torch.cuda.device_count()

#     # single-node distributed training
#     args.rank = 0
#     args.dist_url = f'tcp://localhost:{random.randrange(49152, 65535)}'
#     args.world_size = args.ngpus_per_node
#     torch.multiprocessing.spawn(main_worker, (args,), args.ngpus_per_node)

class SGDNesterovMomentum(optim.Optimizer):
    def __init__(self, params, lr, weight_decay=0, momentum=0.9):
        defaults = dict(lr=lr, weight_decay=weight_decay, momentum=momentum)
        super().__init__(params, defaults)
        self.momentum = momentum
        self.lr = lr


    @torch.no_grad()
    def step(self):
        for g in self.param_groups:
            for p in g['params']:
                dp = p.grad

                if dp is None:
                    continue

                # if self.weight_decay != 0:
                #     dp = dp.add(p, alpha=self.weight_decay)

                if self.momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(dp).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(self.momentum).add_(dp.mul_(-self.lr), alpha=1)
                    # if self.nesterov:
                    buf.mul_(self.momentum).add_(dp.mul_(-self.lr), alpha=1)
                    dp = buf

                p.add_(dp, alpha=1)

def fix_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main_worker(gpu, args):

    torch.distributed.init_process_group(
        backend='nccl', init_method=args.dist_url,
        world_size=args.world_size, rank=args.rank)

    print(' '.join(sys.argv))

    if args.rank == 0:
        logger = tb_logger.Logger(logdir=args.log_dir, flush_secs=2)

    torch.cuda.set_device(gpu)
    torch.backends.cudnn.benchmark = True

    model = models.resnet50().cuda(gpu)
    # remove fc layer
    model.fc = nn.Identity()

    classifier = get_classifier(rep_dim=2048,bayes=args.bayes,bayes_mode='reparam',ensem=False,mlp_depth=args.mlp_depth, mlp_hidden_dim=args.mlp_hidden_dim,\
                        prior_mu=args.prior_mu,prior_sigma=args.prior_sigma,prior_sigma1=1.,prior_sigma2=0.01,prior_pi=0.5,\
                        ensem_samples=1, mimo=False, add_bn=False, num_classes=1000).cuda(gpu)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        ckpt = torch.load(args.checkpoint_dir / 'checkpoint.pth',
                          map_location='cpu')
        start_epoch = ckpt['epoch']
        model.load_state_dict(ckpt['model'])
        classifier.load_state_dict(ckpt['classifier'])
    else:
        # start_epoch = 0
        #
        # state_dict = torch.load(args.pretrained, map_location='cpu')
        # if 'epoch' in str(args.pretrained):
        #     consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.backbone.')
        #     model.load_state_dict(state_dict['model'], strict=False)
        start_epoch = 0
        state_dict = torch.load(args.pretrained, map_location='cpu')
        if 'ep' in str(args.pretrained):
            if 'state_dict' in state_dict:
                dkey = 'state_dict'
            elif 'model' in state_dict:
                dkey = 'model'
            else:
                raise
            filtered_dict = {k.replace("module.", "").replace("encoder_q.","").replace("backbone.",""): v for k, v in state_dict[dkey].items()}
            # consume_prefix_in_state_dict_if_present(state_dict['model'], 'module.backbone.')
            # model.load_state_dict(state_dict['model'], strict=False)
            missing_keys, unexpected_keys = model.load_state_dict(filtered_dict, strict=False)
            print("missing:", missing_keys)
            print("unexpected:",unexpected_keys)
            # assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []

        else:
            missing_keys, unexpected_keys = model.load_state_dict(state_dict["backbone"], strict=False)
            assert missing_keys == ['fc.weight', 'fc.bias'] and unexpected_keys == []

        if not args.bayes:
            classifier.weight.data.normal_(mean=0.0, std=0.01)
            classifier.bias.data.zero_()

    if args.weights == 'freeze':
        model.requires_grad_(False)
        classifier.requires_grad_(True)

    # classifier_parameters, model_parameters = [], []
    # for name, param in model.named_parameters():
    #     if 'fc' in name:
    #         classifier_parameters.append(param)
    #     else:
    #         model_parameters.append(param)
    model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[gpu])
    classifier = torch.nn.parallel.DistributedDataParallel(classifier, device_ids=[gpu])

    criterion = nn.CrossEntropyLoss().cuda(gpu)

    # param_groups = [dict(params=classifier_parameters, lr=args.lr_classifier)]
    # if args.weights == 'finetune':
    #     param_groups.append(dict(params=model_parameters, lr=args.lr_backbone))
    param_groups = [dict(params=classifier.parameters(), lr=args.lr_classifier)]
    if args.weights == 'finetune':
        param_groups.append(dict(params=model.parameters(), lr=args.lr_backbone))
    if args.use_our_sgd:
        optimizer = SGDNesterovMomentum(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    else:
        optimizer = optim.SGD(param_groups, 0, momentum=0.9, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)

    # automatically resume from checkpoint if it exists
    if (args.checkpoint_dir / 'checkpoint.pth').is_file():
        optimizer.load_state_dict(ckpt['optimizer']) # ckpt defined above

    best_acc = argparse.Namespace(top1=0, top5=0)

    # Data loading code
    traindir = args.data / 'train'
    valdir = args.data / 'val'
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
    val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))

    if args.train_percent in {1, 10}:
        train_dataset.samples = []
        for fname in args.train_files:
            fname = fname.decode().strip()
            cls = fname.split('_')[0]
            train_dataset.samples.append(
                (traindir / cls / fname, train_dataset.class_to_idx[cls]))

    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    kwargs = dict(batch_size=args.batch_size // args.world_size, num_workers=args.workers, pin_memory=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler, **kwargs)
    val_loader = torch.utils.data.DataLoader(val_dataset, **kwargs)

    start_time = time.time()
    for epoch in range(start_epoch, args.epochs):
        # train
        classifier.train()
        if args.weights == 'finetune':
            model.train()
        elif args.weights == 'freeze':
            model.eval()
        else:
            assert False
        train_sampler.set_epoch(epoch)
        for step, (images, target) in enumerate(train_loader, start=epoch * len(train_loader)):
            # output = model(images.cuda(gpu, non_blocking=True))
            reps = model(images.cuda(gpu, non_blocking=True))
            if args.bayes:
                output, Lkl = classifier(reps)
            else:
                output = classifier(reps)
            loss = criterion(output, target.cuda(gpu, non_blocking=True))
            if args.bayes:
                loss += Lkl / output.shape[0] * args.kl

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if step % args.print_freq == 0:
                torch.distributed.reduce(loss.div_(args.world_size), 0)
                if args.rank == 0:
                    pg = optimizer.param_groups
                    lr_classifier = pg[0]['lr']
                    lr_backbone = pg[1]['lr'] if len(pg) == 2 else 0
                    stats = dict(epoch=epoch, step=step, lr_backbone=lr_backbone,
                                 lr_classifier=lr_classifier, loss=loss.item(),
                                 time=int(time.time() - start_time))
                    print(json.dumps(stats))

        # evaluate
        model.eval()
        classifier.eval()
        if args.rank == 0:
            # save checkpoint
            state = dict(epoch=epoch + 1, model=model.module.state_dict(),
                         optimizer=optimizer.state_dict(),
                         classifier=classifier.module.state_dict())
            torch.save(state, args.checkpoint_dir / 'checkpoint.pth')

            # save checkpoint to epoch
            if epoch % args.save_freq == 0 and epoch != 0:
                torch.save(state, args.checkpoint_dir / 'checkpoint_epoch{}.pth'.format(epoch))

            top1 = AverageMeter('Acc@1')
            top5 = AverageMeter('Acc@5')
            with torch.no_grad():
                for images, target in val_loader:
                    reps = model(images.cuda(gpu, non_blocking=True))
                    if not args.bayes:
                        output = classifier(reps)
                    else:
                        outputs = [classifier(reps)[0].softmax(dim=-1) for _ in range(args.bayes_samples)]
                        outputs = torch.stack(outputs)
                        output = torch.mean(outputs,0)

                    acc1, acc5 = accuracy(output, target.cuda(gpu, non_blocking=True), topk=(1, 5))
                    top1.update(acc1[0].item(), images.size(0))
                    top5.update(acc5[0].item(), images.size(0))
            best_acc.top1 = max(best_acc.top1, top1.avg)
            best_acc.top5 = max(best_acc.top5, top5.avg)
            stats = dict(epoch=epoch, acc1=top1.avg, acc5=top5.avg, best_acc1=best_acc.top1, best_acc5=best_acc.top5)
            print(json.dumps(stats))

            logger.log_value('Top 1 Accuracy', best_acc.top1, epoch)
            logger.log_value('Top 5 Accuracy', best_acc.top5, epoch)


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


# if __name__ == '__main__':
#     main()
