import os
import sys
import argparse
import time
import math
import pickle

import torch
import torchvision
from torchvision import transforms
from tqdm import tqdm
from pathlib import Path

from datasets import IdxDataset

parser = argparse.ArgumentParser(description='Collecting top k predictions from pre-trained classifier')
parser.add_argument('--data-folder', type=Path, metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', type=str, default='cifar100',
                    help='dataset type (cifar100, imagenet)')
parser.add_argument('--save-folder', type=str, default='./',
                    help='dataset type (cifar100, imagenet)')
parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
parser.add_argument('--batch-size', default=256, type=int, metavar='N',
                    help='mini-batch size')


def get_top5(opt):
    # Load and return if top5_dict exists
    top5_file_path = os.path.join(opt.save_folder, opt.dataset + '_top10.pkl')
    logits_file_path = os.path.join(opt.save_folder, opt.dataset + '_logits.pkl')

    # construct data loader
    if opt.dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif opt.dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif opt.dataset == 'imagenet':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
    elif opt.dataset == 'path':
        mean = eval(opt.mean)
        std = eval(opt.std)
    else:
        raise ValueError('dataset not supported: {}'.format(opt.dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    if opt.dataset == 'imagenet':
        train_transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    train_dataset = IdxDataset(opt.dataset, opt.data_folder, transform=train_transform, train=True)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=opt.batch_size, shuffle=False,
        num_workers=opt.num_workers, pin_memory=True)

    # load pre-trained model for top5 prediction
    if opt.dataset == 'cifar100':
        model = torch.hub.load("chenyaofo/pytorch-cifar-models", "cifar100_resnet56", pretrained=True)
    elif opt.dataset == 'imagenet':
        model = torchvision.models.resnet50(pretrained=True) #torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=True)
    model.eval()

    top5_dict = {}
    logits_dict = {}
    
    # building a dict of index => top 5 predicted labels without the ground truth
    with torch.no_grad():
        for idx, (images, labels, idxs) in enumerate(tqdm(train_loader)):
            outputs = model(images)
            preds_top5 = torch.topk(outputs, 10)[1]
            
            for i in range(len(preds_top5)):
                top5_nogt = preds_top5[i][labels[i]!=preds_top5[i]]
                top5_dict[int(idxs[i])] = top5_nogt.cpu()

                logits_dict[int(idxs[i])] = outputs[i].cpu()


    # save
    with open(top5_file_path, 'wb') as f:
        pickle.dump(top5_dict, f)
    
    with open(logits_file_path, 'wb') as f:
        pickle.dump(logits_dict, f)
    
    return

if __name__ == '__main__':
    args = parser.parse_args()
    get_top5(args)

