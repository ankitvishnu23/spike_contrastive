import os
import shutil

import torch
import torch.nn as nn
import numpy as np
from pathlib import Path
from tqdm.auto import tqdm, trange
import pandas as pd
import shutil
from sklearn.decomposition import PCA

from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV

import matplotlib.pyplot as plt
import yaml
import torch.nn.functional as F

import sys

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
    
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def save_config_file(model_checkpoints_folder, args):
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        with open(os.path.join(model_checkpoints_folder, 'config.yml'), 'w') as outfile:
            yaml.dump(args, outfile, default_flow_style=False)


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


def get_backbone(enc):
    last_layer = list(list(enc.children())[-1].children())[:-1]
    enc.fcpart = nn.Sequential(*last_layer)
    return enc


def get_contr_representations(model, data_set, device):
    reps = []
    model = model.double()
    for item in data_set:
        with torch.no_grad():
            wf = torch.from_numpy(item.reshape(1, 1, -1)).double().to(device)
            rep = model(wf)
        reps.append(rep.detach().cpu().numpy())
    
    return np.squeeze(np.array(reps))


def knn_pca_score(latent_dim, root_path):
    pca = PCA(latent_dim)

    train_data = np.load(os.path.join(root_path, 'spikes_train.npy'))
    test_data = np.load(os.path.join(root_path, 'spikes_test.npy'))

    num_classes = train_data.shape[0] // 1200

    # create labels vectors
    labels_train = np.array([[i for j in range(1200)] \
                                for i in range(num_classes)]).reshape(-1)
    labels_test = np.array([[i for j in range(300)] \
                               for i in range(num_classes)]).reshape(-1)
    
    # fit PCA to raw train
    pca_train = pca.fit_transform(train_data)

    # transform raw test data via learned PCA
    pca_test = pca.transform(test_data)

    # score on PCA represented data
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(pca_train, labels_train)
    pca_score = knn.score(pca_test, labels_test)*100
    print("n-components: {} - PCA classifier accuracy : {:.2f}%".format(latent_dim,pca_score) )

    return pca_score



def validation(model, latent_dim, root_path, device):
    # create labels vectors
    train_data = np.load(os.path.join(root_path, 'spikes_train.npy'))
    test_data = np.load(os.path.join(root_path, 'spikes_test.npy'))

    num_classes = train_data.shape[0] // 1200

    # create labels vectors
    labels_train = np.array([[i for j in range(1200)] \
                                for i in range(num_classes)]).reshape(-1)
    labels_test = np.array([[i for j in range(300)] \
                               for i in range(num_classes)]).reshape(-1)

    # get model without projection head
    enc = get_backbone(model)

    # get detached representations
    contr_train_reps = get_contr_representations(enc, train_data, device)
    contr_test_reps = get_contr_representations(enc, test_data, device)

    # Train and fit KNN classifier
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(contr_train_reps, labels_train)
    contr_score = knn.score(contr_test_reps, labels_test)*100
    print("latent dim: {} - Contrastive reps classifier accuracy: {:.2f}%".format(latent_dim, contr_score) )
    
    return contr_score

# knn monitor as in InstDisc https://arxiv.org/abs/1805.01978
# implementation follows http://github.com/zhirongw/lemniscate.pytorch and https://github.com/leftthomas/SimCLR
def knn_predict(feature, feature_bank, feature_labels, classes, knn_k, knn_t):
    # compute cos similarity between each feature vector and feature bank ---> [B, N]
    sim_matrix = torch.mm(feature, feature_bank)
    # [B, K]
    sim_weight, sim_indices = sim_matrix.topk(k=knn_k, dim=-1)
    # [B, K]
    sim_labels = torch.gather(feature_labels.expand(feature.size(0), -1), dim=-1, index=sim_indices)
    sim_weight = (sim_weight / knn_t).exp()

    # counts for each class
    one_hot_label = torch.zeros(feature.size(0) * knn_k, classes, device=sim_labels.device)
    # [B*K, C]
    one_hot_label = one_hot_label.scatter(dim=-1, index=sim_labels.view(-1, 1), value=1.0)
    # weighted score ---> [B, C]
    pred_scores = torch.sum(one_hot_label.view(feature.size(0), -1, classes) * sim_weight.unsqueeze(dim=-1), dim=1)

    pred_labels = pred_scores.argsort(dim=-1, descending=True)
    return pred_labels

# test using a knn monitor
def knn_monitor(net, memory_data_loader, test_data_loader, device='cuda', k=200, t=0.1, hide_progress=False,
                targets=None):
    if not targets:
        targets = memory_data_loader.dataset.targets

    net.eval()
    classes = 100
    # classes = len(memory_data_loader.dataset.classes)
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        # generate feature bank
        for data, target in memory_data_loader:
            feature = net(data.to(device=device, non_blocking=True).unsqueeze(dim=1))
            feature = F.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device)
        
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            feature = net(data.unsqueeze(dim=1))
            feature = F.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100

