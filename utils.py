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


def get_contr_representations(model, data_set):
    reps = []
    model = model.double()
    for item in data_set:
        with torch.no_grad():
            rep = model(torch.from_numpy(item.reshape(1, 1, -1)).double())
        reps.append(rep.numpy())
    
    return np.squeeze(np.array(reps))

def knn_pca_score(latent_dim, root_path):
    pca = PCA(latent_dim)

    # create labels vectors
    labels_train = np.array([[i for j in range(1200)] \
                                for i in range(5)]).reshape(-1)
    labels_test = np.array([[i for j in range(300)] \
                               for i in range(5)]).reshape(-1)

    train_data = np.load(os.path.join(root_path, 'spikes_train.npy'))
    test_data = np.load(os.path.join(root_path, 'spikes_test.npy'))
    
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


def validation(model, latent_dim, root_path):
    # create labels vectors
    labels_train = np.array([[i for j in range(1200)] \
                                for i in range(5)]).reshape(-1)
    labels_test = np.array([[i for j in range(300)] \
                               for i in range(5)]).reshape(-1)

    train_data = np.load(os.path.join(root_path, 'spikes_train.npy'))
    test_data = np.load(os.path.join(root_path, 'spikes_test.npy'))

    # get model without projection head
    enc = get_backbone(model)

    # get detached representations
    contr_train_reps = get_contr_representations(enc, train_data)
    contr_test_reps = get_contr_representations(enc, test_data)

    # Train and fit KNN classifier
    knn = KNeighborsClassifier(n_neighbors = 10)
    knn.fit(contr_train_reps, labels_train)
    contr_score = knn.score(contr_test_reps, labels_test)*100
    print("n-components: {} - Contrastive reps classifier accuracy: {:.2f}%".format(latent_dim, contr_score) )
    
    return contr_score