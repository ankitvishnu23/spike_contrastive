import logging
import os
import sys

import torch
import torch.nn.functional as F
import torch.multiprocessing as mp
import torch.distributed as dist

from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import (
    save_config_file, validation, 
    save_checkpoint, knn_pca_score, knn_monitor,
    gather_from_all
)

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.gpu = kwargs['gpu']
        # self.model = kwargs['model'].double().cuda(self.args.device)
        self.model = kwargs['model'].double().cuda(self.gpu)
        if self.args.ddp:
            self.model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(self.model)
            self.model = DDP(self.model, device_ids=[self.gpu], find_unused_parameters=True)
        # self.model = kwargs['model'].cuda(self.args.device)
        self.proj = kwargs['proj'].cuda(kwargs['gpu']) if kwargs['proj'] is not None else None
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        # self.writer = SummaryWriter('./logs/'+self.args.exp)
        self.writer = SummaryWriter(self.args.log_dir+self.args.exp)
        
        self.multichan = self.args.multi_chan
        if self.args.rank == 0 or not self.args.ddp:
            logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.gpu)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.cuda(self.gpu)
        features = gather_from_all(features)
        features = torch.squeeze(features)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(self.gpu)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda(self.gpu)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, memory_loader=None, test_loader=None):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        # save_config_file('./runs-args', self.args)

        n_iter = 0
        if self.args.rank == 0 or not self.args.ddp:    
            logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
            logging.info(f"Training with gpu: {not self.args.disable_cuda}.")

        # pca_score = knn_pca_score(self.args.out_dim, self.args.data)
        pca_score = 0

        for epoch_counter in range(self.args.epochs):
            print('Epoch {}'.format(epoch_counter))
            for wf in tqdm(train_loader):
                wf = torch.cat(wf, dim=0)
                wf = torch.squeeze(wf)
                if not self.multichan:
                    wf = torch.unsqueeze(wf, dim=1)

                wf = wf.double().cuda(self.gpu)
                # wf = wf.float().cuda(self.args.device)
                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(wf)
                    if self.proj is not None:
                        features = self.proj(features)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                # knn_score = knn_monitor(net=self.model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True)
                # if n_iter % 10 == 0:
                #     knn_score = knn_monitor(net=self.model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True)
                #     print(f"loss: {loss}, knn_acc:{knn_score}")
                if n_iter % self.args.log_every_n_steps == 0:
                    if self.args.rank == 0 or not self.args.ddp:
                        knn_score = validation(self.model, self.args.out_dim, self.args.data, self.gpu)
                        print(f"loss: {loss}, knn_acc:{knn_score}")
                    
                n_iter += 1

                # warmup for the first 10 epochs
                if epoch_counter >= 10 and self.scheduler != None:
                    self.scheduler.step()   
                    
            if self.args.rank == 0 or not self.args.ddp:
                logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")
                self.writer.add_scalar('loss', loss, epoch_counter)
                # self.writer.add_scalar('pca_knn_score', pca_score, global_step=n_iter)
                self.writer.add_scalar('knn_score', knn_score, epoch_counter)
                curr_lr = self.optimizer.param_groups[0]['lr'] if self.scheduler == None else self.scheduler.get_lr()[0]
                self.writer.add_scalar('learning_rate', curr_lr, epoch_counter)

        if self.args.rank == 0 or not self.args.ddp:
            logging.info("Training has finished.")
            # save model checkpoints
            checkpoint_name = self.args.exp + '_checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.args.checkpoint_dir, checkpoint_name))
            # }, is_best=False, filename=os.path.join('./runs', checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {'./runs'}.")
