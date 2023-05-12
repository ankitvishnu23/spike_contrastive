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
    save_checkpoint, knn_pca_score, 
    gather_from_all
)
from ddp_utils import knn_monitor
import tensorboard_logger as tb_logger
torch.manual_seed(0)
import time
from load_models import save_reps

from ddp_utils import knn_predict
import torch
def knn_monitor2(net, memory_data_loader, test_data_loader, device='cuda', k=200, t=0.1, hide_progress=False,
                targets=None, multi_chan=False):
    if not targets:
        targets = memory_data_loader.dataset.targets

    net.eval()
    classes = 100
    total_top1, total_top5, total_num, feature_bank = 0.0, 0.0, 0, []
    with torch.no_grad():
        for data, target in memory_data_loader:
            if not multi_chan:
                data = torch.squeeze(data, dim=1)
                data = torch.unsqueeze(data, dim=-1)
            else:
                data = data.view(-1, 11*121)
                data = torch.unsqueeze(data, dim=-1)
            
            feature = net(data.to(device=device, non_blocking=True))
            
            feature = torch.nn.functional.normalize(feature, dim=1)
            feature_bank.append(feature)
        # [D, N]
        feature_bank = torch.cat(feature_bank, dim=0).t().contiguous()
        # [N]
        feature_labels = torch.tensor(targets, device=feature_bank.device)
        
        # loop test data to predict the label by weighted knn search
        for data, target in test_data_loader:
            
            data, target = data.to(device=device, non_blocking=True), target.to(device=device, non_blocking=True)
            
            if not multi_chan:
                data = torch.squeeze(data, dim=1)
                data = torch.unsqueeze(data, dim=-1)
            else:
                data = data.view(-1, 11*121)
                data = torch.unsqueeze(data, dim=-1)
            
            feature = net(data)
            
            feature = torch.nn.functional.normalize(feature, dim=1)

            pred_labels = knn_predict(feature, feature_bank, feature_labels, classes, k, t)

            total_num += data.size(0)
            total_top1 += (pred_labels[:, 0] == target).float().sum().item()
    return total_top1 / total_num * 100

class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.gpu = kwargs['gpu']
        self.sampler = kwargs['sampler']
        # self.model = kwargs['model'].double().cuda(self.args.device)
        # self.model = kwargs['model'].double().to(self.gpu)
        
        self.model =  kwargs['model']
        # self.model = kwargs['model'].cuda(self.args.device)
        self.proj = kwargs['proj'].cuda(kwargs['gpu']) if kwargs['proj'] is not None else None 
        if self.proj and self.args.ddp:
            raise "proj needs to be wrapped in ddp"
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        # self.writer = SummaryWriter('./logs/'+self.args.exp)
        if self.args.rank == 0 or not self.args.ddp:
            # self.writer = SummaryWriter(os.path.join(self.args.log_dir,self.args.exp))
            self.logger = tb_logger.Logger(logdir=self.args.log_dir, flush_secs=2)
        self.multichan = self.args.multi_chan
        if self.args.rank == 0 or not self.args.ddp:
            logging.basicConfig(filename=os.path.join(self.args.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().cuda(self.gpu)
        self.start_epoch = kwargs['start_epoch']

    def info_nce_loss(self, features):

        if self.args.ddp:
            features = gather_from_all(features)
        features = torch.squeeze(features)
        features = F.normalize(features, dim=1)
        batch_dim = int(features.shape[0] // 2)

        labels = torch.cat([torch.arange(batch_dim) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        # labels = labels.to(self.gpu)
        labels = labels.cuda()

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        # mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.gpu)
        mask = torch.eye(labels.shape[0], dtype=torch.bool).cuda(non_blocking=True)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        # labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.gpu)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader, memory_loader=None, test_loader=None):

        scaler = GradScaler(enabled=self.args.fp16)

        # save config file
        # save_config_file('./runs-args', self.args)

        n_iter = 0
        if self.args.rank == 0 or not self.args.ddp:
            logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
            logging.info(f"Training with gpu: {not self.args.disable_cuda}.")
            print(f"Start SimCLR training for {self.args.epochs} epochs, starting at {self.start_epoch}.")

        # pca_score = knn_pca_score(self.args.out_dim, self.args.data)
        pca_score = 0

        for epoch_counter in range(self.start_epoch, self.args.epochs):
            if self.args.add_train:
                self.model.train()
            start_time = time.time()
            if self.args.ddp:
                self.sampler.set_epoch(epoch_counter)
            print('Epoch {}'.format(epoch_counter))
            # time4 = time.time()
            for i, (wf, lab) in enumerate(train_loader):
                # print(f"batch {i}")
                chan_pos = None
                if self.args.use_chan_pos:
                    wf, chan_pos = wf
                    chan_pos = torch.cat(chan_pos, dim=0).float()
                wf = torch.cat(wf, dim=0).float()
                lab = torch.cat(lab, dim=0).long().cuda(self.gpu,non_blocking=True)
                # wf = torch.squeeze(wf)
                # if not self.multichan:
                #     wf = torch.unsqueeze(wf, dim=1)
                if self.args.use_gpt:
                    if not self.args.multi_chan:
                        wf = torch.squeeze(wf, dim=1)
                        wf = torch.unsqueeze(wf, dim=-1)
                    else:
                        wf = wf.view(-1, (self.args.num_extra_chans*2+1)*121)
                        wf = torch.unsqueeze(wf, dim=-1)
                wf = wf.cuda(self.gpu,non_blocking=True)
                # wf = wf.float().cuda(self.args.device)
                # time1 = time.time()
                # print("time for loading batch:", time1 - time4)
                
                with autocast(enabled=self.args.fp16):
                    if self.args.online_head:
                        features, cls_loss, online_acc = self.model(wf, lab, chan_pos=chan_pos)
                    else:
                        features = self.model(wf, chan_pos=chan_pos)
                        cls_loss = 0.
                        online_acc = -1
                    if self.proj is not None:
                        raise "projector should be defined in model! "
                        features = self.proj(features)
                    # time2 = time.time()
                    # print("time for fwd:", time2-time1)
                    
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels) + cls_loss
                    # time3 = time.time()
                    # print("time for backward:", time3-time2)
                    
                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()
                
                # time4 = time.time()
                # print("time for optimizer step:", time4-time3)
                
                # for i, param in enumerate(self.model.parameters()):
                #     if i == 0:
                #         print(param.dtype)
                # knn_score = knn_monitor(net=self.model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True)
                # if n_iter % 10 == 0:
                
                # if n_iter % self.args.log_every_n_steps == 0:
                
                n_iter += 1

                # warmup for the first 10 epochs
                if epoch_counter >= 10 and self.scheduler != None:
                    self.scheduler.step()   
            
            if epoch_counter % self.args.eval_knn_every_n_epochs == 0 and epoch_counter != 0:
                if self.args.rank == 0 or not self.args.ddp:
                    # knn_score = validation(self.model, self.args.out_dim, self.args.data, self.gpu)
                    # print(f"loss: {loss}, knn_acc:{knn_score}")
                    knn_score = knn_monitor(net=self.model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, args=self.args)
                    print(f"loss: {loss}, my knn_acc:{knn_score}")
                    self.logger.log_value('knn_score', knn_score, epoch_counter)
                    knn_score = knn_monitor2(net=self.model, memory_data_loader=memory_loader, test_data_loader=test_loader, device='cuda',k=200, hide_progress=True, multi_chan=False)
                    print(f"knn2 acc:{knn_score}")
                    
                    save_dict = {
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'optimizer': self.optimizer.state_dict(),
                    'state_dict': self.model.state_dict()
                    }
                    
                    save_checkpoint(save_dict, is_best=False, filename=os.path.join(self.args.checkpoint_dir, 'checkpoint.pth'))
                    print(f"Model checkpoint and metadata has been saved at {self.args.checkpoint_dir}.")
                    logging.info(f"Model checkpoint and metadata has been saved at {self.args.checkpoint_dir}.")

                    print("now running from load checkpoints")
                    from load_models import load_ckpt, get_dataloader, save_reps
                    single_data_path='/home/gridsan/evanv/charlotte/spike_data/single_mearec_random_neurons_05_10_2023'
                    single_ckpt_path = f'./runs/debugknn/checkpoint.pth'
                    model = load_ckpt(single_ckpt_path, multi_chan = False, rep_after_proj=True, rep_dim=128, proj_dim=5, dropout=0.0)
                    single_train_loader = get_dataloader(single_data_path, multi_chan=False, split='train')
                    single_test_loader = get_dataloader(single_data_path, multi_chan=False, split='test')
                    knn_score = knn_monitor2(net=model.cuda(), memory_data_loader=single_train_loader, test_data_loader=single_test_loader, device='cuda',k=200, hide_progress=True, multi_chan=False)
                    print("Single channel; knn rep AFTER proj:", knn_score)
                    
            if self.args.rank == 0 or not self.args.ddp:
                logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")
                self.logger.log_value('loss', loss, epoch_counter)
                # self.writer.add_scalar('loss', loss, epoch_counter)
                # self.writer.add_scalar('pca_knn_score', pca_score, global_step=n_iter)
                # self.writer.add_scalar('knn_score', knn_score, epoch_counter)
                
                curr_lr = self.optimizer.param_groups[0]['lr'] if self.scheduler == None else self.scheduler.get_lr()[0]
                # self.writer.add_scalar('learning_rate', curr_lr, epoch_counter)
                self.logger.log_value('learning_rate', curr_lr, epoch_counter)
                if online_acc != -1:
                    self.logger.log_value('online_acc', online_acc, epoch_counter)
                    print("loss: ", loss.item(), "online acc: ", online_acc)
            
            print(f"time for epoch {epoch_counter}: {time.time()-start_time}")
            if self.args.rank == 0 or not self.args.ddp:
                # save model checkpoints
                save_dict = {
                    'epoch': epoch_counter,
                    'arch': self.args.arch,
                    'optimizer': self.optimizer.state_dict(),
                    'state_dict': self.model.state_dict()
                    }
                    
                save_checkpoint(save_dict, is_best=False, filename=os.path.join(self.args.checkpoint_dir, 'checkpoint.pth'))
                print(f"Model checkpoint and metadata has been saved at {self.args.checkpoint_dir}.")
                logging.info(f"Model checkpoint and metadata has been saved at {self.args.checkpoint_dir}.")

        if self.args.rank == 0 or not self.args.ddp:
            logging.info("Training has finished.")
            # save model checkpoints
            # checkpoint_name = self.args.exp + '_checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
            save_checkpoint({
                'epoch': self.args.epochs,
                'arch': self.args.arch,
                'state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
            }, is_best=False, filename=os.path.join(self.args.checkpoint_dir, 'final.pth'))
            # }, is_best=False, filename=os.path.join('./runs', checkpoint_name))
            logging.info(f"Model checkpoint and metadata has been saved at {self.args.checkpoint_dir}.")
        