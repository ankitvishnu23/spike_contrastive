import logging
import os
import sys

import torch
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from utils import save_config_file, validation, save_checkpoint, knn_pca_score

torch.manual_seed(0)


class SimCLR(object):

    def __init__(self, *args, **kwargs):
        self.args = kwargs['args']
        self.model = kwargs['model'].double().to(self.args.device)
        self.optimizer = kwargs['optimizer']
        self.scheduler = kwargs['scheduler']
        self.writer = SummaryWriter()
        logging.basicConfig(filename=os.path.join(self.writer.log_dir, 'training.log'), level=logging.DEBUG)
        self.criterion = torch.nn.CrossEntropyLoss().to(self.args.device)

    def info_nce_loss(self, features):

        labels = torch.cat([torch.arange(self.args.batch_size) for i in range(self.args.n_views)], dim=0)
        labels = (labels.unsqueeze(0) == labels.unsqueeze(1)).float()
        labels = labels.to(self.args.device)

        features = F.normalize(features, dim=1)

        similarity_matrix = torch.matmul(features, features.T)
        # assert similarity_matrix.shape == (
        #     self.args.n_views * self.args.batch_size, self.args.n_views * self.args.batch_size)
        # assert similarity_matrix.shape == labels.shape

        # discard the main diagonal from both: labels and similarities matrix
        mask = torch.eye(labels.shape[0], dtype=torch.bool).to(self.args.device)
        labels = labels[~mask].view(labels.shape[0], -1)
        similarity_matrix = similarity_matrix[~mask].view(similarity_matrix.shape[0], -1)
        # assert similarity_matrix.shape == labels.shape

        # select and combine multiple positives
        positives = similarity_matrix[labels.bool()].view(labels.shape[0], -1)

        # select only the negatives the negatives
        negatives = similarity_matrix[~labels.bool()].view(similarity_matrix.shape[0], -1)

        logits = torch.cat([positives, negatives], dim=1)
        labels = torch.zeros(logits.shape[0], dtype=torch.long).to(self.args.device)

        logits = logits / self.args.temperature
        return logits, labels

    def train(self, train_loader):

        scaler = GradScaler(enabled=self.args.fp16_precision)

        # save config file
        save_config_file(self.writer.log_dir, self.args)

        n_iter = 0
        logging.info(f"Start SimCLR training for {self.args.epochs} epochs.")
        logging.info(f"Training with gpu: {self.args.disable_cuda}.")

        pca_score = knn_pca_score(self.args.out_dim, self.args.data)

        for epoch_counter in range(self.args.epochs):
            print('Epoch {}'.format(epoch_counter))
            for wf in tqdm(train_loader):
                wf = torch.cat(wf, dim=0)
                wf = torch.squeeze(wf)
                wf = torch.unsqueeze(wf, dim=1)

                wf = wf.double().to(self.args.device)

                with autocast(enabled=self.args.fp16_precision):
                    features = self.model(wf)
                    print(features.shape)
                    logits, labels = self.info_nce_loss(features)
                    loss = self.criterion(logits, labels)

                self.optimizer.zero_grad()

                scaler.scale(loss).backward()

                scaler.step(self.optimizer)
                scaler.update()

                if n_iter % self.args.log_every_n_steps == 0:
                    contr_score = validation(self.model, self.args.out_dim, self.args.data, self.args.device)
                    self.writer.add_scalar('loss', loss, global_step=n_iter)
                    self.writer.add_scalar('pca_knn_score', pca_score, global_step=n_iter)
                    self.writer.add_scalar('contr_knn_score', contr_score, global_step=n_iter)
                    curr_lr = self.optimizer.param_groups[0]['lr'] if self.scheduler == None else self.scheduler.get_lr()[0]
                    self.writer.add_scalar('learning_rate', curr_lr, global_step=n_iter)

                n_iter += 1

                # warmup for the first 10 epochs
                if epoch_counter >= 10 and self.scheduler != None:
                    self.scheduler.step()   

            logging.debug(f"Epoch: {epoch_counter}\tLoss: {loss}")

        logging.info("Training has finished.")
        # save model checkpoints
        checkpoint_name = 'checkpoint_{:04d}.pth.tar'.format(self.args.epochs)
        save_checkpoint({
            'epoch': self.args.epochs,
            'arch': self.args.arch,
            'state_dict': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
        }, is_best=False, filename=os.path.join(self.writer.log_dir, checkpoint_name))
        logging.info(f"Model checkpoint and metadata has been saved at {self.writer.log_dir}.")
