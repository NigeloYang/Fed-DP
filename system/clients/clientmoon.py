# -*- coding: utf-8 -*-
# @Time    : 2023/5/8

import copy
import numpy as np
import time
import torch
import torch.nn as nn
import torch.nn.functional as F

from system.clients.clientbase import ClientBase
from system.privacy.priv_utils import *


class clientMoon(ClientBase):
    def __init__(self, args, id, train_dataset, label_idxs, **kwargs):
        super().__init__(args, id, train_dataset, label_idxs, **kwargs)
        
        if self.diyldp:
            self.topk = int(self.model_params_length / self.com_rate)
            self.eps_ld = self.epsilon / self.topk
            print(f'local differential privacy epsilon: {self.eps_ld}')
        
        self.tau = args.tau
        self.mu = args.mu
        
        self.global_model = copy.deepcopy(self.model)
        self.old_client_model = copy.deepcopy(self.model)
    
    def train(self, client_id, global_round, metrics):
        local_trainloader = self.local_trainloader
        client_sample_len = 0
        self.model.train()
        
        train_time = time.time()
        
        for local_epoch in range(self.local_epoch):
            acc = 0
            total = 0
            for batch_idx, (images, labels) in enumerate(local_trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                total += len(labels)
                
                # # 预测和计算准确度
                # output = self.model(images)
                # acc += (output.argmax(1) == labels).type(torch.float).sum().item()
                #
                # # 计算损失
                # loss = self.criterion(output, labels)
                #
                # # Moon idea by myself
                # output_glob = self.global_model(images)
                # output_prev = self.old_client_model(images)
                # loss_con = - torch.log(torch.exp(F.cosine_similarity(output, output_glob) / self.tau) / (
                #     torch.exp(F.cosine_similarity(output, output_glob) / self.tau) + torch.exp(
                #     F.cosine_similarity(output, output_prev) / self.tau)))
                # loss += self.mu * torch.mean(loss_con)
                
                # MOON IDEA 预测和计算准确度
                rep_curr = self.model.base(images)
                output = self.model.head(rep_curr)
                
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
                
                loss = self.criterion(output, labels)
                
                rep_glob = self.global_model.base(images).detach()
                rep_prev = self.old_client_model.base(images).detach()
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep_curr, rep_glob) / self.tau) / (
                    torch.exp(F.cosine_similarity(rep_curr, rep_glob) / self.tau) + torch.exp(
                    F.cosine_similarity(rep_curr, rep_prev) / self.tau)))
                
                loss += self.mu * torch.mean(loss_con)
                
                # 将梯度初始化为 0，以便批次之间不会混合梯度
                self.optimizer.zero_grad()
                
                # 后向传递错误
                loss.backward()
                
                # 优化参数
                self.optimizer.step()
                
                if batch_idx % 4 == 0:
                    print(
                        '| Global Round: {:>2} | Client: {:>3} | Local Epoch: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f}'.format(
                            global_round + 1, client_id, local_epoch + 1,
                            100. * (batch_idx + 1) / len(local_trainloader), 100. * acc / total, loss.item()))
            client_sample_len = total
        
        self.old_client_model = copy.deepcopy(self.model)
        
        ctrain_model = copy.deepcopy(self.model)
        delta_ctmodel = self.weight_interpolation(ctrain_model.parameters())
        
        if self.diyldp:
            flattened = self.process_grad(delta_ctmodel)
            delta_ctmodel_noise = self.add_noise(flattened)
            # delta_ctmodel_noise = self.recover_model_shape(delta_ctmodel_noise)
            
            # save train model time cost
            metrics.client_train_time[client_id][global_round] = time.time() - train_time
            
            return delta_ctmodel_noise, client_sample_len
        else:
            # save train model time cost
            metrics.client_train_time[client_id][global_round] = time.time() - train_time
            
            return delta_ctmodel, client_sample_len
    
    def update_client_params(self, global_model):
        for client_m, latest_global_m, global_m in zip(self.model.parameters(), self.client_global_model.parameters(),
                                                       global_model.parameters()):
            client_m.data = global_m.data.clone()
            latest_global_m.data = global_m.data.clone()
        self.global_model = global_model
    
    def train_metrics(self):
        """ Returns the inference accuracy and loss."""
        self.model.eval()
        
        size, acc = 0.0, 0.0
        losses = []
        with torch.no_grad():
            for images, labels in self.local_trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                size += labels.shape[0]
                
                rep_curr = self.model.base(images)
                output = self.model.head(rep_curr)
                
                # acc += (output.argmax(1) == labels).type(torch.float).sum().item()
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
                
                loss = self.criterion(output, labels)
                
                rep_glob = self.global_model.base(images).detach()
                rep_prev = self.old_client_model.base(images).detach()
                loss_con = - torch.log(torch.exp(F.cosine_similarity(rep_curr, rep_glob) / self.tau) / (
                    torch.exp(F.cosine_similarity(rep_curr, rep_glob) / self.tau) + torch.exp(
                    F.cosine_similarity(rep_curr, rep_prev) / self.tau)))
                
                loss += self.mu * torch.mean(loss_con)
                
                losses.append(loss.item())
        
        return acc, sum(losses) / len(losses), size
