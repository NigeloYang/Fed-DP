# -*- coding: utf-8 -*-
# @Time    : 2023/4/29

import copy
import numpy as np
import time

import torch
import torch.nn as nn

from system.clients.clientbase import ClientBase
from system.privacy.priv_utils import *


class clientNova(ClientBase):
    def __init__(self, args, id, train_dataset, label_idxs, **kwargs):
        super().__init__(args, id, train_dataset, label_idxs, **kwargs)
        
        if self.diyldp:
            self.topk = int(self.model_params_length / self.com_rate)
            self.eps_ld = self.epsilon / self.topk
            print(f'local differential privacy epsilon: {self.eps_ld}')
    
    def train(self, client_id, global_round, metrics):
        local_trainloader = self.local_trainloader
        client_sample_len = 0
        self.model.train()
        
        train_time = time.time()
        num_batch = 0
        for local_epoch in range(self.local_epoch):
            acc = 0
            total = 0
            for batch_idx, (images, labels) in enumerate(local_trainloader):
                num_batch += 1
                images, labels = images.to(self.device), labels.to(self.device)
                total += len(labels)
                
                # 预测和计算准确度
                output = self.model(images)
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
                
                # compute loss
                loss = self.criterion(output, labels)
                
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
        
        ctrain_model = copy.deepcopy(self.model)
        delta_ctmodel = self.weight_interpolation(ctrain_model.parameters())
        
        if self.diyldp:
            flattened = self.process_grad(delta_ctmodel)
            delta_ctmodel_noise = self.add_noise(flattened)
            # delta_ctmodel_noise = self.recover_model_shape(delta_ctmodel_noise)
            
            # save train model time cost
            metrics.client_train_time[client_id][global_round] = time.time() - train_time
            
            return delta_ctmodel_noise, client_sample_len, num_batch
        else:
            # save train model time cost
            metrics.client_train_time[client_id][global_round] = time.time() - train_time
            
            return delta_ctmodel, client_sample_len, num_batch
