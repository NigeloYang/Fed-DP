# -*- coding: utf-8 -*-
# @Time    : 2023/4/29

import torch
import torch.nn as nn
import numpy as np
import time
import copy

from system.clients.clientbase import ClientBase
from system.optimizers.fedoptimizer import SCAFFOLDOptimizer
from system.privacy.priv_utils import *


class clientSCAFFOLD(ClientBase):
    def __init__(self, args, id, train_dataset, label_idxs, **kwargs):
        super().__init__(args, id, train_dataset, label_idxs, **kwargs)
        
        self.optimizer = SCAFFOLDOptimizer(self.model.parameters(), lr=self.learn_rate)
        
        self.client_c = []
        for param in self.model.parameters():
            self.client_c.append(torch.zeros_like(param))
        self.global_c = None
        
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
                
                # 计算损失
                loss = self.criterion(output, labels)
                
                # 将梯度初始化为 0，以便批次之间不会混合梯度
                self.optimizer.zero_grad()
                
                # 后向传递错误
                loss.backward()
                
                # for g_c, c_c in zip(self.global_c, self.client_c):
                #     g_c.data -= c_c.data
                # self.optimizer.step(self.global_c)
                
                # 优化参数
                self.optimizer.step(self.global_c, self.client_c)
                
                if batch_idx % 4 == 0:
                    print(
                        '| Global Round: {:>2} | Client: {:>3} | Local Epoch: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f}'.format(
                            global_round + 1, client_id, local_epoch + 1,
                            100. * (batch_idx + 1) / len(local_trainloader), 100. * acc / total, loss.item()))
            client_sample_len = total
        
        train_model = copy.deepcopy(self.model)
        delta_model = self.weight_interpolation(train_model.parameters())
        
        delta_c = self.update_c(num_batch)
        # self.delta_c, self.delta_y = self.delta_yc()
        
        ctrain_model = copy.deepcopy(self.model)
        delta_ctmodel = self.weight_interpolation(ctrain_model.parameters())
        
        if self.diyldp:
            flattened = self.process_grad(delta_ctmodel)
            delta_ctmodel_noise = self.add_noise(flattened)
            # delta_ctmodel_noise = self.recover_model_shape(delta_ctmodel_noise)
            
            # save train model time cost
            metrics.client_train_time[client_id][global_round] = time.time() - train_time
            
            return delta_ctmodel_noise, client_sample_len, delta_c
        else:
            # save train model time cost
            metrics.client_train_time[client_id][global_round] = time.time() - train_time
            
            return delta_ctmodel, client_sample_len, delta_c
    
    def update_client_params(self, global_model, global_c):
        for client_m, latest_global_m, global_m in zip(self.model.parameters(), self.client_global_model.parameters(),
                                                       global_model.parameters()):
            client_m.data = global_m.data.clone()
            latest_global_m.data = global_m.data.clone()
        
        self.global_c = global_c
    
    def update_c(self, num_batches):
        delta_c = copy.deepcopy(self.client_c)
        temp_client_c = copy.deepcopy(self.client_c)
        for ci, c, global_m, client_m in zip(self.client_c, self.global_c, self.client_global_model.parameters(),
                                             self.model.parameters()):
            ci.data = ci.data - c.data + (global_m.data - client_m.data) / (num_batches * self.learn_rate)
        
        for d_c, temp_c, ci_start in zip(delta_c, temp_client_c, self.client_c):
            d_c.data = ci_start.data - temp_c.data
        
        return delta_c
