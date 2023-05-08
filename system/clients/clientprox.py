# -*- coding: utf-8 -*-
# @Time    : 2023/4/23
import copy
import numpy as np
import time
import torch

from system.clients.clientbase import ClientBase
from system.optimizers.fedoptimizer import FedProxOptimizer
from system.utils.priv_utils import *


class clientProx(ClientBase):
    def __init__(self, args, id, train_dataset, label_idxs, **kwargs):
        super().__init__(args, id, train_dataset, label_idxs, **kwargs)
        
        self.mu = args.mu
        self.global_params = copy.deepcopy(list(self.model.parameters()))
        
        self.optimizer = FedProxOptimizer(self.model.parameters(), lr=self.learn_rate, mu=self.mu)
        
        if self.rate > 1 and self.isdiydp:
            self.topk = int(self.model_params_lenght / self.rate)
            print("Topk selecting {} dimensions".format(self.topk))
    
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
                
                # 预测和计算准确度
                output = self.model(images)
                acc += (output.argmax(1) == labels).type(torch.float).sum().item()
                
                # 计算损失
                loss = self.criterion(output, labels)
                
                # 将梯度初始化为 0，以便批次之间不会混合梯度
                self.optimizer.zero_grad()
                
                # 后向传递错误
                loss.backward()
                
                # 优化参数
                self.optimizer.step(self.global_params, self.device)
                
                if batch_idx % 4 == 0:
                    print(
                        '| Client: {:>3} | Global Round: {:>2} | Local Epoch: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f}'.format(
                            client_id, global_round + 1, local_epoch + 1,
                                       100. * (batch_idx + 1) / len(local_trainloader),
                                       100. * acc / total, loss.item()))
            client_sample_len = total
        
        train_model = copy.deepcopy(self.model)
        delta_model = self.weight_interpolation(train_model.parameters())
        
        if self.isdiydp:
            flattened = self.process_grad(delta_model)
            delta_model_noise = self.add_noise(flattened)
            delta_model_noise = self.recover_model_shape(delta_model_noise)
            
            # save train model time cost
            metrics.client_train_time[client_id][global_round] = time.time() - train_time
            
            return delta_model_noise, client_sample_len
        else:
            # save train model time cost
            metrics.client_train_time[client_id][global_round] = time.time() - train_time
            
            return delta_model, client_sample_len
    
    def update_client_params(self, global_model):
        for client_m, latest_global_m, global_p, global_m in zip(self.model.parameters(), self.latest_global_model,
                                                                 self.global_params, global_model.parameters()):
            client_m.data = global_m.data.clone()
            latest_global_m.data = global_m.data.clone()
            global_p.data = global_m.data.clone()
    
    def train_metrics(self):
        """ Returns the inference accuracy and loss."""
        self.model.eval()
        
        size, correct = 0.0, 0.0
        losses = []
        
        with torch.no_grad():
            for images, labels in self.local_trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                size += labels.shape[0]
                
                pred = self.model(images)
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
                
                loss = self.criterion(pred, labels)
                gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm - pm, p=2)
                losses.append(loss.item())
        
        return correct, sum(losses) / len(losses), size
