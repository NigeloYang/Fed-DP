# -*- coding: utf-8 -*-
# @Time    : 2023/4/23
import copy
import numpy as np
import time
import torch

from system.clients.clientbase import ClientBase
from system.optimizers.fedoptimizer import FedProxOptimizer
from system.privacy.priv_utils import sampling_randomizer


class clientProx(ClientBase):
    def __init__(self, args, id, train_dataset, label_idxs, **kwargs):
        super().__init__(args, id, train_dataset, label_idxs, **kwargs)
        
        self.mu = args.mu
        self.global_params = copy.deepcopy(list(self.model.parameters()))
        
        self.optimizer = FedProxOptimizer(self.model.parameters(), lr=self.learn_rate, mu=self.mu)
        
        if self.diyldp:
            self.topk = int(self.model_params_length / self.com_rate)
            self.eps_ld = self.epsilon / self.topk
            print(f'local differential privacy epsilon: {self.eps_ld}')
    
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
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
                
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
            
            return delta_ctmodel_noise, client_sample_len
        else:
            # save train model time cost
            metrics.client_train_time[client_id][global_round] = time.time() - train_time
            
            return delta_ctmodel, client_sample_len
    
    def update_client_params(self, global_model):
        for client_m, latest_global_m, global_p, global_m in zip(self.model.parameters(),
                                                                 self.client_global_model.parameters(),
                                                                 self.global_params, global_model.parameters()):
            client_m.data = global_m.data.clone()
            latest_global_m.data = global_m.data.clone()
            global_p.data = global_m.data.clone()
    
    def train_metrics(self):
        """ Returns the inference accuracy and loss."""
        self.model.eval()
        
        size, acc = 0.0, 0.0
        losses = []
        
        with torch.no_grad():
            for images, labels in self.local_trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                size += labels.shape[0]
                
                output = self.model(images)
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
                
                loss = self.criterion(output, labels)
                gm = torch.cat([p.data.view(-1) for p in self.global_params], dim=0)
                pm = torch.cat([p.data.view(-1) for p in self.model.parameters()], dim=0)
                loss += 0.5 * self.mu * torch.norm(gm - pm, p=2)
                losses.append(loss.item())
        
        return acc, sum(losses) / len(losses), size
