#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 10:57
# @File    : clientbase.py
# @Author  : Richard Yang

import copy
import numpy as np
import os

import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F

from system.utils.data_utils import DatasetSplit


class ClientBase(object):
    def __init__(self, args, id, train_dataset, label_idxs, **kwargs):
        self.dataset = args.dataset
        self.model = copy.deepcopy(args.model)
        
        self.id = id  # integer
        self.device = args.device
        self.rate = args.rate
        
        self.local_epoch = args.local_epoch
        self.learn_rate = args.learn_rate
        
        # Set optimizer for the local updates
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learn_rate)
        self.local_bs = args.local_bs
        self.criterion = nn.CrossEntropyLoss()
        
        self.local_trainloader, self.local_testloader = self.load_client_dataset(train_dataset, list(label_idxs))
        self.data_shape, self.data_len = self.save_model_shape(self.model.state_dict())
        
        self.model_latest = {}
    
    def save_model_shape(self, grads):
        data_shape = {}
        grads_lenght = []
        for name in grads:
            grads_lenght = np.append(grads_lenght, grads[name].reshape(-1).cpu())
            data_shape.setdefault(name, grads[name].size())
        return data_shape, len(grads_lenght)
    
    def load_client_dataset(self, dataset, idxs):
        """
        Returns train, test dataloaders for a given dataset and user indexes.
        """
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_test = idxs[int(0.8 * len(idxs)):]
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=self.local_bs, shuffle=True)
        
        return trainloader, testloader
    
    def weight_update(self, train_weight):
        temp_weight = {}
        for name in train_weight:
            temp = train_weight[name] - self.model_latest[name]
            temp_weight.setdefault(name, temp)
        return temp_weight
    
    def process_grad(self, grads):
        # cpu 操作
        client_grads = []
        for name in grads:
            # print(name,'-------',grads[name].size())
            client_grads = np.append(client_grads, grads[name].reshape(-1).cpu())
        return client_grads
        
        # # gpu 操作
        # grads_temp = torch.tensor([]).cuda()
        # for name in grads:
        #     grads_temp = torch.cat((grads_temp, grads[name].reshape(-1)), dim=0)
        #
        # return grads_temp
    
    def add_noise(self, flattened):
        '''
        DO NOTHING
        1. non-private
        2. no clipping
        3. no sparsification
        (for npsgd)
        '''
        return flattened
    
    def recover_model_shape(self, flattened, shape):
        grads = {}
        start_len = 0
        for name, size in shape.items():
            end_len = 1
            for i in list(size):
                end_len *= i
            temp_data = flattened[start_len:start_len + end_len].reshape(list(size))
            grads.setdefault(name, temp_data)
            start_len = start_len + end_len
        return grads
    
    def update_client_params(self, model_weight):
        self.model.load_state_dict(model_weight)
        self.model_latest = model_weight
    
    def train_metrics(self):
        """ Returns the inference accuracy and loss."""
        self.model.eval()
        
        size, losses, correct = 0.0, 0.0, 0.0
        
        with torch.no_grad():
            for images, labels in self.local_trainloader:
                images, labels = images.to(self.device), labels.to(self.device)
                size += labels.shape[0]
                
                pred = self.model(images)
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
                
                losses += self.criterion(pred, labels).item()
        
        return correct, losses, size
    
    def test_metrics(self):
        """ Returns the inference accuracy and loss."""
        self.model.eval()
        
        size, correct = 0.0, 0.0
        
        with torch.no_grad():
            for images, labels in self.local_testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                size += labels.shape[0]
                
                pred = self.model(images)
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        
        return correct, size
