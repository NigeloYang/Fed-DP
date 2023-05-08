# -*- coding: utf-8 -*-
# @Time    : 2023/4/14

import copy
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from system.utils.data_utils import DatasetSplit
from system.utils.utils import sparsify


class ClientBase(object):
    def __init__(self, args, id, train_dataset, label_idxs, **kwargs):
        self.id = id  # integer
        self.dataset = args.dataset
        
        # Set optimizer for the local updates
        self.model = copy.deepcopy(args.model)
        self.latest_global_model = copy.deepcopy(list(self.model.parameters()))
        self.local_epoch = args.local_epoch
        self.learn_rate = args.local_learn_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learn_rate)
        self.local_bs = args.local_bs
        self.criterion = nn.CrossEntropyLoss()
        self.clip_c = args.norm
        
        self.isdiydp = args.isdiydp
        self.device = args.device
        self.rate = args.rate
        
        self.local_trainloader, self.local_testloader = self.load_client_dataset(train_dataset, list(label_idxs))
        
        self.model_params_shape, self.model_params_lenght = self.save_model_shape(self.model.parameters())
    
    ############################# load dataset ##############################
    def load_client_dataset(self, dataset, idxs):
        """
        Returns train, test dataloaders for a given dataset and user indexes.
        """
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_test = idxs[int(0.8 * len(idxs)):]
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=self.local_bs, shuffle=True)
        
        return trainloader, testloader
    
    ############################# model shape ##############################
    def save_model_shape(self, params):
        params_shape = []
        params_lenght = 0
        for param in params:
            params_shape.append(param.size())
            params_lenght += len(param.data.reshape(-1))
        
        return params_shape, params_lenght
    
    def recover_model_shape(self, flattened):
        grads = []
        start_len = 0
        for size in self.model_params_shape:
            end_len = 1
            for i in size:
                end_len *= i
            temp_data = flattened[start_len:start_len + end_len].reshape(size)
            grads.append(temp_data)
            start_len = start_len + end_len
        return grads
    
    ############################# model interpolation/noise ##############################
    def weight_interpolation(self, train_model):
        delta_model = []
        for train_w, latest_client_w in zip(train_model, self.latest_global_model):
            delta_model.append(train_w.data - latest_client_w.data)
        return delta_model
    
    def process_grad(self, delta_client_model):
        client_grads = []
        for params in delta_client_model:
            client_grads = np.append(client_grads, params.reshape(-1).cpu())
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
        '''
        if self.rate > 1:
            return (sparsify(flattened, self.topk))
        else:
            return flattened
    
    ############################# metrics ##############################
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
                losses.append(loss.item())
        
        return correct, sum(losses) / len(losses), size
    
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
    
    ############################# update model ##############################
    def update_client_params(self, global_model):
        for client_m, latest_global_m, global_m in zip(self.model.parameters(), self.latest_global_model,
                                                       global_model.parameters()):
            client_m.data = global_m.data.clone()
            latest_global_m.data = global_m.data.clone()