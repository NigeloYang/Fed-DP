# -*- coding: utf-8 -*-
# @Time    : 2023/4/14

import copy
import numpy as np

import torch
from torch import nn
from torch.utils.data import DataLoader

from system.utils.data_utils import DatasetSplit
from system.privacy.priv_utils import sampling_randomizer


class ClientBase(object):
    def __init__(self, args, id, train_dataset, label_idxs, **kwargs):
        self.id = id  # integer
        self.dataset = args.dataset
        
        # Set optimizer for the local updates
        self.model = copy.deepcopy(args.model)
        self.client_global_model = copy.deepcopy(self.model)
        self.global_epoch = args.global_epoch
        self.local_epoch = args.local_epoch
        self.learn_rate = args.local_learn_rate
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learn_rate)
        self.local_bs = args.local_bs
        self.criterion = nn.CrossEntropyLoss()
        
        self.train_slow = args.train_slow
        self.send_slow = args.send_slow
        
        self.diyldp = args.diyldp
        self.opacus = args.opacus
        self.epsilon = args.epsilon
        self.delta = args.delta
        self.mechanism = args.mechanism
        self.clip_c = args.clip_c
        
        self.com_rate = args.com_rate
        
        self.device = args.device
        
        self.local_trainloader, self.local_testloader = self.load_client_dataset(train_dataset, list(label_idxs))
        
        self.model_params_shape, self.model_params_length = self.save_model_shape(self.model.parameters())
    
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
        model_params_shape = []
        model_params_lenght = 0
        for param in params:
            model_params_shape.append(param.size())
            model_params_lenght += len(param.data.reshape(-1))
        
        return model_params_shape, model_params_lenght
    
    def recover_model_shape(self, flattened):
        model_params = []
        start_len = 0
        for size in self.model_params_shape:
            end_len = 1
            for i in size:
                end_len *= i
            temp_data = flattened[start_len:start_len + end_len].reshape(size)
            model_params.append(temp_data)
            start_len = start_len + end_len
        return model_params
    
    ############################# model interpolation ##############################
    def weight_interpolation(self, train_model):
        delta_model = []
        for train_w, latest_global_w in zip(train_model, self.client_global_model.parameters()):
            delta_model.append(train_w.data - latest_global_w.data)
        return delta_model
    
    def process_grad(self, delta_client_model):
        client_model_params = []
        for params in delta_client_model:
            client_model_params = np.append(client_model_params, params.reshape(-1).cpu())
        return client_model_params
        
        # gpu 操作
        # client_model_params = torch.tensor([]).cuda()
        # for name in delta_client_model:
        #     client_model_params = torch.cat((client_model_params, delta_client_model[name].reshape(-1)), dim=0)
        
        return client_model_params
    
    ############################# add noise ##############################
    def add_noise(self, flattened):
        choices = np.random.choice(flattened.size, self.topk)
        
        # gpu 操作
        # choices = np.random.choice(flattened.size(0), self.sampling)
        
        return sampling_randomizer(flattened, choices, self.clip_c, self.eps_ld, self.delta, self.mechanism)
    
    ############################# metrics ##############################
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
                acc += (output.argmax(1) == labels).type(torch.float).sum().item()
                
                loss = self.criterion(output, labels)
                losses.append(loss.item())
        
        return acc, sum(losses) / len(losses), size
    
    def test_metrics(self):
        """ Returns the inference accuracy and loss."""
        self.model.eval()
        
        size, acc = 0.0, 0.0
        
        with torch.no_grad():
            for images, labels in self.local_testloader:
                images, labels = images.to(self.device), labels.to(self.device)
                size += labels.shape[0]
                
                output = self.model(images)
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
        
        return acc, size
    
    ############################# update model ##############################
    def update_client_params(self, global_model):
        for client_m, client_global_m, global_m in zip(self.model.parameters(), self.client_global_model.parameters(),
                                                       global_model.parameters()):
            client_m.data = global_m.data.clone()
            client_global_m.data = global_m.data.clone()
