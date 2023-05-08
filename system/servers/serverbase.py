# -*- coding: utf-8 -*-
# @Time    : 2023/4/14

import math

import torch
import os
import numpy as np
import copy
import time
import random
from torch.utils.data import DataLoader

from system.utils.data_utils import get_dataset
from system.utils.utils import transform
from system.utils.data_utils import DatasetSplit


class ServerBase(object):
    def __init__(self, args, metrics):
        self.dataset = args.dataset
        self.global_model = copy.deepcopy(args.model)
        self.global_epoch = args.global_epoch
        self.local_epoch = args.local_epoch
        self.learn_rate = args.local_learn_rate
        self.eval_every = args.eval_every
        self.num_clients = args.num_clients
        self.isrclient = args.isrclient
        
        self.clip_c = args.norm
        self.epsilon = args.epsilon
        self.delta = args.delta
        self.isdiydp = args.isdiydp
        self.isopacus = args.isopacus
        self.mechanism = args.mechanism
        
        self.data_iid = args.dataiid
        self.device = args.device
        self.rate = args.rate
        self.istest = args.istest
        
        self.clients = []
        self.selected_clients = []
        
        self.metrics = metrics
        self.train_dataset, self.test_dataset, self.client_group = get_dataset(self.dataset, self.data_iid,
                                                                               self.num_clients, args)
        self.model_params_shape, self.model_params_lenght = self.save_model_shape(self.global_model.parameters())
    
    ############################# model shape ##############################
    def save_model_shape(self, params):
        params_shape = []
        params_lenght = 0
        for param in params:
            params_shape.append(list(param.size()))
            params_lenght += len(param.data.reshape(-1))
        
        return params_shape, params_lenght
    
    def recover_model_shape(self, flattened):
        grads = []
        start_len = 0
        for size in self.model_params_shape:
            end_len = 1
            for i in list(size):
                end_len *= i
            temp_data = flattened[start_len:start_len + end_len].reshape(size)
            grads.append(temp_data)
            start_len = start_len + end_len
        return grads
    
    ############################# set/select client ##############################
    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            client = clientObj(args, id=i, train_dataset=self.train_dataset, label_idxs=self.client_group[str(i)])
            self.clients.append(client)
    
    def select_clients_id(self):
        '''instantiates clients based on given train and test data directories

        Return:
            list of Clients
        '''
        if self.isrclient:
            return [i for i in range(self.num_clients) if np.random.random() < self.rc_rate]
        elif self.istest:
            # return np.random.randint(0, self.num_clients, 3)
            return [1, 2, 3]
        else:
            return [i for i in range(self.num_clients)]
    
    def send_models(self, epoch, client_id):
        send_time = time.time()
        
        self.clients[client_id].update_client_params(self.global_model)
        
        # save send model time cost
        self.metrics.client_send_time[client_id][epoch] = 2 * (time.time() - send_time)
    
    ############################# metrics ##############################
    def test_metrics(self, epoch):
        tot_correct = []
        num_samples = []
        for client_id in self.selected_clients:
            correct, size = self.clients[client_id].test_metrics()
            tot_correct.append(correct * 1.0)
            num_samples.append(size)
            
            self.metrics.client_test_acc[client_id][epoch] = correct / size
        
        return num_samples, tot_correct
    
    def train_metrics(self, epoch):
        num_samples = []
        tot_correct = []
        tot_losses = []
        for client_id in self.selected_clients:
            correct, loss, size = self.clients[client_id].train_metrics()
            tot_correct.append(correct)
            tot_losses.append(loss)
            num_samples.append(size)
            
            self.metrics.client_train_acc[client_id][epoch] = correct / size
            self.metrics.client_train_loss[client_id][epoch] = loss / size
        
        return num_samples, tot_correct, tot_losses
    
    def evaluate(self, epoch):
        evaluate_time = time.time()
        stats_test = self.test_metrics(epoch)
        stats_train = self.train_metrics(epoch)
        
        test_acc = sum(stats_test[1]) * 1.0 / sum(stats_test[0])
        
        train_acc = sum(stats_train[1]) * 1.0 / sum(stats_train[0])
        train_loss = sum(stats_train[2]) * 1.0 / len(stats_train[0])
        
        self.metrics.local_avg_train_acc.append(train_acc)
        self.metrics.local_avg_train_loss.append(train_loss)
        self.metrics.local_avg_test_acc.append(test_acc)
        
        print("At Global Round {} Evaluate Model time Cost: {:.4f}".format(epoch + 1, time.time() - evaluate_time))
        print("At Global Round {} Averaged Train Acc: {:.4f}".format(epoch + 1, train_acc))
        print("At Global Round {} Averaged Train Loss: {:.4f}".format(epoch + 1, train_loss))
        print("At Global Round {} Averaged Test Acc: {:.4f}".format(epoch + 1, test_acc))
    
    def final_test(self):
        """ Returns the test accuracy and loss. """
        self.global_model.eval()
        criterion = torch.nn.CrossEntropyLoss().to(self.device)
        
        test_loader = DataLoader(self.test_dataset, batch_size=128, shuffle=True)
        loss, total, correct = 0.0, 0.0, 0.0
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                pred = self.global_model(images)
                loss += criterion(pred, labels).item()
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
        loss /= num_batches
        correct /= size
        
        return correct, loss
    
    ################################# Server ##############################
    def server_process(self, client_models, sample_lens):
        agg_client_model = self.aggregate_e(client_models, sample_lens)
        
        if self.isdiydp:
            return self.average(agg_client_model)
        else:
            return agg_client_model
    
    ################################# AVERAGE/AGGREGATE ##############################
    def average(self, agg_client_model):
        for i, v in enumerate(agg_client_model):
            agg_client_model[i] = torch.as_tensor(v).cuda()
        return agg_client_model
    
    def average_cali(self, agg_client_model):
        '''
        total_weight: # of aggregated updates
        base: sum of aggregated updates
        return the average update after transforming back from [0, 1] to [-C, C]
        '''
        for i, v in enumerate(agg_client_model):
            temp_data = transform(v, 0, 1, -self.clip_c, self.clip_c)
            agg_client_model[i] = torch.as_tensor(temp_data).cuda()
        return agg_client_model
    
    def aggregate_e(self, client_models, sample_lens):
        agg_model = [0] * len(client_models[0])
        sample_id = 0
        total_sample = sum(sample_lens)
        for client_model in client_models:
            for i, client_m in enumerate(client_model):
                agg_model[i] = agg_model[i] + client_m * sample_lens[sample_id] / total_sample
            sample_id += 1
        return agg_model
    
    def aggregate_p(self, client_models, sample_lens, choice_list):
        agg_model = self.aggregate_e(client_models, sample_lens)
        m_s = np.bincount(choice_list, minlength=(self.model_params_lenght))
        m_n = np.ones(len(m_s)) * self.m_p - m_s
        assert len(
            np.where(m_n < 0)[
                0]) == 0, 'ERROR: Please choose a larger m_p (smaller mp_rate) and re-run, cause max(m_s): {} > m_p: {}'.format(
            max(m_s), self.m_p)
        dummies = np.zeros(len(m_n))
        
        sigma = (2 * self.clip_c / self.epsilon) * math.sqrt(2 * math.log(1.25 / self.delta))
        for i, v in enumerate(m_n):
            assert self.mechanism == 'laplace', "Please use laplace for v1-v3"
            dummies[i] = sum(np.random.laplace(loc=0.5, scale=1.0 / self.epsilon, size=int(v))) - 0.5 * (
                self.m_p - self.em_s)
        d_noise = self.recover_model_shape(dummies)
        
        return [torch.as_tensor(transform((agg_m + d_n) / self.em_s, 0, 1, -self.clip_c, self.clip_c)).cuda() for
                agg_m, d_n in zip(agg_model, d_noise)]
    
    ############################# update model ##############################
    def update_global_params(self, agg_client_model):
        self.model_merge(self.global_model.parameters(), agg_client_model)
        self.latest_global_model = self.global_model.parameters()
    
    def model_merge(self, server_model, agg_client_model):
        for server_m, agg_client_m in zip(server_model, agg_client_model):
            server_m.data += agg_client_m
