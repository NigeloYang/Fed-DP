#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 10:57
# @File    : serverbase.py
# @Author  : Richard Yang
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
        self.latest_model_weight = self.global_model.state_dict()
        
        self.global_epoch = args.global_epoch
        self.learn_rate = args.learn_rate
        self.eval_every = args.eval_every
        
        self.local_epoch = args.local_epoch
        self.num_clients = args.num_clients
        
        self.isrclient = args.isrclient
        
        self.rate = args.rate
        
        self.data_iid = args.dataiid
        self.device = args.device
        
        self.train_dataset, self.test_dataset, self.client_group = get_dataset(self.dataset, self.data_iid,
                                                                               self.num_clients, args)
        self.metrics = metrics
        
        self.clients = []
        self.selected_clients = []
        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []
    
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
            # return [i for i in range(50) if np.random.random() < self.rc_rate]
            return [i for i in range(self.num_clients) if np.random.random() < self.rc_rate]
        else:
            # return [i for i in range(4)]
            return np.random.randint(0, self.num_clients, 5)
            # return [i for i in range(self.num_clients)]
    
    def send_models(self, epoch, client_id):
        send_time = time.time()
        
        self.clients[client_id].update_client_params(self.latest_model_weight)
        
        # save send model time cost
        self.metrics.client_send_time[client_id][epoch] = 2 * (time.time() - send_time)
    
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
        losses = []
        for client_id in self.selected_clients:
            correct, loss, size = self.clients[client_id].train_metrics()
            tot_correct.append(correct)
            losses.append(loss)
            num_samples.append(size)
            
            self.metrics.client_train_acc[client_id][epoch] = correct / size
            self.metrics.client_train_loss[client_id][epoch] = loss / size
        
        return num_samples, tot_correct, losses
    
    def evaluate(self, epoch):
        evaluate_time = time.time()
        stats_test = self.test_metrics(epoch)
        stats_train = self.train_metrics(epoch)
        
        test_acc = np.sum(stats_test[1]) * 1.0 / np.sum(stats_test[0])
        
        train_acc = np.sum(stats_train[1]) * 1.0 / np.sum(stats_train[0])
        # train_loss = np.dot(stats_train[2], stats_train[0]) * 1.0 / np.sum(stats_train[0])
        train_loss = np.sum(stats_train[2]) * 1.0 / np.sum(stats_train[0])
        
        self.metrics.local_avg_train_acc.append(train_acc)
        self.metrics.local_avg_train_loss.append(train_loss)
        self.metrics.local_avg_test_acc.append(test_acc)
        
        print("At Global {} Evaluate Model time Cost: {:.4f}".format(epoch + 1, time.time() - evaluate_time))
        print("At Global {} Averaged Train Acc: {:.4f}".format(epoch + 1, train_acc))
        print("At Global {} Averaged Train Loss: {:.4f}".format(epoch + 1, train_loss))
        print("At Global {} Averaged Test Acc: {:.4f}".format(epoch + 1, test_acc))
    
    def server_process(self, messages):
        '''
        ONLY AGGREGATE
        weighted or evenly-weighted by num_samples
        '''
        if len(messages) == 1:
            weight_data, total_client = self.aggregate_e(messages)
        else:
            weight_data, total_client = self.aggregate_w(messages)
        return self.average(weight_data, total_client)
    
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
    
    def update_global_params(self, agg_model_weight):
        self.latest_model_weight = self.model_merge(self.latest_model_weight, agg_model_weight)
        self.global_model.load_state_dict(self.latest_model_weight)
    
    ################################# AVERAGE/AGGREGATE ##############################
    def model_merge(self, original_weight, new_weight):
        for name in new_weight:
            temp = original_weight[name] + new_weight[name]
            original_weight[name] = temp
        return original_weight
    
    def average(self, merge_weight, total_client):
        for name in merge_weight:
            merge_weight[name] = torch.as_tensor(merge_weight[name] / total_client).cuda()
        return merge_weight
    
    def average_cali(self, origin_weight, total_client, clip):
        '''
        total_weight: # of aggregated updates
        base: sum of aggregated updates
        return the average update after transforming back from [0, 1] to [-C, C]
        '''
        for name in origin_weight:
            temp = transform(origin_weight[name] / total_client, 0, 1, -self.clip_c, self.clip_c)
            origin_weight[name] = torch.Tensor(temp).cuda()
        return origin_weight
    
    def aggregate_e(self, client_weights):
        total_client = 0.0
        merge_weight = {}
        for weight in client_weights:
            total_client += 1.0
            for name in weight:
                if name in merge_weight:
                    merge_weight[name] = merge_weight[name] + weight[name]
                else:
                    merge_weight.setdefault(name, weight[name])
        return merge_weight, total_client
    
    def aggregate_w(self, weights):
        total_client = 0.0
        merge_weight = {}
        for w, weight in weights:
            total_client += 1.0
            for name in weight:
                if name in merge_weight:
                    merge_weight.setdefault(name, merge_weight.get(name) + w * weight[name])
                else:
                    merge_weight.setdefault(name, weight[name])
        
        return merge_weight, total_client
    
    def aggregate_p(self, weights):
        merge_weight, total_client = self.aggregate_e(weights)
        m_s = np.bincount(self.choice_list, minlength=(self.data_len))
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
        d_noise = self.recover_modelshape(dummies, self.data_shape)
        
        self.choice_list = []  # empty the choise list after each aggregation
        final_weight = {}
        for name in merge_weight:
            temp = transform((merge_weight[name] + d_noise[name]) / self.em_s, 0, 1, -self.clip_c, self.clip_c)
            final_weight.setdefault(name, torch.as_tensor(temp).cuda())
        
        return final_weight
    
    def aggregate_rp(self, weights):
        merge_weight, total_client = self.aggregate_e(weights)
        m_s = np.bincount(self.choice_list, minlength=(self.data_len))
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
        d_noise = self.recover_modelshape(dummies, self.data_shape)
        
        self.choice_list = []  # empty the choise list after each aggregation
        
        final_weight = {}
        for name in merge_weight:
            temp = transform((merge_weight[name] + d_noise[name]) / self.em_s, 0, 1, -self.clip_c, self.clip_c)
            final_weight.setdefault(name, torch.as_tensor(temp).cuda())
        
        return final_weight
