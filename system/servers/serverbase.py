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
from system.utils.dlg import DLG
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
        self.data_iid = args.dataiid
        self.major_classes_num = args.major_classes_num
        
        self.rclient = args.rclient
        self.rc_rate = args.rc_rate
        self.train_slow = args.train_slow
        self.send_slow = args.send_slow
        
        self.diyldp = args.diyldp
        self.diycdp = args.diycdp
        self.opacus = args.opacus
        self.epsilon = args.epsilon
        self.delta = args.delta
        self.mechanism = args.mechanism
        self.clip_c = args.clip_c
        self.sigma = (2 * self.clip_c / self.epsilon) * math.sqrt(2 * math.log(1.25 / self.delta))
        
        self.com_rate = args.com_rate
        self.mp_rate = args.mp_rate
        
        self.device = args.device
        self.test = args.test
        
        self.clients = []
        self.selected_clients = []
        
        self.metrics = metrics
        self.train_dataset, self.test_dataset, self.client_group = get_dataset(self.dataset, self.data_iid,
                                                                               self.num_clients, self.major_classes_num,
                                                                               args)
        self.model_params_shape, self.model_params_length = self.save_model_shape(self.global_model.parameters())
    
    ############################# model shape ##############################
    def save_model_shape(self, params):
        model_params_shape = []
        model_params_lenght = 0
        for param in params:
            model_params_shape.append(list(param.size()))
            model_params_lenght += len(param.data.reshape(-1))
        
        return model_params_shape, model_params_lenght
    
    def recover_model_shape(self, flattened):
        model_params = []
        start_len = 0
        for size in self.model_params_shape:
            end_len = 1
            for i in list(size):
                end_len *= i
            temp_data = flattened[start_len:start_len + end_len].reshape(size)
            model_params.append(temp_data)
            start_len = start_len + end_len
        return model_params
    
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
        if self.test:
            return [i for i in range(10)]
        elif self.rclient:
            return [i for i in range(self.num_clients) if np.random.random() <= self.rc_rate]
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
        
        test_loader = DataLoader(self.test_dataset, batch_size=64, shuffle=True)
        loss, acc = 0.0, 0.0
        size = len(test_loader.dataset)
        num_batches = len(test_loader)
        
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(test_loader):
                images, labels = images.to(self.device), labels.to(self.device)
                output = self.global_model(images)
                loss += criterion(output, labels).item()
                acc += (torch.sum(torch.argmax(output, dim=1) == labels)).item()
        loss /= num_batches
        acc /= size
        
        return acc, loss
    
    ################################# Server ##############################
    def server_process(self, client_models, sample_weights):
        if self.diyldp:
            return self.aggregate_model_ldp(client_models, sample_weights)
        elif self.diycdp:
            return self.aggregate_model_cdp(client_models, sample_weights)
        else:
            return self.aggregate_model(client_models, sample_weights)
    
    ################################# AGGREGATE ##############################
    def aggregate_model(self, client_models, sample_weights):
        total_weights = sum(sample_weights)
        agg_model = copy.deepcopy(client_models[0])
        for params in agg_model:
            params.data.zero_()
        
        for sw, client_model in zip(sample_weights, client_models):
            for a_m, c_m in zip(agg_model, client_model):
                a_m.data += (c_m.data * sw) / total_weights
        
        return agg_model
    
    def aggregate_model_cdp(self, client_models, sample_weights):
        total_weights = sum(sample_weights)
        agg_model = copy.deepcopy(client_models[0])
        for params in agg_model:
            params.data.zero_()
        
        for sw, client_model in zip(sample_weights, client_models):
            for a_m, c_m in zip(agg_model, client_model):
                a_m.data += (c_m.data * sw) / total_weights
        
        for a_m in agg_model:
            a_m.data = a_m.data + torch.as_tensor(
                np.random.normal(0, self.sigma, np.array(a_m.data.cpu()).shape)).cuda()
        return agg_model
    
    def aggregate_model_ldp(self, client_models, sample_weights):
        total_weights = sum(sample_weights)
        agg_model = [0] * len(client_models[0])
        
        for sw, client_model in zip(sample_weights, client_models):
            agg_model += client_model * sw / total_weights
        
        agg_model = self.recover_model_shape(agg_model)
        
        # total_weight: of aggregated updates
        # base: sum of aggregated updates
        # return the average update after transforming back from [0, 1] to [-C, C]
        for i, v in enumerate(agg_model):
            temp_data = transform(v, 0, 1, -self.clip_c, self.clip_c)
            agg_model[i] = torch.as_tensor(temp_data).cuda()
        
        return agg_model
    
    ############################# update model ##############################
    def update_global_params(self, agg_client_model):
        for server_m, agg_client_m in zip(self.global_model.parameters(), agg_client_model):
            server_m.data += agg_client_m.data.clone()
    
    ############################# attack model ##############################
    def call_dlg(self, R):
        # items = []
        cnt = 0
        psnr_val = 0
        for cid, client_model in zip(self.uploaded_ids, self.uploaded_models):
            client_model.eval()
            origin_grad = []
            for gp, pp in zip(self.global_model.parameters(), client_model.parameters()):
                origin_grad.append(gp.data - pp.data)
            
            target_inputs = []
            trainloader = self.clients[cid].load_train_data()
            with torch.no_grad():
                for i, (x, y) in enumerate(trainloader):
                    if i >= self.batch_num_per_client:
                        break
                    
                    if type(x) == type([]):
                        x[0] = x[0].to(self.device)
                    else:
                        x = x.to(self.device)
                    y = y.to(self.device)
                    output = client_model(x)
                    target_inputs.append((x, output))
            
            d = DLG(client_model, origin_grad, target_inputs)
            if d is not None:
                psnr_val += d
                cnt += 1
            
            # items.append((client_model, origin_grad, target_inputs))
        
        if cnt > 0:
            print('PSNR value is {:.2f} dB'.format(psnr_val / cnt))
        else:
            print('PSNR error')
