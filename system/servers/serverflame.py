# -*- coding: utf-8 -*-
# @Time    : 2024/4/11

import math
import time
import numpy as np
import torch
from tqdm import tqdm

from system.servers.serverbase import ServerBase
from system.clients.clientflame import clientFlame
from system.utils.utils import transform


class FedFlame(ServerBase):
    def __init__(self, args, metrics):
        super().__init__(args, metrics)
        
        # select client
        self.set_clients(args, clientFlame)
        
        self.clients_per_round = 0
        self.m_p = 1
        self.em_s = 1
        
        print(f"total clients: {self.num_clients}")
        print("Finished creating server and clients. \n ")
    
    def train(self):
        for epoch in tqdm(range(self.global_epoch), desc='Processing'):
            print(f'\n--------------- Global training Round: {epoch + 1}th ------------------------')
            epoch_time = time.time()
            
            # select client
            self.selected_clients = self.select_clients_id()
            print(f'selected client: {self.selected_clients} \n')
            
            # evaluate model
            if epoch % self.eval_every == 0:
                print("Model is Evaluating")
                self.evaluate(epoch)
            
            ############ local client process ###########
            client_models = []
            client_sample_lens = []
            client_idx_choices = []
            for client_id in self.selected_clients:
                # send global model
                if self.send_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.send_models(epoch, client_id)
                
                # local iteration train
                delta_ctmodel, csample_len, idx_choice = self.clients[client_id].train(client_id, epoch, self.metrics)
                client_models.append(delta_ctmodel)
                client_sample_lens.append(csample_len)
                client_idx_choices.append(idx_choice)
            
            ############################# server process / weight process / rceive model ##############################
            agg_client_model = self.server_process(client_models, client_sample_lens, client_idx_choices)
            
            self.update_global_params(agg_client_model)
            
            self.metrics.global_epoch_time.append(time.time() - epoch_time)
            print("Global Training Round: {:>3} | Cost Time: {:>4.4f}".format(epoch + 1, time.time() - epoch_time))
            
            print('\n--------------Test Model-----------------')
            test_acc, test_loss = self.final_test()
            print(
                "Global Training Round: {:>3} | Test Model Acc: {:>4.4f}% | Test Model Loss: {:>4.4f}".format(epoch + 1,
                                                                                                              100 * test_acc,
                                                                                                              test_loss))
            
            self.metrics.final_accuracies.append(test_acc)
            self.metrics.final_loss.append(test_loss)
    
    def update_flame_param(self):
        self.clients_per_round = len(self.selected_clients)
        self.m_p = int(self.clients_per_round / self.mp_rate)
        print("Setting the padding size for each dimension with ", self.m_p)
        self.em_s = self.clients_per_round / self.com_rate
    
    def server_process(self, client_models, sample_weights, client_idx_choices):
        return self.aggregate_model_ldp(client_models, sample_weights, client_idx_choices)
    
    def aggregate_model_ldp(self, client_models, sample_weights, client_idx_choices):
        # get flame parameters
        self.update_flame_param()
        
        total_weights = sum(sample_weights)
        agg_model = [0] * len(client_models[0])
        cidx_choices = []
        
        for sw, client_model in zip(sample_weights, client_models):
            agg_model += client_model * sw / total_weights
        
        for idx in client_idx_choices:
            cidx_choices.extend(idx)
        
        m_s = np.bincount(cidx_choices, minlength=self.model_params_length)
        
        m_n = np.ones(len(m_s)) * self.m_p - m_s
        assert len(
            np.where(m_n < 0)[
                0]) == 0, 'ERROR: Please choose a larger m_p (smaller mp_rate) and re-run, cause max(m_s): {} > m_p: {}'.format(
            max(m_s), self.m_p)
        dummies = np.zeros(len(m_n))
        
        sigma = (2 * self.clip_c / self.epsilon) * math.sqrt(2 * math.log(1.25 / self.delta))
        for i, v in enumerate(m_n):
            assert self.mechanism == 'laplace', "Please use laplace for FedFlame"
            dummies[i] = sum(np.random.laplace(loc=1, scale=1.0 / self.epsilon, size=int(v))) - 0.5 * (
                self.m_p - self.em_s)
        
        d_noise = self.recover_model_shape(dummies)
        agg_model = self.recover_model_shape(agg_model)
        
        for i, v in enumerate(agg_model):
            temp_data = transform((v + d_noise[i]) / self.em_s, 0, 1, -self.clip_c, self.clip_c)
            agg_model[i] = torch.as_tensor(temp_data).cuda()
        
        return agg_model
        
        # return [torch.as_tensor(transform((agg_m + d_n) / self.em_s, 0, 1, -self.clip_c, self.clip_c)).cuda() for
        #         agg_m, d_n in zip(agg_model, d_noise)]

