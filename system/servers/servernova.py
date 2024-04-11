# -*- coding: utf-8 -*-
# @Time    : 2023/4/29
import copy
import time
import numpy as np
import torch
from tqdm import tqdm

from system.servers.serverbase import ServerBase
from system.clients.clientnova import clientNova
from system.utils.utils import transform


class FedNova(ServerBase):
    def __init__(self, args, metrics):
        super().__init__(args, metrics)
        
        # select client
        self.set_clients(args, clientNova)
        
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
            client_batches = []
            for client_id in self.selected_clients:
                # send global model
                if self.send_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.send_models(epoch, client_id)
                
                # local iteration train
                client_model, sample_len, client_batch = self.clients[client_id].train(client_id, epoch, self.metrics)
                client_models.append(client_model)
                client_sample_lens.append(sample_len)
                client_batches.append(client_batch)
            
            ############ server process / weight process / rceive model ###########
            agg_client_model = self.server_process(client_models, client_sample_lens, client_batches)
            
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
    
    def server_process(self, client_models, sample_lens, client_batches):
        '''
        basic aggregate, but enlarge the learning rate when Top-k is applied
        '''
        if self.diyldp:
            return self.aggregate_model_ldp(client_models, sample_lens, client_batches)
        elif self.diycdp:
            return self.aggregate_model_cdp(client_models, sample_lens, client_batches)
        else:
            return self.aggregate_model(client_models, sample_lens, client_batches)
    
    def aggregate_model(self, client_models, sample_weights, client_batches):
        total_weights = sum(sample_weights)
        total_batch = 0
        for sample_len, client_batch in zip(sample_weights, client_batches):
            total_batch += (sample_len * client_batch) / total_weights
        
        agg_model = copy.deepcopy(client_models[0])
        for params in agg_model:
            params.data.zero_()
        
        for w, c_b, client_model in zip(sample_weights, client_batches, client_models):
            for a_m, c_m in zip(agg_model, client_model):
                a_m.data += total_batch * (c_m.data * w) / (total_weights * c_b)
        return agg_model
    
    def aggregate_model_cdp(self, client_models, sample_weights, client_batches):
        total_weights = sum(sample_weights)
        total_batch = 0
        for sample_len, client_batch in zip(sample_weights, client_batches):
            total_batch += (sample_len * client_batch) / total_weights
        
        agg_model = copy.deepcopy(client_models[0])
        for params in agg_model:
            params.data.zero_()
        
        for sw, c_b, client_model in zip(sample_weights, client_batches, client_models):
            for a_m, c_m in zip(agg_model, client_model):
                a_m.data += total_batch * (c_m.data * sw) / (total_weights * c_b)
        
        for a_m in agg_model:
            a_m.data = a_m.data + torch.as_tensor(
                np.random.normal(0, self.sigma, np.array(a_m.data.cpu()).shape)).cuda()
        return agg_model
    
    def aggregate_model_ldp(self, client_models, sample_weights, client_batches):
        total_weights = sum(sample_weights)
        total_batch = 0
        for sample_len, client_batch in zip(sample_weights, client_batches):
            total_batch += (sample_len * client_batch) / total_weights
        
        agg_model = [0] * len(client_models[0])
        
        for sw, c_b, client_model in zip(sample_weights, client_batches, client_models):
            agg_model += total_batch * (client_model * sw) / (total_weights * c_b)
        
        agg_model = self.recover_model_shape(agg_model)
        
        # total_weight: of aggregated updates
        # base: sum of aggregated updates
        # return the average update after transforming back from [0, 1] to [-C, C]
        for i, v in enumerate(agg_model):
            temp_data = transform(v, 0, 1, -self.clip_c, self.clip_c)
            agg_model[i] = torch.as_tensor(temp_data).cuda()
        
        return agg_model
