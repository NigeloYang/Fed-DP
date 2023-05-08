# -*- coding: utf-8 -*-
# @Time    : 2023/4/29

import time
import torch
from tqdm import tqdm

from system.servers.serverbase import ServerBase
from system.clients.clientnova import clientNova


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
        
        print('\n--------------Test Final Model-----------------')
        test_acc, test_loss = self.final_test()
        print(f"After Global Epoch,Test Final Model Acc: {100 * test_acc:.4f}% | Loss: {test_loss:.4f} ")
        
        self.metrics.final_accuracies.append(test_acc)
        self.metrics.final_loss.append(test_loss)
    
    def server_process(self, client_models, sample_lens, client_batches):
        '''
        basic aggregate, but enlarge the learning rate when Top-k is applied
        '''
        agg_client_model = self.aggregate_e(client_models, sample_lens, client_batches)
        
        if self.isdiydp:
            return self.average(agg_client_model)
        else:
            return agg_client_model
    
    def aggregate_e(self, client_models, sample_lens, client_batches):
        agg_model = [0] * len(client_models[0])
        sample_id = 0
        total_sample = sum(sample_lens)
        
        avg_batch = 0
        for sample_len, client_batch in zip(sample_lens, client_batches):
            avg_batch += (sample_len * client_batch) / total_sample
        
        for client_model in client_models:
            for i, client_m in enumerate(client_model):
                agg_model[i] = agg_model[i] + avg_batch * (client_m * sample_lens[sample_id]) / (
                    total_sample * client_batches[sample_id])
            sample_id += 1
        return agg_model
