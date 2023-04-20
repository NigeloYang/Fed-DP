#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 11:20
# @File    : serveravg.py
# @Author  : Richard Yang

import time
from tqdm import tqdm
from system.servers.serverbase import ServerBase
from system.clients.clientavg import clientAvg


class FedAvg(ServerBase):
    def __init__(self, args, metrics):
        super().__init__(args, metrics)
        
        # select client
        self.set_clients(args, clientAvg)
        
        print(f"total clients: {self.num_clients}")
        print("Finished creating server and clients. \n ")
    
    def train(self):
        for epoch in tqdm(range(self.global_epoch), desc='Processing'):
            print(f'\n--------------- Global training Round: {epoch + 1}th ------------------------')
            epoch_time = time.time()
            
            # select client
            self.selected_clients = self.select_clients_id()
            print(f'selected client: {self.selected_clients}')
            
            # evaluate model
            if epoch % self.eval_every == 0:
                print("Model is Evaluating")
                self.evaluate(epoch)
            
            ############ local client process ###########
            client_noise_weights = []
            for client_id in self.selected_clients:
                # send global model
                self.send_models(epoch, client_id)
                
                # local iteration train
                weight_noise = self.clients[client_id].train(client_id, epoch, self.metrics)
                client_noise_weights.append(weight_noise)
            
            ############ server process / weight process / rceive model ###########
            agg_model_weight = self.server_process(client_noise_weights)
            self.update_global_params(agg_model_weight)
            self.metrics.global_epoch_time.append(time.time() - epoch_time)
            print("Global Training Round: {:>3} | Cost Time: {:>4.4f}".format(epoch + 1, time.time() - epoch_time))
        
        print('\n--------------Test Final Model-----------------')
        test_acc, test_loss = self.final_test()
        print(f"After Global Epoch,Test Final Model Acc: {100 * test_acc:.4f}% | Loss: {test_loss:.4f} ")
        
        self.metrics.final_accuracies.append(test_acc)
        self.metrics.final_loss.append(test_loss)
    
    def server_process(self, client_weights):
        '''
        basic aggregate, but enlarge the learning rate when Top-k is applied
        '''
        merge_weight, total_client = self.aggregate_e(client_weights)
        return self.average(merge_weight, total_client / self.rate)
