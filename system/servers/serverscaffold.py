# -*- coding: utf-8 -*-
# @Time    : 2023/4/29

import copy
import random
import time
import numpy as np
import torch
from tqdm import tqdm

from system.clients.clientscaffold import clientSCAFFOLD
from system.servers.serverbase import ServerBase


class FedScaffold(ServerBase):
    def __init__(self, args, metrics):
        super().__init__(args, metrics)
        
        # select client
        self.set_clients(args, clientSCAFFOLD)
        
        print(f"total clients: {self.num_clients}")
        print("Finished creating server and clients. \n ")
        
        self.server_learning_rate = args.server_learn_rate
        self.global_c = []
        for param in self.global_model.parameters():
            self.global_c.append(torch.zeros_like(param))
    
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
            delta_cs = []
            for client_id in self.selected_clients:
                # send global model
                if self.send_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.send_models(epoch, client_id)
                
                # local iteration train
                delta_ctmodel, csample_len, delta_c = self.clients[client_id].train(client_id, epoch, self.metrics)
                client_models.append(delta_ctmodel)
                client_sample_lens.append(csample_len)
                delta_cs.append(delta_c)
            
            ############ server process / weight process / rceive model ###########
            agg_client_model = self.server_process(client_models, client_sample_lens)
            
            self.update_global_params(agg_client_model)
            self.update_global_c(len(client_models), delta_cs)
            
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
    
    def send_models(self, epoch, client_id):
        send_time = time.time()
        
        self.clients[client_id].update_client_params(self.global_model, self.global_c)
        
        # save send model time cost
        self.metrics.client_send_time[client_id][epoch] = 2 * (time.time() - send_time)
    
    def update_global_c(self, num_clients, delta_cs):
        for delta_c in delta_cs:
            for g_c, d_c in zip(self.global_c, delta_c):
                g_c.data += d_c.data / num_clients
