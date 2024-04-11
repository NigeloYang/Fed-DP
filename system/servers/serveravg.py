# -*- coding: utf-8 -*-
# @Time    : 2023/4/14

import time

import numpy as np
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
            print(f'selected client: {self.selected_clients} \n')
            
            # evaluate model
            if epoch % self.eval_every == 0:
                print("Model is Evaluating")
                self.evaluate(epoch)
            
            ############################# local client process ##############################
            client_models = []
            client_sample_lens = []
            for client_id in self.selected_clients:
                # send global model
                if self.send_slow:
                    time.sleep(0.1 * np.abs(np.random.rand()))
                self.send_models(epoch, client_id)
                
                # local iteration train
                delta_ctmodel, csample_len = self.clients[client_id].train(client_id, epoch, self.metrics)
                client_models.append(delta_ctmodel)
                client_sample_lens.append(csample_len)
            
            ############################# server process / weight process / rceive model ##############################
            agg_client_model = self.server_process(client_models, client_sample_lens)
            
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
