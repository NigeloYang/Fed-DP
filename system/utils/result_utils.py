# -*- coding: utf-8 -*-
# @Time    : 2023/4/14

import datetime
import json
import os


class Metrics(object):
    def __init__(self, args):
        self.args = args
        global_epoch = self.args.global_epoch
        self.client_train_loss = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.client_train_acc = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.client_test_acc = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.client_train_time = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.client_send_time = {c: [0] * global_epoch for c in range(self.args.num_clients)}
        self.global_epoch_time = []
        self.local_avg_train_acc = []
        self.local_avg_train_loss = []
        self.local_avg_test_acc = []
        self.final_accuracies = []
        self.final_loss = []
        self.clients_per_round = []
        self.clients_rate = []
        self.epsilon_rate = []
        self.all_time = []
        self.dataidi_name = ['balanced_iid', 'unbalanced_iid', 'shard_noniid', 'unbanlanced_shard_noniid',
                             'Hetero_Dirichlet_noniid', 'Balanced_Dirichlet_noniid',
                             'Quantity_based_Label_Distribution_Noniid']
        self.path = './result/'
    
    def check_dir(self):
        path = os.path.join(self.path, self.args.dataset, str(self.args.dataiid))
        if not os.path.exists(path):
            os.makedirs(path)
            print('The directory is created ')
        else:
            print('The directory exists')
    
    def write(self):
        # 创建存储文件目录
        self.check_dir()
        
        '''write existing history records into a json file'''
        time = datetime.datetime.now().strftime("%Y%m%d%H%M")
        metrics = {}
        metrics['dataset'] = self.args.dataset
        metrics['algorithm'] = self.args.algorithm
        metrics['model_name'] = self.args.model
        metrics['global_epoch'] = self.args.global_epoch
        metrics['eval_every'] = self.args.eval_every
        metrics['learn_rate'] = self.args.learn_rate
        metrics['local_epoch'] = self.args.local_epoch
        metrics['batch_size'] = self.args.local_bs
        metrics['num_clients'] = self.args.num_clients
        metrics['init_client_rate'] = self.args.rc_rate
        metrics['norm'] = self.args.norm
        metrics['rate'] = self.args.rate
        metrics['mp_rate'] = self.args.mp_rate
        metrics['epsilon'] = self.args.epsilon
        metrics['delta'] = self.args.delta
        metrics['mu'] = self.args.mu
        metrics['clients_per_round'] = self.clients_per_round
        metrics['clients_rate'] = self.clients_rate
        metrics['epsilon_rate'] = self.epsilon_rate
        metrics['client_train_loss'] = self.client_train_loss
        metrics['client_train_acc'] = self.client_train_acc
        metrics['client_test_acc'] = self.client_test_acc
        metrics['client_train_time'] = self.client_train_time
        metrics['client_send_time'] = self.client_send_time
        metrics['global_epoch_time'] = self.global_epoch_time
        metrics['local_avg_train_acc'] = self.local_avg_train_acc
        metrics['local_avg_train_loss'] = self.local_avg_train_loss
        metrics['local_avg_test_acc'] = self.local_avg_test_acc
        metrics['final_accuracies'] = self.final_accuracies
        metrics['final_loss'] = self.final_loss
        metrics['all_time'] = self.all_time
        metrics_dir = os.path.join(self.path, self.args.dataset, str(self.args.dataiid),
                                   '{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.json'.format(self.args.dataset,
                                                                                     self.args.algorithm,
                                                                                     self.args.learn_rate,
                                                                                     self.args.num_clients,
                                                                                     self.args.epsilon,
                                                                                     self.args.delta,
                                                                                     self.args.rc_rate,
                                                                                     self.args.norm,
                                                                                     self.args.rate,
                                                                                     self.args.mp_rate,
                                                                                     self.args.mechanism,
                                                                                     time))
        with open(metrics_dir, 'w') as ouf:
            json.dump(metrics, ouf)


if __name__ == "__main__":
    pass
