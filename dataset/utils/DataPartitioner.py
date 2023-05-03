# -*- coding: utf-8 -*-
# @Time    : 2023/4/27
import argparse
import numpy as np
import torch
from matplotlib import pyplot as plt
from torchvision import datasets, transforms
from . import dataset_util as F


class DataPartitioner(object):
    def __init__(self, args):
        self.dataset, self.dir_path = self.get_dataset(args.dataset)
        self.dataset_name = args.dataset
        self.partition = args.partition
        self.isbalance = None if args.isbalance is None else args.isbalance
        self.num_clients = args.num_clients
        self.num_classes = len(self.dataset.classes)
        self.num_shard = args.num_shard
        self.num_img = args.num_img
        self.alpha = args.alpha
        self.sigma = args.sigma
        self.min_require_size = args.min_require_size
        self.major_classes_num = args.major_classes_num
        self.dataplot = args.dataplot
        
        if self.isbalance is None:
            assert args.partition in ['shards', 'unequal_shards', 'skew-base-label', 'hetero'], \
                "When balance=None, partition only accepts shards, unequal_shards, and skew-base-label."
        elif isinstance(self.isbalance, bool):
            assert args.partition in ['iid', 'dirichlet', 'homo'], \
                "When balance is bool, partition only accepts dirichlet and iid."
        else:
            raise ValueError(f"'balance' can only be NoneType or bool, not {type(self.isbalance)}.")
        
        self.clients = self.get_partition()
    
    def get_dataset(self, datasetname):
        if datasetname == 'cifar10':
            dir_path = 'formatdata/cifar10/'
            
            train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomGrayscale(),
                                        transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
            dataset = datasets.CIFAR10(
                root='local_dataset/', train=True, download=True,
                transform=train
            )
        elif datasetname == 'mnist':
            dir_path = 'formatdata/mnist/'
            
            dataset = datasets.MNIST(
                root='local_dataset/', train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            )
        elif datasetname == 'fmnist':
            dir_path = 'formatdata/fmnist/'
            
            dataset = datasets.MNIST(
                root='local_dataset/', train=True, download=True,
                transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
            )
        else:
            exit(f"The {datasetname} dataset is not available at this time")
        
        return dataset, dir_path
    
    def get_partition(self):
        if self.isbalance is None:
            if self.partition == 'shards':
                client_dict = F.shard_noniid(self.dir_path, self.dataset, self.num_clients, self.num_shard,
                                             self.num_img)
            elif self.partition == 'unequal_shards':
                client_dict = F.unequal_shard_noniid(self.dir_path, self.dataset, self.num_clients,
                                                     self.num_shard, self.num_img)
            elif self.partition == 'skew-base-label':
                # label-distribution-skew:quantity-based
                client_dict = F.quantity_based_label_distribution_noniid(self.dir_path, self.dataset,
                                                                         self.num_clients, self.num_classes,
                                                                         self.major_classes_num)
            elif self.partition == 'hetero':
                # label-distribution-skew:distributed-based (Dirichlet)
                client_dict = F.hetero_dirichlet_noniid(self.dir_path, self.dataset, self.num_clients,
                                                        self.num_classes, self.alpha)
        elif self.isbalance == True:
            if self.partition == "iid":
                client_dict = F.balanced_iid(self.dir_path, self.dataset, self.num_clients, self.num_classes)
            elif self.partition == "dirichlet":
                client_dict = F.balanced_dirichlet_noniid(self.dir_path, self.dataset, self.num_clients,
                                                          self.num_classes, self.alpha)
            elif self.partition == "homo":
                client_dict = F.balance_homo_partition(self.dir_path, self.dataset, self.num_clients,
                                                       self.alpha)
        elif self.isbalance == False:
            if self.partition == "iid":
                client_dict = F.unbalanced_iid(self.dir_path, self.dataset, self.num_clients, self.num_classes,
                                               self.sigma)
            elif self.partition == "dirichlet":
                client_dict = F.unbalanced_dirichlet_noniid(self.dir_path, self.dataset, self.num_clients,
                                                            self.num_classes, self.alpha, self.sigma)
            elif self.partition == 'homo':
                # quantity-skew (Dirichlet)
                client_dict = F.unbalance_homo_partition(self.dir_path, self.dataset, self.num_clients, self.sigma)
        else:
            exit('Error: There are no other data distributions for the time being')
        
        return client_dict
    
    def data_plot(self):
        labels = np.array(self.dataset.targets)
        if self.isbalance is None:
            distri_id = 1 + ['shards', 'unequal_shards', 'skew-base-label', 'hetero'].index(self.partition)
        elif self.isbalance == True:
            distri_id = 5 + ['iid', 'dirichlet', 'homo'].index(self.partition)
        elif self.isbalance == False:
            distri_id = 8 + ['iid', 'dirichlet', 'homo'].index(self.partition)
        distri_id -= 1
        
        data_tri_name = ['Non-IID-Shard', 'Non-IID-Unequal-Shard', 'Non-IID-Quantity-Based-Label-Distribution',
                         'Non-IID-Hetero-Dirichlet', 'IID-Balanced', 'Non-IID-Balanced-Dirichlet', 'IID-Balance-Homo',
                         'IID-Unbalanced', 'Non-IID-Unbalanced-Dirichlet', 'IID-Unbalance-Homo']
        
        if self.dataplot == 1:
            plt.figure(figsize=(20, 6))  # 3
            label_distribution = [[] for _ in range(10)]
            client_labels = {clientid: [] for clientid in range(self.num_clients)}
            for c_id, idc in self.clients.items():
                for idx in idc:
                    label_distribution[labels[int(idx)]].append(c_id)
                    client_labels[c_id].append(labels[int(idx)])
            print('The client owns the label content')
            for key, value in client_labels.items():
                print('Client Id: {:>3} | Dataset Classes: {}'.format(key, set(value)))
            
            plt.hist(label_distribution, stacked=True, bins=np.arange(-0.7, self.num_clients + 2, 1),
                     label=self.dataset.classes,
                     rwidth=0.5)
            plt.xticks(np.arange(self.num_clients), ["%d" % c_id for c_id in range(self.num_clients)])
            plt.ylabel("Number of samples")
            plt.xlabel("Client ID")
            plt.legend()
            plt.title("Dataset {} Distribution: {}".format(self.dataset_name, data_tri_name[distri_id]))
            plt.savefig('datadistribution/{}_{}.png'.format(self.dataset_name, data_tri_name[distri_id]))
            plt.show()
        else:
            pass
