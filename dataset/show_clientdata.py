# -*- coding: utf-8 -*-
# @Time    : 2024/4/11

import json
import os
import argparse
import numpy as np
import torch

from matplotlib import pyplot as plt
from torchvision import datasets, transforms


def get_dataset(dataset_name):
    if dataset_name == 'cifar10':
        dir_path = './formatdata/cifar10/'
        
        train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomGrayscale(),
                                    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        dataset = datasets.CIFAR10(
            root='local_dataset/', train=True, download=True,
            transform=train
        )
    elif dataset_name == 'mnist':
        dir_path = './formatdata/mnist/'
        
        dataset = datasets.MNIST(
            root='local_dataset/', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        )
    elif dataset_name == 'fmnist':
        dir_path = './formatdata/fmnist/'
        
        dataset = datasets.MNIST(
            root='local_dataset/', train=True, download=True,
            transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
        )
    else:
        exit(f"The {dataset_name} dataset is not available at this time")
    
    clients = get_clients(dir_path)
    
    return dataset, clients


def get_clients(dir_path):
    clients = []
    for fileName in os.listdir(dir_path):
        client = json.load(open(dir_path + fileName, 'rb'))
        clients.append(client)
    return clients


def data_plot_client(dataset_name, num_clients):
    dataset, clients_dict = get_dataset(dataset_name)
    labels = np.array(dataset.targets)

    data_tri_name = ['Client-IID-Balanced',
                     'Client-IID-Balanced-Homo',
                     'Client-IID-Unbalanced',
                     'Client-IID-Unbalanced-Homo',
                     'Client-Non-IID-Balanced-Dirichlet',
                     'Client-Non-IID-Hetero-Dirichlet',
                     'Client-Non-IID-Quantity-Based-Label-Distribution-1',
                     'Client-Non-IID-Quantity-Based-Label-Distribution-2',
                     'Client-Non-IID-Quantity-Based-Label-Distribution-3',
                     'Client-Non-IID-Shard',
                     'Client-Non-IID-Unbalanced-Dirichlet',
                     'Client-Non-IID-Unequal-Shard']
    for distri_id, name in enumerate(data_tri_name):
        save_path = 'datadistribution/' + dataset_name + '/'+ data_tri_name[distri_id] + '.png'
        clients = clients_dict[distri_id]

        plt.figure(figsize=(20, 6))  # 3
        label_distribution = [[] for _ in range(10)]
        client_labels = {clientid: [] for clientid in range(num_clients)}
        for c_id in range(num_clients):
            for idx in clients[str(c_id)]:
                label_distribution[labels[int(idx)]].append(c_id)
                client_labels[c_id].append(labels[int(idx)])
        print('The client owns the label content')
        for key, value in client_labels.items():
            print('Client Id: {:>3} | Dataset Classes: {} | Number of Dataset: {}'.format(key, set(value), len(value)))

        plt.hist(label_distribution, stacked=True, bins=np.arange(-0.7, num_clients + 2, 1),
                 label=dataset.classes,
                 rwidth=0.5)
        plt.xticks(np.arange(num_clients), ["%d" % c_id for c_id in range(num_clients)])
        plt.ylabel("Number of samples")
        plt.xlabel("Client ID")
        plt.legend()
        plt.title(
            "Display Label Distribution on Different Clients\n Dataset {}\n Distribution Type: {}".format(dataset_name,
                                                                                                          data_tri_name[
                                                                                                              distri_id]))
        # plt.savefig(save_path)
        plt.show()


def data_plot_label(dataset_name, num_clients):
    dataset, clients = get_dataset(dataset_name)
    num_classes = len(dataset.classes)
    labels = np.array(dataset.targets)

    data_tri_name = ['Lable-IID-Balanced',
                     'Lable-IID-Balanced-Homo',
                     'Lable-IID-Unbalanced',
                     'Lable-IID-Unbalanced-Homo',
                     'Lable-Non-IID-Balanced-Dirichlet',
                     'Lable-Non-IID-Hetero-Dirichlet',
                     'Lable-Non-IID-Quantity-Based-Label-Distribution-1',
                     'Lable-Non-IID-Quantity-Based-Label-Distribution-2',
                     'Lable-Non-IID-Quantity-Based-Label-Distribution-3',
                     'Lable-Non-IID-Shard',
                     'Lable-Non-IID-Unbalanced-Dirichlet',
                     'Lable-Non-IID-Unequal-Shard']

    for distri_id, name in enumerate(data_tri_name):
        print('************ Dataset Distribution Type:' + name + '************')
        save_path = 'datadistribution/' + dataset_name + '/' + data_tri_name[distri_id] + '.png'
        clients_dict = clients[distri_id]

        plt.figure(figsize=(20, 6))
        label_distribution = [[] for _ in range(num_classes)]
        client_labels = {clientid: [] for clientid in range(num_clients)}
        for c_id in range(num_clients):
            for label_idx in clients_dict[str(c_id)]:
                client_labels[c_id].append(labels[int(label_idx)])
                label_distribution[labels[int(label_idx)]].append(c_id)
        print('The client owns the label content')
        for key, value in client_labels.items():
            print('Client Id: {:>3} | Dataset Classes: {} | Number of Dataset: {}'.format(key, set(value), len(value)))

        plt.hist([client_labels[c_id] for c_id in range(num_clients)], stacked=True,
                 bins=np.arange(min(labels) - 0.5, max(labels) + 1.5, 1),
                 label=["Client {}".format(i) for i in range(num_clients)],
                 rwidth=0.5)
        plt.xticks(np.arange(num_classes), dataset.classes)
        plt.ylabel("Number of samples")
        plt.xlabel("Label type")
        plt.legend(loc="upper right")
        plt.title(
            "Display Label Distribution on Different Clients\n Dataset {}\n Distribution Type: {}".format(dataset_name,
                                                                                                          data_tri_name[
                                                                                                              distri_id]))
        # plt.savefig(save_path)
        plt.show()


if __name__ == "__main__":
    # get_clients('./formatdata/cifar10/')
    data_plot_client('mnist', 10)
    # data_plot_label('mnist', 10)
