# -*- coding: utf-8 -*-
# @Time    : 2023/4/14

import json
import os
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""
    
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


def get_dataset(dataset_name, data_iid, num_users, args):
    """ Returns train and test datasets and a user group which is a dict where
    the keys are the user index and the values are the corresponding data for
    each of those users.
    data_iid 1: IID
    data_iid 2: NonIID
    data_iid 3: NonIID_unequal
    """
    source_data = os.path.join('./dataset/local_dataset/')
    format_data = os.path.join('./dataset/formatdata/' + args.dataset + '/')
    
    if dataset_name == 'cifar10':
        train = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.RandomGrayscale(),
                                    transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        test = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
        
        train_dataset = datasets.CIFAR10(source_data, train=True, download=True, transform=train)
        test_dataset = datasets.CIFAR10(source_data, train=False, download=True, transform=test)
        if data_iid == 1:
            file = open(format_data + 'iid_balanced_homo_%s.json' % num_users, 'rb')
        elif data_iid == 2:
            file = open(format_data + 'iid_balanced_%s.json' % num_users, 'rb')
        elif data_iid == 3:
            file = open(format_data + 'iid_unbalanced_homo_%s.json' % num_users, 'rb')
        elif data_iid == 4:
            file = open(format_data + 'iid_balanced_%s.json' % num_users, 'rb')
        elif data_iid == 5:
            file = open(format_data + 'noniid_balanced_dirichlet_%s.json' % num_users, 'rb')
        elif data_iid == 6:
            file = open(format_data + 'noniid_hetero_dirichlet_%s.json' % num_users, 'rb')
        elif data_iid == 7:
            file = open(format_data + 'noniid_quantity_label_%s.json' % num_users, 'rb')
        elif data_iid == 8:
            file = open(format_data + 'noniid_shard_%s.json' % num_users, 'rb')
        elif data_iid == 9:
            file = open(format_data + 'noniid_unbalanced_dirichlet_%s.json' % num_users, 'rb')
        elif data_iid == 10:
            file = open(format_data + 'noniid_unbalanced_shard_%s.json' % num_users, 'rb')
        else:
            exit('Error: no others dataset format')
            
        client_groups = json.load(file)
        return train_dataset, test_dataset, client_groups
    elif dataset_name == 'mnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.MNIST(source_data, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.MNIST(source_data, train=False, download=True, transform=apply_transform)
        if data_iid == 1:
            file = open(format_data + 'iid_balance_homo_%s.json' % num_users, 'rb')
        elif data_iid == 2:
            file = open(format_data + 'iid_balanced_%s.json' % num_users, 'rb')
        elif data_iid == 3:
            file = open(format_data + 'iid_unbalance_homo_%s.json' % num_users, 'rb')
        elif data_iid == 4:
            file = open(format_data + 'iid_unbalanced_%s.json' % num_users, 'rb')
        elif data_iid == 5:
            file = open(format_data + 'noniid_balanced_dirichlet_%s.json' % num_users, 'rb')
        elif data_iid == 6:
            file = open(format_data + 'noniid_hetero_dirichlet_%s.json' % num_users, 'rb')
        elif data_iid == 7:
            file = open(format_data + 'noniid_quantity_label_%s.json' % num_users, 'rb')
        elif data_iid == 8:
            file = open(format_data + 'noniid_shard_%s.json' % num_users, 'rb')
        elif data_iid == 9:
            file = open(format_data + 'noniid_unbalanced_dirichlet_%s.json' % num_users, 'rb')
        elif data_iid == 10:
            file = open(format_data + 'noniid_unbalanced_shard_%s.json' % num_users, 'rb')
        else:
            exit('Error: no others dataset format')

        client_groups = json.load(file)
        return train_dataset, test_dataset, client_groups
    elif dataset_name == 'fmnist':
        apply_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))]
        )
        train_dataset = datasets.FashionMNIST(source_data, train=True, download=True, transform=apply_transform)
        test_dataset = datasets.FashionMNIST(source_data, train=False, download=True, transform=apply_transform)
        if data_iid == 1:
            file = open(format_data + 'iid_balanced_%s.json' % num_users, 'rb')
            client_groups = json.load(file)
        elif data_iid == 2:
            print('not exist unbalanced-iid')
        elif data_iid == 3:
            file = open(format_data + 'noniid_shard_%s.json' % num_users, 'rb')
            client_groups = json.load(file)
        elif data_iid == 4:
            file = open(format_data + 'noniid_unbalanced_shard_%s.json' % num_users, 'rb')
            client_groups = json.load(file)
        else:
            exit('Error: no others dataset format')
        return train_dataset, test_dataset, client_groups
    else:
        exit('Error: no others dataset')


if __name__ == "__main__":
    source_data = os.path.join('../../dataset/local_dataset/')
    format_data = os.path.join('../../dataset/formatdata/mnist/')
    train_dataset = datasets.CIFAR10(source_data, train=True, download=True)
