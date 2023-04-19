#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/5 21:37
# @File    : dataset_util.py
# @Author  : Richard Yang
import os
import json
import numpy as np
import random


def mkdirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('dir contend exist')


def check(config_path, train_path, test_path, num_clients, num_classes, niid=False, balance=True, partition=None):
    # check existing dataset
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            config = json.load(f)
        if config['num_clients'] == num_clients and \
            config['num_classes'] == num_classes and \
            config['non_iid'] == niid and \
            config['balance'] == balance and \
            config['partition'] == partition:
            print("\nDataset already generated.\n")
            return True
    
    dir_path = os.path.dirname(train_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    dir_path = os.path.dirname(test_path)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    
    return False


def balanced_iid(dir_path, dataset, num_clients, num_classes):
    """
    Sample I.I.D. client data from dataset,
    Each client has same number of samples, and same distribution for all class samples.
    
    :param dataset:
    :param num_clients:
    :return: dict of image index
    """
    
    mkdirs(dir_path)
    
    classes_data = []
    samples_per_client = int(len(dataset) / num_clients)
    
    labels = np.array(dataset.targets)
    
    for i in range(10):
        idx = np.argwhere(labels == i).reshape(-1)
        print(f'the {i}th class before shuffle: {idx}')
        random.shuffle(idx)
        print(f'the {i}th class after shuflle: {idx} ')
        classes_data.append(idx)
    print('每个类别的数量', [len(v) for v in classes_data])
    
    dict_clients = {user: np.array([]) for user in range(num_clients)}
    for user in range(num_clients):
        for i in range(samples_per_client):
            l = (user + i) % num_classes
            data = np.random.choice(classes_data[l], 1, replace=False)
            dict_clients[user] = np.append(dict_clients[user], data)
    
    dict_json = {}
    for key in dict_clients.keys():
        dict_json[key] = dict_clients[key].tolist()
    
    with open(dir_path + 'iid_{}_{}.json'.format('balanced', num_clients), 'w') as f:
        json.dump(dict_json, f)
        print('finish write dataset')
    
    return dict_json


def unbalanced_iid():
    '''
    Assign different sample number for each client using Log-Normal distribution $Log-N(0,\sigma^2)$, while keep same distribution for different class samples.

    '''
    pass


def shard_noniid(dir_path, dataset, num_clients, num_shards, num_imgs):
    """
    Sample non-I.I.D client data from dataset. Non-iid partition based on shards
    
    :param dataset:
    :param num_clients:
    :return:
    """
    
    mkdirs(dir_path)
    
    idx_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # divide and assign
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_clients[i] = np.concatenate(
                (dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    
    dict_json = {}
    for key in dict_clients.keys():
        dict_json[key] = dict_clients[key].tolist()
    
    with open(dir_path + 'noniid_{}_{}.json'.format('shard', num_clients), 'w') as f:
        json.dump(dict_json, f)
    
    return dict_json


def unbanlanced_shard_noniid(dir_path, dataset, num_clients, num_shards, num_imgs):
    mkdirs(dir_path)
    
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    # num_shards, num_imgs = 1000, 50
    idx_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30
    
    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_clients)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)
    
    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:
        for i in range(num_clients):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[i] = np.concatenate((dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        
        random_shard_size = random_shard_size - 1
        
        # Next, randomly assign the remaining shards
        for i in range(num_clients):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[i] = np.concatenate((dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    else:
        for i in range(num_clients):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[i] = np.concatenate((dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_clients, key=lambda x: len(dict_clients.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[k] = np.concatenate((dict_clients[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    
    dict_json = {}
    for key in dict_clients.keys():
        dict_json[key] = dict_clients[key].tolist()
    
    with open(dir_path + 'noniid_{}_{}.json'.format('unbalanced_shard', num_clients), 'w') as f:
        json.dump(dict_json, f)
    
    return dict_json


def Hetero_Dirichlet_noniid():
    pass


def Balanced_Dirichlet_noniid():
    pass


def Unbalanced_Dirichlet_noniid():
    '''


    '''
    pass


def Quantity_based_Label_Distribution_Noniid():
    pass


def non_iid(dir_path, dataset, num_clients, num_shards, num_imgs):
    """
    Sample non-I.I.D client data from CIFAR10 dataset
    :param dataset:
    :param num_clients:
    :return:
    """
    
    mkdirs(dir_path)
    
    idx_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # divide and assign
    for i in range(num_clients):
        rand_set = set(np.random.choice(idx_shard, 2, replace=False))
        idx_shard = list(set(idx_shard) - rand_set)
        for rand in rand_set:
            dict_clients[i] = np.concatenate(
                (dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    
    dict_json = {}
    for key in dict_clients.keys():
        dict_json[key] = dict_clients[key].tolist()
    
    with open(dir_path + 'noniid{}.json'.format(num_clients), 'w') as f:
        json.dump(dict_json, f)
    
    return dict_json


def noniid_unequal(dir_path, dataset, num_clients, num_shards, num_imgs):
    mkdirs(dir_path)
    
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    # num_shards, num_imgs = 1000, 50
    idx_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([]) for i in range(num_clients)}
    idxs = np.arange(num_shards * num_imgs)
    labels = np.array(dataset.targets)
    
    # sort labels
    idxs_labels = np.vstack((idxs, labels))
    idxs_labels = idxs_labels[:, idxs_labels[1, :].argsort()]
    idxs = idxs_labels[0, :]
    
    # Minimum and maximum shards assigned per client:
    min_shard = 1
    max_shard = 30
    
    # Divide the shards into random chunks for every client
    # s.t the sum of these chunks = num_shards
    random_shard_size = np.random.randint(min_shard, max_shard + 1, size=num_clients)
    random_shard_size = np.around(random_shard_size / sum(random_shard_size) * num_shards)
    random_shard_size = random_shard_size.astype(int)
    
    # Assign the shards randomly to each client
    if sum(random_shard_size) > num_shards:
        for i in range(num_clients):
            # First assign each client 1 shard to ensure every client has
            # atleast one shard of data
            rand_set = set(np.random.choice(idx_shard, 1, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[i] = np.concatenate((dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        
        random_shard_size = random_shard_size - 1
        
        # Next, randomly assign the remaining shards
        for i in range(num_clients):
            if len(idx_shard) == 0:
                continue
            shard_size = random_shard_size[i]
            if shard_size > len(idx_shard):
                shard_size = len(idx_shard)
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[i] = np.concatenate((dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    else:
        for i in range(num_clients):
            shard_size = random_shard_size[i]
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[i] = np.concatenate((dict_clients[i], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
        if len(idx_shard) > 0:
            # Add the leftover shards to the client with minimum images:
            shard_size = len(idx_shard)
            # Add the remaining shard to the client with lowest data
            k = min(dict_clients, key=lambda x: len(dict_clients.get(x)))
            rand_set = set(np.random.choice(idx_shard, shard_size, replace=False))
            idx_shard = list(set(idx_shard) - rand_set)
            for rand in rand_set:
                dict_clients[k] = np.concatenate((dict_clients[k], idxs[rand * num_imgs:(rand + 1) * num_imgs]), axis=0)
    
    dict_json = {}
    for key in dict_clients.keys():
        dict_json[key] = dict_clients[key].tolist()
    
    with open(dir_path + 'noniid_unequal{}.json'.format(num_clients), 'w') as f:
        json.dump(dict_json, f)
    
    return dict_json


if __name__ == "__main__":
    pass
