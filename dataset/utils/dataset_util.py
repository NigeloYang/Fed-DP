# -*- coding: utf-8 -*-
# @Time    : 2023/4/5
import os
import json
import numpy as np
import random


def make_dirs(path):
    if not os.path.exists(path):
        os.makedirs(path)
    else:
        print('Directory content is Exist')


def balanced_iid(dir_path, dataset, num_clients, num_classes):
    make_dirs(dir_path)
    
    classes_data = []
    samples_per_client = int(len(dataset) / num_clients)
    
    labels = np.array(dataset.targets)
    
    for i in range(num_classes):
        idx = np.argwhere(labels == i).reshape(-1)
        print(f'the {i}th class before shuffle: {idx}')
        random.shuffle(idx)
        print(f'the {i}th class after shuflle: {idx} ')
        classes_data.append(idx)
    print('classes numbers: ', [len(v) for v in classes_data])
    
    dict_clients = {user: np.array([]).astype(int) for user in range(num_clients)}
    for user in range(num_clients):
        for i in range(samples_per_client):
            l = (user + i) % num_classes
            data = np.random.choice(classes_data[l], 1, replace=False)
            dict_clients[user] = np.append(dict_clients[user], data)
    
    clients = {}
    for key in dict_clients.keys():
        clients[key] = dict_clients[key].tolist()
    
    with open(dir_path + 'iid_{}_{}.json'.format('balanced', num_clients), 'w') as f:
        json.dump(clients, f)
        print('finish write dataset')
    
    return clients


def unbalanced_iid(dir_path, dataset, num_clients, num_classes, unbalance_sigma=0.3):
    make_dirs(dir_path)
    
    classes_data = []
    samples_per_client = int(len(dataset) / num_clients)
    
    labels = np.array(dataset.targets)
    
    for i in range(num_classes):
        idx = np.argwhere(labels == i).reshape(-1)
        print(f'the {i}th class before shuffle: {idx}')
        random.shuffle(idx)
        print(f'the {i}th class after shuflle: {idx} ')
        classes_data.append(idx)
    print('每个类别的数量', [len(v) for v in classes_data])
    
    dict_clients = {user: np.array([]).astype(int) for user in range(num_clients)}
    for i in range(samples_per_client):
        for user in range(num_clients):
            l = (user + i) % num_classes
            if np.random.uniform() >= unbalance_sigma:
                data = np.random.choice(classes_data[l], 1, replace=False)
                dict_clients[user] = np.append(dict_clients[user], data)
    
    clients = {}
    for key in dict_clients.keys():
        clients[key] = dict_clients[key].tolist()
    
    with open(dir_path + 'iid_{}_{}.json'.format('unbalanced', num_clients), 'w') as f:
        json.dump(clients, f)
        print('finish write dataset')
    
    return clients


def balance_homo_partition(dir_path, dataset, num_clients, alpha=0.3):
    make_dirs(dir_path)
    
    num_samples = len(dataset.targets)
    
    num_samples_per_client = int(num_samples / num_clients)
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)
    
    rand_perm = np.random.permutation(num_samples)
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    
    client_indices_pairs = [(cid, idxs.tolist()) for cid, idxs in enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    clients = dict(client_indices_pairs)
    
    with open(dir_path + 'iid_{}_{}.json'.format('balance_homo', num_clients), 'w') as f:
        json.dump(clients, f)
        print('finish write dataset')
    
    return clients


def unbalance_homo_partition(dir_path, dataset, num_clients, sigma, alpha=0.3):
    make_dirs(dir_path)
    
    num_samples = len(dataset.targets)
    
    num_samples_per_client = int(num_samples / num_clients)
    if sigma != 0:
        client_sample_nums = np.random.lognormal(mean=np.log(num_samples_per_client),
                                                 sigma=sigma,
                                                 size=num_clients)
        client_sample_nums = (
            client_sample_nums / np.sum(client_sample_nums) * num_samples).astype(int)
        diff = np.sum(client_sample_nums) - num_samples  # diff <= 0
        
        # Add/Subtract the excess number starting from first client
        if diff != 0:
            for cid in range(num_clients):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break
    else:
        client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)
    
    rand_perm = np.random.permutation(num_samples)
    num_cumsum = np.cumsum(client_sample_nums).astype(int)
    
    client_indices_pairs = [(cid, idxs.tolist()) for cid, idxs in enumerate(np.split(rand_perm, num_cumsum)[:-1])]
    clients = dict(client_indices_pairs)
    
    with open(dir_path + 'iid_{}_{}.json'.format('unbalance_homo', num_clients), 'w') as f:
        json.dump(clients, f)
        print('finish write dataset')
    
    return clients


def shard_noniid(dir_path, dataset, num_clients, num_shards, num_imgs):
    make_dirs(dir_path)
    
    idx_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([]).astype(int) for i in range(num_clients)}
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
    
    clients = {}
    for key in dict_clients.keys():
        clients[key] = dict_clients[key].tolist()
    
    with open(dir_path + 'noniid_{}_{}.json'.format('shard', num_clients), 'w') as f:
        json.dump(clients, f)
        print('finish write dataset')
    
    return clients


def unequal_shard_noniid(dir_path, dataset, num_clients, num_shards, num_imgs):
    make_dirs(dir_path)
    
    # 60,000 training imgs --> 50 imgs/shard X 1200 shards
    # num_shards, num_imgs = 1000, 50
    idx_shard = [i for i in range(num_shards)]
    dict_clients = {i: np.array([]).astype(int) for i in range(num_clients)}
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
    
    clients = {}
    for key in dict_clients.keys():
        clients[key] = dict_clients[key].tolist()
    
    with open(dir_path + 'noniid_{}_{}.json'.format('unbalanced_shard', num_clients), 'w') as f:
        json.dump(clients, f)
        print('finish write dataset')
    
    return clients


def hetero_dirichlet_noniid(dir_path, dataset, num_clients, num_classes, alpha, min_require_size=None):
    make_dirs(dir_path)
    
    if min_require_size is None:
        min_require_size = num_classes
    
    num_samples = len(dataset.targets)
    min_size = 0
    
    labels = dataset.targets
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    while min_size < min_require_size:
        client_idx = [[] for _ in range(num_clients)]
        for k in range(num_classes):
            idx_k = np.argwhere(labels == k).reshape(-1)
            np.random.shuffle(idx_k)
            
            proportions = np.random.dirichlet(np.repeat(alpha, num_clients))
            
            proportions = np.array(
                [p * (len(idx_j) < num_samples / num_clients) for p, idx_j in zip(proportions, client_idx)])
            proportions = proportions / proportions.sum()
            proportions = (np.cumsum(proportions) * len(idx_k)).astype(int)[:-1]
            client_idx = [idx_j + idx.tolist() for idx_j, idx in zip(client_idx, np.split(idx_k, proportions))]
            min_size = min([len(idx_j) for idx_j in client_idx])
    
    clients = {}
    for j in range(num_clients):
        np.random.shuffle(client_idx[j])
        clients[j] = client_idx[j]
    
    with open(dir_path + 'noniid_{}_{}.json'.format('hetero_dirichlet', num_clients), 'w') as f:
        json.dump(clients, f)
        print('finish write dataset')
    
    return clients


def balanced_dirichlet_noniid(dir_path, dataset, num_clients, num_classes, alpha, verbose=True):
    make_dirs(dir_path)
    
    num_samples_per_client = int(len(dataset) / num_clients)
    client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)
    
    labels = np.array(dataset.targets)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    rand_perm = np.random.permutation(labels.shape[0])
    targets = labels[rand_perm]
    
    class_priors = np.random.dirichlet(alpha=[alpha] * num_classes, size=num_clients)
    prior_cumsum = np.cumsum(class_priors, axis=1)
    
    idx_list = [np.argwhere(targets == i).reshape(-1) for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]
    
    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in range(num_clients)]
    
    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = idx_list[curr_class][class_amount[curr_class]]
            
            break
    
    clients = {cid: client_indices[cid].tolist() for cid in range(num_clients)}
    
    with open(dir_path + 'noniid_{}_{}.json'.format('balanced_dirichlet', num_clients), 'w') as f:
        json.dump(clients, f)
        print('finish write dataset')
    
    return clients


def unbalanced_dirichlet_noniid(dir_path, dataset, num_clients, num_classes, alpha, sigma, verbose=True):
    make_dirs(dir_path)
    
    num_samples = len(dataset.targets)
    num_samples_per_client = int(num_samples / num_clients)
    
    if sigma != 0:
        client_sample_nums = np.random.lognormal(mean=np.log(num_samples_per_client), sigma=sigma, size=num_clients)
        client_sample_nums = (client_sample_nums / np.sum(client_sample_nums) * num_samples).astype(int)
        diff = np.sum(client_sample_nums) - num_samples  # diff <= 0
        
        # Add/Subtract the excess number starting from first client
        if diff != 0:
            for cid in range(num_clients):
                if client_sample_nums[cid] > diff:
                    client_sample_nums[cid] -= diff
                    break
    else:
        client_sample_nums = (np.ones(num_clients) * num_samples_per_client).astype(int)
    
    # handle labels
    labels = np.array(dataset.targets)
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    rand_perm = np.random.permutation(labels.shape[0])
    labels = labels[rand_perm]
    
    class_priors = np.random.dirichlet(alpha=[alpha] * num_classes, size=num_clients)
    
    prior_cumsum = np.cumsum(class_priors, axis=1)
    idx_list = [np.argwhere(labels == i).reshape(-1) for i in range(num_classes)]
    class_amount = [len(idx_list[i]) for i in range(num_classes)]
    
    client_indices = [np.zeros(client_sample_nums[cid]).astype(np.int64) for cid in range(num_clients)]
    
    while np.sum(client_sample_nums) != 0:
        curr_cid = np.random.randint(num_clients)
        # If current node is full resample a client
        if verbose:
            print('Remaining Data: %d' % np.sum(client_sample_nums))
        if client_sample_nums[curr_cid] <= 0:
            continue
        client_sample_nums[curr_cid] -= 1
        curr_prior = prior_cumsum[curr_cid]
        while True:
            curr_class = np.argmax(np.random.uniform() <= curr_prior)
            # Redraw class label if no rest in current class samples
            if class_amount[curr_class] <= 0:
                continue
            class_amount[curr_class] -= 1
            client_indices[curr_cid][client_sample_nums[curr_cid]] = idx_list[curr_class][class_amount[curr_class]]
            
            break
    
    clients = {cid: client_indices[cid].tolist() for cid in range(num_clients)}
    
    with open(dir_path + 'noniid_{}_{}.json'.format('unbalanced_dirichlet', num_clients), 'w') as f:
        json.dump(clients, f)
        print('finish write dataset')
    
    return clients


def quantity_based_label_distribution_noniid(dir_path, dataset, num_clients, num_classes, major_classes_num):
    make_dirs(dir_path)
    
    labels = dataset.targets
    if not isinstance(labels, np.ndarray):
        labels = np.array(labels)
    
    idx_batch = [np.ndarray(0, dtype=np.int64) for _ in range(num_clients)]
    
    # only for major_classes_num < num_classes.
    # if major_classes_num = num_classes, it equals to IID partition
    times = [0 for _ in range(num_classes)]
    contain = []
    for cid in range(num_clients):
        current = [cid % num_classes]
        times[cid % num_classes] += 1
        j = 1
        while j < major_classes_num:
            ind = np.random.randint(num_classes)
            if ind not in current:
                j += 1
                current.append(ind)
                times[ind] += 1
        contain.append(current)
    
    for k in range(num_classes):
        idx_k = np.argwhere(labels == k).reshape(-1)
        
        np.random.shuffle(idx_k)
        split = np.array_split(idx_k, times[k])
        ids = 0
        for cid in range(num_clients):
            if k in contain[cid]:
                idx_batch[cid] = np.append(idx_batch[cid], split[ids])
                ids += 1
    
    clients = {cid: idx_batch[cid].tolist() for cid in range(num_clients)}
    
    with open(dir_path + 'noniid_{}_{}.json'.format('quantity_label', num_clients), 'w') as f:
        json.dump(clients, f)
        print('finish write dataset')
    
    return clients


def noise_based_feature_distribution_skew_partition():
    pass


if __name__ == "__main__":
    pass
