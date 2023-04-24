#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/13 21:37
# @File    : generate_fmnist.py
# @Author  : Richard Yang
import argparse
import torch
from torchvision import datasets, transforms

from utils.dataset_util import balanced_iid, shard_noniid, unbanlanced_shard_noniid


def fmnist_100(args):
    print('------start fmnist format------')
    
    fmnist = datasets.MNIST(
        root='local_dataset/', train=True, download=True,
        transform=transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))])
    )
    
    dir_path = 'formatdata/fmnist/'
    
    if args.dataiid == 1:
        client = balanced_iid(dir_path, fmnist, args.num_clients, num_classes=args.num_classes)
        print('-----------------finished fmnist iid-balanced-----------------')
        print(f'client[0]: \n {client[0]}  \n length: {len(client[0])} \n type: {type(client[0])}')
        print(f'client[1]: \n {client[1]}  \n length: {len(client[1])} \n type: {type(client[1])}')
    elif args.dataiid == 2:
        print('not exist unbalanced-iid')
    elif args.dataiid == 3:
        client = shard_noniid(dir_path, fmnist, args.num_clients, num_shards=args.num_shard, num_imgs=args.num_img)
        print('-----------------finished fmnist noniid-shard-----------------')
        print(f'client[0]: \n {client[0]}  \n length: {len(client[0])} \n type: {type(client[0])}')
        print(f'client[1]: \n {client[1]}  \n length: {len(client[1])} \n type: {type(client[1])}')
    elif args.dataiid == 4:
        client = unbanlanced_shard_noniid(dir_path, fmnist, args.num_clients, num_shards=1200, num_imgs=50)
        print('-----------------finished fmnist noniid-unbalanced-shard-----------------')
        print(f'client[0]: \n {client[0]}  \n length: {len(client[0])} \n type: {type(client[0])}')
        print(f'client[1]: \n {client[1]}  \n length: {len(client[1])} \n type: {type(client[1])}')
    else:
        print('dataformat iid = [1,2,3], generate dataformat [balanced_iid, shard_noniid, unbanlanced_shard_noniid]')


if __name__ == "__main__":
    # parse arguements
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_clients', type=int, default=100, help='Number of clients')
    parser.add_argument('--num_shard', type=int, default=200, help='Shards partition, number of shards')
    parser.add_argument('--num_img', type=int, default=300, help='Shards partition, number of images')
    parser.add_argument('--num_classes', type=int, default=10, help='Shards partition, number of images')
    parser.add_argument('--dataiid', type=int, default=3,
                        help='Data distribution target,1:balanced_iid,2:unbalanced_iid...,default: 3')
    args = parser.parse_args()
    
    fmnist_100(args)
