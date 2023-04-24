#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2023/4/14 11:04
# @File    : main.py
# @Author  : Richard Yang

import os
import argparse
import random
import time
import numpy as np
import torch

import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from system.servers.serveravg import FedAvg
from system.servers.serverprox import FedProx

from system.model.models import *
from system.utils.result_utils import Metrics


def start_train(args):
    print('--------------------- start ------------------------')
    start = time.time()
    model_name = args.model
    metrics = Metrics(args)
    
    # generated model
    if model_name == "CNNMnist1":
        args.model = CNNMnist1().to(args.device)
    elif model_name == "CNNFmnist1":
        args.model = CNNFmnist1().to(args.device)
    elif model_name == "CNNCifar1":
        args.model = CNNCifar1().to(args.device)
    else:
        raise NotImplementedError
    
    print(args.model.parameters())
    
    # select algorithm
    if args.algorithm == 'FedAvg':
        server = FedAvg(args, metrics)
    elif args.algorithm == 'FedProx':
        server = FedProx(args, metrics)
    else:
        raise NotImplementedError
    
    server.train()
    
    metrics.all_time.append(time.time() - start)
    args.model = model_name
    metrics.write()
    print(f'\n All done! All Epoch Costs Time: {time.time() - start:.2f} \n')


if __name__ == '__main__':
    # 计算模型整体训练时间
    total_start = time.time()
    
    # parse arguements
    parser = argparse.ArgumentParser()
    
    # main setting
    parser.add_argument('--algorithm', type=str, default='FedAvg', help='name of training framework;')
    parser.add_argument('--dataset', type=str, default='mnist', help='name of dataset;')
    parser.add_argument('--model', type=str, default='CNNMnist1', help='name of model;')
    
    # global
    parser.add_argument('--global_epoch', type=int, default=100, help="number of rounds of global training")
    parser.add_argument('--learn_rate', type=float, default=0.001, help="model learning rate")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--eval_every', type=int, default=1, help='evaluate every ____ rounds;')
    parser.add_argument('--seed', type=int, default=0, help='seed for randomness;')
    
    # local
    parser.add_argument('--local_epoch', type=int, default=5, help="number of rounds of local training")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size")
    
    # client rate
    parser.add_argument('--isrclient', type=bool, default=False, help="random choose client number")
    parser.add_argument('--rc_rate', type=float, default=0.75,
                        help="The ratio which the client randomly participates in training")
    
    # for sparsification
    parser.add_argument('--norm', type=float, default=10, help='L2 norm clipping threshold')
    parser.add_argument('--rate', type=int, default=1, help='compression rate, 1 for no compression')
    
    # for padding
    parser.add_argument('--mp_rate', type=float, default=1, help='under factor for mp=m/mp_rate')
    
    # Differential privay
    parser.add_argument('--delta', type=float, default=5e-6, help='use dp with train. dp, ldp, RDP')
    parser.add_argument('--epsilon', type=float, default=0.81251, help='use dp with train. dp, ldp, RDP')
    parser.add_argument('--mechanism', type=str, default='gaussian',
                        help='type of local randomizer: gaussian, laplace, krr')
    
    # other parameter
    parser.add_argument('--dataiid', type=int, default=3, help="chosse dataset format")
    parser.add_argument('--device', help="device is gpu or cpu", type=str, default='cuda')
    parser.add_argument('--num_clients', type=int, default=100, help="number of users: K")
    
    # personalized FL parameters
    parser.add_argument("--mu", type=float, default=0, help="Proximal rate for FedProx")
    
    args = parser.parse_args()
    
    print("=" * 100)
    
    if args.device == "cuda" and torch.cuda.is_available():
        print('Using Device is: '.rjust(50), args.device)
        print("Count Cuda Device: ".rjust(50), torch.cuda.device_count())
        print("Using Cuda Device index: ".rjust(50), torch.cuda.current_device())
    else:
        args.device = 'cpu'
        print('Using Device is: '.rjust(50), args.device)
    
    print("Algorithm: ".rjust(50), args.algorithm)
    print("Dataset: ".rjust(50), args.dataset)
    print("Model name: ".rjust(50), args.model)
    print("Global epoch: ".rjust(50), args.global_epoch)
    print("Total number of clients: ".rjust(50), args.num_clients)
    
    print("Local epoch: ".rjust(50), args.local_epoch)
    print("Local batch size: ".rjust(50), args.local_bs)
    print("Local learing rate: ".rjust(50), args.learn_rate)
    
    print("Client random participation probability: ".rjust(50), args.rc_rate)
    print("L2 norm clipping threshold: ".rjust(50), args.norm)
    print("compression rate: ".rjust(50), args.rate)
    
    print("=" * 100)
    
    # set random seed
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    torch.manual_seed(123 + args.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(123 + args.seed)
    
    start_train(args)
