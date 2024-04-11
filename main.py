# -*- coding: utf-8 -*-
# @Time    : 2023/4/14

import copy
import os
import argparse
import random
import time
import numpy as np
import torch
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

from system.servers.serveravg import FedAvg
from system.servers.serverflame import FedFlame
from system.servers.serverprox import FedProx
from system.servers.servernova import FedNova
from system.servers.serverscaffold import FedScaffold
from system.servers.servermoon import FedMoon

from system.model.models import *
from system.utils.result_utils import Metrics


def start_train(args):
    print('--------------------- start ------------------------')
    start = time.time()
    model_name = args.model
    metrics = Metrics(args)
    
    # generated model
    if model_name == "CNNMnist":
        args.model = CNNMnist().to(args.device)
    elif model_name == "CNNFmnist":
        args.model = CNNFmnist().to(args.device)
    elif model_name == "CNNCifar10":
        args.model = CNNCifar10().to(args.device)
    else:
        raise NotImplementedError
    
    print(args.model.parameters())
    
    # select algorithm
    if args.algorithm == 'FedAvg':
        server = FedAvg(args, metrics)
    elif args.algorithm == 'FedFlame':
        server = FedFlame(args, metrics)
    elif args.algorithm == 'FedProx':
        server = FedProx(args, metrics)
    elif args.algorithm == 'FedNova':
        server = FedNova(args, metrics)
    elif args.algorithm == 'FedScaffold':
        server = FedScaffold(args, metrics)
    elif args.algorithm == 'FedMoon':
        head = copy.deepcopy(args.model.fc)
        args.model.fc = nn.Identity()
        args.model = BaseHeadSplit(args.model, head)
        server = FedMoon(args, metrics)
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
    
    # model setting
    parser.add_argument('--algorithm', type=str, default='FedAvg', help='name of training framework;')
    parser.add_argument('--dataset', type=str, default='fmnist', help='name of dataset;')
    parser.add_argument('--model', type=str, default='CNNFmnist1', help='name of model;')
    parser.add_argument('--global_epoch', type=int, default=100, help="number of rounds of global training")
    parser.add_argument('--server_learn_rate', type=float, default=0.005, help="model learning rate")
    parser.add_argument('--local_epoch', type=int, default=5, help="number of rounds of local training")
    parser.add_argument('--local_learn_rate', type=float, default=0.01, help="model learning rate")
    parser.add_argument('--local_bs', type=int, default=16, help="local batch size")
    parser.add_argument('--dataiid', type=int, default=6, help="chosse dataset format")
    parser.add_argument('--major_classes_num', type=int, default=3,
                        help="lable distribution skew,1: class 1, 2:class 2")
    parser.add_argument('--verbose', type=int, default=1, help='verbose')
    parser.add_argument('--eval_every', type=int, default=1, help='evaluate every ____ rounds;')

    # client rate
    parser.add_argument('--num_clients', type=int, default=100, help="number of users: K")
    parser.add_argument('--rclient', type=bool, default=False, help="random choose client number")
    parser.add_argument('--rc_rate', type=float, default=1.1,
                        help="The ratio which the client randomly participates in training")
    parser.add_argument("--train_slow", type=float, default=0.0,
                        help="The rate for slow clients when training locally")
    parser.add_argument("--send_slow", type=float, default=0.0,
                        help="The rate for slow clients when sending global model")

    # Differential privay
    parser.add_argument('--diyldp', type=bool, default=False, help='DIY adds LDP')
    parser.add_argument('--diycdp', type=bool, default=False, help='DIY adds CDP')
    parser.add_argument('--opacus', type=bool, default=False, help='using opacus adds DP')
    parser.add_argument('--epsilon', type=float, default=1, help='use dp with train. dp, ldp, RDP')
    parser.add_argument('--delta', type=float, default=5e-6, help='use dp with train. dp, ldp, RDP')
    parser.add_argument('--mechanism', type=str, default='laplace',
                        help='type of local randomizer: gaussian, laplace, krr')

    # sparsification technology
    parser.add_argument('--clip_c', type=float, default=0.05, help='L2 norm clipping threshold')
    parser.add_argument('--com_rate', type=int, default=5, help='parameters compression rate, 1 for no compression')
    parser.add_argument('--mp_rate', type=float, default=1, help='under factor for mp=m/mp_rate')

    # paper parameters in PFL
    parser.add_argument("--mu", type=float, default=0.01, help="Proximal rate for paper FedProx")
    parser.add_argument("--tau", type=float, default=1.0, help="ratio for paper MOON")
    

    # Others parameters
    parser.add_argument('--seed', type=int, default=0, help='seed for randomness;')
    parser.add_argument('--device', help="device is gpu or cpu", type=str, default='cuda')
    parser.add_argument('--test', type=bool, default=True, help="test model")

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
    print("Local learning rate: ".rjust(50), args.local_learn_rate)
    print("dataset distribution: ".rjust(50), args.dataiid)

    print("Client random participation probability: ".rjust(50), args.rc_rate)
    print("L2 norm clipping threshold: ".rjust(50), args.clip_c)
    print("compression rate: ".rjust(50), args.com_rate)

    print("=" * 100)

    # set random seed
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    torch.manual_seed(123 + args.seed)  # 为CPU设置随机种子
    torch.cuda.manual_seed(123 + args.seed)

    start_train(args)
