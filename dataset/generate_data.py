# -*- coding: utf-8 -*-
# @Time    : 2023/4/27

import argparse
from utils.DataPartitioner import DataPartitioner


def run():
    # parse arguements
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--dataset', type=str, default='mnist', help='dataset type')
    parser.add_argument('--isbalance', type=bool, default=True, help='dataset distribution is balanced')
    parser.add_argument('--partition', type=str, default='iid', help='dataset partition type')
    parser.add_argument('--num_clients', type=int, default=100, help='Number of clients')
    parser.add_argument('--num_classes', type=int, default=10, help='Shards partition, number of images')
    parser.add_argument('--num_shard', type=int, default=300,
                        help='Shard partition, number of shards. When num_shard = 1000, num_img = 50')
    parser.add_argument('--num_img', type=int, default=200,
                        help='Shards partition, number of images. When num_img = 50,num_shard = 1000')
    parser.add_argument('--alpha', type=int, default=0.3, help='Dirichlet ratio')
    parser.add_argument('--sigma', type=int, default=0.3, help='unbalanced ratio')
    parser.add_argument('--dataplot', type=int, default=1, help='plot data distribution type')
    parser.add_argument('--min_require_size', type=int, default=10, help='min_require_size')
    parser.add_argument('--major_classes_num', type=int, default=3, help='major_classes_num')
    
    args = parser.parse_args()
    
    partition_data = DataPartitioner(args)
    
    client = partition_data.get_partition()
    print(f'client[0]: \n {client[0]}  \n length: {len(client[0])}')
    print(f'client[1]: \n {client[1]}  \n length: {len(client[1])}')
    
    partition_data.data_plot()


if __name__ == '__main__':
    run()
