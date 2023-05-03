# -*- coding: utf-8 -*-
# @Time    : 2023/4/14

import torch
from torch.optim import Optimizer


class FedProxOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, mu=0.0):
        default = dict(lr=lr, mu=mu)
        super(FedProxOptimizer, self).__init__(params, default)
    
    @torch.no_grad()
    def step(self, global_params, device):
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                g = g.to(device)
                d_p = p.grad.data + group['mu'] * (p.data - g.data)
                p.data.add_(d_p, alpha=-group['lr'])


class SCAFFOLDOptimizer(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(SCAFFOLDOptimizer, self).__init__(params, defaults)
    
    def step(self, global_c, client_c):
        for group in self.param_groups:
            for p, g_c, c_c in zip(group['params'], global_c, client_c):
                d_p = p.grad.data - c_c.data + g_c.data
                p.data.add_(d_p, alpha=-group['lr'])
