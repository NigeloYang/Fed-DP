# -*- coding: utf-8 -*-
# @Time    : 2024/4/11

from .fedflame_privacy_budget import *

def get_budget(FedName):
    if FedName == 'FedFLAME':
        dim = 50618
        e_l = 506.18
        flame = FedFLAME(rate=50, m_p=int(1000), dim=dim, m=1000, e_l=e_l)
        flame.print()
        flame.no_sub_amplification()
        
if __name__ == "__main__":
    get_budget('FedFLAME')
