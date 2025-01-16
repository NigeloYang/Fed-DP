# Federated Learning Platform

**We provide a friendly foundational platform for beginners who intend to start Federated Learning (FL).[[1]](#1)**

Federated learning (FL), proposed by Google at the very beginning, is recently a burgeoning research area of machine
learning, which aims to protect individual data privacy in the distributed machine learning processes, especially in
finance, smart healthcare, and edge computing. Different from traditional data-centered distributed machine learning,
participants in the FL setting utilize localized data to train local models, then leverages specific strategies with
other participants to acquire the final model collaboratively, avoiding direct data-sharing behavior.

To ease the hassle of implementing the FL algorithm for beginners, we set up a highly customizable framework Fed-DP 
in this work. Fed-DP provides the necessary modules for FL, such as basic model, model optimization, data partitioning,
etc. In the future, our framework will also include the following:

- Shuffle Model (SM) [[2]](#2)
- Differential Privacy (DP) [[3]](#3)
- - Shuffle Model and Differential Privacy (SDP) [[2,5]](#5)
- Personalized Federated Learning (PFL) [[4]](#4)



## Methods with Code (updating)
| Model    | Paper                                                                                                                         | Time         | Type     | Official Code                                  | Fed-DP Code       | Methods                | DP          |
|:---------|:------------------------------------------------------------------------------------------------------------------------------|:-------------|:---------|:-----------------------------------------------|:------------------|:-----------------------|:------------|
| FedAvg   | [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html) | AISTATS 2017 | FL       | None                                           | [Code](./main.py) |                        | CDP,LDP     |
| FedProx  | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)                                          | MLsys 2020   | PFL      | [Code](https://github.com/litian96/FedProx)    | [Code](./main.py) | Regularized Local Loss | CDP,LDP     |
| FedNova  | [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481)      | NeurIPS 2020 | PFL      | [Code](https://github.com/JYWa/FedNova)        | [Code](./main.py) | Regularized Local Loss | CDP,LDP     |
| SCAFFOLD | [SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a.html)     | ICML 2020    | PFL      | None                                           | [Code](./main.py) | Regularized Local Loss | CDP,LDP     |
| MOON     | [Model-Contrastive Federated Learning](https://arxiv.org/abs/2103.16257)                                                      | CVPR 2021    | PFL      | [Code](https://github.com/QinbinLi/MOON)       | [Code](./main.py) | Regularized Local Loss | CDP,LDP     |
| FLAME    | [FLAME: Differentially Private Federated Learning in the Shuffle Model](https://arxiv.org/abs/2009.08063)                     | AAAI 2021    | LDP-FL   | [Code](https://github.com/Rachelxuan11/FLAME)  | [Code](./main.py) | SDP                    | CDP,LDP,SDP |


## Create Environment 
Command to create a new environment in anaconda
  ```bash
  git clone https://github.com/NigeloYang/Fed-DP.git
  cd Fed-DP
  conda env create -f Fed_dp_linux.yaml/Fed_dp_win.yaml
  conda activate env_name
  ```

## Datasets and Separation
This project has generated Ten data distributions of MNIST, Fashion-Mnist and Cifar-10, as follows:
- Balanced-IID (two)
- Unbalanced-IID (two)
- Balanced Shards
- Unbalanced Shards
- Quantity-Based-Label-Distribution
- Hetero-Dirichlet
- Balanced-Dirichlet
- Unbalanced-Dirichlet

Generation Data Methods:
```bash
cd Fed-DP/dataset
python generate_data.py  # for MNIST iid_balanced_100
```
For details, please see: [Generate Datasets](./dataset/README.md)

## How to start simulating

- Build dataset: [Datasets](./dataset/README.md)
- Train and evaluate the model:
    ```bash
    python main.py --algorithm=FedAvg  # for FedAvg and MNIST,dataset distribution --dataiid=1
    ```

## Development progress
- [x] Base Federated Learning Framework
- [x] Integrated Differential Privacy (Updating,next add meta`s DP tool -> opacus )
- [x] Integrated shuffle Model
- [x] Integrated HE (Updating)
- [X] Integrated MPC (Updating) 
- [x] Integrated Personalized Federated Learning to solve Non-IID
- [x] Generated Data: IID, Non-IID


## Citation
```
@misc{yang2023, 
  author = {Qiantao Yang}, 
  title = {Fed-DP}, 
  howpublished = {https://github.com/NigeloYang/Fed-DP}, 
  year = {2023}
}
```

## Contact

For technical issues related to Fed-DP development, please contact me through Github issues or email :satisfied: 

📧 yangqiantao@outlook.com


## References
<a id="1">[1]</a> Yang, Q., Liu, Y., Chen, T., & Tong, Y. (2019). Federated Machine Learning: Concept and Applications. ACM Trans. Intell. Syst. Technol., 10(2), 12:11-12:19.

<a id="1">[2]</a> Cheu, A. (2021). Differential Privacy in the Shuffle Model: A Survey of Separations. CoRR, abs/2107.11839.

<a id="1">[3]</a> Cynthia, D., & Aaron, R. (2014). The Algorithmic Foundations of Differential Privacy.

<a id="1">[4]</a> Tan, A. Z., Yu, H., Cui, L., & Yang, Q. (2022). Towards Personalized Federated Learning. IEEE Transactions on Neural Networks and Learning Systems, 1-17. [Chinese](https://zhuanlan.zhihu.com/p/621188058)

<a id="1">[5]</a> Balle, B., et al. (2019). The Privacy Blanket of the Shuffle Model: 638-667.