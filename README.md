# Federated Learning Platform

***We expose this user-friendly platform for beginners who intend to start federated learning (FL) study with shuffle
model and differential privacy (DP).***

## Introuduction

Federated learning (FL), proposed by Google at the very beginning, is recently a burgeoning research area of machine
learning, which aims to protect individual data privacy in the distributed machine learning processes, especially in
finance, smart healthcare, and edge computing. Different from traditional data-centered distributed machine learning,
participants in the FL setting utilize localized data to train local models, then leverages specific strategies with
other participants to acquire the final model collaboratively, avoiding direct data-sharing behavior.

## Methods with Code (updating)

| Model    | Paper                                                                                                                         | Time         | Type | Source Code                                 | Fed-DP Code          |
|:---------|:------------------------------------------------------------------------------------------------------------------------------|:-------------|:-----|:--------------------------------------------|:---------------------|
| FedAvg   | [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html) | AISTATS 2017 | FL   | None                                        | [Code](./main.py)    |
| FedProx  | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)                                          | MLsys 2020   | PFL  | [Code](https://github.com/litian96/FedProx) | [Code](./main.py)    |
| FedNova  | [Tackling the Objective Inconsistency Problem in Heterogeneous Federated Optimization](https://arxiv.org/abs/2007.07481)      | NeurIPS 2020 | PFL  | [Code](https://github.com/JYWa/FedNova)     | [Code](./main.py)    |
| SCAFFOLD | [ SCAFFOLD: Stochastic Controlled Averaging for Federated Learning](http://proceedings.mlr.press/v119/karimireddy20a.html)    | ICML 2020    | PFL  | None                                        | [Code](./main.py)    |


## Datasets and Separation (updating)

This project has generated three data distributions of MNIST, Fashion-Mnist and Cifar-10, namely IID, Balanced Shards
partition, and Unbalanced Shards partition.

- Generation Data:
    ```bash
    git clone https://github.com/NigeloYang/Fed-DP.git
    cd Fed-DP/dataset
    python generate_data.py --datatset=mnist --num_clients=100 --num_shard=200 --num_img=300 --num_classes=10 --dataiid=3  # for MNIST noniid_shard_100
    ```

For details, please see: dataset/generate_data.py

## How to start simulating

- Build dataset: [Datasets](#Datasets and Separation) (updating)

- Train and evaluate the model:
    ```bash
    python main.py --algorithm=FedAvg --dataset=mnist --model=CNNMnist1 # for FedAvg and MNIST
    ```

## Contact

For technical issues related to Fed-DP development, please contact me through Github issues or email:

- 📧 yangqiantao@126.com

## Development progress

- [x] Base Federated Learning Framework
- [x] Integrated Differential Privacy (Updating)
- [ ] Integrated shuffle Model
- [x] Integrated Personalized Federated Learning to solve Non-IID (Updating)
- [x] Generated Data: IID, Non-IID (Updating)