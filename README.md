# Federated Learning Platform

***We expose this user-friendly platform for beginners who intend to start federated learning (FL) study with shuffle model and differential privacy (DP).*** 


## Introuduction
Federated learning (FL), proposed by Google at the very beginning, is recently a burgeoning research area of machine 
learning, which aims to protect individual data privacy in the distributed machine learning processes, especially in 
finance, smart healthcare, and edge computing. Different from traditional data-centered distributed machine learning, 
participants in the FL setting utilize localized data to train local models, then leverages specific strategies with 
other participants to acquire the final model collaboratively, avoiding direct data-sharing behavior.


## Methods with Code (updating)
| Model    | Paper                                                                                                                         | Time         | Source Code                                 | Fed-DP Code                            |
|:---------|:------------------------------------------------------------------------------------------------------------------------------|:-------------|:--------------------------------------------|:---------------------------------------|
| FedAvg   | [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html) | AISTATS 2017 | None                                        | [Code](./system/servers/serveravg.py)  |
| FedProx  | [Federated Optimization in Heterogeneous Networks](https://arxiv.org/abs/1812.06127)                                          | MLsys 2020   | [Code](https://github.com/litian96/FedProx) | [Code](./system/servers/serverprox.py) |


## Datasets and Separation (updating)
This project has generated three data distributions of MNIST, Fashion-Mnist and Cifar-10, namely IID, Balanced Shards 
partition, and Unbalanced Shards partition.

- Generation Data:
    ```bash
    git clone https://github.com/NigeloYang/Fed-DP.git
    cd Fed-DP/dataset
    python generate_mnist.py --num_clients=100 --num_shard=200 --num_img=300 --num_classes=10 --dataiid=3  # for MNIST noniid_shard_100
    ```

For details, please see: dataset/generate_**.py


## How to start simulating
- Build dataset: [Datasets](#Datasets and Separation (updating))

- Train and evaluate the model:
    ```bash
    python main.py --algorithm=FedAvg --dataset=mnist --model=CNNMnist1 # for FedAvg and MNIST
    ```


## Contact
For technical issues related to Fed-DP development, please contact me through Github issues or email:
- 📧 yangqiantao@126.com


## Development progress
- [x] Base Federated Learning Framework
- [ ] Integrated Differential Privacy
- [ ] Integrated shuffle Model
- [ ] Integrated Personalized Federated Learning to solve Non-IID
- [x] Generated Data: IID, Non-IID