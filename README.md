# Federated Learning Platform

***We expose this user-friendly platform for beginners who intend to start federated learning (FL) study with shuffle model and differential privacy (DP).*** 


## Introuduction
Federated learning (FL), proposed by Google at the very beginning, is recently a burgeoning research area of machine 
learning, which aims to protect individual data privacy in the distributed machine learning processes, especially in 
finance, smart healthcare, and edge computing. Different from traditional data-centered distributed machine learning, 
participants in the FL setting utilize localized data to train local models, then leverages specific strategies with 
other participants to acquire the final model collaboratively, avoiding direct data-sharing behavior.


## Methods with Code (updating)
| Model     | Paper                                                                                                                          | Time          | Source Code | Fed-DP Code       |
|:----------|:-------------------------------------------------------------------------------------------------------------------------------|:--------------|:------------|:------------------|
| FedAvg    | [Communication-Efficient Learning of Deep Networks from Decentralized Data](http://proceedings.mlr.press/v54/mcmahan17a.html)  | AISTATS 2017  | None        | [code](./main.py) |


## Datasets and Separation (updating)
No instructions have been provided at this time


## How to start simulating
No instructions have been provided at this time


## Contact
For technical issues related to Fed-DP development, please contact me through Github issues or email:
- 📧 yangqiantao@126.com


## Development progress
- [x] Base Federated Learning Framework
- [ ] Integrated Differential Privacy
- [ ] Integrated shuffle Model
- [ ] Integrated Personalized Federated Learning to solve Non-IID
- [x] Generated Data: IID, Non-IID