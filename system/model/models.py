import torch
from torch import nn
import torch.nn.functional as F
from torchinfo import summary


class CNNMnist1(nn.Module):
    def __init__(self):
        super(CNNMnist1, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.fc1 = nn.Linear(14 * 14 * 8, 32)
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNMnist2(nn.Module):
    def __init__(self):
        super(CNNMnist2, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 256)
        self.fc2 = nn.Linear(256, 50)
        self.fc3 = nn.Linear(50, 10)
    
    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, x.shape[1] * x.shape[2] * x.shape[3])
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNFmnist1(nn.Module):
    def __init__(self):
        super(CNNFmnist1, self).__init__()
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 4 * 4, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 4 * 4)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


class CNNFmnist2(nn.Module):
    def __init__(self):
        super(CNNFmnist2, self).__init__()
        self.conv1 = nn.Conv2d(1, 8, kernel_size=(3, 3), stride=(1, 1), padding=1)
        self.fc1 = nn.Linear(14 * 14 * 8, 32)
        self.fc2 = nn.Linear(32, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), kernel_size=2, stride=2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar1(nn.Module):
    def __init__(self):
        super(CNNCifar1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 200)
        self.fc2 = nn.Linear(200, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)


class CNNCifar2(nn.Module):
    def __init__(self):
        super(CNNCifar2, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3)
        self.fc1 = nn.Linear(15 * 15 * 16, 64)
        self.fc2 = nn.Linear(64, 10)
    
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), 2, 2)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class CNNCifar3(nn.Module):
    def __init__(self):
        super(CNNCifar3, self).__init__()
        # 卷积层 (32x32x3的图像)
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        # 卷积层(16x16x16)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        # 卷积层(8x8x32)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        # 最大池化层
        self.pool = nn.MaxPool2d(2, 2)
        # linear layer (64 * 4 * 4 -> 500)
        self.fc1 = nn.Linear(64 * 4 * 4, 500)
        # linear layer (500 -> 10)
        self.fc2 = nn.Linear(500, 10)
        # dropout层 (p=0.3)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        # add sequence of convolutional and max pooling layers
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = self.pool(F.relu(self.conv3(x)))
        # flatten image input
        x = x.view(-1, 64 * 4 * 4)
        # add dropout layer
        x = self.dropout(x)
        # add 1st hidden layer, with relu activation function
        x = F.relu(self.fc1(x))
        # add dropout layer
        x = self.dropout(x)
        # add 2nd hidden layer, with relu activation function
        x = self.fc2(x)
        return x


if __name__ == '__main__':
    print('\n ----------------mnist 1--------------------- \n')
    mnist1 = CNNMnist1()
    for name, parameter in mnist1.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(mnist1, input_size=(1, 1, 28, 28))
    
    print('\n ----------------mnist 1--------------------- \n')
    mnist2 = CNNMnist2()
    for name, parameter in mnist2.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(mnist2, input_size=(1, 1, 28, 28))
    
    print('\n ----------------fmnist 1--------------------- \n')
    fmnist1 = CNNFmnist1()
    for name, parameter in fmnist1.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(fmnist1, input_size=(1, 1, 28, 28))
    
    print('\n ----------------fmnist 2--------------------- \n')
    fmnist2 = CNNFmnist2()
    for name, parameter in fmnist2.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(fmnist2, input_size=(1, 1, 28, 28))
    
    print('\n ----------------Cifar10 1--------------------- \n')
    cifar1 = CNNCifar1()
    for name, parameter in cifar1.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(cifar1, input_size=(1, 3, 32, 32))
    
    print('\n ----------------Cifar10 2--------------------- \n')
    cifar2 = CNNCifar2()
    for name, parameter in cifar2.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(cifar2, input_size=(1, 3, 32, 32))
    
    print('\n ----------------Cifar10 3--------------------- \n')
    cifar3 = CNNCifar3()
    for name, parameter in cifar3.named_parameters():
        print(f'name: {name} --- parameter size: {parameter.size()}')
    summary(cifar3, input_size=(1, 3, 32, 32))
