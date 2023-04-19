import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class DatasetSplit(Dataset):
    """An abstract Dataset class wrapped around Pytorch Dataset class."""
    
    def __init__(self, dataset, idxs):
        self.dataset = dataset
        self.idxs = [int(i) for i in idxs]
    
    def __len__(self):
        return len(self.idxs)
    
    def __getitem__(self, item):
        image, label = self.dataset[self.idxs[item]]
        return torch.tensor(image), torch.tensor(label)


class ModelUpdate(object):
    def __init__(self, params, dataset, idxs):
        for key, val in params.items():
            setattr(self, key, val)
        
        self.trainloader, self.testloader = self.train_test(dataset, list(idxs))
        
        self.criterion = nn.CrossEntropyLoss()
    
    def train_test(self, dataset, idxs):
        """
        Returns train, test dataloaders for a given dataset and user indexes.
        """
        idxs_train = idxs[:int(0.8 * len(idxs))]
        idxs_test = idxs[int(0.8 * len(idxs)):]
        
        trainloader = DataLoader(DatasetSplit(dataset, idxs_train), batch_size=self.local_bs, shuffle=True)
        testloader = DataLoader(DatasetSplit(dataset, idxs_test), batch_size=self.local_bs, shuffle=True)
        
        return trainloader, testloader
    
    def update_weights(self, model, global_round):
        # Set mode to train model
        model.train()
        epoch_loss = []
        epoch_acc = []
        
        # Set optimizer for the local updates
        if self.local_optimizer == 'sgd':
            optimizer = torch.optim.SGD(model.parameters(), lr=self.learn_rate)
        elif self.local_optimizer == 'adam':
            optimizer = torch.optim.Adam(model.parameters(), lr=self.learn_rate, weight_decay=1e-4)
        
        for local_epoch in range(self.local_epoch):
            batch_loss = []
            acc = 0
            total = 0
            for batch_idx, (images, labels) in enumerate(self.trainloader):
                images, labels = images.to(self.device), labels.to(self.device)
                total += len(labels)
                
                # 预测和计算准确度
                log_probs = model(images)
                acc += (log_probs.argmax(1) == labels).type(torch.float).sum().item()
                
                # 计算损失
                loss = self.criterion(log_probs, labels)
                
                # 将梯度初始化为 0，以便批次之间不会混合梯度
                optimizer.zero_grad()
                
                # 后向传递错误
                loss.backward()
                
                # 优化参数
                optimizer.step()
                
                if batch_idx % 5 == 0:
                    print(
                        '| Global Round: {:>2} | Local Epoch: {:>2} | Process: {:>3.0f}% | Acc: {:>3.0f}% | Loss: {:.3f}'.format(
                            global_round + 1, local_epoch + 1, 100. * (batch_idx + 1) / len(self.trainloader),
                                                       100. * acc / total, loss.item()))
                batch_loss.append(loss.item())
            epoch_loss.append(sum(batch_loss) / len(batch_loss))
            epoch_acc.append(acc / total)
        return model.state_dict(), sum(epoch_loss) / len(epoch_loss), sum(epoch_acc) / len(epoch_acc)
    
    def inference(self, model):
        """ Returns the inference accuracy and loss."""
        loss, correct = 0.0, 0.0
        size = len(self.testloader.dataset)
        num_batches = len(self.testloader)
        with torch.no_grad():
            for batch_idx, (images, labels) in enumerate(self.testloader):
                images, labels = images.to(self.device), labels.to(self.device)
                
                pred = model(images)
                correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
                
                loss += self.criterion(pred, labels).item()
                
        loss /= num_batches
        correct /= size
        
        return correct, loss


def test_inference(args, model, test_dataset):
    """ Returns the test accuracy and loss. """
    model.eval()
    device = args.get('device')
    criterion = nn.CrossEntropyLoss().to(device)
    
    testloader = DataLoader(test_dataset, batch_size=128, shuffle=True)
    loss, total, correct = 0.0, 0.0, 0.0
    size = len(testloader.dataset)
    num_batches = len(testloader)
    
    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(testloader):
            images, labels = images.to(device), labels.to(device)
            pred = model(images)
            loss += criterion(pred, labels).item()
            correct += (pred.argmax(1) == labels).type(torch.float).sum().item()
    loss /= num_batches
    correct /= size
    
    return correct, loss
