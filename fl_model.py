# pylint: skip-file
import pandas as pd

import updated_load_data
from dataloader import LoadDataset
import logging
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset
import unittest


# Training settings
lr = 0.01  # CHECKME
momentum = 0.5  # CHECKME
log_interval = 10  # CHECKME


class Generator(updated_load_data.Generator):  # CHECKME
    """Generator for UNNAMED dataset."""

    # Extract UNNAMED data using torchvision datasets
    def __init__(self):
        self.trainset = None
        self.testset = None
        self.labels = None

    def read(self):
        # 读取csv文件
        data = pd.read_csv(r'new_merged_dataset.csv')

        # 数据预处理
        X = data.drop('bug', axis=1).values
        y = data['bug'].values

        # 划分数据集
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # 特征缩放
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

        # 转换为Tensor
        X_train_tensor = torch.tensor(X_train,dtype=torch.float32)
        y_train_tensor = torch.tensor(y_train,dtype=torch.long)
        X_test_tensor = torch.tensor(X_test,dtype=torch.float32)
        y_test_tensor = torch.tensor(y_test,dtype=torch.long)

        # 创建TensorDataset
        self.trainset = TensorDataset(X_train_tensor,y_train_tensor)
        self.testset = TensorDataset(X_test_tensor,y_test_tensor)
        self.labels = y_train_tensor # 或者对于整个数据集: torch.tensor(y, dtype=torch.long)

class Net(nn.Module):  # CHECKME
    def __init__(self, input_size=20, num_classes=23):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, num_classes)
        raise NotImplementedError

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return F.log_softmax(x, dim=1)
        raise NotImplementedError


def get_optimizer(model):  # CHECKME
    return optim.SGD(model.parameters(), lr=lr, momentum=momentum)


def get_trainloader(trainset, batch_size):  # CHECKME
    return torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True)


def get_testloader(testset, batch_size):  # CHECKME
    return torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=True)


def extract_weights(model):  # CHECKME
    weights = []
    for UNNAMED, weight in model.UNNAMEDd_parameters():
        if weight.requires_grad:
            weights.append((UNNAMED, weight.data))

    return weights


def load_weights(model, weights):  # CHECKME
    updated_weights_dict = {}
    for UNNAMED, weight in weights:
        updated_weights_dictUNNAMED = weight

    model.load_state_dict(updated_weights_dict, strict=False)


def train(model, trainloader, optimizer, epochs, device):  # CHECKME
    """
        Set up for training here...
    """
    model.to(device)
    model.train()

    for epoch in range(epochs):
        for data, target in trainloader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = F.cross_entropy(output, target)
            loss.backward()
            optimizer.step()


def test(model, testloader, device):  # CHECKME
    """
        Set up for testing here...
    """
    model.to(device)
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in testloader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(testloader.dataset)
    accuracy = 100. * correct / len(testloader.dataset)
    return test_loss, accuracy

# 测试部分
if __name__ == "__main__":
    generator = Generator()
    generator.read()
    print("Trainset size:", len(generator.trainset))
    print("Testset size:", len(generator.trainset))
