import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

file_train = open(sys.argv[1], "r")
data_train = torch.Tensor([[float(cell) for cell in row.split(",")] for row in file_train])
file_test = open(sys.argv[2], "r")
data_test = torch.Tensor([[float(cell) for cell in row.split(",")] for row in file_test])

class CNN(nn.Module):

    def __init__(self, n, batch_size, r):
        super(CNN, self).__init__()
        self.root_n = int(math.sqrt(n))
        self.conv = nn.Sequential(
            nn.Conv2d(1, 64, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Dropout(p=0.5),
            nn.Conv2d(64, 128, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Dropout(p=0.5),
            nn.Conv2d(128, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Dropout(p=0.5),
            nn.Conv2d(256, 512, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, (3, 3), stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.MaxPool2d((2, 2), stride=2, padding=0),
            nn.Dropout(p=0.5)
        )
        self.linear = nn.Sequential(
            nn.Linear(512*3*3,512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512,256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(128,64),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(64,r)
        )
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-5)
        self.loss_function = nn.CrossEntropyLoss()

    def forward(self, X):
        X = X.view(-1, 1, self.root_n, self.root_n)
        X = self.conv(X)
        X = X.view(-1, 3*3*512)
        X = self.linear(X)
        return X

    def train(self, data):
        if torch.cuda.is_available():
            self = self.cuda()
            data = data.cuda()
        max_epoch = 200
        for epoch in range(max_epoch):
            cost = 0
            data = data[torch.randperm(data.shape[0]),:]
            for batch in range(math.ceil(data.shape[0]/self.batch_size)):
                self.optimizer.zero_grad()
                output = self(data[batch*self.batch_size:(batch+1)*self.batch_size,1:])
                loss = self.loss_function(output, data[batch*self.batch_size:(batch+1)*self.batch_size,0].long())
                loss.backward()
                self.optimizer.step()
                cost += float(loss.data) * data[batch*self.batch_size:(batch+1)*self.batch_size,:].shape[0]
            # print(f"Epochs = {epoch+1} and Loss = {round(cost/data.shape[0],6)}")
        if torch.cuda.is_available():
            self = self.cpu()
            data = data.cpu()

    def test(self, data):
        if torch.cuda.is_available():
            self = self.cuda()
            data = data.cuda()
        cost = 0
        correct = 0
        for batch in range (math.ceil(data.shape[0]/self.batch_size)):
            output = self(data[batch*self.batch_size:(batch+1)*self.batch_size,1:])
            loss = self.loss_function(output, data[batch*self.batch_size:(batch+1)*self.batch_size,0].long())
            cost += float(loss.data) * data[batch*self.batch_size:(batch+1)*self.batch_size,:].shape[0]
            prediction = F.softmax(output.data, dim=1).max(1)[1]
            correct += float(prediction.eq(data[batch*self.batch_size:(batch+1)*self.batch_size,0]).sum())
        # print(f"Accuracy = {round(100*correct/data.shape[0],2)}% and Loss = {round(cost/data.shape[0],6)}")
        if torch.cuda.is_available():
            self = self.cpu()
            data = data.cpu()

    def predict(self, data):
        if torch.cuda.is_available():
            self = self.cuda()
            data = data.cuda()
        y = []
        for batch in range (math.ceil(data.shape[0]/self.batch_size)):
            output = self(data[batch*self.batch_size:(batch+1)*self.batch_size,1:])
            prediction = F.softmax(output.data, dim=1).max(1)[1]
            y.extend(prediction)
        y = [f"Id,Prediction"] + [f"{i+1},{y[i]}" for i in range(len(y))]
        if torch.cuda.is_available():
            self = self.cpu()
            data = data.cpu()
        return y

cnn = CNN(data_train.shape[1]-1, 64, 7)
cnn.train(data_train)
# cnn.test(data_train)
# cnn.test(data_test)
np.savetxt(sys.argv[3], cnn.predict(data_test), fmt="%s", delimiter="\n")
