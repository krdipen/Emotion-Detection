import sys
import math
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from skimage.feature import hog
from skimage.filters import edges
from skimage.transform import resize
from skimage.filters import gabor_kernel

file_train = open(sys.argv[2], "r")
data_train = torch.Tensor([[float(cell) for cell in row.split(",")] for row in file_train])
file_test = open(sys.argv[3], "r")
data_test = torch.Tensor([[float(cell) for cell in row.split(",")] for row in file_test])

if sys.argv[1] == "3": # (20 points) Convolutional Neural Network

    class CNN(nn.Module):

        def __init__(self, n, batch_size, r):
            super(CNN, self).__init__()
            self.root_n = int(math.sqrt(n))
            self.conv1 = nn.Conv2d(1, 64, (3, 3), stride=3, padding=0)
            self.norm1 = nn.BatchNorm2d(64)
            self.pool1 = nn.MaxPool2d((2, 2), stride=2, padding=0)
            self.conv2 = nn.Conv2d(64, 128, (2, 2), stride=2, padding=0)
            self.norm2 = nn.BatchNorm2d(128)
            self.pool2 = nn.MaxPool2d((2, 2), stride=2, padding=0)
            self.fc1 = nn.Linear(128 * ((((((((self.root_n-3)//3+1)-2)//2+1)-2)//2+1)-2)//2+1)**2, 256)
            self.norm3 = nn.BatchNorm1d(256)
            self.fc2 = nn.Linear(256, r)
            self.batch_size = batch_size
            self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
            self.loss_function = nn.CrossEntropyLoss()

        def forward(self, X):
            X = X.view(-1, 1, self.root_n, self.root_n)
            X = F.relu(self.conv1(X))
            X = self.norm1(X)
            X = self.pool1(X)
            X = F.relu(self.conv2(X))
            X = self.norm2(X)
            X = self.pool2(X)
            X = X.view(-1, 128 * ((((((((self.root_n-3)//3+1)-2)//2+1)-2)//2+1)-2)//2+1)**2)
            X = F.relu(self.fc1(X))
            X = self.norm3(X)
            X = self.fc2(X)
            return X

        def train(self, data):
            if torch.cuda.is_available():
                self = self.cuda()
                data = data.cuda()
            max_epoch = 75
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
            if torch.cuda.is_available():
                self = self.cpu()
                data = data.cpu()
            return y

    cnn = CNN(data_train.shape[1]-1, 64, 7)
    cnn.train(data_train)
    # cnn.test(data_train)
    # cnn.test(data_test)
    np.savetxt(sys.argv[4], cnn.predict(data_test), fmt="%d", delimiter="\n")

if sys.argv[1] == "2": # (10 points) Feature Engineering

    count = 0
    filter = "gabor"
    def apply(filter, image):
        if filter == "gabor":
            accum = np.zeros_like(image)
            for kernel in kernels:
                filtered = edges.convolve(image, kernel, mode='wrap')
                np.maximum(accum, filtered, accum)
            image = accum
        elif filter == "hog":
            image = resize(image, (128, 64))
            image = hog(image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), multichannel=False)[1]
            image = resize(image, (48, 48))
        global count
        count += 1
        # print(f"Images Filtered = {count}")
        return image

    kernels = []
    for theta in np.arange(0, np.pi, np.pi/4):
        for frequency in (0.05, 0.25):
            for sigma in (1, 3):
                kernel = np.real(gabor_kernel(frequency, theta=theta, sigma_x=sigma, sigma_y=sigma))
                kernels.append(kernel)

    image_train = data_train[:,1:].view(-1, 48, 48).numpy()
    image_new_train = torch.Tensor([apply(filter, image) for image in image_train])
    data_train = torch.cat([data_train[:,0].unsqueeze(dim=1), image_new_train.view(-1, 2304)], dim=1).float()

    image_test = data_test[:,1:].view(-1, 48, 48).numpy()
    image_new_test = torch.Tensor([apply(filter, image) for image in image_test])
    data_test = torch.cat([data_test[:,0].unsqueeze(dim=1), image_new_test.view(-1, 2304)], dim=1).float()

if sys.argv[1] == "1" or sys.argv[1] == "2": # (10 points) Vanilla Neural Network

    class NN_Model(nn.Module):

        def __init__(self, n, batch_size, r):
            super(NN_Model, self).__init__()
            self.fc1 = nn.Linear(n,100)
            self.norm1 = nn.BatchNorm1d(100)
            self.fc2 = nn.Linear(100,r)
            self.batch_size = batch_size
            self.optimizer = torch.optim.SGD(self.parameters(), lr=0.01, momentum=0.9)
            self.loss_function = nn.CrossEntropyLoss()

        def forward(self, X):
            X = F.relu(self.fc1(X))
            X = self.norm1(X)
            X = self.fc2(X)
            return X

        def train(self, data):
            if torch.cuda.is_available():
                self = self.cuda()
                data = data.cuda()
            max_epoch = 100
            if sys.argv[1] == "2":
                max_epoch = 500
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
            if torch.cuda.is_available():
                self = self.cpu()
                data = data.cpu()
            return y

    nn_model = NN_Model(data_train.shape[1]-1, 256, 7)
    nn_model.train(data_train)
    # nn_model.test(data_train)
    # nn_model.test(data_test)
    np.savetxt(sys.argv[4], nn_model.predict(data_test), fmt="%d", delimiter="\n")
