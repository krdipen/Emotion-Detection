# Non Competitive Part

import numpy as np
from collections import Counter
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F

class Neuralnetwork(nn.Module):
    def __init__(self, input_size):
        super(Neuralnetwork, self).__init__()
        self.first = nn.Linear(input_size,100)
        self.hidden = nn.Linear(100,10)

    def forward(self, X):
        X = F.relu(self.first(X))
        out = F.relu(self.first(X))
        return out #without softmax

def main():
    print(sys.argv[1])
    print(sys.argv[2])

    file_train = open(sys.argv[1], "r")
    data_train = np.array([[cell for cell in row.split(",")] for row in file_train], np.float)

    file_test = open(sys.argv[2], "r")
    data_test = np.array([[cell for cell in row.split(",")] for row in file_test], np.float)

    print(data_train.shape)
    print(data_test.shape)

    y_train = Counter(data_train[:,0])
    y_test = Counter(data_test[:,0])

    nn = Neuralnetwork(data_train.shape[1]-1)    

main()