# Competitive Part

import numpy as np
from collections import Counter
import sys

print(sys.argv[1])
print(sys.argv[2])
print(sys.argv[3])
print(sys.argv[4])

file_train = open(sys.argv[1], "r")
data_train = np.array([[cell for cell in row.split(",")] for row in file_train], np.float)

file_test = open(sys.argv[2], "r")
data_test = np.array([[cell for cell in row.split(",")] for row in file_test], np.float)

file_predict = open(sys.argv[3], "r")
data_predict = np.array([[cell for cell in row.split(",")] for row in file_predict], np.float)

print(data_train.shape)
print(data_test.shape)
print(data_predict.shape)

y_train = Counter(data_train[:,0])
y_test = Counter(data_test[:,0])
y_predict = Counter(data_predict[:,0])
print(y_train)
print(y_test)
print(y_predict)
