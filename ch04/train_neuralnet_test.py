# coding: utf-8
import sys, os
sys.path.append(os.pardir)  # 为了导入父目录的文件而进行的设定
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from two_layer_net import TwoLayerNet

# 读入数据
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

import pickle
dataset_dir = os.path.dirname(os.path.abspath(__file__))
save_file = dataset_dir + "/data2.pkl"
with open(save_file, 'rb') as f:
    network = pickle.load(f)

# error = 0
# index = []
# for i in range(x_test.shape[0]):
#     x = x_test[i]
#     t = t_test[i]
#     y = network.predict(x)
#     if(np.argmax(t) != np.argmax(y)):
#         error = error + 1
#         index.append(i)
# print(error)
# print(index)

n = 146
x = x_test[n]
t = t_test[n]

print("t: ", t, np.argmax(t))
y = network.predict(x)
print("y: ", y, np.argmax(y))




