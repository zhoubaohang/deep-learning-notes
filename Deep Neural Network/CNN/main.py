# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 20:34:05 2018

@author: 周宝航
"""
import numpy as np
from nn.model import CNN, Sequence
from nn.layers import Conv2DLayer, Pool2DLayer, Dense
from nn.utils import Activation, Flatten, Droupout, BatchNormalization

from mnist_loader import load_data_wrapper

# a = np.array(
#     [[[[0,1,1,0,2],
#       [2,2,2,2,1],
#       [1,0,0,2,0],
#       [0,1,1,0,0],
#       [1,2,0,0,2]],
#      [[1,0,2,2,0],
#       [0,0,0,2,0],
#       [1,2,1,2,1],
#       [1,0,0,0,0],
#       [1,2,1,1,1]],
#      [[2,1,2,0,0],
#       [1,0,0,1,0],
#       [0,2,1,0,1],
#       [0,1,2,2,2],
#       [2,1,0,0,1]]]])

# conv = Conv2DLayer(filtersize=(2,2), mode='valid')
# output = conv.forward(a)
# sensity = np.ones(output.shape, dtype=np.float64)

# conv.backward(sensity)

# epsilon = 1e-8

# for i in range(conv.W.shape[0]):
#     for j in range(conv.W.shape[1]):
#         for k in range(conv.W.shape[2]):
#             conv.W[i,j,k,:] += epsilon
#             err1 = conv.forward(a).sum()
#             conv.W[i,j,k,:] -= 2*epsilon
#             err2 = conv.forward(a).sum()
#             err = (err1 - err2 ) / (2 * epsilon)
#             conv.W[i,j,k,:] += epsilon
#             print(err.sum(), conv.dW[i,j,k,:].sum())

training_data, _, test_data = load_data_wrapper()

learning_rate = 0.001
# 学习衰减率
decay_rate = 0.
# 迭代次数
iter_num = 3
# 批大小
batch_size = 10
#
# 只训练了前 50 张图片
cnn = CNN(training_data[:batch_size], input_dim=(28,28,1), \
         learning_rate=learning_rate, decay_rate=decay_rate, optimizer='adam')
#
## 升级版 LeNet
cnn.add(Conv2DLayer(filternum=16, filtersize=(5,5), mode='valid'))
cnn.add(Activation('relu'))
cnn.add(Pool2DLayer(f=2, stride=1, mode='max'))

## cnn.add(Conv2DLayer(filternum=16, filtersize=(3,3), stride=1, pad=0))
## cnn.add(BatchNormalization())
## cnn.add(Activation('relu'))
## cnn.add(Pool2DLayer(f=2, stride=1, mode='max'))
#
## cnn.add(Conv2DLayer(filternum=120, filtersize=(5,5), stride=1, pad=0))
## cnn.add(Activation('relu'))
cnn.add(Flatten())

cnn.add(Dense(84))
cnn.add(Activation('relu'))

cnn.add(Dense(10))
cnn.add(Activation('softmax'))

cnn.fit(iter_num, batch_size, plot_loss=True, mode='sgd')