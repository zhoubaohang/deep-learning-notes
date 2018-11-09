
# AutoEncoder(自编码器)

- 今天介绍一下应用比较广泛的一种无监督神经网络模型——AutoEncoder(自编码器)。
- 这种模型可以应用在降维、去噪等方向。不过，其一些变种也可以用作生成模型，如：VAE(变分自编码器)。

## 模型结构

![png](AE_model.png)

- 上图为标准的自编码器结构，左侧蓝色为输入层，中间为隐含层，右侧为输出层。
- 我们将隐含层看作是编码器部分，而将输出层看作是解码器部分。

### 编码器

- 其主要任务是提取输入样本的“精华”，即：降维。所以，编码器部分的各层神经元数量往往是越来越低的。这样起到压缩维度，提取数据集中的重要信息的作用。

- 在实际使用中，我们在训练完AE后，只使用编码器部分。因为“浓缩的都是精华嘛”，使用降维后的数据来进行监督学习往往能起到更高的正确率。

### 解码器

- 其主要任务是将降维后的数据（编码器的输出）进行重建，并输出与原数据维度相同的重构样本。解码器当然希望重建数据的概率分布与原数据相同，因此将重构样本与输入样本的误差（衡量压缩数据造成的信息损失）进行反向传播。

### 工作流程

1. 前向传播的过程与普通神经网络相同，将样本送入输入层，传播至隐含层，最后通过输出层输出。（自编码器的输出层维度与输入层维度相同，因为解码器的工作就是尽可能地重构样本。）

2. 我们知道标准AE为无监督模型，所以我们将样本与输出层的输出（即：重构样本）比较，并以两者之间的差异作为误差进行反向传播。

## 变种模型

1. 欠完备自编码器
2. 正则自编码器
	2.1 稀疏自编码器（L1正则）
	2.2 去噪自编码器（引入噪声）
	2.3 收缩自编码器

## 实验部分

- 使用MNIST数据集，最后得到重构后的样本。

![png](reconstruct.png)

- 上图为实验结果。上方为原始样本，下方为重构样本。我们可以看出训练的效果还不错。

```python
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  8 09:36:13 2018

@author: 周宝航
"""

import os
os.chdir('../')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist_loader import load_data


(train_X, _), (test_X, test_Y), _ = load_data()
#%%
lr = 1e-3
batch_size = 128
data_size, input_size = train_X.shape

n_hidden1 = 256
n_hidden2 = 128
n_hidden3 = 64

tf.reset_default_graph()
#%%
x_ = tf.placeholder(tf.float32, shape=[None, input_size])

with tf.variable_scope('encoder'):
    ec_hidden_layer1 = tf.layers.dense(x_, n_hidden1, activation=tf.nn.leaky_relu)
    ec_hidden_layer2 = tf.layers.dense(ec_hidden_layer1, n_hidden2, activation=tf.nn.leaky_relu)
    encoder = tf.layers.dense(ec_hidden_layer2, n_hidden3, activation=tf.nn.leaky_relu)

with tf.variable_scope('decoder'):
    dc_hidden_layer1 = tf.layers.dense(encoder, n_hidden2, activation=tf.nn.leaky_relu)
    dc_hidden_layer2 = tf.layers.dense(dc_hidden_layer1, n_hidden1, activation=tf.nn.leaky_relu)
    decoder = tf.layers.dense(dc_hidden_layer2, input_size, activation=tf.nn.sigmoid)

loss = tf.losses.mean_squared_error(x_, decoder)
op = tf.train.AdamOptimizer(lr).minimize(loss)

sess = tf.Session()
sess.run(tf.global_variables_initializer())
#%%
epoch = 10
iteration = data_size // batch_size

for i in range(1, epoch+1):
    
    errors = 0.
    
    for j in range(iteration):
        start = j * batch_size
        end = (j+1) * batch_size
        feed_dict = {x_:train_X[start:end]}
        
        error, _ = sess.run([loss, op], feed_dict=feed_dict)
        errors += error
    
    print('epoch {} loss:{}'.format(i, '%.6f'%(errors / iteration)))

test_image_num = 10
decode_img, = sess.run([decoder], feed_dict={x_:test_X})

fig = plt.figure()
for i in range(test_image_num):
    index = i + 1
    ax_real = fig.add_subplot(2,10,index)
    ax_real.imshow(test_X[i].reshape((28,28)), cmap='gray')
    ax_real.axis('off')
    ax_decode = fig.add_subplot(2,10,index+10)
    ax_decode.imshow(decode_img[i].reshape((28,28)), cmap='gray')
    ax_decode.axis('off')
fig.savefig('reconstruct.png')
```

[参考文章]
[Deep Learning（深度学习）学习笔记整理系列之（四）](https://blog.csdn.net/zouxy09/article/details/8775524)
[深度学习之自编码器AutoEncoder](https://blog.csdn.net/marsjhao/article/details/73480859)