# -*- coding: utf-8 -*-
import os

os.chdir('../../')

import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import load_data
from nn.layers import Dense
from nn.utils import Activation, Droupout
from nn.gan import GAN, Generator, Discriminator

# 加载 MNIST 训练数据
(train_X,_), _, _ = load_data()

train_X = train_X * 2. - 1.
# 添加生成器
generator = Generator(layers=[Dense(256),
                              Activation('relu', leaky_rate=0.01),
                              Dense(784),
                              Activation('tanh')])
# 添加判别器
discriminator = Discriminator(layers=[Dense(64),
                                      Activation('relu', leaky_rate=0.01),
                                      Dense(1),
                                      Activation('sigmoid')])

# 实例化网络
gan = GAN(generator, discriminator, lr=0.01, decay_rate=1e-4)

#%%
gan.train(train_X, epoch=100, k=1, mini_batch_size=100)

test_x = np.random.uniform(-1.,1.,size=(100,1))
img_generate = gan.generate(test_x)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.imshow(img_generate, cmap='gray')
ax.axis['off']