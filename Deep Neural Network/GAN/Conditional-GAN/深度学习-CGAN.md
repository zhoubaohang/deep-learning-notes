
# Conditional-GAN(条件生成式对抗网络)

[参考文章]
[GAN论文阅读——CGAN](https://blog.csdn.net/stalbo/article/details/79359380)
[详解GAN代码之搭建并详解CGAN代码](https://blog.csdn.net/jiongnima/article/details/80209239)
[Conditional Generative Adversarial Nets论文笔记](https://blog.csdn.net/wspba/article/details/54666907)

- 今天聊的内容是**GAN**模型的一个变种——**C-GAN**。
- 标准GAN中，未做对生成数据的限制，对于高维数据的生成过于自由，使得整个生成数据的过程不可控。如：标准GAN中，基于MNIST数据集生成的手写数字是随机的。
- 对此，Mirza等人就提出了一种Conditional Generative Adversarial Networks。

## 模型结构

![png](cgan_model.png)

- 由上图可以看出，这是一种带条件约束的生成对抗模型，它在生成模型（G）和判别模型（D）的建模中均引入了条件变量y，这里y可以是label，可以是tags，可以是来自不同模态是数据，甚至可以是一张图片，使用这个额外的条件变量，对于生成器对数据的生成具有指导作用，因此，Conditional Generative Adversarial Networks也可以看成是把无监督的GAN变成有监督模型的一种改进。

#### SGAN 目标函数

$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim  p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log(1-D(G(z)))]
$$

#### CGAN 目标函数

$$\min_G \max_D V(D,G) = \mathbb{E}_{x \sim  p_{data}(x)}[\log D(x|y)] + \mathbb{E}_{z \sim p_z(z)}[\log(1-D(G(z|y)))]
$$

- 由上面的目标函数也可以看出，在判别器与生成器的输入中加入条件概率。如此一来，最后得到的模型可以通过输入条件向量来控制样本的输出。

## 实验部分

- 这里还是用的论文中的实验过程，不过模型还是自己调的。一开始效果不是很好，最后改小了batch size，效果才变得可以。
- 先放一张生成过程的gif，每10个epoch保存一张生成结果。

![gif](cgan_process.gif)

- 最后是样本多样性输出，每个数字生成8个样本。最终的生成效果还不错。

![png](result.png)

```python
# -*- coding: utf-8 -*-
"""
Created on Sun Nov  4 14:54:36 2018

@author: 周宝航
"""
import os
os.chdir('../../')

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from mnist_loader import load_data_wrapper
from nn.tf_gan import init_GAN_generator, init_GAN_discriminator, init_GAN_optimizer

#%% 数据预处理
training_data, _, _ = load_data_wrapper()
X = []
Y = []
noise_size = 100
data_size = len(training_data)

for i in range(data_size):
    x, y = training_data[i]
    X.append(x)
    Y.append(y)

X = np.array(X).reshape((data_size, -1))
Y = np.array(Y).reshape((data_size, -1))
N = np.random.uniform(-1.0, 1.0, size=(data_size, noise_size))
#%%
def plot_result(img_matrix, epoch=0):
    
    fig = plt.figure(figsize=(8, 6))
    for i in range(5):
        for j in range(2):
            index = i*2+j
            ax = fig.add_subplot(5,2,index+1)
            img = img_matrix[index].reshape((28,28))
            ax.imshow(img, cmap='gray')
            ax.axis('off')
    fig.tight_layout()
    fig.savefig('{}.png'.format(epoch))
#%% 模型参数
lr = 1e-3
batch_size = 20
hidden_size = 300
input_size = 784
output_size = 10

VS_GEN = 'generator'
VS_DIS = 'discriminator'

tf.reset_default_graph()

# 生成网络
with tf.variable_scope(VS_GEN):
    n_ = tf.placeholder(tf.float32, shape=[None, noise_size], name='noise')
    y_ = tf.placeholder(tf.float32, shape=[None, output_size], name='output')
    G_input = tf.concat([y_, n_], 1)
    
    G_output = init_GAN_generator(hidden_size, input_size, G_input, active_fn=tf.nn.sigmoid)

# 判别网络
with tf.variable_scope(VS_DIS):
    x_ = tf.placeholder(tf.float32, shape=[None, input_size], name='real')
    D_real_input = tf.concat([y_, x_], 1)
    D_fake_input = tf.concat([y_, G_output], 1)
    
    D_real, D_fake = init_GAN_discriminator(hidden_size, D_real_input, D_fake_input,
                                            active_fn=tf.nn.leaky_relu)

# 损失函数
(G_loss, G_optimizer), (D_loss, D_optimizer), clip_weight = \
                         init_GAN_optimizer(lr, VS_GEN, VS_DIS, D_real, D_fake)
                         
sess = tf.Session()
sess.run(tf.global_variables_initializer())

iteration = data_size // batch_size
#%%
KD = 1
KG = 1
epoch = 300
for i in range(1+add, epoch+add+1):
    
    loss_gs = []
    loss_ds = []
    
    for j in range(iteration):
        # 判别网络训练
        ds = 0.
        for d in range(KD):
            start = (j+d) * batch_size
            end = (j+d+1) * batch_size
            feed_dict = {x_:X[start:end], n_:N[start:end], y_:Y[start:end]}
            d_loss, _ = sess.run([D_loss, D_optimizer], feed_dict=feed_dict)
            ds += d_loss
        loss_ds.append(ds / KD)
        
        # 生成网络训练
        gs = 0.
        for g in range(KG):
            start = (j+g) * batch_size
            end = (j+g+1) * batch_size
            feed_dict = {x_:X[start:end], n_:N[start:end], y_:Y[start:end]}
            g_loss, _ = sess.run([G_loss, G_optimizer], feed_dict=feed_dict)
            gs += g_loss
        loss_gs.append(gs / KG)
    
    loss_gs = np.mean(loss_gs)
    loss_ds = np.mean(loss_ds)
    
    print('epoch {} G_loss {} D_loss {}'.format(i, '%.6f'%loss_gs, '%.6f'%loss_ds))
    
    if i == 1 or i % 10 == 0:
        test_noise = N[:10]
        test_y = np.eye(output_size)
        generate, = sess.run([G_output], feed_dict={n_:test_noise, y_:test_y})
        plot_result(generate, i)

test_noise = N[:80]
test_y = []

for j in range(10):
    for i in range(8):
        vec = np.zeros(10)
        vec[j] = 1.
        test_y.append(vec)

test_y = np.array(test_y).reshape((80, 10))

generate, = sess.run([G_output], feed_dict={n_:test_noise, y_:test_y})
fig = plt.figure(figsize=(8, 6))
for i in range(10):
    for j in range(8):
        index = i*8+j
        ax = fig.add_subplot(10, 8, index+1)
        img = generate[index].reshape((28,28))
        ax.imshow(img, cmap='gray')
        ax.axis('off')
fig.savefig('result.png')
```