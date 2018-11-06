
# GAN(生成式对抗网络)

[Github地址](https://github.com/zhoubaohang/deep-learning-notes/tree/master/Deep%20Neural%20Network)

参考：
[生成式对抗网络（Generative Adversarial Networks，GANs）](https://blog.csdn.net/u010089444/article/details/78946039)
[Generative Adversarial Networks-Ian GoodFellow](http://papers.nips.cc/paper/5423-generative-adversarial-nets
)

- 不要怂，就是**GAN(Generative Adversarial Networks)**！
- 时隔一个多月，更新一篇博客。一个多月忙完保研，看了机器学习，一直在做卷积神经网络的实现。自己写了一个MINI神经网络框架，上周刚跑了MNIST，还没来得及更，之后一并更新。
- 接到南开实验室的一个题目，要用到GAN。所以刷完GoodFellow的原文，用自己的小框架实现一哈GAN生成MNIST的实验。

先放上生成过程的gif图：一个100维的随机向量，送入生成网络，输出一个28*28的图片。由一开始输出不知道是啥的生成图片，到最后稳定输出数字‘0’。GAN的魅力可见一斑。
![gif](output.gif)

## 网络结构

![png](model.png)

- 上图展示了GAN的流程，涉及两个关键的函数$D(x)$和$G(x)$.所以GAN的两个关键角色为：生成器网络**Generator Network**与判别器网络**Discriminator Network**。

##### Generator Network

- 不断学习训练集中真实数据的概率分布，目标是将输入的随机噪声转换为以假乱真的数据（如：生成的图片与训练集中的图片越相似越好）。

##### Discriminator Network

- 判断一个数据是否是真是的数据，目标是将生成网络产生的“假”数据与训练集中的“真”数据区分开。

##### Train

- GAN基于**二人零和极小极大博弈**。其训练过程中让**D**和**G**进行博弈，相互的竞争让这两个模型同时得到增强。
- 由于判别网络**G**的存在，使得**G**在没有大量先验知识以及先验分布的前提下也能很好的去逼近真实数据的分布，并最终生成的数据达到以假乱真的效果。(即**D**无法区分**G**生成的数据与真实数据，从而两个网络达到某种平衡。)

### 目标函数

$$
\min_G \max_D V(D,G) = \mathbb{E}_{x \sim p_{data}(x)}[\log D(x)] + \mathbb{E}_{z \sim p_z(z)}[\log (1 - D(G(z)))]
$$

- 这是原论文提出的优化目标函数，要更新**D**时最大化上式；更新**G**时最小化上式。
- 在对判别模型**D**的参数进行更新时：对于来自真实分布$p_{data}$的样本$x$而言，我们希望$D(x)$的输出越接近**1**越好，即$\log D(x)$越大越好；对于噪声$z$生成的数据$G(z)$而言，我们希望$D(G(z))$尽量接近0，因此$\log (1-D(G(z)))$也是越大越好，所以需要$\max D$。
- 在对生成模型**G**的参数进行更新时：我们希望$G(z)$尽可能和真实数据一样，即分布相同：$p_g = p_{data}$。因此，我们希望$D(G(z))$尽量接近1，既$\log (1-D(G(z)))$越小越好，所以需要$\min G$.

## 算法流程

![png](algorithm.png)

- 算法流程截取自论文原文，详细描述了训练GAN的过程。
- 网络的训练方法采用反向传播，一次iteration中，先训练**判别网络**，再训练**生成网络**。
- 由于要最大化**D**的目标函数，所以采用梯度上升。不过，实际实现中往往在该目标函数前加上负号，从而统一使用梯度下降。

## 实验部分

- 实验部分只是将MNIST作为输入，未加入条件信息。所以最终的输出结果不一定为‘3’，也可能为上面的‘0’.

```python
# 导入必备的库
import os

os.chdir('../')

import numpy as np
import matplotlib.pyplot as plt
from mnist_loader import load_data
from nn.layers import Dense
from nn.utils import Activation, Droupout
from nn.gan import GAN, Generator, Discriminator
```


```python
# 加载 MNIST 训练数据
(train_X,_), _, _ = load_data()

# 将 训练数据从[0,1]范围转换到[-1,1]
train_X = train_X * 2. - 1.
```


```python
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
```


```python
# 训练网络
gan.train(train_X, epoch=100, k=1, mini_batch_size=100)
```

    epoch 1/100:[##################################################]100.00% 	loss_g 0.658039	loss_d 0.764890
	//省略部分
    epoch 100/100:[##################################################]100.00% 	loss_g 0.689197	loss_d 0.712495


![png](output_4_1.png)


- PS: 上面的为损失函数曲线，蓝色为判别网络的损失曲线，黄色为生成网络的损失曲线。


```python
# 生成测试图像
test_x = np.random.uniform(-1,1,size=(100,1))
img = gan.generate(test_x)
fig = plt.figure()
ax = fig.add_subplot(1,1,1)
ax.imshow(img.reshape((28,28)), cmap='gray')
ax.axis('off')
```

![png](output_6_1.png)

- 上图为 随机向量作为输入，经过生成器输出的图片结果。已经可以很明显的看出为：3，GAN的作用已经显现出来了。
