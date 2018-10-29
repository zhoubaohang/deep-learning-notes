
# RNN(循环神经网络)

参考文章：
[[tensorflow应用之路]RNN预测时间序列原理及LSTM/GRU算法实现](https://blog.csdn.net/h8832077/article/details/80400462?utm_source=blogkpcl9)
[深度学习（Deep Learning）读书思考六：循环神经网络一（RNN）](https://blog.csdn.net/fangqingan_java/article/details/53014085)
[LSTM与GRU结构](https://blog.csdn.net/qq_28743951/article/details/78974058)
[GRU神经网络](https://blog.csdn.net/wangyangzhizhou/article/details/77332582)

- 在昨天，终于实现了学长发表论文的模型。效果还行，这里记下了中途学习的RNN模型。
- 一开始做的时候，自己心心念的想实现一下RNN、GRU，结果跑出来的结果很不好，很受伤。最后还是投向**tensorflow**的怀抱。

## 标准RNN

- 现实世界中，在不同时间点上收集到的数据，我们称之为：时序数据。由于CNN与DNN处理的均为固定维数的数据，对于变长时序数据无能为力，因此提出了**RNN（循环神经网络）**。

![png](RNN.png)

- 上图展示了标准RNN的架构，左侧为未按时间展开的结构，而右侧为按时间展开后的运行过程。
- **关键1**：由上图可以看出，在处理一个或一批时序数据时，RNN的一个单元的内部参数是共享的。
- **关键2**：每个时刻的输出只与其输入或状态相对应，而不会和其它时刻的神经元相联系。

### 各种变式

![png](RNN_mode.png)

1. 一对一：每一个输入都有对应的输出。
2. 多对一：整个序列只有一个输出，例如：文本分类。
3. 一对多：一个输入产生一个时序，常用于**seq2seq**的解码阶段。
4. 多对多：不是每一个输入对应一个输出，可能对应到延后的多个输出。

### 缺陷问题

- RNN训练比较困难，主要原因在于隐藏层参数W，无论在前向传播过程还是在反向传播过程中都会乘上多次。这样就会导致
	1. 前向传播某个小于1的值乘上多次，对输出影响变小。
	2. 反向传播时会导致梯度弥散问题，参数优化变得比较困难。

1. **梯度爆炸**：梯度截断
2. **梯度弥散**：初始化、模型改进等等

- 以上都是模型训练上的困难，但是时序数据存在时间前后的依赖关系。在理论上，RNN 绝对可以处理这样的 长期依赖 问题。人们可以仔细挑选参数来解决这类问题中的最初级形式，但在实践中，RNN 肯定不能够成功学习到这些知识。Bengio, et al. (1994)等人对该问题进行了深入的研究，他们发现一些使训练 RNN 变得非常困难的相当根本的原因。而**LSTM、GRU**并不存在这类问题。

## GRU

![png](GRU.png)

- 由于论文模型的实现使用的GRU，故本篇只讨论一下GRU的结构。

### 前向传播过程

**重置门**

$$r_t = \sigma ( W_r X_t + U_r h_{t-1} + b_r)$$

**更新门**

$$z_t = \sigma ( W_z X_t + U_z h_{t-1} + b_z)$$

**候选记忆门**

$$\hat h_t = \tanh (W X_t + r_t U h_{t-1} + b)$$

**当前时刻记忆单元**

$$h_t = (1 - z_t) h_{t-1} + z_t \hat h_t$$

- GRU模型是由LSTM变形而来，它只有两个门，分别为更新门和重置门，即图中的$z_t$和$r_t$。
- 更新门用于控制前一时刻的状态信息被带入到当前状态中的程度，更新门的值越大说明前一时刻的状态信息带入越多。
- 重置门用于控制忽略前一时刻的状态信息的程度，重置门的值越小说明忽略得越多。

## 实验部分

- 这里做的实验很简单，预测sin序列。用前5个sin值，预测第6个值。

```python
# 导入必要的库
import os
os.chdir('../')

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
```


```python
# 训练数据个数
data_size = 10000
# 测试数据个数
testing_examples = 1000
# sin函数的采样间隔
sample_gap = 0.01
# 每个训练样本的长度
time_step_size = 5

def generate_data(seq):
    X = []
    y = []

    for i in range(len(seq) - time_step_size):
        X.append(seq[i : i+time_step_size])
        y.append(seq[i+time_step_size])

    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

test_start = data_size*sample_gap
test_end = test_start + data_size*sample_gap

train_x, train_y = generate_data(np.sin(np.linspace(0, test_start, data_size)))
test_x, test_y = generate_data(np.sin(np.linspace(test_start, test_end, testing_examples)))

print('train_x : {}, train_y : {}'.format(train_x.shape, train_y.shape),
      'test_x : {}, test_y : {}'.format(test_x.shape, test_y.shape))
```

    train_x : (9995, 5), train_y : (9995,) test_x : (995, 5), test_y : (995,)
    


```python
###############################################################################
#                            Model   Parameter                                #
#                               lr : 学习率                                   #
#                       input_size : 输入维数                                 #
#                      output_size : 输出维数                                 #
#                      hidden_size : 隐藏层维数                               #
#                       batch_size : 批大小                                   #
###############################################################################
lr = 1e-3
input_size = 1
output_size = 1
hidden_size = 64
batch_size = 64

# 重置 tensorflow 计算图
tf.reset_default_graph()
```


```python
###############################################################################
#                               RNN   architecture                            #
###############################################################################
x = tf.placeholder(tf.float32, [None, time_step_size, input_size])
y = tf.placeholder(tf.float32, [None, output_size])
# RNN cell
cell = tf.contrib.rnn.GRUCell(hidden_size)
# defining initial state
initial_state = cell.zero_state(batch_size, dtype=tf.float32)
# 'state' is a tensor of shape [batch_size, cell_state_size]
outputs, state = tf.nn.dynamic_rnn(cell,
                                   x,
                                   initial_state=initial_state,
                                   dtype=tf.float32)
# output layer
W = tf.get_variable(name = 'output_layer_W',
                    shape = (hidden_size, output_size),
                    initializer=tf.contrib.layers.xavier_initializer())
b = tf.get_variable(name = 'output_layer_b',
                    shape = (1, output_size),
                    initializer=tf.constant_initializer(0))
output = tf.matmul(outputs[:,-1], W) + b

loss = tf.reduce_mean(tf.abs(output - y))
optimizer = tf.train.AdamOptimizer(lr).minimize(loss)
```


```python
###############################################################################
#                               Model train                                   #
###############################################################################
iteration = data_size // batch_size

session = tf.Session()

with session.as_default() as sess:
    sess.run(tf.global_variables_initializer())

    epoch = 120
    for i in range(1, epoch+1):

        losses = []

        for j in range(iteration):
            index = np.random.choice(iteration-1)
            start = index * batch_size
            end = (index+1) * batch_size
            feed_dict = {x:train_x[start:end,:,None], y:train_y[start:end,None]}

            cost, _ = sess.run([loss, optimizer], feed_dict=feed_dict)
            losses.append(cost)
        if i % (epoch / 10) == 0:
            print('epoch {} loss : {}'.format(i, np.mean(losses)))

    py, = sess.run([output], feed_dict={x:test_x[:batch_size,:,None]})
    plt.plot(py, 'r', label='predicted')
    plt.plot(test_y[:batch_size], 'b', label='real')
    plt.legend(loc=1)
```

    epoch 12 loss : 0.02230345830321312
    epoch 24 loss : 0.019675523042678833
    epoch 36 loss : 0.01689651608467102
    epoch 48 loss : 0.012267905287444592
    epoch 60 loss : 0.010291705839335918
    epoch 72 loss : 0.008620242588222027
    epoch 84 loss : 0.007919215597212315
    epoch 96 loss : 0.005264157894998789
    epoch 108 loss : 0.005410676822066307
    epoch 120 loss : 0.00481170741841197
    


![png](output_4_1.png)

- 由上图可以看出，已经可以很好的预测sin序列了。
- 之后的话，如果得到学长的同意，会把最近做的模型写成博客，供大家一起讨论。