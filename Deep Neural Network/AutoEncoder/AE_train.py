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