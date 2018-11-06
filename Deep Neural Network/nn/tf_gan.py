# -*- coding: utf-8 -*-

import tensorflow as tf

def init_GAN_generator(hidden_size, output_size, noise_data, active_fn=tf.nn.tanh):
    
    _, n = noise_data.get_shape().as_list()
    
    W1 = tf.get_variable(name = 'G_W1',
                         shape = (n, hidden_size),
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(name = 'G_b1',
                         shape = (1, hidden_size),
                         initializer=tf.constant_initializer(0))
    
    W2 = tf.get_variable(name = 'G_W2',
                         shape = (hidden_size, output_size),
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name = 'G_b2',
                         shape = (1, output_size),
                         initializer=tf.constant_initializer(0))
    
    hidden_output = tf.nn.leaky_relu(tf.matmul(noise_data, W1) + b1)
    output = active_fn(tf.add(tf.matmul(hidden_output, W2), b2))
    
    return output

def init_GAN_discriminator(hidden_size, real_data, fake_data, active_fn=tf.nn.tanh):
    
    batch_size, n = real_data.get_shape().as_list()
    
    W1 = tf.get_variable(name = 'D_W1',
                         shape = (n, hidden_size),
                         initializer=tf.contrib.layers.xavier_initializer())
    b1 = tf.get_variable(name = 'D_b1',
                         shape = (1, hidden_size),
                         initializer=tf.constant_initializer(0))
    
    W2 = tf.get_variable(name = 'D_W2',
                         shape = (hidden_size, 1),
                         initializer=tf.contrib.layers.xavier_initializer())
    b2 = tf.get_variable(name = 'D_b2',
                         shape = (1, 1),
                         initializer=tf.constant_initializer(0))
    # confirm real data as true
    real_data_hidden_output = active_fn(tf.matmul(real_data, W1) + b1)
    D_real = tf.matmul(real_data_hidden_output, W2) + b2
    # confirm fake data as false
    fake_data_hidden_output = active_fn(tf.matmul(fake_data, W1) + b1)
    D_fake = tf.matmul(fake_data_hidden_output, W2) + b2
    
    return D_real, D_fake

def init_GAN_fp_process(hidden_size, real_data, noise_data):
    
    G_params, fake_data = init_GAN_generator(hidden_size, real_data, noise_data)
    D_params, D_real, D_fake_fake, D_fake_real = \
        init_GAN_discriminator(hidden_size, real_data, fake_data, tf.nn.sigmoid)

    return  G_params, D_params, fake_data, D_real, D_fake_fake, D_fake_real

def init_GAN_optimizer(lr, G_name, D_name, D_real, D_fake, c=0.01, smooth=0.):
    # get the trainable variables
    t_vars = tf.trainable_variables()
    d_vars = [var for var in t_vars if var.name.startswith(D_name)]
    g_vars = [var for var in t_vars if var.name.startswith(G_name)]
    
    D_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_real,
                                                                         labels=tf.ones_like(D_real)*(1-smooth)))
    D_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake,
                                                                         labels=tf.zeros_like(D_fake)))
#    D_loss_real = - tf.reduce_mean(D_real)
#    D_loss_fake = tf.reduce_mean(D_fake)
    D_loss = D_loss_real + D_loss_fake
    D_optimizer = tf.train.AdamOptimizer(lr).minimize(D_loss, var_list=d_vars)
    clip_discriminator_var_op = [var.assign(tf.clip_by_value(var,-c,c)) for var in d_vars]

    G_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=D_fake,
                                                                    labels=tf.ones_like(D_fake)*(1-smooth)))
#    G_loss = - tf.reduce_mean(D_fake)
    G_optimizer = tf.train.AdamOptimizer(lr).minimize(G_loss, var_list=g_vars)
    
    return (G_loss, G_optimizer), (D_loss, D_optimizer), clip_discriminator_var_op