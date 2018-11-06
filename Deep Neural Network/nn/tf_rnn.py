# -*- coding: utf-8 -*-
import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn

def init_dynamic_RNN_cell(hidden_size, input_data, batch_size, name=None, reuse=False):
    """
        init dynamic rnn cell
    """
    cell = rnn.GRUCell(hidden_size, name=name, reuse=reuse)
    # defining initial state
    initial_state = cell.zero_state(batch_size, dtype=tf.float32)
    # 'state' is a tensor of shape [batch_size, cell_state_size]
    outputs, state = tf.nn.dynamic_rnn(cell,
                                       input_data,
                                       initial_state=initial_state,
                                       dtype=tf.float32)
    return outputs

def init_o2m_RNN(hidden_size,
                 output_size,
                 batch_size,
                 input_data, 
                 time_step_size,
                 time_lag=1.,
                 reuse=False,
                 keep_prob=1.,
                 name='o2m_rnn',
                 cell_reuse=False,
                 active_fn=tf.nn.tanh):
    """
        init rnn in 'one to many' mode
    """
    # get the input data's shape
    _, n = input_data.get_shape().as_list()
    # define namespace
    with tf.variable_scope(name, reuse=reuse):
        # GRUI beta weight
        beta_W = tf.get_variable(name = 'GRUI_beta_W',
                            shape = (n, hidden_size),
                            initializer=tf.contrib.layers.xavier_initializer())
        # GRUI beta bias
        beta_b = tf.get_variable(name = 'GRUI_beta_b',
                            shape = (1, hidden_size),
                            initializer=tf.constant_initializer(0))
        # GRU cell
        cell = rnn.GRUCell(hidden_size, name='rnn/gru_cell', reuse=cell_reuse)
        # init the output layer weight
        W = tf.get_variable(name = 'o2m_rnn_output_layer_W',
                            shape = (hidden_size, output_size),
                            initializer=tf.contrib.layers.xavier_initializer())
        # init the output layer bias
        b = tf.get_variable(name = 'o2m_rnn_output_layer_b',
                            shape = (1, output_size),
                            initializer=tf.constant_initializer(0))
        # input the 'very first input data'
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        cell_output, state = cell(input_data, initial_state)
        # output through output_layer
        layer_output = tf.nn.dropout(active_fn(tf.matmul(cell_output, W) + b), keep_prob)
        delta = None
        # save the output
        outputs = []
        # begin for looping by time
        for i in range(time_step_size):
            # GRUI operation
            delta = tf.zeros_like(input_data) if i==0 else\
                    tf.where(tf.equal(layer_output, 0.),\
                             delta+time_lag, tf.ones_like(input_data)*time_lag)
            beta = 1. / tf.exp(tf.nn.relu(tf.add(tf.matmul(delta, beta_W), beta_b)))
            # use the post output as the input
            cell_output, state = cell(layer_output, state*beta)
            # output through output_layer
            layer_output = tf.nn.dropout(tf.nn.leaky_relu(
                                         tf.matmul(cell_output, W) + b), keep_prob)
            # save the output
            outputs.append(layer_output)
    # transpose the output data's shape to [batch_size, time_steps, n]
    return tf.transpose(tf.convert_to_tensor(outputs), [1,0,2])

def init_m2m_RNN(hidden_size,
                 output_size,
                 batch_size,
                 input_data,
                 reuse=False,
                 name='m2m_rnn',
                 cell_reuse=False,
                 time_lag_matrix=None,
                 active_fn=tf.nn.tanh):
    """
        init rnn in 'many to many' mode
    """
    _, time_step_size, n = input_data.get_shape().as_list()
    # transpose the matrix's dimension
    input_data = tf.transpose(input_data, [1,0,2])
    time_lag_matrix = tf.transpose(time_lag_matrix, [1,0,2])
    # define namespace
    with tf.variable_scope(name, reuse=reuse):
        # GRUI beta weight
        beta_W = tf.get_variable(name = 'GRUI_beta_W',
                            shape = (n, hidden_size),
                            initializer=tf.contrib.layers.xavier_initializer())
        # GRUI beta bias
        beta_b = tf.get_variable(name = 'GRUI_beta_b',
                            shape = (1, hidden_size),
                            initializer=tf.constant_initializer(0))
        # init the rnn cell
        cell = rnn.GRUCell(hidden_size, name='rnn/gru_cell', reuse=cell_reuse)
    
        W = tf.get_variable(name = 'm2m_rnn_output_layer_W',
                            shape = (hidden_size, output_size),
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name = 'm2m_rnn_output_layer_b',
                            shape = (1, output_size),
                            initializer=tf.constant_initializer(0))
        
        # input the 'very first input data'
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        cell_output, state = None, initial_state
        
        output = []
        # begin for looping by time
        for i in range(time_step_size):
            # GRUI operation
            raw_beta = time_lag_matrix[i]
            beta = 1. / tf.exp(tf.nn.relu(tf.add(tf.matmul(raw_beta, beta_W), beta_b)))
            # use the post output as the input
            cell_output, state = cell(input_data[i], state*beta)
            output.append(active_fn(tf.matmul(cell_output, W) + b))
        
    return tf.transpose(tf.convert_to_tensor(output), [1,0,2])

def init_m2o_RNN(hidden_size,
                 output_size,
                 batch_size,
                 input_data,
                 reuse=False,
                 name='m2o_rnn',
                 cell_reuse=False,
                 time_lag_matrix=None,
                 active_fn=tf.nn.tanh):
    """
        init rnn in 'many to one' mode
    """
    _, time_step_size, n = input_data.get_shape().as_list()
    # transpose the matrix's dimension
    input_data = tf.transpose(input_data, [1,0,2])
    time_lag_matrix = tf.transpose(time_lag_matrix, [1,0,2])
    # define namespace
    with tf.variable_scope(name, reuse=reuse):
        # GRUI beta weight
        beta_W = tf.get_variable(name = 'GRUI_beta_W',
                            shape = (n, hidden_size),
                            initializer=tf.contrib.layers.xavier_initializer())
        # GRUI beta bias
        beta_b = tf.get_variable(name = 'GRUI_beta_b',
                            shape = (1, hidden_size),
                            initializer=tf.constant_initializer(0))
        # init the rnn cell
        cell = rnn.GRUCell(hidden_size, name='rnn/gru_cell', reuse=cell_reuse)
    
        W = tf.get_variable(name = 'm2o_rnn_output_layer_W',
                            shape = (hidden_size, output_size),
                            initializer=tf.contrib.layers.xavier_initializer())
        b = tf.get_variable(name = 'm2o_rnn_output_layer_b',
                            shape = (1, output_size),
                            initializer=tf.constant_initializer(0))
        
        # input the 'very first input data'
        initial_state = cell.zero_state(batch_size, dtype=tf.float32)
        cell_output, state = None, initial_state
        # begin for looping by time
        for i in range(time_step_size):
            # GRUI operation
            raw_beta = time_lag_matrix[i]
            beta = 1. / tf.exp(tf.nn.relu(tf.add(tf.matmul(raw_beta, beta_W), beta_b)))
            # use the post output as the input
            cell_output, state = cell(input_data[i], state*beta)
            
        logits = tf.add(tf.matmul(cell_output, W), b)
        pred = active_fn(logits)

    return (logits, pred)