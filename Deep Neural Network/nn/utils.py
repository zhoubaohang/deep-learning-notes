# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 20:33:48 2018

@author: 周宝航
"""

import numpy as np
from .optimizer import Momentem

rotate_180 = lambda mat: np.rot90(np.rot90(mat))

class Activation(object):
    """
        Activation layer
        function type : ReLU  Sigmoid  Softmax tanh
    """
    def __init__(self, mode, leaky_rate=0.):
        
        self.mode = mode
        self.func = getattr(self, mode)
        self.leaky_rate = leaky_rate

    def forward(self, Z):
        """
            forward propagation
            input  : Z
            return : A = func(Z)
            like   : ReLU(Z)  Sigmoid(Z)  Softmax(Z)
        """
        self.Z = Z
        self.A = self.func(Z)
        return self.A
    
    def backward(self, delta):
        """
            backward propagation
            input  : dA
            return : dZ = dA*func`(dZ)
            like   : dA*ReLU`(Z)  dA*Sigmoid`(Z)  dA*Softmax`(Z)
        """
        return self.func(delta, derive=True)
    
    def relu(self, z, derive=False):
        """
            ReLU function
            input  : Z, derive
            return : ReLU(Z) or ReLU`(Z)
        """
        result = z
        if derive:
            result[self.Z < 0.] *= self.leaky_rate
        else:
            result[z < 0.] *= self.leaky_rate
#            result = np.maximum(z, 0.)
        return result
    
    def sigmoid(self, z, derive=False):
        """
            Sigmoid function
            input  : Z, derive
            return : Sigmoid(Z) or Sigmoid`(Z)
        """
        result = 0
        if derive:
            result = z * self.A * (1.0 - self.A)
        else:
            result = 1.0 / (1.0 + np.exp(-z))
        return result
    
    def softmax(self, z, derive=False):
        """
            Softmax function
            input  : Z, derive
            return : Softmax(Z) or Softmax`(Z)
        """
        result = 0
        if derive:
            result = z
        else:
            m = z.shape[1]
            for i in range(m):
                ele = z[:, i]
                ele -= np.max(ele)
                exp_e = np.exp(ele)
                sum_exp_e = np.sum(exp_e)
                z[:, i] = exp_e / sum_exp_e
            result = z
        return result
    
    def tanh(self, z, derive=False):
        """
            Tanh function
            input  : Z, derive
            return : tanh(Z) or tanh`(Z)
        """
        result = 0
        if derive:
            result = (1. - self.A**2) * z
        else:
            result = np.tanh(z)
        return result

class Flatten(object):
    """
        Flatten layer
        transferform PoolLayer or ConvLayer to full connect layer
    """
    def __init__(self):
        
        pass
        
    def forward(self, A_prev):
        """
            forward propagation
            input  : activation value from the post layer, like : pooling
            return : the value of the activation value flattend
        """
        self.shape = A_prev.shape
        m, H, W, C = self.shape
        # flatten_A = A_prev.flatten().reshape((-1, 1))
        flatten_A = np.zeros((H*W*C, m))

        for i in range(m):
            tmp = A_prev[i, :, :, :]
            flatten_A[:, i] = tmp.flatten()

        assert(flatten_A.shape == (H*W*C, m))
        return flatten_A
    
    def backward(self, dA):
        """
            backward propagation
            input  : the error from the next layer
            return : reshape the error by the post layer's activation value
        """
        m, H, W, C = self.shape
        
        dA_prev = np.zeros(self.shape)
        
        for i in range(m):
            tmp = dA[:, i]
            dA_prev[i, :, :, :] = tmp.reshape((H, W, C))
        return dA_prev

class Droupout(object):

    def __init__(self, p=0.5):

        self.p = p

    def forward(self, A_prev):

        self.U = np.random.rand(*(A_prev.shape)) <= self.p
        return A_prev * self.U / self.p

    def backward(self, dA):

        return self.U * dA

class BatchNormalization(object):

    def __init__(self, epsilon=1e-5, momentom=0.9, learning_rate=0.09):

        self.beta = None
        self.gamma = None
        self.epsilon = epsilon
        self.momentom = momentom
        self.learning_rate = learning_rate

    def forward(self, x, train_flag=True):

        out = None

        if train_flag:
            self.x = x

            if self.gamma is None and self.beta is None:
                *shape, _ = x.shape
                shape.append(1)
                self.gamma = np.random.uniform(-1e-4, 1e-4, shape)
                self.beta = np.random.uniform(-1e-4, 1e-4, shape)
                self.optimizer = Momentem(shape, shape, self.momentom)

            self.x_mean = np.mean(x, axis=-1, keepdims=True)
            self.x_var = np.var(x, axis=-1, keepdims=True)
            self.x_normalized = (x - self.x_mean) / np.sqrt(self.x_var + self.epsilon)
            out = self.gamma * self.x_normalized + self.beta

            self.optimizer.getWeight(self.x_mean, self.x_var)
        else:
            mean, var = self.optimizer.vw, self.optimizer.vb
            x_normalized = (x - mean) / np.sqrt(var + self.epsilon)
            out = self.gamma * x_normalized + self.beta

        return out

    def backward(self, dout):

        C = self.x.shape[-1]
        dbeta = np.mean(dout, axis=-1, keepdims=True)
        dgamma = np.mean(self.x_normalized * dout, axis=-1, keepdims=True)
        dx_normalized = self.gamma * dout
        dvar = np.sum(-1.0/2 * dx_normalized * (self.x - self.x_mean) \
                    / (self.x_var + self.epsilon)**(3.0/2), axis=-1, keepdims=True)
        dmean = np.sum(-1.0/np.sqrt(self.x_var + self.epsilon) * dx_normalized, axis=-1, keepdims=True) \
                + 1.0/C * dvar * np.sum(-2 * (self.x - self.x_mean), axis=-1, keepdims=True)
        dx = 1.0 / np.sqrt(self.x_var + self.epsilon) * dx_normalized \
             + dvar * 2.0 / C *(self.x - self.x_mean) + 1.0 / C * dmean
             
        self.beta -= self.learning_rate * dbeta
        self.gamma -= self.learning_rate * dgamma

        return dx