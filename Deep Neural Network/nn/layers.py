# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 20:33:48 2018

@author: 周宝航
"""

import numpy as np
from scipy.signal import convolve2d
from .utils import rotate_180
from .optimizer import Adam, Momentem, RMSProp

class Layer(object):
    """
        base layer object
        define the basic methods
    """
    def __init__(self):
        pass
    
    def forward(self, A_prev, optimizer):
        """
            forward propagation
            input  : A_prev    (activation value from the post layer)
                     optimizer (name of the optimizer, we support like: 'adam','rmsprop','momentom')
            return : value operated by this layer
        """
        pass
    
    def backward(self, dZ):
        """
            backward propagation
            input  : dZ         (error from the next layer)
            return : error of the post layer
        """
        pass
    
    def getOptimizer(self, shape_vw, shape_vb, optimizer='adam'):
        """
            get the optimizer object
            input  : shape_vw, shape_vb, optimizer
            return : Adam . Momentom . RMSProp object
        """
        opt = None
        if optimizer == 'adam':
            opt = Adam(shape_vw, shape_vb, self.beta1, self.beta2, self.epsilon)
        elif optimizer == 'momentem':
            opt = Momentem(shape_vw, shape_vb, self.beta1)
        elif optimizer == 'rmsprop':
            opt = RMSProp(shape_vw, shape_vb, self.beta2)
        return opt

class Dense(Layer):
    """
        full connect layer
    """
    def __init__(self, output_dim, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # weight matrix
        self.W = None
        # weight matrix
        self.W_cache = []
        # bias matrix
        self.b = None
        # hyper parameter (momentom)
        self.beta1 = beta1
        # hyper parameter (RMSProp)
        self.beta2 = beta2
        # hyper parameter (Adam)
        self.epsilon = epsilon
        # dimension of the layer
        self.output_dim = output_dim
        # hyper parameter (learning rate)
        self.learning_rate = learning_rate
        self.A_prev_cache = []
        
    def forward(self, A_prev, optimizer='adam'):
        
        # get the dimension of the post layer
        n, m = A_prev.shape
        if self.W is None and self.b is None:
            # Xavier initialize weights
            self.W = np.random.randn(self.output_dim, n) * 1. / n#np.sqrt(6. / (self.output_dim + n))
            self.b = np.zeros((self.output_dim, 1))
            shape_vw = (self.output_dim, n)
            shape_vb = (self.output_dim, 1)
            # initialize optimizer
            self.optimizer = self.getOptimizer(shape_vw, shape_vb, optimizer)
        # delta W
        self.dW = np.zeros((self.output_dim, n))
        # delta b
        self.db = np.zeros((self.output_dim, 1))
        # Z = W*X + b
        Z = self.W.dot(A_prev) + self.b
        
        assert(Z.shape == (self.output_dim, m))
        # save the activation value from the post layer
        self.A_prev_cache.append(A_prev)
        self.W_cache.append(self.W.copy())
        return Z
    
    def backward(self, dZ, t=1):

        # get the samples number
        m = dZ.shape[1]
        W = self.W_cache.pop()
        A_prev = self.A_prev_cache.pop()
        # get the error of the post layer
        delta = W.T.dot(dZ)
        # get the error of the weights
        dW = dZ.dot(A_prev.T) / m
        # get the error of the bias
        db = np.sum(dZ, axis=1, keepdims=True) / m
        self.dW += dW
        self.db += db
        if not (len(self.W_cache) and len(self.A_prev_cache)):
            # get the delta weights and bias corrected by the optimizer
            vw, vb = self.optimizer.getWeight(self.dW, self.db, t)
            # update the weights and bias
            self.W -= self.learning_rate * vw
            self.b -= self.learning_rate * vb
        return delta
        
class Pool2DLayer(object):
    
    def __init__(self, f=1, stride=1, mode='max'):
        
        self.f = f
        self.mode = mode
        self.stride = stride
    
    def create_mask_from_window(self, x):
        
        return x == np.max(x)
    
    def distribute_value(dz, shape):

        (n_H, n_W) = shape
        
        average = dz / (n_H * n_W)
        
        a = np.ones(shape) * average
        return a
    
    def forward(self, A_prev):
        
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
    
        f = self.f
        stride = self.stride
        
        n_H = int(1 + (n_H_prev - f) / stride)
        n_W = int(1 + (n_W_prev - f) / stride)
        n_C = n_C_prev
        
        A = np.zeros((m, n_H, n_W, n_C))              
        
        for i in range(m):
            for h in range(n_H):
                for w in range(n_W):
                    for c in range (n_C):
                        
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        
                        a_prev_slice = A_prev[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        
                        if self.mode == "max":
                            A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "average":
                            A[i, h, w, c] = np.mean(a_prev_slice)

        assert(A.shape == (m, n_H, n_W, n_C))
        
        self.A_prev = A_prev
        return A
    
    def backward(self, dA):

        stride = self.stride
        f = self.f
        
        m, n_H_prev, n_W_prev, n_C_prev = self.A_prev.shape
        m, n_H, n_W, n_C = dA.shape
        
        dA_prev = np.zeros_like(self.A_prev)
        
        for i in range(m):
            
            a_prev = self.A_prev[i]
            
            for h in range(n_H):
                for w in range(n_W):
                    for c in range(n_C):
                        
                        vert_start = h * stride
                        vert_end = vert_start + f
                        horiz_start = w * stride
                        horiz_end = horiz_start + f
                        
                        if self.mode == "max":
                            a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                            mask = self.create_mask_from_window(a_prev_slice)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += \
                            mask * dA[i, h, w, c]
                            
                        elif self.mode == "average":
                            da = dA[i, h, w, c]
                            shape = (f, f)
                            dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += \
                            self.distribute_value(da, shape)
                            
        assert(dA_prev.shape == self.A_prev.shape)
        
        return dA_prev

class Conv2DLayer(Layer):
    
    def __init__(self, learning_rate=0.1, beta1=0.9, beta2=0.999, epsilon=1e-8, \
                 filternum=1, filtersize=(3,3), mode='valid'):
        
        self.W = None
        self.b = None
        self.mode = mode
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.filternum = filternum
        self.filtersize = filtersize
        self.learning_rate = learning_rate
    
    def zero_pad(self, X, pad):
        
        X_pad = np.pad(X, ((0,0), (pad, pad), (pad,pad), (0,0)), 'constant', constant_values=0)
        return X_pad
    
    def single_step(self, a_slice_prev, W, b):
        
        s = a_slice_prev * W + b
        Z = np.sum(s)
        return Z
    
    def forward(self, A_prev, optimizer='adam'):
        
        (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape
        
        if self.W is None and self.b is None:
            nin = n_H_prev * n_W_prev * n_C_prev
            self.W = np.random.randn(self.filtersize[0], self.filtersize[1], n_C_prev, self.filternum)\
                                                            * 1. / nin#np.sqrt(6. / (n_C_prev + self.filternum))
            self.b = np.zeros((1, 1, 1, self.filternum))
            shape_vw = (self.filtersize[0], self.filtersize[1], n_C_prev, self.filternum)
            shape_vb = (1, 1, 1, self.filternum)
            self.optimizer = self.getOptimizer(shape_vw, shape_vb, optimizer)
    
        (f, f, n_C_prev, n_C) = self.W.shape
        
        Z = []
        
        for i in range(m):
            a_prev = A_prev[i]
            channels = []
            for j in range(n_C):
                sum_maps = None
                n_H, n_W = None, None
                for k in range(n_C_prev):
                    a_slice_prev = a_prev[:,:,k]
                    t = convolve2d(a_slice_prev, rotate_180(self.W[:,:,k,j]), mode=self.mode)\
                                    + self.b[:,:,0,j]
                    n_H, n_W = t.shape
                    if sum_maps is None:
                        sum_maps = t
                    else:
                        sum_maps += t
                channels.append(sum_maps)
            Z.append(np.array(channels).reshape((n_H, n_W, n_C)))
        
        Z = np.array(Z)
        
        self.A_prev = A_prev
        return Z
    
    def backward(self, dZ, t=1):
        
        (m, n_H_prev, n_W_prev, n_C_prev) = self.A_prev.shape
    
        (f, f, n_C_prev, n_C) = self.W.shape
        
        (m, n_H, n_W, n_C) = dZ.shape
        
        dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))                           
        dW = np.zeros((f, f, n_C_prev, n_C))
        db = np.zeros((1, 1, 1, n_C))
        
        for i in range(m):
            
            a_prev = self.A_prev[i]
            da_prev = dA_prev[i]
            
            for j in range(n_C_prev):
            
                for k in range(n_C):
                    
                    da_prev[:,:,j] += convolve2d(dZ[i,:,:,k], self.W[:,:,j,k])
                    dW[:,:,j,k] += convolve2d(dZ[i,:,:,k], a_prev[:,:,j], mode='valid')
                    db[:,:,:,k] += np.sum(dZ[i,:,:,k])
        
        assert(dA_prev.shape == (m, n_H_prev, n_W_prev, n_C_prev))
        
        dW /= m
        db /= m

        vw, vb = self.optimizer.getWeight(dW, db, t)
        self.W -= self.learning_rate * vw
        self.b -= self.learning_rate * vb
        return dA_prev