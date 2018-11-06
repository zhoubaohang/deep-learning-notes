# -*- coding: utf-8 -*-
"""
Created on Sat Sep 22 20:33:48 2018

@author: 周宝航
"""

import numpy as np
import matplotlib.pyplot as plt
from .progressbar import ProgressBar
from .loss_func import LossFunction
from .utils import BatchNormalization
from .layers import Conv2DLayer, Dense

class Model(object):
    """
        define the basic model object
    """
    def __init__(self):
        pass
    
    def forward(self, data):
        """
            forward propagation
            input  : data    (activation value from the post layer)
        """
        pass
    
    def backward(self, delta, t):
        """
            backward propagation
            input  : delta   (error from the output layer)
                     t       (the iteration number performed)
        """
        pass
    
    def updateLearningrate(self, t):
        """
            update the learning rate
            input : t - the iteration number performed
        """
        self.learning_rate = 1. / (1. + self.decay_rate * t) * self.learning_rate


class Sequence(Model):
    
    def __init__(self, training_data, input_dim=(1,1), layers=[], \
                       optimizer='adam', learning_rate=0.009, decay_rate=0,\
                       loss_func='mean_abs_error'):
        
        self.layers = layers
        self.M, self.N = input_dim
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.training_data = training_data
        self.learning_rate = learning_rate
        self.lossFunction = LossFunction(loss_func)
    
    def add(self, layer):
        
        # Set learning rate to ConvLayer or DenseLayer
        if type(layer) is Conv2DLayer or type(layer) is Dense \
                        or type(layer) is BatchNormalization:
            layer.learning_rate = self.learning_rate
        # append the layer to the model's layer list
        self.layers.append(layer)
    
    def forward(self, data):
        
        for layer in self.layers:
            if type(layer) is Conv2DLayer or type(layer) is Dense \
                            or type(layer) is BatchNormalization:
                layer.learning_rate = self.learning_rate
                data = layer.forward(data, self.optimizer)
            else:
                data = layer.forward(data)
        return data
    
    def backward(self, delta, t):
        
        for layer in reversed(self.layers):
            if type(layer) is Conv2DLayer or type(layer) is Dense:
                delta = layer.backward(delta, t)
            else:
                delta = layer.backward(delta)
        return delta
    
#    def loss(self, _y, y):
#        
#        m = y.shape[1]
#        return np.sum(np.abs(y - _y)) / (2 * m)

    def generateTrainingData(self, batch_size, start=0):

        X, y = self.training_data[start]
        for tx, ty in self.training_data[start+1:start+batch_size]:
            X = np.c_[X, tx]
            y = np.c_[y, ty]
        X = X.reshape((self.N, batch_size))
        y = y.reshape((-1, batch_size))
        return (X, y)

    def MBGD(self, epoch, batch_size, plot_loss=False):

        m = len(self.training_data)
        steps = int(m / batch_size)
        losses = []
        threshold = 100.
        pbar = ProgressBar(title='MBGD Training', max_steps=steps)
        pbar.show(0)
        for i in range(1, epoch+1):
            threshold /= 100.
            sum_correct = 0
            pbar.setTitle("epoch {0}/{1}".format(i, epoch))
            np.random.shuffle(self.training_data)
            for j in range(steps):
                X, _y = self.generateTrainingData(batch_size, start=j*batch_size)
                y = self.forward(X)
                cost = y - _y
                loss = self.lossFunction.getLoss(_y, y) #self.loss(_y, y)
                y = self.softmax2ones(y)

                sum_correct += np.sum(y * _y)
                acc = sum_correct / ((j+1) * batch_size)

                pbar.setInfo("loss {0}\tacc {1}".format('%.6f'%loss, '%.6f'%acc))
                pbar.show(j+1)
                losses.append(loss)

                if loss <= threshold:
                    continue
                self.backward(cost, i)
            self.updateLearningrate(i)

        if plot_loss:
            plt.title('loss')
            plt.plot(np.arange(len(losses)), losses)
            plt.show()

    def SGD(self, epoch, plot_loss=False):

        m = len(self.training_data)
        steps = m
        losses = []
        threshold = 100.
        pbar = ProgressBar(title='SGD Training', max_steps=steps)
        pbar.show(0)
        for i in range(1, epoch+1):
            threshold /= 100.
            sum_correct = 0
            pbar.setTitle("epoch {0}/{1}".format(i, epoch))
            np.random.shuffle(self.training_data)
            for j in range(steps):
                X, _y = self.generateTrainingData(1, j)
                y = self.forward(X)
                cost = y - _y
                loss = self.lossFunction.getLoss(_y, y) #self.loss(_y, y)
                y = self.softmax2ones(y)

                sum_correct += np.sum(y * _y)
                acc = sum_correct / (j+1)

                pbar.setInfo("loss {0}\tacc {1}".format('%.6f'%loss, '%.6f'%acc))
                pbar.show(j+1)
                losses.append(loss)

                if loss <= threshold:
                    continue
                self.backward(cost, i)

        if plot_loss:
            plt.title('loss')
            plt.plot(np.arange(len(losses)), losses)
            plt.show()

    
    def fit(self, epoch, batch_size=100, plot_loss=False, mode='sgd'):

        if mode == 'sgd':
            self.SGD(epoch, plot_loss)
        elif mode == 'mbgd':
            self.MBGD(epoch, batch_size, plot_loss)
        
    def softmax2ones(self, y):
        
        n, m = y.shape
        result = np.zeros_like(y)
        for i in range(m):
            ele = y[:, i]
            index = np.where(ele == np.max(ele))[0]
            index = index[0] if len(index) > 0 else 0
            result[index, i] = 1.
        return result
    
    def predict(self, test_data):
        
        y = self.forward(test_data)
        return y
    
    def evaluate(self, test_data):
        
        acc = 0
        m = len(test_data)
        pbar = ProgressBar(title='Testing', max_steps=m)
        pbar.show(0)
        for i, data in enumerate(test_data):
            X, _y = data
            X = X.reshape((1, self.H, self.W, self.C))
            y = self.predict(X)
            y = self.softmax2ones(y)
            acc += y[_y] == 1.
            pbar.setInfo('acc : {0}'.format('%.6f' % (acc / (i+1))))
            pbar.show(i+1)
        return acc / m