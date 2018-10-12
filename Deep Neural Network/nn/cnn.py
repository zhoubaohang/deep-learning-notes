# -*- coding: utf-8 -*-

import numpy as np
from .model import Sequence

class CNN(Sequence):
    
    def __init__(self, training_data, input_dim, layers=[], \
                       optimizer='adam', learning_rate=0.009, decay_rate=0):
        
        self.layers = layers
        self.optimizer = optimizer
        self.decay_rate = decay_rate
        self.training_data = training_data
        self.learning_rate = learning_rate
        self.H, self.W, self.C = input_dim

    def loss(self, _y, y):
        
        m = y.shape[1]
        y[y < 1e-10] = 1e-10
        return - np.log(np.sum(_y * y) / m)

    def generateTrainingData(self, batch_size, start=0):

        X, y = self.training_data[start]
        for tx, ty in self.training_data[start+1:start+batch_size]:
            X = np.c_[X, tx]
            y = np.c_[y, ty]
        X = X.reshape((batch_size, self.H, self.W, self.C))
        y = y.reshape((-1, batch_size))
        return (X, y)
