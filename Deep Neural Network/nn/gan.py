# -*- coding: utf-8 -*-

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from .model import Model
from .progressbar import ProgressBar
from .layers import Dense, Conv2DLayer

class Discriminator(object):
    
    def __init__(self, layers=[], lr=0.01, optimizer='momentem'):
        
        self.lr = lr
        self.layers = layers
        self.optimizer = optimizer
    
    def forward(self, A_prev):
        
        A = A_prev
        for layer in self.layers:
            if type(layer) is Dense or type(layer) is Conv2DLayer:
                layer.learning_rate = self.lr
                A = layer.forward(A, self.optimizer)
            else:
                A = layer.forward(A)
        return A
    
    def backward(self, dA):
        
        delta = dA
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta
    
class Generator(object):
    
    def __init__(self, layers=[], lr=0.01, optimizer='momentem'):
        
        self.lr = lr
        self.layers = layers
        self.optimizer = optimizer
    
    def forward(self, A_prev):
        
        A = A_prev
        for layer in self.layers:
            if type(layer) is Dense or type(layer) is Conv2DLayer:
                layer.learning_rate = self.lr
                A = layer.forward(A, self.optimizer)
            else:
                A = layer.forward(A)
        return A
    
    def backward(self, dA):
        
        delta = dA
        for layer in reversed(self.layers):
            delta = layer.backward(delta)
        return delta
    
class GAN(Model):
    
    def __init__(self, generator, discriminator, lr=0.01, optimizer='momentem',\
                 decay_rate=0.):
        
        self.learning_rate = lr
        generator.lr = lr
        generator.optimizer = optimizer
        self.generator = generator
        discriminator.lr = lr
        discriminator.optimizer = optimizer
        self.discriminator = discriminator
        self.decay_rate = decay_rate
    
    def loss_d(self, Dx, Dz):
        
        return np.mean(-np.log(Dx) - np.log(1. - Dz))
        
    
    def loss_g(self, Dz):
        
        return np.mean(-np.log(Dz))
        
    def train(self, data, mini_batch_size=100, k=1, epoch=1, \
              noise_sample_size=100, verbose=False):
        
        m = data.shape[0]
        Z = np.random.uniform(-1., 1., size=(noise_sample_size, m))
        loss_ds = []
        loss_gs = []
        iter_num = m // mini_batch_size
        pbar = ProgressBar(title='MBGD Training', max_steps=iter_num)
        pbar.show(0)
        
        for i in range(1, epoch+1):
            np.random.shuffle(data)
            X = data.T
            pbar.setTitle("epoch {0}/{1}".format(i, epoch))
            for j in range(iter_num):
            
                loss_d = 0
                
                for t in range(k):
                    sample_z = Z[:, j*mini_batch_size:(j+1)*mini_batch_size]
                    sample_x = X[:, j*mini_batch_size:(j+1)*mini_batch_size]
                    
                    Dx = self.discriminator.forward(sample_x)
                    self.discriminator.backward(Dx - 1.)
                    Gz = self.generator.forward(sample_z)
                    Dz = self.discriminator.forward(Gz)
                    self.discriminator.backward(Dz)
                    
                    loss_d += self.loss_d(Dx, Dz)
                loss_d /= k
                loss_ds.append(loss_d)
                
                sample_z = Z[:, j*mini_batch_size:(j+1)*mini_batch_size]
                
                Gz = self.generator.forward(sample_z)
                Dz = self.discriminator.forward(Gz)
                loss_g = self.loss_g(Dz)
                loss_gs.append(loss_g)
                
                tmp_discriminator = Discriminator(layers=self.discriminator.layers.copy())
                self.generator.backward(tmp_discriminator.backward(Dz - 1.))
                
                pbar.setInfo("loss_g {0}\tloss_d {1}".format('%.6f'%loss_g, '%.6f'%loss_d))
                pbar.show(j+1)

            self.updateLearningrate(i)
        
        fig = plt.figure()
        ax = fig.add_subplot(1,1,1)
        ax.plot(np.arange(epoch*iter_num), loss_ds, '-')
        ax.plot(np.arange(epoch*iter_num), loss_gs, '-')

    def generate(self, test_data):
        
        return self.generator.forward(test_data)
    
    def updateLearningrate(self, t):
        
        super().updateLearningrate(t)
        self.generator.lr = self.learning_rate
        self.discriminator.lr = self.learning_rate
                
        