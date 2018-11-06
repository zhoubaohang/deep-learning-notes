# -*- coding: utf-8 -*-

import numpy as np

class LossFunction(object):
    
    def __init__(self, name):
        
        self.name = name
        self.func = getattr(self, name)
    
    def getLoss(self, _y, y):
        
        return self.func(_y, y)
    
    def mean_square_error(self, _y, y):
        
        return np.mean((_y - y)**2)
    
    def mean_abs_error(self, _y, y):
        
        return np.mean(np.abs(_y - y))
    
    def cross_entropy_loss(self, _y, y):
        
        return -np.mean(_y*np.log(y) + (1. - _y)*np.log(1. - y))
    
    def softmax_cross_entropy_loss(self, _y, y):
        
        return -np.log(np.mean(_y * y))