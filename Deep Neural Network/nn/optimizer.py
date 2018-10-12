# -*- coding: utf-8 -*-
"""
Created on Thu Sep 27 18:34:19 2018

@author: 周宝航
"""

import numpy as np

class Momentem(object):
    
    def __init__(self, shape_vw, shape_vb, beta=0.9):
        
        self.beta = beta
        self.vw = np.zeros(shape_vw)
        self.vb = np.zeros(shape_vb)
    
    def getWeight(self, dW, db, t=0):
        
        self.vw = self.beta * self.vw + (1. - self.beta) * dW
        self.vb = self.beta * self.vb + (1. - self.beta) * db
        return (self.vw, self.vb)

class RMSProp(object):
    
    def __init__(self, shape_sw, shape_sb, beta=0.999):
        
        self.beta = beta
        self.sw = np.zeros(shape_sw)
        self.sb = np.zeros(shape_sb)
    
    def getWeight(self, dW, db, t=0):
        
        self.sw = self.beta * self.sw + (1. - self.beta) * dW**2
        self.sb = self.beta * self.sb + (1. - self.beta) * db**2
        return (self.sw, self.sb)

class Adam(object):
    
    def __init__(self, shape_w, shape_b, beta1=0.9, beta2=0.999, epsilon=1e-8):
        
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.momentom = Momentem(shape_w, shape_b, beta1)
        self.rmsprop = RMSProp(shape_w, shape_b, beta2)
    
    def getWeight(self, dW, db, t):
        
        vw, vb = self.momentom.getWeight(dW, db)
        sw, sb = self.rmsprop.getWeight(dW, db)
        
        vw_correct = vw / (1. - self.beta1**t)
        vb_correct = vb / (1. - self.beta1**t)
        sw_correct = sw / (1. - self.beta2**t)
        sb_correct = sb / (1. - self.beta2**t)
        
        update_w = vw_correct / np.sqrt(sw_correct + self.epsilon)
        update_b = vb_correct / np.sqrt(sb_correct + self.epsilon)
        return (update_w, update_b)
        