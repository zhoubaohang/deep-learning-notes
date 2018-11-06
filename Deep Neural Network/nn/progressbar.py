# -*- coding: utf-8 -*-
"""
Created on Mon Sep 24 13:38:37 2018

@author: 周宝航
"""
import sys

class ProgressBar(object):

    def __init__(self, title='Process', width=50, symbol='=', max_steps=100):
        
        self.info = ''
        # self.cache = []
        self.title = title
        self.width = width
        self.symbol = symbol
        self.max_steps = max_steps
        self.max_percent = 100.0
        
    def show(self, progress):
        
        num_arrow = int(progress * self.width / self.max_steps)
        num_line = self.width - num_arrow
        percent = progress * self.max_percent / self.max_steps
        process_bar = "{0}:[{1}{2}]{3}% \t{4}\r"\
                    .format(self.title, self.symbol * num_arrow, '-' * num_line, '%.2f'%percent, self.info)
        sys.stdout.write("{0}".format(process_bar))
        sys.stdout.flush()
        if percent >= self.max_percent:
            sys.stdout.write("{0}\n".format(process_bar))
    
    def setInfo(self, info):
        
        self.info = info
    
    def setTitle(self, title):
        
        self.title = title
    
