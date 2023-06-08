#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:41:34 2023

@author: tiantian
"""

from tensorflow import keras
from keras import layers
from keras.models import Sequential
#from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten, Dropout



class NN(object):

    def __init__(self, classes, config, pretrained_embedding):
        self.classes = classes
        self.num_class = len(classes)
        self.config = config
        self.model = self._build(pretrained_embedding)
    
    def _build(self, pretrained_embedding):
        pass
    
    def fit(self, train_x, train_y):
        print(train_y.shape)
        self.model.fit(train_x, train_y, epochs = self.config['epochs'], verbose = True, batch_size = self.config['batch_size'])
        
        
        
    def predict(self, test_x):
        probs = self.model.predict(test_x)
        return  probs >= 0.5
    
    def predict_prob(self, test_x):
        return self.model.predict(test_x)