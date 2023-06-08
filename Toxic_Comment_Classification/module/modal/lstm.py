#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 14:15:04 2023

@author: tiantian
"""

from tensorflow import keras
from keras import layers
from keras.models import Sequential
#from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Bidirectional,  Dense, Dropout
from .nn import NN

class biLSTM(NN):
    
    
    def _build(self, pretrained_embedding):
        model = Sequential()
        if pretrained_embedding is not None:
            print(self.config['vocab_size'])
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], weights = [pretrained_embedding],input_length = self.config['maxlen'], trainable = False))
        else:
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], input_length = self.config['maxlen'], trainable = True))
        model.add(Bidirectional(LSTM(128)))
        model.add(Dense(self.num_class, activation = 'sigmoid'))
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        model.summary()
        return model
    

