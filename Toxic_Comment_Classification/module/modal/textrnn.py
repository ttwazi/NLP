#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun  2 10:44:25 2023

@author: tiantian
"""

from tensorflow import keras
from keras import layers
from keras.models import Sequential
#from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, SimpleRNN,  Dense, Dropout
from .nn import NN

class TextRNN(NN):
    
    
    def _build(self, pretrained_embedding):
        model = Sequential()
        if pretrained_embedding is not None:
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], weights = [pretrained_embedding],input_length = self.config['maxlen'], trainable = False))
        else:
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], input_length = self.config['maxlen'], trainable = True))
        model.add(SimpleRNN(128))
        model.add(Dense(self.num_class, activation = 'sigmoid'))
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        model.summary()
        return model
    



    
