#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 15:06:44 2023

@author: tiantian
"""

from tensorflow import keras
from keras import layers
from keras.models import Sequential
#from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Conv1D, MaxPooling1D, Dense, Flatten, Dropout
from .nn import NN

class TextCNN(NN):
    
    
    def _build(self, pretrained_embedding):
        model = Sequential()
        if pretrained_embedding is not None:
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], weights = [pretrained_embedding],input_length = self.config['maxlen'], trainable = False))
        else:
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], input_length = self.config['maxlen'], trainable = True))

        model.add(Conv1D(128, 7, activation = 'relu', padding = 'same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(256, 5, activation = 'relu', padding = 'same'))
        model.add(MaxPooling1D())
        model.add(Conv1D(512, 3, activation = 'relu', padding = 'same'))
        model.add(MaxPooling1D())
        model.add(Flatten())
        model.add(Dense(128, activation = 'relu'))
        model.add(Dropout(0.5))
        model.add(Dense(self.num_class, activation = None))
        model.add(Dense(self.num_class, activation = 'sigmoid'))
        model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])
        model.summary()
        return model
    

        
        