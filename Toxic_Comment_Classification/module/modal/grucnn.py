#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun  3 15:54:30 2023

@author: tiantian
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense, Flatten, GRU, Bidirectional
from tensorflow.keras.optimizers import Adam
from .nn import NN

class GRUCNN(NN):
    def _build(self, pretrained_embedding):
        model = Sequential()
        if pretrained_embedding is not None:
            print(self.config['vocab_size'])
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], weights = [pretrained_embedding], input_length = self.config['maxlen'], trainable = False))        
        else:
            model.add(Embedding(self.config['vocab_size'], self.config['embedding_dim'], embeddings_initializer = 'uniform', input_length = self.config['maxlen'], trainable = True))
        #static embedding => encoder network +> Contextual Embedding
        model.add(Bidirectional(GRU(128, return_sequences = True, dropout = 0.1, recurrent_dropout = 0.1 )))
        #classifier
        model.add(Conv1D(64, kernel_size = 3, padding = 'valid', kernel_initializer = 'glorot_uniform'))
        model.add(GlobalMaxPooling1D())
        model.add(Dense(self.num_class, activation = 'sigmoid'))
        model.compile(optimizer = Adam(lr = 1e-3), loss = 'binary_crossentropy', metrics = ['accuracy'])
        model.summary()
        return model
        