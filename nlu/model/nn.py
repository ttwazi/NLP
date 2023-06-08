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
import os
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
import numpy as np


class NN(object):

    def __init__(self, classes, config, model_name):
        self.classes = classes
        self.num_class = len(classes)
        self.config = config
        self.model_path = os.path.join(config['output_path'], model_name)
        self.model = self._build()
    
    def _build(self):
        pass
    
    def fit_and_validate(self, train_x, train_y, validate_x, validate_y):

        self.model.fit(train_x, train_y, epochs = self.config['epochs'], verbose = True,\
                       batch_size = self.config['batch_size'], validation_data = (validate_x, validate_y))
        
        predictions = self.predict(validate_x)
        return predictions
    
    def evaluate(self, predictions, validate_x, validate_y):
        accuracy = accuracy_score( validate_y, predictions)
        cls_report = classification_report(validate_y, predictions, zero_division=1)
        return accuracy, cls_report
    
    def predict(self, test_x):
        probs = self.model.predict(test_x)
        return np.argmax(probs, axis = -1)
    
    def predict_prob(self, test_x):
        return self.model.predict(test_x)
    
    def save(self):
        self.model.save(self.model_path)
        
    def load(self):
        return keras.models.load_model(self.model_path)