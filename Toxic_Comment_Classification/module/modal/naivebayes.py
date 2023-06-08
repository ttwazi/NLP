#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jun  1 08:47:10 2023

@author: tiantian
"""
from sklearn.naive_bayes import MultinomialNB
import numpy as np



class NaiveBayer(object):
    
    def __init__(self, classes):
        self.models = {}
        self.classes = classes
        
        for cls in self.classes:
            model = MultinomialNB()
            self.models[cls] = model
        
    def fit(self, train_x, train_y):
        for idx, cls in enumerate(self.classes):
            class_labels = train_y[:, idx]
            self.models[cls].fit(train_x, class_labels)

        
    def predict(self, test_x):
        predictions = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            predictions[:, idx] = self.models[cls].predict(test_x)
        return predictions
    
    def predict_prob(self, test_x):
        predictions = np.zeros((test_x.shape[0], len(self.classes)))
        for idx, cls in enumerate(self.classes):
            predictions[:, idx] = self.models[cls].predict_proba(test_x)[:,1] 
        return predictions