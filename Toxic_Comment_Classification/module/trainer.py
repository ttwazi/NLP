#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:10:22 2023

@author: tiantian
"""


from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from module.modal import NaiveBayer, TextCNN, TextRNN, biLSTM, TransformerClassifier, GRUCNN


class Trainer(object):
    
    def __init__(self, config, logger, classes, pretrained_embedding, train_ds = None):
        self.config = config
        self.logger = logger
        self.classes = classes
        self.pretrained_embedding = pretrained_embedding

        self._create_model(classes, train_ds)
        
        
    def _create_model(self, classes, train_ds = None):
        if self.config['model_name'] == 'naivebayse':
            self.model = NaiveBayer(classes)
        elif self.config['model_name'] == 'textcnn':
            self.model = TextCNN(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'textrnn':
            self.model = TextRNN(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'bilstm':
            self.model = biLSTM(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'transformer':
            self.model = TransformerClassifier(classes, self.config, self.pretrained_embedding)
        elif self.config['model_name'] == 'grucnn':
            self.model = GRUCNN(classes, self.config, self.pretrained_embedding)
#        elif self.config['model_name'] == 'smallbert':
#            self.model = SmallBert(classes, self.config, train_ds)
        else:
            self.logger.warning("Model Type: {} is not supported yet".format(self.config['model_name']))
    
#    def fit_with_tf_dataset(self, train_ds, validate_ds):
 #       self.model.fit wi
        
    def fit(self, train_x, train_y):
        print(train_y.shape)
        self.model.fit(train_x, train_y)
        return self.model
        
    def validate(self, validate_x, validate_y):
        predictions = self.model.predict(validate_x)
        return self.metrics(predictions, validate_y)
    
    def metrics(self, predictions, labels):
        accuracy = accuracy_score(labels, predictions)
        cls_report= classification_report(labels, predictions, zero_division = 1)
        return accuracy, cls_report
        
        