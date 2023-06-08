#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 20:10:10 2023

@author: tiantian
"""
import csv
from sklearn.metrics import classification_report as cls_report
from .calibrator import Calibrator
import numpy as np


class Predictor(object):
    
    def __init__(self, config, logger, model):
        self.config = config
        self.logger = logger
        self.model = model
        if self.config['enable_calibration']:
            self.calibrators = []
            for i in range(len(self.config['classes'])):
                self.calibrators.append(Calibrator(model_type = self.config['calibrator_type']))

        
    def predict(self, test_x):
        predictions = self.model.predict(test_x)
        return predictions
    
    def predict_raw_prob(self, test_x):
        if hasattr(self.model, 'predict_prob'):
            predictions = self.model.predict_prob(test_x)
        else:
            predictions = self.model.predict_proba(test_x)
            
        return predictions
    
    def predict_prob(self, test_x):
        prob = self.predict_raw_prob(test_x)
        if self.config['enable_calibration']:
            prob = self._calibrate(prob)
        return prob
  
    def save_result(self, test_ids, probs):
        with open(self.config['output_path'], 'w') as output_csv_file:
            header = [id, 'toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']
            
            writer = csv.writer(output_csv_file)
            writer.writerow(header)
            for test_id, prob in zip(test_ids, probs.tolist()):
                writer.writerow([test_id] + prob)
                
    def train_calibrators(self, x, y):
        self.logger.info("train calibrators")
        prob = self.predict_raw_prob(x)
        for i in range(len(self.config['classes'])):
            category = self.config['classes'][i]
            pred_prob = prob[:,i]
            truth_label= y[:,i]
            self.calibrators[i].plot_reliability_diagrams(truth_label, pred_prob, category, self.config['calibrators_output_path'])
            uncalibrated_ece, calibrated_ece = self.calibrators[i].fit(truth_label, pred_prob)
            self.logger.info("class:{}, uncalibrated_ece:{} calibrated_ece:{}".format(category, uncalibrated_ece, calibrated_ece))            

    def _calibrate(self, prob):
        calibrated_prob_list = []
        for i in range(len(self.config['classes'])):
            category = self.config['classes'][i]
            pred_prob = prob[:, i]
            calibrated_prob = self.calibrators[i].calibrate(pred_prob)
            calibrated_prob_list.append(calibrated_prob[:, 1])
        calibrated_prob = np.stack(calibrated_prob_list, axis=1)
        return calibrated_prob   
        
