#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jun  5 11:47:56 2023

@author: tiantian
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression


class Calibrator:
    
    ISOTONIC_REGRESSION = 'isotonic_regression'
    PLATT_SCALING = 'platt_scaling'
    
    def __init__(self, num_bins = 10, model_type = None):
        self.num_bins = num_bins
        if model_type == self.ISOTONIC_REGRESSION:
            self.model = IsotonicRegression()
        else:
            self.model = LogisticRegression()
            
    def _get_bin_sizes(self, pred_prob, num_bins):
        bins = np.linspace(0., 1. + 1e-9, num_bins + 1)
        bin_indies = np.digitize(pred_prob, bins) - 1
        bin_sizes = np.bincount(bin_indies, minlength = len(bins))
        bin_sizes = [i for i in bin_sizes if i != 0]
        return bin_sizes
        
    
    def plot_reliability_diagrams(self, truth_label, pred_prob, cls_label, output_path):
        truth_label, pred_prob = calibration_curve(truth_label, pred_prob, n_bins = self.num_bins)
        plt.figure(figsize = (10, 10))
        plt.gca()
        plt.plot([0,1], [0,1], color = 'r', linestyle = ":", label = "Perfect Calibration")
        plt.plot(pred_prob, truth_label, label = cls_label)
        plt.xlabel("Confidence", fontsize = 16)
        plt.ylabel("Accuracy", fontsize = 16)
        plt.grid(True, color = 'b')
        plt.legend(fontsize = 16)
        save_path = os.path.join(output_path, "{}.png".format(cls_label))
        plt.savefig(fname = save_path, format = 'png')
    
    def cal_ece(self, truth_label, pred_prob, bin_sizes):
        truth_label, pred_prob = calibration_curve(truth_label, pred_prob, n_bins = self.num_bins)
        total_samples = sum(bin_sizes)
        ece = np.float32(0)
        for m in range(len(bin_sizes)):
            ece += bin_sizes[m] / total_samples * np.abs(truth_label[m] - pred_prob[m])
        return ece.item()
    
    def fit(self, truth_label, pred_prob):
        bin_sizes = self._get_bin_sizes(pred_prob, self.num_bins)
        uncalibrated_ece = self.cal_ece(truth_label, pred_prob, bin_sizes)

        expanded_pred_prob = np.expand_dims(pred_prob, axis = 1)

        self.model.fit(expanded_pred_prob, truth_label)

        calibrated_prob = self.model.predict_proba(expanded_pred_prob)[:,1]
        bin_sizes = self._get_bin_sizes(calibrated_prob.flatten(), self.num_bins)
        calibrated_ece = self.cal_ece(truth_label, calibrated_prob.flatten(), bin_sizes) 
        
        return uncalibrated_ece, calibrated_ece
        

        
        
    
        
    def calibrate(self, pred_prob):
        expanded_pred_prob = np.expand_dims(pred_prob, axis = 1)
        return self.model.predict_proba(expanded_pred_prob)
        