#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed May 31 17:45:23 2023

@author: tiantian
"""

import yaml
import logging
import argparse
from module import Preprocessor, Trainer, Predictor, Calibrator

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description = 'Process commandline')
    parser.add_argument('--config', type = str, required = True)
    parser.add_argument('--log_level', type = str, default = 'INFO')
    args = parser.parse_args()
    
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format=FORMAT, level = args.log_level)
    logger = logging.getLogger('global_logger')
    
    logger.info("Start!")
    
    with open(args.config, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            preprocessor = Preprocessor(config['preprocessing'], logger)
            if config['preprocessing']['input_convertor'] == 'tf_dataset':
                train_ds, validate_ds, validate_x_ds, validate_y, test_x_ds = preprocessor.process()
                trainer = Trainer(config['training'], logger, preprocessor.classes, None,train_ds)
                model = trainer.fit_with_tf_dataset(train_ds, validate_ds)
                accuracy, cls_report = trainer.validate_with_tf_dataset(validate_x_ds, validate_y)
            else:            
                data_x, data_y, train_x, train_y, validate_x, validate_y, test_x, test_ids  = preprocessor.process()
                if config['training']['model_name'] != 'naivebayes':
                    config['training']['vocab_size'] = len(preprocessor.word2ind.keys())
                    
                pretrained_embedding = preprocessor.embedding_matrix if config['preprocessing'].get('pretrained_embedding', None) else None
                
               
                trainer = Trainer(config['training'],logger, preprocessor.classes, pretrained_embedding)
       
            
            model = trainer.fit(train_x, train_y)
            accuracy, cls_report = trainer.validate(validate_x, validate_y)
            logger.info("validation accuracy is {}".format(accuracy))
            logger.info("classification report: {}".format(cls_report))

            logger.info("fit_completed")
            predictor = Predictor(config['predict'], logger, model)
            logger.info("predictor_created")
            if config['predict']['enable_calibration']:
                predictor.train_calibrators(validate_x, validate_y)
            probs = predictor.predict_prob(test_x)
            logger.info("prediction result get")
            predictor.save_result(preprocessor.test_ids, probs)
        except yaml.YAMLError as err:
            logger.warning("Config file error: {}".format(err))
            
            