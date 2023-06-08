#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  6 19:19:15 2023

@author: tiantian
"""

import argparse
import yaml
import pickle
import logging
from model import biLSTM, TextCNN
from keras.utils import to_categorical
import numpy as np
import sys
import requests
from tensorflow import keras
import json



def execute_domain(city):
    
     url ="http://api.openweathermap.org/data/2.5/weather?q={}&appid=1135522aaff37f15fc091ecca9334dbf".format(city)
     print(url)
     ret = requests.get(url).content
     print(ret)
     response = json.loads(ret)
     if response["cod"] == 200:
         return "City:{} Weather:{} Temperate:{} Humidity:{}".format(city, \
                            response['weather'][0]['description'], response['main']['temp'], \
                            response['main']['humidity'])
     else:
        return "city not found"
         
     

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description = 'Process commandline')
    parser.add_argument('--config', type = str, required = True)
    parser.add_argument('--log_level', type = str, default = 'INFO')
    args = parser.parse_args()
    
    FORMAT = '%(asctime)-15s %(message)s'
    logging.basicConfig(format = FORMAT, level = args.log_level)
    logger = logging.getLogger('global_logger')
    
    with open(args.config, 'r') as config_file:
        try:
            config = yaml.safe_load(config_file)
            cls_data = pickle.load(open(config['preprocessing']['cls_data_file'], 'rb'))
            weather_seq_data = pickle.load(open(config['preprocessing']['weather_seq_data_file'], 'rb'))

            cls_labels = cls_data['label_list']
            cls_config = config['training'].copy()
            cls_config['epochs'] = config['training']['cls_epochs']
            vocab_size = len(cls_data['word2ind'].keys())
            cls_config['vocab_size'] = vocab_size
            classifier = TextCNN(cls_labels, cls_config, 'classification' )
            
            weather_seq_labels = weather_seq_data['label_list']
            weather_seq_config = config['training'].copy()
            weather_seq_config['epochs'] = config['training']['seq_epochs']
            vocab_size = len(weather_seq_data['word2ind'].keys())
            weather_seq_config['vocab_size'] = vocab_size
            seq_parser = biLSTM(weather_seq_labels, weather_seq_config, 'seq_parser')
            weather_seq_sample_weight = weather_seq_data['sample_weight']
     
            if not config['training']['predict_only']:
                cls_train_y = to_categorical(cls_data['train_y'])
                cls_val_y = to_categorical(cls_data['val_y'])
                predictions = classifier.fit_and_validate(cls_data['train_x'], \
                                                         cls_train_y, cls_data['val_x'], cls_val_y)
                accuracy, cls_report = classifier.evaluate(predictions,cls_data['val_x'], cls_data['val_y'])
                logger.info("classifier validation accuracy: {}".format(accuracy))
                logger.info("classifier validation classification report: {}".format(cls_report))  
                classifier.save()
                
                weather_seq_train_y = to_categorical(weather_seq_data['train_y'])
                weather_seq_val_y = to_categorical(weather_seq_data['val_y'])                
                predictions = seq_parser.fit_and_validate(weather_seq_data['train_x'], weather_seq_train_y, weather_seq_sample_weight,\
                                            weather_seq_data['val_x'],weather_seq_val_y)
                accuracy, seq_report = seq_parser.evaluate(predictions, weather_seq_data['val_x'], weather_seq_data['val_y'])
                logger.info("slot labelling validation accuracy: {}".format(accuracy))
                logger.info("slot labelling report: {}".format(seq_report)) 
                seq_parser.save()

            else:
                classifier.load()
                seq_parser.load()
                cls_val_y = to_categorical(cls_data['val_y'])
                weather_seq_val_y = to_categorical(weather_seq_data['val_y'])
                cls_predictions = classifier.predict(cls_val_y)
                weather_seq_predictions = seq_parser.predict(weather_seq_val_y)
                
                accuracy, cls_report = classifier.evaluate(predictions,cls_data['val_x'], cls_data['val_y'])
                logger.info("classifier validation accuracy: {}".format(accuracy))
                logger.info("classifier validation classification report: {}".format(cls_report))  
                
                accuracy, seq_report = seq_parser.evaluate(predictions,weather_seq_data['val_x'], weather_seq_data['val_y'])
                logger.info("slot labelling validation accuracy: {}".format(accuracy))
                logger.info("slot labelling eport: {}".format(seq_report)) 
                

            logger.info(("=" * 20 + '\n') * 5)
            logger.info("Service is up: ")
            cls_word2ind = cls_data['word2ind']
            weather_seq_word2ind = weather_seq_data['word2ind']
            cls_label_list = cls_data['label_list']
            weather_seq_label_list = weather_seq_data['label_list']
            
            while True:
                query = sys.stdin.readline()
                tokens = query.strip().split()
                tokens_ids = [cls_word2ind.get(token, cls_word2ind['<unk>']) for token in tokens]
                tokens_ids = np.array([tokens_ids], dtype = object)
                tokens_ids = keras.preprocessing.sequence.pad_sequences(tokens_ids, maxlen = 64, padding = 'post', value = cls_word2ind['<pad>'])

                cls_predictions = classifier.predict_prob(tokens_ids)
                pred_cls = np.argmax(cls_predictions[0])
                logger.info("pred_cls: {}".format(cls_label_list[pred_cls]))
                if cls_label_list.index('GetWeather') == pred_cls:
                    logger.info('It is Weather Query!')
                    tokens_ids = [weather_seq_word2ind.get(token, weather_seq_word2ind['<unk>']) for token in tokens]
                    tokens_ids = np.array([tokens_ids], dtype = object) 
                    tokens_ids = keras.preprocessing.sequence.pad_sequences(tokens_ids, maxlen = 64, padding = 'post', value = weather_seq_word2ind['<pad>'])

                    seq_predictions = seq_parser.predict_prob(tokens_ids)
                    seq_labels_idx = np.argmax(seq_predictions[0], -1)
                    seq_labels= [weather_seq_label_list[idx] for idx in seq_labels_idx]
                    seq_labels = seq_labels[0:len(tokens)]

                    city_name = []     

                    for token_, label_ in zip(tokens, seq_labels):

                        logger.info("Query: {} -> Label:{}".format(token_, label_))
                        #logger.info("Query:{}".format(token))

                        if label_ == 'city':
                            city_name.append(token_)
                    city_name = ' '.join(city_name)
                    if len(city_name):
                        response = execute_domain(city_name)
                        logger.info("Response: {}".format(response))
                    else:
                        logger.info("Query not supported")
                
                else:
                    logger.info("Query not supported")

    
    
        except yaml.YAMLError as err:
            logger.warning("Config file err: {}".format(err))