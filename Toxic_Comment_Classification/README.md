# [Kaggle Toxic Comment Classification Challenge](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge)

This project developed a classification framework to detect toxic comments in natual language.

## Dataset
Please refer to this [Kaggle competition page](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge) to find details about dataset used in this project or to download the dataset.

Input data: natual language comments <br>
Class labels: "toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate" (The six labels are not mutually exclusive, which means a comment  can be classified into more than one category, and the comment is non-toxic if it's not classified into any of the catefories.)

## Method
The classification framework consists of four modules: **Preprocessor, Trainer, Predictor and Calibrator** based on object-oriented programming concept.

A basic machine learning classification model **Naive Bayes** and multiple deep learning models **TextCNN, TextRNN, biLSTM, Transformer** were tested to train the classifier to compare performance. <br>

Other techniques were tested with the above mentioned models to improve performance, which include: <br>.
* Use pretrained static word embeddings such as [**GloVe**](https://github.com/stanfordnlp/GloVe).
* Create new samples to solve the issue of unbalanced dataset.
* Calibrate the model using Calibrator (Calibrator use **IsotonicRegression** model).
* Use contextual embeddings trained from **GRU** or **BERT**.

Different combination of model type and techniques were tested to compare performance.

## Result

The combination of **BERT** + New Samples + Calibration got the best result with a competition score as high as 0.982, which is close to the top team in the competition.

