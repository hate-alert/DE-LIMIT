# DELIMIT--DeEpLearning models for MultIlingual haTespeech

This repository contains the codes related to Deep learning models used for hatespeech detection task in a multilingual settings.

The models included here are:
1. Logistic regression applied on top of LASER embeddings.
2. Training a bert-base model on data translated from other language to English.
3. Training of multilingual BERT (mBERT) on the dataset directly.

These models are used in two settings.
1. Monolingual or the baseline setting - The training and the testing is done on the same language
2. Multilingual setting - The data from other languages is included in the training for the target language

The codes related to model 1 can be found in the folder `LASER+LR`
The codes related to model 2 and 3 can be found in the `BERT_training_inference.py`(Training part) and `BERT_inference.py`(Model evaluation part). 


