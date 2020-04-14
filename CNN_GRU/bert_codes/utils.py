import time
import datetime
import numpy as np
from sklearn.metrics import f1_score
import random
import torch
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from sklearn.pipeline import Pipeline
import os


def fix_the_random(seed_val = 42):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)



def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))
    
    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))


# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

# def top_n_accuracy(preds,labels,n=5):
#     labels_flat = labels.flatten()
#     preds_flat = np.argsort()

def flat_fscore(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return f1_score(labels_flat, pred_flat, average='macro')


class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)



    
def save_model(model,tokenizer,params):
    if(params['to_save']==True):   
        if(params['csv_file']=='*_full.csv'):
            translate='actual'
        elif(params['csv_file']=='*_full_target.csv'):
            translate='actual_target'
        elif(params['csv_file']=='*_translated.csv'):
            translate='translated'
        # Saving best-practices: if you use defaults names for the model, you can reload it using from_pretrained()
        if(params['how_train']!='all'):
            output_dir = 'models_saved/'+params['path_files']+'_'+params['language']+'_'+translate+'_'+params['how_train']+'_'+str(params['sample_ratio'])
        else:
            output_dir = 'models_saved/'+params['path_files']+'_'+translate+'_'+params['how_train']+'_'+str(params['sample_ratio'])
    
        if(params['save_only_bert']):
            model=model.bert
            output_dir=output_dir+'_only_bert/'
        else:
            output_dir=output_dir+'/'
        print(output_dir)
        # Create output directory if needed
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

def save_model_target(model,tokenizer,params):
    if(params['to_save']):
        if params['translated']:
            output_dir = 'Target_identification_models/translated_'+params['what_bert']
        else:
            output_dir = 'Target_identification_models/actual_'+params['what_bert']
        if(params['save_only_bert']):
            model=model.bert
            output_dir=output_dir+'_only_bert/'
        else:
                output_dir=output_dir+'/'
        print(output_dir)
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        print("Saving model to %s" % output_dir)

        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        
        model_to_save = model.module if hasattr(model, 'module') else model  # Take care of distributed/parallel training
        model_to_save.save_pretrained(output_dir)
        tokenizer.save_pretrained(output_dir)

def multihot_encode(df):
    target_map = {
    'other':0,
    'african_descent':1,
    'arabs':2,
    'asians':3,
    'christian':4,
    'gay':5,
    'hispanics':6,
    'immigrants':7,
    'indian/hindu':8,
    'individual':9,
    'jews':10,
    'left_wing_people':11,
    'muslims':12,
    'refugees':13,
    'special_needs':14,
    'women':15
    }
    labels = df['label'].values
    labels = labels.reshape(labels.shape[0],1)
    targets = df.drop('label',axis=1)
    targets = targets.replace({col:target_map for col in targets.columns})
    targets = targets.values
    mb = MultiLabelBinarizer(classes = list(range(16)))
    targets = mb.fit_transform(targets)
    labels_target = np.concatenate([labels,targets],axis=1)
    return labels_target
    # assert False

def take_top_n(logits,n=5):
    # print(logits)
    top_n_indices = np.argsort(logits,axis=1)[:,::-1][:,:n]
    preds = np.zeros(logits.shape)
    # print(top_n_indices)
    np.put_along_axis(preds,top_n_indices,1,axis=1)
    # print(preds)
    return preds
    # assert False
