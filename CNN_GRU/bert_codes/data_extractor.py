import pandas as pd
import numpy as np


def stratified_sample_df(df, col, n_samples,sampled='stratified',random_state=1):
    # n = min(n_samples, df[col].value_counts().min())
    # df_ = df.groupby(col).apply(lambda x: x.sample(n))
    # df_.index = df_.index.droplevel(0)
    #df.sample(n=n_samples, weights=col, random_state=1).reset_index(drop=True)
    if(sampled=='stratified'):
        df_=df.groupby(col, group_keys=False).apply(lambda x: x.sample(int(np.rint(n_samples*len(x)/len(df))))).sample(frac=1,random_state=random_state).reset_index(drop=True)
    
    elif(sampled=='equal'):
        df_=df.groupby(col, group_keys=False).apply(lambda x: x.sample(int(n_samples/2))).sample(frac=1,random_state=random_state).reset_index(drop=True)
    
    return df_

# def data_collector(file_names,language,is_train=False,sample_ratio=0.5,type_train='baseline',sampled='stratified',take_ratio=False):
   
#     if(is_train!=True):
#         df_test=[]
#         for file in file_names:
#             lang_temp=file.split('/')[-1][:-12]
#             if(lang_temp==language):
#                 df_test.append(pd.read_csv(file))
#         df_test=pd.concat(df_test,axis=0)
#         return df_test
#     else:
#         if(type_train=='baseline'):
#             df_test=[]
#             print(file_names)
#             for file in file_names:
#                 lang_temp=file.split('/')[-1][:-12]
#                 if(lang_temp==language):
#                     temp=pd.read_csv(file)
#                     df_test.append(temp)

#             df_test=pd.concat(df_test,axis=0)
#             return df_test
#         if(type_train=='zero_shot'):
#             df_test=[]
#             for file in file_names:
#                 lang_temp=file.split('/')[-1][:-12]
#                 if(lang_temp=='English'):
#                     temp=pd.read_csv(file)

#                     if(take_ratio==True):
#                         n_samples=int(len(temp)*sample_ratio/100)
#                     else:
#                         n_samples=sample_ratio

#                     if(n_samples==0): 
#                         n_samples+=1
#                     temp_sample=stratified_sample_df(temp, 'label', n_samples,sampled)
#                     df_test.append(temp_sample)
#             df_test=pd.concat(df_test,axis=0)
#             return df_test
#         if(type_train=='all_but_one'):
#             df_test=[]
#             for file in file_names:
#                 lang_temp=file.split('/')[-1][:-12]
#                 if(lang_temp!=language):
#                     temp=pd.read_csv(file)
#                     if(take_ratio==True):
#                         n_samples=int(len(temp)*sample_ratio/100)
#                     else:
#                         n_samples=sample_ratio

#                     if(n_samples==0): 
#                         n_samples+=1
#                     temp_sample=stratified_sample_df(temp, 'label', n_samples,sampled)
#                     df_test.append(temp_sample)
#             df_test=pd.concat(df_test,axis=0)
#             return df_test
#         if(type_train=='all'):
#             df_test=[]
#             for file in file_names:
#                 temp=pd.read_csv(file)
#                 df_test.append(temp)
#             df_test=pd.concat(df_test,axis=0)

#             return df_test
#         if(type_train=='all_multitask'):
#             df_test=[]
#             for file in file_names:
#                 temp=pd.read_csv(file)
#                 df_test.append(temp)
#             df_test=pd.concat(df_test,axis=0)
#             return df_test
#         if(type_train=='all_multitask_own'):
#             df_test=[]
#             for file in file_names:
#                 temp=pd.read_csv(file)
#                 df_test.append(temp)
#             df_test=pd.concat(df_test,axis=0)
#             return df_test



###### data collection taking all at a time

def data_collector(file_names,params,is_train):
    if(params['csv_file']=='*_full.csv'):
        index=12
    elif(params['csv_file']=='*_translated.csv'):
        index=23
    elif(params['csv_file']=='*_full_target.csv'):
        index = 19
    elif(params['csv_file']=='*_translated_target.csv'):
        index = 30
    sample_ratio=params['sample_ratio']
    type_train=params['how_train']
    sampled=params['samp_strategy']
    take_ratio=params['take_ratio']
    language=params['language']

    if(is_train!=True):
        df_test=[]
        for file in file_names:
            lang_temp=file.split('/')[-1][:-index]
            if(lang_temp==language):
                df_test.append(pd.read_csv(file))
        df_test=pd.concat(df_test,axis=0)
        return df_test
    else:
        if(type_train=='baseline'):
            df_test=[]
            for file in file_names:

                lang_temp=file.split('/')[-1][:-index]
                print(lang_temp)

                if(lang_temp==language):
                    temp=pd.read_csv(file)
                    df_test.append(temp)
            df_test=pd.concat(df_test,axis=0)
        if(type_train=='zero_shot'):
            df_test=[]
            for file in file_names:
                lang_temp=file.split('/')[-1][:-index]
                if(lang_temp=='English'):
                    temp=pd.read_csv(file)

                    df_test.append(temp)
            df_test=pd.concat(df_test,axis=0)
        if(type_train=='all_but_one'):
            df_test=[]
            for file in file_names:
                lang_temp=file.split('/')[-1][:-index]
                if(lang_temp!=language):
                    temp=pd.read_csv(file)
                    df_test.append(temp)
            df_test=pd.concat(df_test,axis=0)
        


        if(take_ratio==True):
            n_samples=int(len(df_test)*sample_ratio/100)
        else:
            n_samples=sample_ratio

        if(n_samples==0): 
            n_samples+=1
        df_test=stratified_sample_df(df_test, 'label', n_samples,sampled,params['random_seed'])
        return df_test

def data_collector_target(file_names,params,is_train):
    df_test=[]    
    for file in file_names:
        temp=pd.read_csv(file)
        df_test.append(temp)
    df_test=pd.concat(df_test,axis=0)
    return df_test