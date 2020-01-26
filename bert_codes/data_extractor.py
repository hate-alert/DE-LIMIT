import pandas as pd
import numpy as np


def stratified_sample_df(df, col, n_samples):
    # n = min(n_samples, df[col].value_counts().min())
    # df_ = df.groupby(col).apply(lambda x: x.sample(n))
    # df_.index = df_.index.droplevel(0)
    #df.sample(n=n_samples, weights=col, random_state=1).reset_index(drop=True)
    df_=df.groupby(col, group_keys=False).apply(lambda x: x.sample(int(np.rint(n_samples*len(x)/len(df))))).sample(frac=1).reset_index(drop=True)
    return df_

def data_collector(file_names,language,is_train=False,sample_ratio=0.5,type_train='baseline'):
   
    if(is_train!=True):
        df_test=[]
        for file in file_names:
            lang_temp=file.split('/')[-1][:-12]
            if(lang_temp==language):
                df_test.append(pd.read_csv(file))
        df_test=pd.concat(df_test,axis=0)
        return df_test
    else:
        if(type_train=='baseline'):
            df_test=[]
            for file in file_names:
                lang_temp=file.split('/')[-1][:-12]
                if(lang_temp==language):
                    temp=pd.read_csv(file)
                    n_samples=int(len(temp)*sample_ratio/100)
                    if(n_samples==0): 
                        n_samples+=1
                    temp_sample=stratified_sample_df(temp, 'label', n_samples)
                    df_test.append(temp_sample)
            df_test=pd.concat(df_test,axis=0)
            return df_test
        if(type_train=='zero_shot'):
            df_test=[]
            for file in file_names:
                lang_temp=file.split('/')[-1][:-12]
                if(lang_temp=='English'):
                    temp=pd.read_csv(file)
                    n_samples=int(len(temp)*sample_ratio/100)
                    if(n_samples==0): 
                        n_samples+=1
                    temp_sample=stratified_sample_df(temp, 'label', n_samples)
                    df_test.append(temp_sample)
            df_test=pd.concat(df_test,axis=0)
            return df_test
        if(type_train=='all_but_one'):
            df_test=[]
            for file in file_names:
                lang_temp=file.split('/')[-1][:-12]
                if(lang_temp!=language):
                    temp=pd.read_csv(file)
                    n_samples=int(len(temp)*sample_ratio/100)
                    print("n_samples,total dataframe",n_samples,len(temp))
                    if(n_samples==0): 
                        n_samples+=1
                    temp_sample=stratified_sample_df(temp, 'label', n_samples)
                    df_test.append(temp_sample)
            df_test=pd.concat(df_test,axis=0)
            return df_test
        if(type_train=='all'):
            df_test=[]
            for file in file_names:
                temp=pd.read_csv(file)
                n_samples=int(len(temp)*sample_ratio/100)
                print("n_samples,total dataframe",n_samples,len(temp))
                if(n_samples==0): 
                    n_samples+=1
                temp_sample=stratified_sample_df(temp, 'label', n_samples)
                df_test.append(temp_sample)
            df_test=pd.concat(df_test,axis=0)
            return df_test