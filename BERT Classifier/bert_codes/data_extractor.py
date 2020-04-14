import pandas as pd
import numpy as np

# Sample the given dataframe df to select n_sample number of points. 
def stratified_sample_df(df, col, n_samples,sampled='stratified',random_state=1):
    if(sampled=='stratified'):
        df_=df.groupby(col, group_keys=False).apply(lambda x: x.sample(int(np.rint(n_samples*len(x)/len(df))))).sample(frac=1,random_state=random_state).reset_index(drop=True)
    
    elif(sampled=='equal'):
        df_=df.groupby(col, group_keys=False).apply(lambda x: x.sample(int(n_samples/2))).sample(frac=1,random_state=random_state).reset_index(drop=True)
    
    return df_

###### data collection taking all at a time

def data_collector(file_names,params,is_train):
    if(params['csv_file']=='*_full.csv'):
        index=12
    elif(params['csv_file']=='*_translated.csv'):
        index=23
    sample_ratio=params['sample_ratio']
    type_train=params['how_train']
    sampled=params['samp_strategy']
    take_ratio=params['take_ratio']
    language=params['language']

    # If the data being loaded is not train, i.e. either val or test, load everything and return
    if(is_train!=True):
        df_test=[]
        for file in file_names:
            lang_temp=file.split('/')[-1][:-index]
            if(lang_temp==language):
                df_test.append(pd.read_csv(file))
        df_test=pd.concat(df_test,axis=0)
        return df_test
    # If train data is being loaded, 
    else:
        # Baseline setting - only target language data is loaded
        if(type_train=='baseline'):
            df_test=[]
            for file in file_names:

                lang_temp=file.split('/')[-1][:-index]
                print(lang_temp)

                if(lang_temp==language):
                    temp=pd.read_csv(file)
                    df_test.append(temp)
            df_test=pd.concat(df_test,axis=0)
        # Zero shot setting - all except target language loaded
        if(type_train=='zero_shot'):
            df_test=[]
            for file in file_names:
                lang_temp=file.split('/')[-1][:-index]
                if(lang_temp=='English'):
                    temp=pd.read_csv(file)

                    df_test.append(temp)
            df_test=pd.concat(df_test,axis=0)

        # All_but_one - all other languages fully loaded, target language sampled
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
