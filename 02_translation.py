#!/usr/bin/env python
# coding: utf-8

# In[22]:


#export 
import re
import emoji
from gtrans import translate_text, translate_html


# In[23]:


#export

import random
import pandas as pd
import numpy as np
from multiprocessing import  Pool
import time

def remove_emoji(text):
    return emoji.get_emoji_regexp().sub(u'', text)


def approximate_emoji_insert(string, index,char):
    if(index<(len(string)-1)):
        
        while(string[index]!=' ' ):
            if(index+1==len(string)):
                break
            index=index+1
        return string[:index] + ' '+char + ' ' + string[index:]
    else:
        return string + ' '+char + ' ' 
    


def extract_emojis(str1):
    try:
        return [(c,i) for i,c in enumerate(str1) if c in emoji.UNICODE_EMOJI]
    except AttributeError:
        return []


# In[30]:


#export 
def parallelize_dataframe(df, func, n_cores=4):
    '''parallelize the dataframe'''
    df_split = np.array_split(df, n_cores)
    pool = Pool(n_cores)
    df = pd.concat(pool.map(func, df_split))
    pool.close()
    pool.join()
    return df


def translate(x,lang):
    '''provide the translation given text and the language'''
    #x=preprocess_lib.preprocess_multi(x,lang,multiple_sentences=False,stop_word_remove=False, tokenize_word=False, tokenize_sentence=False)
    emoji_list=extract_emojis(x)
    try:
        translated_text=translate_text(x,lang,'en')
    except:
        translated_text=x
    for ele in emoji_list:
        translated_text=approximate_emoji_insert(translated_text, ele[1],ele[0])
    return translated_text



def add_features(df):
    '''adding new features to the dataframe'''
    translated_text=[]
    for index,row in df.iterrows():
        if(row['lang']in ['en','unk']):
            translated_text.append(row['text'])
        else:
            translated_text.append(translate(row['text'],row['lang']))    
    df["translated"]=translated_text
    #df=pd.concat([df_english,df_hindi],axis=0)
    #df=df.sort_values(['timestamp'], ascending=True)
    #print("done")
    return df


# In[3]:


import glob 


files=glob.glob('*.csv')


# In[ ]:


from tqdm import tqdm
size=1000


for file in files:
    wp_data=pd.read_csv(file)
    list_df=[]
    print(file, len(wp_data))
    for i in tqdm(range(0,len(wp_data),size)):
                print(i,"_iteration")
                df_new=parallelize_dataframe(wp_data[i:i+size],add_features,n_cores=20)
                list_df.append(df_new)
    df_translated=pd.concat(list_df,ignore_index=True)
    file_name='Translation/translated_'+file
    df_translated.to_csv(file_name)






