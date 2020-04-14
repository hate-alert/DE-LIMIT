import sys, os
# sys.path.append(os.path.abspath(os.path.join('..', 'utils')))
# from utils import pad
from tqdm import tqdm
import numpy as np
from torch.utils.data import Dataset
import pandas as pd
from collections import Counter
from itertools import chain, repeat, islice
from tqdm import tqdm_notebook,tqdm


def most_frequent(List): 
    counter=Counter(List)
    max_freq= max(set(List), key = List.count)
    return max_freq,counter[max_freq]


def CheckForGreater(list1, val):  
    return(all(x > val for x in list1))  

def pad_infinite(iterable, padding=None):
       return chain(iterable, repeat(padding))

def pad(iterable, size, padding=None):
       return islice(pad_infinite(iterable, padding), size)







#### class that returns the imitates the dataloader function of pytorch
#input: path of the dataframe pickle    ##pickle since the dataframe contains list




def encode_data(df,word2id):
        max_len=0
        for index,row in tqdm(df.iterrows(),total=len(df)):
            
            if(max_len<len(row["text"].split(' '))):
                max_len=len(row["text"].split(' '))
        
        tuple_new_data=[]
        
        
        for index,row in df.iterrows():
            list_token_id=[]
            words=row['text'].split(' ')
            for word in words:
                try:
                    index=word2id[word]
                except KeyError:
                    index=len(list(word2id.keys()))
                list_token_id.append(index)
            with_padding_text=list(pad(list_token_id, max_len, len(list(word2id.keys()))+1))
            tuple_new_data.append((with_padding_text,row['label'],row['text']))
        return tuple_new_data




class SCRAT_Dataset(Dataset):
    def __init__(self, data_path):
        self.samples = []
        self.data_path=data_path
        self.max_len=0
        self.get_data()
        
        
    #### gets the data from the file and stores in samples
    def get_data(self):
        total_data=pd.read_pickle(self.data_path)
        for index,row in total_data.iterrows():
            
            self.samples.append((row["Text"],row["Attention"],row["Label"]))
            if(self.max_len<len(row["Text"])):
                self.max_len=len(row["Text"])
                
    #### returns length of the samples           
    def __len__(self):
        return len(self.samples)

    #### returns a particular item from the dataset
    def __getitem__(self, idx):
        return self.samples[idx]
    

    ### 1.encodes data into word ids using vocab 
    ### 2.adds padding to both attention masks and the list of the words
    ### input: train data/val data/test data tuples and vocab generated using only the train dataset
    ### output: encoded and padded train data/val data/test data
    def encode_data(self,tuple_data,vocab):
        tuple_new_data=[]

        for data in tuple_data:
            list_token_id=[]
            list_mask=[]
            for word in data[0]:
                try:
                    index=vocab.stoi[word]
                except KeyError:
                    index=vocab.stoi['unk']
                list_token_id.append(index)
                list_mask.append(1.0)
            with_padding_text=list(pad(list_token_id, self.max_len, vocab.stoi['<pad>']))
            with_padding_attention=list(pad(data[1], self.max_len, 0.0))
            with_padding_mask=list(pad(list_mask, self.max_len, 0.0))
            tuple_new_data.append((with_padding_text,with_padding_attention,data[2],with_padding_mask,data[0]))
        return tuple_new_data




### This class deals with the 
### 1.creating vocab  
### 2.stoi dict
### 3.itos dict 
### 4.embeddings matrix


class Vocab_own():
    def __init__(self,list_of_tuples, model,word2id):
        self.itos={}
        self.stoi={}
        self.vocab={}
        self.embeddings=[]
        self.tuples =list_of_tuples
        self.model=model
        self.word2id=word2id
    
    ### load embedding given a word and unk if word not in vocab
    ### input: word
    ### output: embedding,word or embedding for unk, unk
    def load_embeddings(self,word):
        try:
            
            return self.model[self.word2id[word]],word
        except KeyError:
            return np.zeros((300,)),'unk'
    # def load_embeddings(self,word):
    #     try:
    #         return np.array(self.model.emb(word),dtype=float),word
    #     except KeyError:
    #         return np.array(self.model.emb('unk'),dtype=float),'unk'
    
    ### create vocab,stoi,itos,embedding_matrix
    ### input: **self
    ### output: updates class members

    def create_vocab(self):
        count=0
        for tuple1 in tqdm(self.tuples):
            words=tuple1.split(' ')
            for word in words:
                # print(word)
                vector,word=self.load_embeddings(word)  
                try:
                    self.vocab[word]+=1
                except KeyError:
                    # if(word=='unk'):
                    #     print(word)
                    self.vocab[word]=1
                    self.stoi[word]=count
                    self.itos[count]=word
                    self.embeddings.append(vector)
                    count+=1
        self.vocab['<pad>']=1
        self.stoi['<pad>']=count
        self.itos[count]='<pad>'
        self.embeddings.append(np.zeros((300,), dtype=float))
        self.embeddings=np.array(self.embeddings)
