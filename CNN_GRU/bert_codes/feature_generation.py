import torch
import transformers
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
batch_size = 8

# Set the maximum sequence length.
# I've chosen 64 somewhat arbitrarily. It's slightly larger than the
# maximum training sentence length of 47...
MAX_LEN = 512


def custom_tokenize(sentences,tokenizer,max_length=512):
    input_ids = []
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        try:

            encoded_sent = tokenizer.encode(
                                sent ,                     # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_length,
                                # This function also supports truncation and conversion
                                # to pytorch tensors, but we need to do padding, so we
                                # can't use these features :( .
                                #max_length = 128,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                           
                           )

            # Add the encoded sentence to the list.
            
        except ValueError:
            encoded_sent = tokenizer.encode(
                                ' ',                      # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_length,
                                # This function also supports truncation and conversion
                                # to pytorch tensors, but we need to do padding, so we
                                # can't use these features :( .
                                #max_length = 128,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                           )
              ### decide what to later
        
        input_ids.append(encoded_sent)

    return input_ids



####### pair of sentences 
def custom_tokenize_pair(sentences,tokenizer,max_length=512):
    input_ids = []
    # For every sentence...
    for sent in sentences:
        # `encode` will:
        #   (1) Tokenize the sentence.
        #   (2) Prepend the `[CLS]` token to the start.
        #   (3) Append the `[SEP]` token to the end.
        #   (4) Map tokens to their IDs.
        try:

            encoded_sent = tokenizer.encode(
                                sent[0],  
                                sent[1],                    # Sentence to encode.
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_length,
                                # This function also supports truncation and conversion
                                # to pytorch tensors, but we need to do padding, so we
                                # can't use these features :( .
                                #max_length = 128,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                           
                           )

            # Add the encoded sentence to the list.
            
        except ValueError:
            encoded_sent = tokenizer.encode(
                                ' ',                      # Sentence to encode.
                                ' ',
                                add_special_tokens = True, # Add '[CLS]' and '[SEP]'
                                max_length = max_length,
                                # This function also supports truncation and conversion
                                # to pytorch tensors, but we need to do padding, so we
                                # can't use these features :( .
                                #max_length = 128,          # Truncate all sentences.
                                #return_tensors = 'pt',     # Return pytorch tensors.
                           )
              ### decide what to later
        
        input_ids.append(encoded_sent)

    return input_ids


def custom_tokenize_pair_two(sentences,tokenizer,max_length=512):
    ml_sent1=int(max_length*2/3)
    ml_sent2=int(max_length*1/3)
    
    print('length of sent', ml_sent1,ml_sent2)
    
    sentences_0=[sent[0] for sent in sentences]
    sentences_1=[sent[1] for sent in sentences]
    
    
    
    encode_sent1=custom_tokenize(sentences_0,tokenizer,ml_sent1)
    encode_sent2=custom_tokenize(sentences_1,tokenizer,ml_sent2)
    
    print('encoded sent', encode_sent1[0])
    input_ids=[]
    for sent1,sent2 in zip(encode_sent1,encode_sent2):
        sent=list(sent1)+list(sent2)
        input_ids.append(sent)
    return input_ids





















def custom_att_masks(input_ids):
    attention_masks = []

    # For each sentence...
    for sent in input_ids:

        # Create the attention mask.
        #   - If a token ID is 0, then it's padding, set the mask to 0.
        #   - If a token ID is > 0, then it's a real token, set the mask to 1.
        att_mask = [int(token_id > 0) for token_id in sent]

        # Store the attention mask for this sentence.
        attention_masks.append(att_mask)
    return attention_masks

def combine_features(sentences,tokenizer,max_length=512, take_pair=True,take_target=False):
    if(take_pair):
        input_ids=custom_tokenize_pair_two(sentences,tokenizer,max_length)
    elif(take_target):
        input_ids=custom_tokenize_pair(sentences,tokenizer,max_length)
    else:
        input_ids=custom_tokenize(sentences,tokenizer,max_length)
    print('Input shape before truncating',input_ids[0:5])
    input_ids = pad_sequences(input_ids, dtype="long", 
                          value=0, truncating="post", padding="post")
    print(input_ids.shape)
    att_masks=custom_att_masks(input_ids)
    return input_ids,att_masks

def return_dataloader(input_ids,labels,att_masks,batch_size=8,is_train=False):
    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels,dtype=torch.long)
    masks = torch.tensor(np.array(att_masks))
    data = TensorDataset(inputs, masks, labels)
    if(is_train==False):
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

def return_cnngru_dataloader(tuples,batch_size=8,is_train=False):

    input_ids=[ele[0] for ele in tuples]
    labels=[ele[1] for ele in tuples]

    inputs = torch.tensor(input_ids)
    labels = torch.tensor(labels,dtype=torch.long)
    data = TensorDataset(inputs, labels)
    if(is_train==False):
        sampler = SequentialSampler(data)
    else:
        sampler = RandomSampler(data)
    dataloader = DataLoader(data, sampler=sampler, batch_size=batch_size)
    return dataloader

