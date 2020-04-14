import torch
import transformers
from keras.preprocessing.sequence import pad_sequences
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
batch_size = 8

MAX_LEN = 512

# Function to tokenize given sentences
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
                                sent,                      # Sentence to encode.
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

# Create mask for the given inputs.
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

# Truncate and Tokenize sentences, then pad them
def combine_features(sentences,tokenizer,max_length=512):
    input_ids=custom_tokenize(sentences,tokenizer,max_length)
    input_ids = pad_sequences(input_ids, dtype="long", 
                          value=0, truncating="post", padding="post")
    print(input_ids.shape)
    att_masks=custom_att_masks(input_ids)
    return input_ids,att_masks

# Generate pytorch data loader with the given dataset.
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

