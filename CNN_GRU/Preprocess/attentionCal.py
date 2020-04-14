import numpy as np

###this files contain different attention mask calculation from the n masks from n annotators. In this code there are 3 annotators

##softmax calculate 

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)


###softmax attention
## input:
#### at mask: attention masks from 3 annotators,  
#### label: 
from numpy import array, exp


def sigmoid(z):
      g = 1 / (1 + exp(-z))
      return g
    
    
def softmax_attention(at_mask,label,variance=1):
    if(len(at_mask[0])==len(at_mask[1])==len(at_mask[2])):
        if(label=="normal"):
            at_mask_fin=[1/len(at_mask[0]) for x in at_mask[0]]
        else:
            print(at_mask)
            at_mask=variance*at_mask
            print(at_mask)
            at_mask_fin=np.sum(at_mask,axis=0)
            print(at_mask_fin)
            at_mask_fin=softmax(at_mask_fin)
        return at_mask_fin
    else:
        print("error")
        print(len(at_mask[0]),len(at_mask[1]),len(at_mask[2]))
        return []

    
def sigmoid_attention(at_mask,label):
    if(len(at_mask[0])==len(at_mask[1])==len(at_mask[2])):
            at_mask_fin=np.sum(at_mask,axis=0)
            at_mask_fin=sigmoid(at_mask_fin)
            return at_mask_fin
    else:
        print("error")
        print(len(at_mask[0]),len(at_mask[1]),len(at_mask[2]))
        return []



if __name__ == '__main__':
    print(softmax_attention(np.array([[0,0,0],[0,1,0],[0,1,1]]),"offensive",1))