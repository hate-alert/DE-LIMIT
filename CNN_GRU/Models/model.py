import torch
import torch.nn as nn
import numpy as np

def global_max_pooling(tensor, dim, topk):
    """Global max pooling"""
    ret, _ = torch.topk(tensor, topk, dim)
    return ret

class CNN_GRU(nn.Module):
    def __init__(self,args,vector):
        super(CNN_GRU, self).__init__()
        self.embedsize = vector.shape[1]
        self.conv1 = nn.Conv1d(self.embedsize,100, 2)
        self.conv2 = nn.Conv1d(self.embedsize,100, 3,padding=1)
        self.conv3 = nn.Conv1d(self.embedsize,100, 4,padding=2)
        self.maxpool1D = nn.MaxPool1d(4, stride=4)
        self.seq_model = nn.GRU(100, 100, bidirectional=False, batch_first=True)
        self.embedding = nn.Embedding(args["vocab_size"], self.embedsize)
        self.embedding.weight = nn.Parameter(torch.tensor(vector.astype(np.float32), dtype=torch.float32))
        self.embedding.weight.requires_grad = args["train_embed"]
        self.num_labels=2
        self.weights=args['weights']
        self.out = nn.Linear(100, self.num_labels)

        
    def forward(self,x,labels=None):
        batch_size=x.size(0)
        h_embedding = self.embedding(x)
        new_conv1=self.maxpool1D(self.conv1(h_embedding.permute(0,2,1)))
        new_conv2=self.maxpool1D(self.conv2(h_embedding.permute(0,2,1)))
        new_conv3=self.maxpool1D(self.conv3(h_embedding.permute(0,2,1)))
        concat=self.maxpool1D(torch.cat([new_conv1, new_conv2,new_conv3], dim=2))
        h_seq, _ = self.seq_model(concat.permute(0,2,1))
        global_h_seq=torch.squeeze(global_max_pooling(h_seq, 1, 1)) 
        output=self.out(global_h_seq)
        
        if labels is not None:
        	loss_fct = nn.CrossEntropyLoss(weight=torch.tensor(self.weights).cuda())
        	loss = loss_fct(output.view(-1, self.num_labels), labels.view(-1))
        	return loss,output
        return output

        
