import torch
import torch.nn as nn


def save_model(model, model_path):
    """Save model."""
    torch.save(model.state_dict(), model_path)


def load_model(model, model_path, use_cuda=False):
    """Load model."""
    map_location = 'cpu'
    if use_cuda and torch.cuda.is_available():
        map_location = 'cuda:0'
    model.load_state_dict(torch.load(model_path, map_location))
    return model

def cross_entropy(input, target, size_average=True):
    """ Cross entropy that accepts soft targets
    Args:
         pred: predictions for neural network
         targets: targets, can be soft
         size_average: if false, sum is returned instead of mean

    Examples::

        input = torch.FloatTensor([[1.1, 2.8, 1.3], [1.1, 2.1, 4.8]])
        input = torch.autograd.Variable(out, requires_grad=True)

        target = torch.FloatTensor([[0.05, 0.9, 0.05], [0.05, 0.05, 0.9]])
        target = torch.autograd.Variable(y1)
        loss = cross_entropy(input, target)
        loss.backward()
    """
    logsoftmax = nn.LogSoftmax()
    return torch.sum(-target * logsoftmax(input))
    # if size_average:
    #     return torch.mean(torch.sum(-target * logsoftmax(input), dim=1))
    # else:
    #     return torch.sum(torch.sum(-target * logsoftmax(input), dim=1))


def masked_cross_entropy(input1,target,mask):
    list_count=[]
    #print("len mask",len(mask))
    for h in range(len(mask)):
        count=0
        #print(x_batch_mask[h])
        for element in mask[h]:
            if element==0:
                break
            else:
                count+=1
        list_count.append(count)
    #print(list_count)
    cr_ent=0
    for h in range(0,len(list_count)):
        #print(input1.shape)
        cr_ent+=cross_entropy(input1[h][0:list_count[h]],target[h][0:list_count[h]])
    return cr_ent/len(list_count)


