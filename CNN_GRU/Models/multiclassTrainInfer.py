import neptune
import numpy as np
from Models.modelUtils import *
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score,classification_report,accuracy_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder
import torch
import time
from Models.model import *
from collections import Counter
import collections
debug=False

def select_model(args):
    text=args["model"]
    if(text=="birnn"):
        model=BiRNN(args)
    if(text == "birnnatt"):
        model=BiAtt_RNN(args,return_att=False)
    if(text == "birnnscrat"):
        model=BiAtt_RNN(args,return_att=True)
    if(text == "cnngru"):
        model=CNN_GRU(args)
    if(text == "lstm_bad"):
        model=LSTM_bad(args)
    return model




        
                
#model = BiLSTMGRUAttention(args)
def cross_validation(args,params=None):
    n_epochs = args["epochs"] # how many times to iterate over all samples
    batch_size=args["batch_size"]
    n_epochs = args["epochs"] # how many times to iterate over all samples
    batch_size=args["batch_size"]
    # matrix for the out-of-fold predictions
    train_preds = np.zeros((len(args["data_train"])))

    x_train=np.array([ele[0] for ele in args["data_train"]])
    ### training data attention
    x_train_att=np.array([ele[1] for ele in args["data_train"]])
    ### mask to identify padding section
    x_train_mask=np.array([ele[3] for ele in args["data_train"]])
    ###y_labels
    y_train=np.array([ele[2] for ele in args["data_train"]])
    le = LabelEncoder()
    y_train = le.fit_transform(y_train)
    y_dumm = pd.get_dummies(y_train).values
    
    ####cross val folds
    splits = list(StratifiedKFold(n_splits=4, shuffle=True, random_state=10).split(x_train, y_train))
    for m, (train_idx, valid_idx) in enumerate(splits): 
        
        if(args["logging"]=="neptune"):
            tag_one='cross_val_'+str(m)
            name_one = args['model']+"_"+tag_one
            neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
            neptune.append_tag(tag_one)
            neptune.append_tag(args['model'])
        
                
        #model = BiLSTMGRUAttention(args)
        #model =BiGRUAttention(args)
        model=select_model(args)
        # split data in train / validation according to the KFold indeces
        # also, convert them to a torch tensor and store them on the GPU (done with .cuda())
        x_train_fold_pos = torch.tensor(x_train[train_idx], dtype=torch.long).cuda()
        y_train_fold = torch.tensor(y_dumm[train_idx, np.newaxis], dtype=torch.float32).cuda()
        x_val_fold_pos = torch.tensor(x_train[valid_idx], dtype=torch.long).cuda()
        y_val_fold = torch.tensor(y_dumm[valid_idx, np.newaxis], dtype=torch.float32).cuda()
        x_train_fold_att = torch.tensor(x_train_att[train_idx], dtype=torch.float32).cuda()
        x_val_fold_att = torch.tensor(x_train_att[valid_idx], dtype=torch.float32).cuda()
        x_train_fold_mask = torch.tensor(x_train_mask[train_idx], dtype=torch.float32).cuda()
        x_val_fold_mask = torch.tensor(x_train_mask[valid_idx], dtype=torch.float32).cuda()
        
        # make sure everything in the model is running on the GPU
        model.cuda()
        # define cross entropy loss
        # for numerical stability in the loss
        loss_fn = torch.nn.CrossEntropyLoss(reduction='mean')
        optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"])
        train = torch.utils.data.TensorDataset(x_train_fold_pos,x_train_fold_att,x_train_fold_mask,y_train_fold)
        valid = torch.utils.data.TensorDataset(x_val_fold_pos,x_val_fold_att,x_val_fold_mask, y_val_fold)

        train_loader= torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)
        valid_loader = torch.utils.data.DataLoader(valid, batch_size=batch_size, shuffle=False)

        #print(f'Fold {m + 1}')            
        for epoch in range(n_epochs):
            # set train mode of the model. This enables operations which are only applied during training like dropout
            start_time = time.time()
            model.train()
            avg_loss = 0.  
            train_preds_fold = np.zeros((x_train_fold_pos.size(0)))
            for j,(x_batch_pos,x_batch_att,x_batch_mask,y_batch) in enumerate(train_loader):
                # Forward pass: compute predicted y by passing x to the model.
                if(args["model"]!='birnnscrat'):
                    y_pred = model(x_batch_pos,x_batch_mask)
                    loss = loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1])
                else:
                    y_pred,pred_att = model(x_batch_pos,x_batch_mask)

                    attention_loss=args["att_loss"]*masked_cross_entropy(pred_att,x_batch_att.float(),x_batch_mask)
                    #attention_loss=args["att_loss"]*cr_ent/len(list_count)


                    loss = loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1])+attention_loss
                    if(debug==True):
                        print("attention_loss: ",attention_loss)
                        print("total loss: ",loss)
                        
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the Tensors it will update (which are the learnable weights
                # of the model)
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
                
                _, pred_label = torch.max(y_pred.cpu().data, 1)
                train_preds_fold[j * batch_size:(j+1) * batch_size] = pred_label
            
            # set evaluation mode of the model. This disabled operations which are only applied during training like dropout
            model.eval()

            # predict all the samples in y_val_fold batch per batch
            valid_preds_fold = np.zeros((x_val_fold_pos.size(0)))
            test_preds_fold = np.zeros((len(args["data_test"])))
            #test_att_fold = np.zeros((len(args["data_test"])))


            avg_val_loss = 0.
            
            for k, (x_batch_pos,x_batch_att,x_batch_mask, y_batch) in enumerate(valid_loader):
                
                if(args["model"]!='birnnscrat'):
                    y_pred = model(x_batch_pos,x_batch_mask).detach()
                    avg_val_loss += loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1]).item() / len(valid_loader)
                else:
                    y_pred,y_att = model(x_batch_pos,x_batch_mask)
                    y_pred=y_pred.detach()
                    y_att=y_att.detach()
                    
                    avg_val_loss += (loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1]).item()
                                     +args["att_loss"]*masked_cross_entropy(y_att,x_batch_att.float(),x_batch_mask).item()) / len(valid_loader)
                    
                
                _, pred_label = torch.max(y_pred.cpu().data, 1)
                
                valid_preds_fold[k * batch_size:(k+1) * batch_size] = pred_label
            f1score_val=f1_score(y_train[valid_idx], valid_preds_fold, average='macro')
            f1score_train=f1_score(y_train[train_idx], train_preds_fold, average='macro')
            accuracy_val=accuracy_score(y_train[valid_idx], valid_preds_fold)
            accuracy_train=accuracy_score(y_train[train_idx], train_preds_fold)
            
            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t train loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss, avg_val_loss,elapsed_time))
            
            if(args["logging"]=="neptune"):
                neptune.log_metric('epoch', epoch)
                neptune.log_metric('train_loss',avg_loss)
                neptune.log_metric('val_loss',avg_val_loss)
                neptune.log_metric('accuracy_train',accuracy_train)
                neptune.log_metric('accuracy_val',accuracy_val)
                neptune.log_metric('f1score_train',f1score_train)
                neptune.log_metric('f1score_val',f1score_val)
            else:
                print('Epoch {}/{} \t f1score_train={:.4f} \t f1score_val={:.4f}'.format(epoch + 1,n_epochs,f1score_train, f1score_val))
                print('Epoch {}/{} \t accuracy_train={:.4f} \t accuracy_val={:.4f}'.format(epoch + 1,n_epochs,accuracy_train, accuracy_val))
                print('================================================================')
        if(args["logging"]=="neptune"):
            neptune.stop()
        train_preds[valid_idx] = valid_preds_fold
        print(f"Confusion Matrix \n{classification_report(y_train[valid_idx], valid_preds_fold)}")
    return train_preds


import torch
import torch.utils.data



def return_data_loader(tuples_list,batch_size):
    x_train=torch.tensor(np.array([ele[0] for ele in tuples_list]), dtype=torch.long).cuda()
    x_train_att=torch.tensor(np.array([ele[1] for ele in tuples_list]), dtype=torch.float32).cuda()
    x_train_mask=torch.tensor(np.array([ele[3] for ele in tuples_list]), dtype=torch.float32).cuda()
    y_train=[ele[2] for ele in tuples_list]
    y_dumm = pd.get_dummies(y_train)
    print(y_dumm.head(5))
    y_train=torch.tensor(y_dumm.values, dtype=torch.float32).cuda()
    train = torch.utils.data.TensorDataset(x_train,x_train_att,x_train_mask,y_train)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=False)        
    return train_loader


def give_weights(list1):
    counter=Counter(list1)
    counter_new=collections.OrderedDict(sorted(counter.items()))

    print(counter_new)
    list_weight=[]
    for key in counter_new:
        list_weight.append(len(list1)/counter[key])
    
    print(list_weight)
    return list_weight




###### Final steps to be modified ##### 
def fixed_validation_inference(args,params=None):
    n_epochs = args["epochs"] # how many times to iterate over all samples
    batch_size=args["batch_size"]
    # matrix for the out-of-fold predictions
    train_preds = np.zeros((len(args["data_train"])))
    val_preds = np.zeros((len(args["data_val"])))
    test_preds = np.zeros((len(args["data_test"])))
    ### colect training data
    train_loader=return_data_loader(args["data_train"],batch_size)
    val_loader=return_data_loader(args["data_val"],batch_size)
    test_loader=return_data_loader(args["data_test"],batch_size)
    
    #### get label encoding 
    y_train=np.array([ele[2] for ele in args["data_train"]])
    y_val=np.array([ele[2] for ele in args["data_val"]])
    y_test=np.array([ele[2] for ele in args["data_test"]])
    
    le = LabelEncoder()
    #### label encoding to transform label name into integer
    y_train = le.fit_transform(y_train)
    y_val=le.transform(y_val)
    y_test=le.transform(y_test)
    
    #### trying to find weights for different classes
    y_all=list(y_train)+list(y_val)+list(y_test)
    
    list_weights=give_weights(y_all)
    
    
    if(args["logging"]=="neptune"):
        name_one = args['model']
        neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
        neptune.append_tag(args['model'])
        neptune.append_tag('Fixed val result')
        neptune.append_tag('weighted loss')
        
    ### select_model
    model=select_model(args)
    model.cuda()
    # define cross entropy loss
    # for numerical stability in the loss
    weight = torch.tensor([list_weights]).cuda()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight,reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"])
    
    for epoch in range(n_epochs):
            # set train mode of the model. This enables operations which are only applied during training like dropout
            start_time = time.time()
            model.train()
            avg_loss = 0.  
            train_preds_fold = np.zeros((len(args["data_train"])))
            for j,(x_batch_pos,x_batch_att,x_batch_mask,y_batch) in enumerate(train_loader):
                # Forward pass: compute predicted y by passing x to the model.
                if(args["model"]!='birnnscrat'):
                    y_pred = model(x_batch_pos,x_batch_mask)
                    loss = loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1])
                else:
                    y_pred,pred_att = model(x_batch_pos,x_batch_mask)

                    attention_loss=args["att_loss"]*masked_cross_entropy(pred_att,x_batch_att.float(),x_batch_mask)
                    #attention_loss=args["att_loss"]*cr_ent/len(list_count)


                    loss = loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1])+attention_loss
                    if(debug==True):
                        print("attention_loss: ",attention_loss)
                        print("total loss: ",loss)
                        
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the Tensors it will update (which are the learnable weights
                # of the model)
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
                
                _, pred_label = torch.max(y_pred.cpu().data, 1)
                train_preds_fold[j * batch_size:(j+1) * batch_size] = pred_label
            
            # set evaluation mode of the model. This disabled operations which are only applied during training like dropout
            model.eval()

            # predict all the samples in y_val_fold batch per batch
            valid_preds_fold = np.zeros((len(args["data_val"])))
            #test_att_fold = np.zeros((len(args["data_test"])))


            avg_val_loss = 0.
            
            for k, (x_batch_pos,x_batch_att,x_batch_mask, y_batch) in enumerate(val_loader):
                
                if(args["model"]!='birnnscrat'):
                    y_pred = model(x_batch_pos,x_batch_mask).detach()
                    avg_val_loss += loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1]).item() / len(val_loader)
                else:
                    y_pred,y_att = model(x_batch_pos,x_batch_mask)
                    y_pred=y_pred.detach()
                    y_att=y_att.detach()
                    
                    avg_val_loss += (loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1]).item()
                                     +args["att_loss"]*masked_cross_entropy(y_att,x_batch_att.float(),x_batch_mask).item()) / len(val_loader)
                    
                
                _, pred_label = torch.max(y_pred.cpu().data, 1)
                
                valid_preds_fold[k * batch_size:(k+1) * batch_size] = pred_label
            f1score_val=f1_score(y_val, valid_preds_fold, average='macro')
            f1score_train=f1_score(y_train, train_preds_fold, average='macro')
            accuracy_val=accuracy_score(y_val, valid_preds_fold)
            accuracy_train=accuracy_score(y_train, train_preds_fold)
            
            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t train loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss, avg_val_loss,elapsed_time))
            
            if(args["logging"]=="neptune"):
                neptune.log_metric('epoch', epoch)
                neptune.log_metric('train_loss',avg_loss)
                neptune.log_metric('val_loss',avg_val_loss)
                neptune.log_metric('accuracy_train',accuracy_train)
                neptune.log_metric('accuracy_val',accuracy_val)
                neptune.log_metric('f1score_train',f1score_train)
                neptune.log_metric('f1score_val',f1score_val)
            else:
                print('Epoch {}/{} \t f1score_train={:.4f} \t f1score_val={:.4f}'.format(epoch + 1,n_epochs,f1score_train, f1score_val))
                print('Epoch {}/{} \t accuracy_train={:.4f} \t accuracy_val={:.4f}'.format(epoch + 1,n_epochs,accuracy_train, accuracy_val))
                print('================================================================')
            


            test_preds= np.zeros((len(args["data_test"])))
            avg_test_loss = 0.

            for k, (x_batch_pos,x_batch_att,x_batch_mask, y_batch) in enumerate(test_loader):

                if(args["model"]!='birnnscrat'):
                    y_pred = model(x_batch_pos,x_batch_mask).detach()
                    avg_val_loss += loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1]).item() / len(test_loader)
                else:
                    y_pred,y_att = model(x_batch_pos,x_batch_mask)
                    y_pred=y_pred.detach()
                    y_att=y_att.detach()

                    avg_test_loss += (loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1]).item()
                                     +args["att_loss"]*masked_cross_entropy(y_att,x_batch_att.float(),x_batch_mask).item()) / len(test_loader)


                _, pred_label = torch.max(y_pred.cpu().data, 1)

                test_preds[k * batch_size:(k+1) * batch_size] = pred_label

            f1score_test=f1_score(y_test, test_preds, average='macro')
            accuracy_test=accuracy_score(y_test, test_preds)

            elapsed_time = time.time() - start_time 
            print('test loss={:.4f} \t time={:.2f}s'.format(avg_test_loss,elapsed_time))

            if(args["logging"]=="neptune"):
                neptune.log_metric('epoch', epoch)
                neptune.log_metric('test_loss',avg_test_loss)
                neptune.log_metric('accuracy_test',accuracy_test)
                neptune.log_metric('f1score_test',f1score_test)
            else:
                print('f1score_test={:.4f} \t accuracy_test={:.4f}'.format(f1score_test,accuracy_test))
            
            
            
    if(args["save_model"]):
        model.train()
        PATH='DNN_model/'+args["model"]+'_'+args['seq_model']+'_'+str(args['att_loss'])+'.pth'
        torch.save(model.state_dict(), PATH)
        

    
        
    if(args["logging"]=="neptune"):
            neptune.stop()
#     train_preds[valid_idx] = valid_preds_fold
#     print(f"Confusion Matrix \n{classification_report(y_val, valid_preds_fold)}")


def fixed_finetuning(args,params=None):
    n_epochs = args["epochs"] # how many times to iterate over all samples
    batch_size=args["batch_size"]
    # matrix for the out-of-fold predictions
    train_preds = np.zeros((len(args["data_train"])))
    val_preds = np.zeros((len(args["data_val"])))
    test_preds = np.zeros((len(args["data_test"])))
    ### colect training data
    train_loader=return_data_loader(args["data_train"],batch_size)
    val_loader=return_data_loader(args["data_val"],batch_size)
    test_loader=return_data_loader(args["data_test"],batch_size)
    
    #### get label encoding 
    y_train=np.array([ele[2] for ele in args["data_train"]])
    y_val=np.array([ele[2] for ele in args["data_val"]])
    y_test=np.array([ele[2] for ele in args["data_test"]])
    
    le = LabelEncoder()
    #### label encoding to transform label name into integer
    y_train = le.fit_transform(y_train)
    y_val=le.transform(y_val)
    y_test=le.transform(y_test)
    
    #### trying to find weights for different classes
    y_all=list(y_train)+list(y_val)+list(y_test)
    
    list_weights=give_weights(y_all)
    
    
    if(args["logging"]=="neptune"):
        name_one = args['model']
        neptune.create_experiment(name_one,params=params,send_hardware_metrics=False,run_monitoring_thread=False)
        neptune.append_tag(args['model'])
        neptune.append_tag('Finetune result')
        neptune.append_tag('weighted loss')
        
    ### select_model
    model=select_model(args)
    model.cuda()
    # define cross entropy loss
    # for numerical stability in the loss
    weight = torch.tensor([list_weights]).cuda()
    loss_fn = torch.nn.CrossEntropyLoss(weight=weight,reduction='mean')
    optimizer = torch.optim.Adam(model.parameters(),lr=args["lr"])
    
    max_fscore=0
    epoch_reached=0
    test_fscore=0
    for epoch in range(n_epochs):
            # set train mode of the model. This enables operations which are only applied during training like dropout
            start_time = time.time()
            model.train()
            avg_loss = 0.  
            train_preds_fold = np.zeros((len(args["data_train"])))
            for j,(x_batch_pos,x_batch_att,x_batch_mask,y_batch) in enumerate(train_loader):
                # Forward pass: compute predicted y by passing x to the model.
                if(args["model"]!='birnnscrat'):
                    y_pred = model(x_batch_pos,x_batch_mask)
                    loss = loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1])
                else:
                    y_pred,pred_att = model(x_batch_pos,x_batch_mask)

                    attention_loss=args["att_loss"]*masked_cross_entropy(pred_att,x_batch_att.float(),x_batch_mask)
                    #attention_loss=args["att_loss"]*cr_ent/len(list_count)


                    loss = loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1])+attention_loss
                    if(debug==True):
                        print("attention_loss: ",attention_loss)
                        print("total loss: ",loss)
                        
                # Before the backward pass, use the optimizer object to zero all of the
                # gradients for the Tensors it will update (which are the learnable weights
                # of the model)
                optimizer.zero_grad()

                # Backward pass: compute gradient of the loss with respect to model parameters
                loss.backward()

                optimizer.step()
                avg_loss += loss.item() / len(train_loader)
                
                _, pred_label = torch.max(y_pred.cpu().data, 1)
                train_preds_fold[j * batch_size:(j+1) * batch_size] = pred_label
            
            # set evaluation mode of the model. This disabled operations which are only applied during training like dropout
            model.eval()

            # predict all the samples in y_val_fold batch per batch
            valid_preds_fold = np.zeros((len(args["data_val"])))
            #test_att_fold = np.zeros((len(args["data_test"])))


            avg_val_loss = 0.
            
            for k, (x_batch_pos,x_batch_att,x_batch_mask, y_batch) in enumerate(val_loader):
                
                if(args["model"]!='birnnscrat'):
                    y_pred = model(x_batch_pos,x_batch_mask).detach()
                    avg_val_loss += loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1]).item() / len(val_loader)
                else:
                    y_pred,y_att = model(x_batch_pos,x_batch_mask)
                    y_pred=y_pred.detach()
                    y_att=y_att.detach()
                    
                    avg_val_loss += (loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1]).item()
                                     +args["att_loss"]*masked_cross_entropy(y_att,x_batch_att.float(),x_batch_mask).item()) / len(val_loader)
                    
                
                _, pred_label = torch.max(y_pred.cpu().data, 1)
                
                valid_preds_fold[k * batch_size:(k+1) * batch_size] = pred_label
            f1score_val=f1_score(y_val, valid_preds_fold, average='macro')
            f1score_train=f1_score(y_train, train_preds_fold, average='macro')
            accuracy_val=accuracy_score(y_val, valid_preds_fold)
            accuracy_train=accuracy_score(y_train, train_preds_fold)
            
            elapsed_time = time.time() - start_time 
            print('Epoch {}/{} \t train loss={:.4f} \t val_loss={:.4f} \t time={:.2f}s'.format(epoch + 1, n_epochs, avg_loss, avg_val_loss,elapsed_time))
            
            if(args["logging"]=="neptune"):
                neptune.log_metric('epoch', epoch)
                neptune.log_metric('train_loss',avg_loss)
                neptune.log_metric('val_loss',avg_val_loss)
                neptune.log_metric('accuracy_train',accuracy_train)
                neptune.log_metric('accuracy_val',accuracy_val)
                neptune.log_metric('f1score_train',f1score_train)
                neptune.log_metric('f1score_val',f1score_val)
            else:
                print('Epoch {}/{} \t f1score_train={:.4f} \t f1score_val={:.4f}'.format(epoch + 1,n_epochs,f1score_train, f1score_val))
                print('Epoch {}/{} \t accuracy_train={:.4f} \t accuracy_val={:.4f}'.format(epoch + 1,n_epochs,accuracy_train, accuracy_val))
                print('================================================================')
            
            test_preds= np.zeros((len(args["data_test"])))
            avg_test_loss = 0.

            for k, (x_batch_pos,x_batch_att,x_batch_mask, y_batch) in enumerate(test_loader):

                if(args["model"]!='birnnscrat'):
                    y_pred = model(x_batch_pos,x_batch_mask).detach()
                    avg_val_loss += loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1]).item() / len(test_loader)
                else:
                    y_pred,y_att = model(x_batch_pos,x_batch_mask)
                    y_pred=y_pred.detach()
                    y_att=y_att.detach()

                    avg_test_loss += (loss_fn(y_pred, y_batch.view(-1,args["num_classes"]).max(1)[1]).item()
                                     +args["att_loss"]*masked_cross_entropy(y_att,x_batch_att.float(),x_batch_mask).item()) / len(test_loader)


                _, pred_label = torch.max(y_pred.cpu().data, 1)

                test_preds[k * batch_size:(k+1) * batch_size] = pred_label

            f1score_test=f1_score(y_test, test_preds, average='macro')
            accuracy_test=accuracy_score(y_test, test_preds)

            elapsed_time = time.time() - start_time 
            print('test loss={:.4f} \t time={:.2f}s'.format(avg_test_loss,elapsed_time))

            if(args["logging"]=="neptune"):
                neptune.log_metric('epoch', epoch)
                neptune.log_metric('test_loss',avg_test_loss)
                neptune.log_metric('accuracy_test',accuracy_test)
                neptune.log_metric('f1score_test',f1score_test)
            else:
                print('f1score_test={:.4f} \t accuracy_test={:.4f}'.format(f1score_test,accuracy_test))
            if(f1score_val>max_fscore):
                max_fscore=f1score_val
                epoch_reached=epoch
                test_fscore=f1score_test
                if(args["save_model"]):
                        model.train()
                        PATH='DNN_model/'+args["model"]+'_'+args['seq_model']+'_'+str(args['att_loss'])+'.pth'
                        torch.save(model.state_dict(), PATH)

    if(args["logging"]=="neptune"):
            neptune.log_metric('max_f1score_val',max_fscore)
            neptune.log_metric('epoch_at_max',epoch_reached)
            neptune.log_metric('test_fscore_at_max',test_fscore)
            
            
            
        

    
        
    if(args["logging"]=="neptune"):
            neptune.stop()
#     train_preds[valid_idx] = valid_preds_fold
#     print(f"Confusion Matrix \n{classification_report(y_val, valid_preds_fold)}")





def select_model_inference(args):
    text=args["model"]
    if(text=="birnn"):
        model=BiRNN(args)
    elif(text=="cnngru"):
        model=CNN_GRU(args)
    else:
        model=BiAtt_RNN(args,return_att=True)
    return model



def fixed_inference(args,params=None):
    n_epochs = args["epochs"] # how many times to iterate over all samples
    batch_size=args["batch_size"]
    # matrix for the out-of-fold predictions
    train_preds = np.zeros((len(args["data_train"])))
    val_preds = np.zeros((len(args["data_val"])))
    test_preds = np.zeros((len(args["data_test"])))
    ### colect training data
    train_loader=return_data_loader(args["data_train"],batch_size)
    val_loader=return_data_loader(args["data_val"],batch_size)
    test_loader=return_data_loader(args["data_test"],batch_size)
    
    #### get label encoding 
    y_train=np.array([ele[2] for ele in args["data_train"]])
    y_val=np.array([ele[2] for ele in args["data_val"]])
    y_test=np.array([ele[2] for ele in args["data_test"]])
    
    le = LabelEncoder()
    #### label encoding to transform label name into integer
    y_train = le.fit_transform(y_train)
    y_val=le.transform(y_val)
    y_test=le.transform(y_test)
    
    y_all=list(y_train)       
    
    
    print(Counter(y_all))
    
    model = select_model_inference(args)
    if(args["save_model"]):
            print("hello loading model")
            PATH='DNN_model/'+args["model"]+'_'+args['seq_model']+'_'+str(args['att_loss'])+'.pth'
            model.load_state_dict(torch.load(PATH))
           
    
    model.cuda()
    model.eval()

       
    test_preds= np.zeros((len(args["data_test"])))
    
    avg_test_loss = 0.
    tuple_wrong=[]
    tuple_right=[]
    for k, (x_batch_pos,x_batch_att,x_batch_mask, y_batch) in enumerate(test_loader):
        
        if(args["model"] in ['birnn','cnngru']):
            y_pred = model(x_batch_pos,x_batch_mask).detach()
            y_att =[0]*len(y_pred)
        else:
            y_pred,y_att = model(x_batch_pos,x_batch_mask)
            y_pred=y_pred.detach()
            y_att=y_att.detach()
            
        
        _, pred_label = torch.max(y_pred.cpu().data, 1)
        
        for i in range(len(pred_label)):
            if(y_test[k * batch_size+i]!=pred_label[i]):
                  tuple_wrong.append((k*batch_size+i,y_att[i],x_batch_att[i].detach(),y_test[k * batch_size+i],pred_label[i]))
            if(y_test[k * batch_size+i]==pred_label[i]):
                  tuple_right.append((k*batch_size+i,y_att[i],x_batch_att[i].detach(),y_test[k * batch_size+i],pred_label[i]))
                    
        test_preds[k * batch_size:(k+1) * batch_size] = pred_label
    
    
    f1score_test=f1_score(y_test, test_preds, average='macro')
    print('f1score_test={:.4f}'.format(f1score_test))       
    return tuple_wrong,tuple_right
    
    
        
       
    
        
  














