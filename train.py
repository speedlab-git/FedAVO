import torchvision
import torchvision.transforms as transforms
import torchvision.transforms as transforms
import json
import os
import torch
from torchvision.datasets import VisionDataset
from torchvision.datasets.utils import check_integrity, download_and_extract_archive

from data import *
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from loaders.LISA import *

from torch.utils.data import TensorDataset, DataLoader

from numpy import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from models.cnn import *
from utils import *
import pickle

if torch.cuda.is_available():
        device = torch.device("cuda")
else:
        device = torch.device("cpu")

print(style.GREEN+ "\n Runtime Device:" + str(device))

def augmentClientData(dataloader,batch_size):
    from PIL import Image
    import torchvision.transforms as T
    augmented_x=[]
    augmented_y=[]
    transform=transforms.Compose([
                              transforms.RandomHorizontalFlip(),
                              transforms.RandomVerticalFlip(),
                              transforms.RandomRotation((30,90)),
                              transforms.RandomCrop([28, 28]),transforms.CenterCrop(10),
                              transforms.ColorJitter(brightness=(0.5,1.5),contrast=(1),saturation=(0.5,2),hue=(-0.1,0.4)),
                              transforms.Resize([32,32]),
                          ])
    for i in range(len((dataloader))):
        batch_x, batch_y = next(iter(dataloader))
        for j in range (len(batch_x)):
            batch_x[j]= transform(batch_x[j])
            augmented_x.append(batch_x[j])
            augmented_y.append(batch_y[j])
    tensor_x = torch.stack(augmented_x) # transform to torch tensor
    tensor_y = torch.stack(augmented_y)
    augmented_dataset = TensorDataset(tensor_x,tensor_y) # create your datset
    augmented_dataloader = DataLoader(augmented_dataset,batch_size=batch_size, shuffle=True)
    return augmented_dataloader



def loss_classifier(predictions,labels):
    
#     m = nn.LogSoftmax(dim=1)
    loss = nn.CrossEntropyLoss()
    
    return loss(predictions ,labels)


def loss_dataset(model, dataset, loss_f):
    """Compute the loss of `model` on `dataset`"""
    loss=0
    
    for idx,(features,labels) in enumerate(dataset):
        features = features.to(device)
        labels=labels.to(device)
        predictions= model(features)
        loss+=loss_f(predictions,labels)
    
    loss/=idx+1
    return loss


def accuracy_dataset(model, dataset):
    """Compute the accuracy of `model` on `dataset`"""
    
    correct=0
    
    for features,labels in iter(dataset):
        features = features.to(device)
        labels=labels.to(device)
        predictions= model(features)
        
        _,predicted=predictions.max(1,keepdim=True)
        
        correct+=torch.sum(predicted.view(-1,1)==labels.view(-1, 1)).item()
        
    accuracy = 100*correct/len(dataset.dataset)
        
    return accuracy


def train_step(model, model_0, mu:int, optimizer, train_data, loss_f,k):
    """Train `model` on one epoch of `train_data`"""
    gradients = {}
    grads=[]
    total_loss=0
    
    for idx, (features,labels) in enumerate(train_data):
        
        optimizer.zero_grad()
        features = features.to(device)
        predictions= model(features)
        labels=labels.to(device)
        loss=loss_f(predictions,labels)
        loss+=mu/2*difference_models_norm_2(model,model_0)
        total_loss+=loss
        
        loss.backward()
#         for p in model.parameters():
#             print(f'====> {k}===> {p.grad.norm()}')
            
#             grads.append(p.grad.norm())
#         print(k)
        x=(list(model.parameters())[-1])
        if k == 2:
            grads.append(x)
        optimizer.step()
        
    return total_loss/(idx+1),grads



def local_learning(model, mu:float, optimizer, train_data, epochs:int, loss_f,k):
    
    model_0=deepcopy(model)
    
    for e in range(epochs):
        local_loss,grads=train_step(model,model_0,mu,optimizer,train_data,loss_f,k)
        
    return float(local_loss.detach().cpu().numpy()),grads


def difference_models_norm_2(model_1, model_2):
    """Return the norm 2 difference between the two model parameters
    """
    
    tensor_1=list(model_1.parameters())
    tensor_2=list(model_2.parameters())
    
    norm=sum([torch.sum((tensor_1[i]-tensor_2[i])**2) 
        for i in range(len(tensor_1))])
    
    return norm


def set_to_zero_model_weights(model):
    """Set all the parameters of a model to 0"""

    for layer_weigths in model.parameters():
        layer_weigths.data.sub_(layer_weigths.data)

def average_models(model, clients_models_hist:list , weights:list):

    """Creates the new model of a given iteration with the models of the other

    clients"""
    
    new_model=deepcopy(model)
    set_to_zero_model_weights(new_model)

    for k,client_hist in enumerate(clients_models_hist):
        
        for idx, layer_weights in enumerate(new_model.parameters()):

            contribution=client_hist[idx].data*weights[k]
            layer_weights.data.add_(contribution)
            
    return new_model
      

def FedProx(model, training_sets:list, n_iter:int,  testing_sets:list, mu=0, 
    file_name="test", epochs=5, lr=10**-2):
    """ all the clients are considered in this implementation of FedProx
    Parameters:
        - `model`: common structure used by the clients and the server
        - `training_sets`: list of the training sets. At each index is the 
            training set of client "index"
        - `n_iter`: number of iterations the server will run
        - `testing_set`: list of the testing sets. If [], then the testing
            accuracy is not computed
        - `mu`: regularization term for FedProx. mu=0 for FedAvg
        - `epochs`: number of epochs each client is running
        - `lr`: learning rate of the optimizer
        - `decay`: to change the learning rate at each iteration
    
    returns :
        - `model`: the final global model 
    """
        
    loss_f=loss_classifier
    
    #Variables initialization
    K=len(training_sets) #number of clients
    n_samples=sum([len(db.dataset) for db in training_sets])
    weights=([len(db.dataset)/n_samples for db in training_sets])
    print("Clients' weights:",weights)
    
    
    loss_hist=[[float(loss_dataset(model, dl, loss_f).detach()) 
        for dl in training_sets]]
    acc_hist=[[accuracy_dataset(model, dl) for dl in testing_sets]]
    server_hist=[[tens_param.detach().cpu().numpy()
        for tens_param in list(model.parameters())]]
    models_hist = []
    
    
    server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
    server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])
    print(f'====> i: 0 Loss: {server_loss} Server Test Accuracy: {server_acc}')
    server_loss_list=[]
    server_accuracy_list=[]  
    for i in range(n_iter):
        
        clients_params=[]
        clients_models=[]
        clients_losses=[]
        print(clients_losses)
        for k in range(K):
            local_model=deepcopy(model)
            local_optimizer=optim.Adam(local_model.parameters(),lr=0.0005)
            
            local_loss,grads=local_learning(local_model,mu,local_optimizer,
                training_sets[k],epochs,loss_f,k)
            
            clients_losses.append(local_loss)
                
            #GET THE PARAMETER TENSORS OF THE MODEL
            list_params=list(local_model.parameters())
            list_params=[tens_param.detach() for tens_param in list_params]
            clients_params.append(list_params)    
            clients_models.append(deepcopy(local_model))

            print(f"{k}---local_loss--- {local_loss}" )
        
        
        #CREATE THE NEW GLOBAL MODEL
        model = average_models(deepcopy(model), clients_params, 
            weights=weights)
        models_hist.append(clients_models)
#         print(clients_params)
#         print(weights)
        #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
        loss_hist+=[[float(loss_dataset(model, dl, loss_f).detach()) 
            for dl in training_sets]]
        acc_hist+=[[accuracy_dataset(model, dl) for dl in testing_sets]]

        server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
        server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])

        print(style.GREEN+f'====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}')
        server_accuracy_list.append(server_acc)
        server_loss_list.append(server_loss)        

        server_hist.append([tens_param.detach().cpu().numpy() 
            for tens_param in list(model.parameters())])
        
        #DECREASING THE LEARNING RATE AT EACH SERVER ITERATION
            
    return model, loss_hist, acc_hist,server_accuracy_list,server_loss_list




def train (batch_size,poison,data_split,optimizer,comm_rounds,local_epochs,lr,num_clients): 
   



    lisa_iid_train_dls, lisa_iid_test_dls = get_LISA(data_split,
    n_samples_train =300, n_samples_test=50, n_clients =num_clients, 
    batch_size =batch_size, shuffle =True)


    if poison>0:

        poison_idx=random.randint(num_clients-1, size=(poison))

        augs = []

        for i in range (len(poison_idx)):
            lisa_iid_train_dls[i] = augmentClientData(lisa_iid_train_dls[i],batch_size)

    model = Model().to(device)
    n_iter = comm_rounds
    model_f, loss_hist_FA_iid, acc_hist_FA_iid,server_accuracy_list,server_loss_list,grads,gradients = FedProx( model, 
    lisa_iid_train_dls, n_iter, lisa_iid_test_dls, epochs =local_epochs)


    with open('acc-10-epoch-local-1.pickle', 'wb') as handle:
        pickle.dump(acc_hist_FA_iid, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('loss-10-epoch-local-1.pickle', 'wb') as handle:
        pickle.dump(loss_hist_FA_iid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('server_acc_hist.pickle', 'wb') as handle:
        pickle.dump(server_accuracy_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('server_loss_list.pickle', 'wb') as handle:
        pickle.dump(server_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)