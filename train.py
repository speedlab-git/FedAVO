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

from loaders.loader import *
from models.cifar import *
from models.mnist import *
from models.lisa import *
from models.fmnist import *
from torch.utils.data import TensorDataset, DataLoader

from numpy import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from models.lisa import *
from utils import *
import pickle
from optimizers.fedavg import *
from optimizers.fedavo import *



if torch.cuda.is_available():
        device = torch.device("cuda")
else:
        device = torch.device("cpu")

print(style.GREEN+ "\n Runtime Device:" + str(device))


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
      










def train (dataset,batch_size,poison,data_split,
n_samples_train,n_samples_test,
optimizer,comm_rounds,local_epochs,
lr,num_clients,tuning_epoch): 
   

    if(dataset == "mnist"):
        train_dls, test_dls = get_MNIST(data_split,
        n_samples_train, n_samples_test, n_clients =num_clients, 
        batch_size =batch_size, shuffle =True)

        model = MNISTCNN().to(device)

    elif(dataset == "cifar10"):
        train_dls, test_dls = get_CIFAR(data_split,
        n_samples_train, n_samples_test=50, n_clients =num_clients, 
        batch_size =batch_size, shuffle =True)  

        model = CIFARCNN().to(device)     

    elif(dataset=="fmnist"):
        train_dls, test_dls = get_FASHION(data_split,
        n_samples_train , n_samples_test, n_clients =num_clients, 
        batch_size =batch_size, shuffle =True)

        model = FMNISTCNN().to(device)
    else:
        train_dls, test_dls = get_LISA(data_split,
        n_samples_train, n_samples_test, n_clients =num_clients, 
        batch_size =batch_size, shuffle =True) 

        model = LISACNN().to(device)
          

    n_iter = comm_rounds

    if(optimizer=="fedavo"):

        model_f, loss_hist_FA_iid, acc_hist_FA_iid,server_accuracy_list,server_loss_list,grads,gradients = FedAVO(model, training_sets, n_iter, testing_sets)( model, 
    train_dls, n_iter, test_dls, epochs =local_epochs,tuning_epoch=tuning_epoch,data_split=data_split)
    else:
        model_f, loss_hist_FA_iid, acc_hist_FA_iid,server_accuracy_list,server_loss_list,grads,gradients = FedAVG(model, training_sets, n_iter, testing_sets)(model, training_sets, n_iter, testing_sets)( model, 
    train_dls, n_iter, test_dls, epochs =local_epochs)
    
    
    
    with open('client_accuracy.pickle', 'wb') as handle:
        pickle.dump(acc_hist_FA_iid, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('client_loss.pickle', 'wb') as handle:
        pickle.dump(loss_hist_FA_iid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('server_accuracy.pickle', 'wb') as handle:
        pickle.dump(server_accuracy_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('server_loss.pickle', 'wb') as handle:
        pickle.dump(server_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)