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
      

def FedProx(model, training_sets:list, n_iter:int, testing_sets:list, mu=0, 
    file_name="test", epochs=5,tuning_epoch=1,data_split='iid'):

#     model = OriginalAVOA(problem,epoch=50,pop_size=50)
#     model.solve()
#     ''' Code for Hyper-Parameter Optimization'''
    
    loss_f=loss_classifier
    
    #Variables initialization
    K=len(training_sets) #number of clients
    print(K)
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
    for i in tqdm(range(n_iter)):

            clients_params=[]
            clients_models=[]
            clients_losses=[]
           
            clients_param=[]
            clients_model=[]
            clients_loss =[]
            for k in range(K):
                def objective_function(solution):
                    lr = solution[0]
                    momentum = solution[1]
                    decay =solution[2]
                    local_model=deepcopy(model)
                    local_optimizer=optim.SGD(local_model.parameters(),lr=lr,momentum=momentum,weight_decay=decay)

                    local_loss=local_learning(local_model,mu,local_optimizer,
                        training_sets[k],epochs,loss_f)

                        
                    clients_loss.append(local_loss)
                    list_param=1
                    #GET THE PARAMETER TENSORS OF THE MODEL
                    list_params=list(local_model.parameters())
                    list_params=[tens_param.detach() for tens_param in list_params]
                    clients_param.append(list_params)    
                    clients_model.append(deepcopy(local_model))
                    return local_loss
                    
                
                
                # problem = {
                #     "fit_func":objective_function ,
                #     "lb": [0.1,0.0000000001,0.0000000001 ],
                #     "ub": [0.1,0.000000001 ,0.000000001],
                #     "minmax": "min",
                # }

                
                
                # problem = {
                #     "fit_func":objective_function ,
                #     "lb": [0.01,0.5,0.000000001 ],
                #     "ub": [0.01,0.9 ,0.000001],
                #     "minmax": "min",
                # }
                model_hyper = OriginalAVOA(epoch=tuning_epoch,pop_size=50)

                if(data_split=='iid'):
                    model_hyper.solve(problem_iid)
                if(data_split=='non_iid'):
                    model_hyper.solve(problem_niid)

                
                
                min_loss = min(clients_loss)
                min_index = clients_loss.index(min_loss)
                
                clients_params.append(clients_param[min_index])
                clients_models.append(clients_model[min_index])
                clients_losses.append(clients_loss[min_index])
                clients_param=[]
                clients_model=[]
                clients_loss =[]                
                
                print(f"Best Learning Rate:{model_hyper.solution[0][0]},Best momentum Rate:{model_hyper.solution[0][1]},Best Decay Rate:{model_hyper.solution[0][2]}")
#                 CREATE THE NEW GLOBAL MODEL
                print(f"Lowest Loss:{model_hyper.solution[0]},Lowest:{model_hyper.solution[1]}")
                print(len(clients_loss))

            model = average_models(deepcopy(model), clients_params, 
                    weights=weights)
            models_hist.append(clients_models)

                #COMPUTE THE LOSS/ACCURACY OF THE DIFFERENT CLIENTS WITH THE NEW MODEL
            loss_hist+=[[float(loss_dataset(model, dl, loss_f).detach()) 
                    for dl in training_sets]]
            acc_hist+=[[accuracy_dataset(model, dl) for dl in testing_sets]]

            server_loss=sum([weights[i]*loss_hist[-1][i] for i in range(len(weights))])
            server_acc=sum([weights[i]*acc_hist[-1][i] for i in range(len(weights))])

            print(f'====> i: {i+1} Loss: {server_loss} Server Test Accuracy: {server_acc}')
            server_accuracy_list.append(server_acc)
            server_loss_list.append(server_loss)


            server_hist.append([tens_param.detach().cpu().numpy() 
                    for tens_param in list(model.parameters())])




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
    model_f, loss_hist_FA_iid, acc_hist_FA_iid,server_accuracy_list,server_loss_list,grads,gradients = FedProx( model, 
    train_dls, n_iter, test_dls, epochs =local_epochs,tuning_epoch=tuning_epoch,data_split=data_split)


    with open('acc-10-epoch-local-1.pickle', 'wb') as handle:
        pickle.dump(acc_hist_FA_iid, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('loss-10-epoch-local-1.pickle', 'wb') as handle:
        pickle.dump(loss_hist_FA_iid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('server_acc_hist.pickle', 'wb') as handle:
        pickle.dump(server_accuracy_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('server_loss_list.pickle', 'wb') as handle:
        pickle.dump(server_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)