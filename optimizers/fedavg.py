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

from training_utils import *
from tqdm import tqdm


def FedAVG(model, training_sets:list, n_iter:int,  testing_sets:list, mu=0, 
    file_name="test", epochs=5, lr=10**-2):

        
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
