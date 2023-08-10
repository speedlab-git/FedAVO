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



def FedAVO(model, training_sets:list, n_iter:int, testing_sets:list, mu=0, 
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