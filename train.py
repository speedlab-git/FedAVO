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



      










def train (dataset,batch_size,data_split,
n_samples_train,n_samples_test,
optimizer,comm_rounds,local_epochs,
lr,num_clients,tuning_epoch): 
   
    print(batch_size)
    if(dataset == "mnist"):
        train_dls, test_dls = get_MNIST(data_split,
        n_samples_train=n_samples_train, n_samples_test=n_samples_test, n_clients =num_clients, 
        batch_size =batch_size, shuffle =True)

        model = MNISTCNN().to(device)

    elif(dataset == "cifar10"):
        train_dls, test_dls = get_CIFAR(data_split,
        n_samples_train=n_samples_train, n_samples_test=n_samples_test, n_clients =num_clients, 
        batch_size =batch_size, shuffle =True)  

        model = CIFARCNN().to(device)     

    elif(dataset=="fmnist"):
        train_dls, test_dls = get_FASHION(data_split,
       n_samples_train=n_samples_train, n_samples_test=n_samples_test, n_clients =num_clients, 
        batch_size =batch_size, shuffle =True)

        model = FMNISTCNN().to(device)
    else:
        train_dls, test_dls = get_LISA(data_split,
        n_samples_train=n_samples_train, n_samples_test=n_samples_test, n_clients =num_clients, 
        batch_size =batch_size, shuffle =True) 

        model = LISACNN().to(device)
          

    n_iter = comm_rounds

    if(optimizer=="fedavo"):

        model_f, loss_hist_FA_iid, acc_hist_FA_iid,server_accuracy_list,server_loss_list,grads,gradients = FedAVO( model, 
    train_dls, n_iter, test_dls, epochs =local_epochs,tuning_epoch=tuning_epoch,data_split=data_split)
    else:
        model_f, loss_hist_FA_iid, acc_hist_FA_iid,server_accuracy_list,server_loss_list,grads,gradients = FedAVG( model, 
    train_dls, n_iter, test_dls, epochs =local_epochs)
    
    
    
    with open('client_accuracy.pickle', 'wb') as handle:
        pickle.dump(acc_hist_FA_iid, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('client_loss.pickle', 'wb') as handle:
        pickle.dump(loss_hist_FA_iid, handle, protocol=pickle.HIGHEST_PROTOCOL)

    with open('server_accuracy.pickle', 'wb') as handle:
        pickle.dump(server_accuracy_list, handle, protocol=pickle.HIGHEST_PROTOCOL)
    
    with open('server_loss.pickle', 'wb') as handle:
        pickle.dump(server_loss_list, handle, protocol=pickle.HIGHEST_PROTOCOL)