from data.data import *
import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

from data.clients import *


def  get_LISA(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=4, shuffle=True):
        transform = transforms.Compose(
        [    transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize((0.4563, 0.4076, 0.3895), (0.2298, 0.2144, 0.2259))])
        
        dataset_loaded_train = LISA(root='./data', download=True, train=True,transform=transform)
        dataset_loaded_test = LISA(root='./data', download=True, train=False,transform=transform)

        if type=="iid":
            train=iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
            test=iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)
        elif type=="non_iid":
            train=non_iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
            test=non_iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)
        else:
            train=[]
            test=[]

        return train, test

def  get_CIFAR(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=25, shuffle=True):
        dataset_loaded_train = datasets.CIFAR10(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor()
    )
        dataset_loaded_test = datasets.CIFAR10(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor()
    )
        if type=="iid":
            train=iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
            test=iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)
        elif type=="non_iid":
            train=non_iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
            test=non_iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)
        else:
            train=[]
            test=[]

        return train, test
    
def  get_FASHION(type="iid", n_samples_train=200, n_samples_test=100, n_clients=3, batch_size=25, shuffle=True):
        dataset_loaded_train = datasets.FashionMNIST(
            root="./data",
            train=True,
            download=True,
            transform=transforms.ToTensor()
    )
        dataset_loaded_test = datasets.FashionMNIST(
            root="./data",
            train=False,
            download=True,
            transform=transforms.ToTensor()
    )
        if type=="iid":
            train=iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
            test=iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)
        elif type=="non_iid":
            train=non_iid_split(dataset_loaded_train, n_clients, n_samples_train, batch_size, shuffle)
            test=non_iid_split(dataset_loaded_test, n_clients, n_samples_test, batch_size, shuffle)
        else:
            train=[]
            test=[]

        return train, test