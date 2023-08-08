import torch
from torchvision import datasets
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np



def non_iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle, shuffle_digits=False):
    assert(nb_nodes>0 and nb_nodes<=10)

    digits=torch.arange(10) if shuffle_digits==False else torch.randperm(10, generator=torch.Generator().manual_seed(0))
    digits2=torch.arange(3) if shuffle_digits==False else torch.randperm(10, generator=torch.Generator().manual_seed(0))

    digits = torch.cat((digits,digits2))

    # split the digits in a fair way
    digits_split=list()
    i=0
    for n in range(nb_nodes, 0, -1):
        inc=int((10-i)/n)
#         print(i)
#         print(n)
        digits_split.append(digits[i:i+3])
        i+=inc
 

    # load and shuffle nb_nodes*n_samples_per_node from the dataset
    loader = torch.utils.data.DataLoader(dataset,
                                        batch_size=nb_nodes*n_samples_per_node,
                                        shuffle=shuffle)
    dataiter = iter(loader)
    images_train_mnist, labels_train_mnist = next(dataiter)

    data_splitted=list()
    for i in range(nb_nodes):
        idx=torch.stack([y_ == labels_train_mnist for y_ in digits_split[i]]).sum(0).bool() # get indices for the digits
        data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(images_train_mnist[idx], labels_train_mnist[idx]), batch_size=batch_size, shuffle=shuffle))
    return data_splitted


def iid_split(dataset, nb_nodes, n_samples_per_node, batch_size, shuffle):
        # load and shuffle n_samples_per_node from the dataset

        loader = torch.utils.data.DataLoader(dataset,
                                            batch_size=n_samples_per_node,
                                            shuffle=shuffle)
        dataiter = iter(loader)
        X,Y= next(iter(loader))
        data_splitted=list()
        for i in range(nb_nodes):
            data_splitted.append(torch.utils.data.DataLoader(torch.utils.data.TensorDataset(*(next(dataiter))), batch_size=batch_size, shuffle=shuffle,))

        

        return data_splitted