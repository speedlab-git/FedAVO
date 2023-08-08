from numpy import random
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from copy import deepcopy
import numpy as np
import matplotlib.pyplot as plt
from models.cnn import *

def ConvLayer(inp, out, ks=3, s=1, p=1):
  return nn.Conv2d(inp, out, kernel_size=ks, stride=s, padding=p)
class Model(nn.Module):
  def __init__(self):
    super().__init__()
    self.neural_net = nn.Sequential(
        ConvLayer(3, 32), nn.ReLU(), 
        ConvLayer(32, 64), nn.ReLU(),
        nn.MaxPool2d(2, 2), # 16
        ConvLayer(64, 128), nn.ReLU(), 
        ConvLayer(128, 256), nn.ReLU(), 
        nn.MaxPool2d(2, 2), # 8
        ConvLayer(256, 512), nn.ReLU(), 
        ConvLayer(512, 1024), nn.ReLU(), 
        nn.MaxPool2d(2, 2), # 4
        nn.Flatten(), 
        nn.Linear(1024*4*4, 512), nn.ReLU(), 
        nn.Linear(512, 128), nn.ReLU(), 
        nn.Linear(128, 47)
    )
  def forward(self, x):
    return self.neural_net(x)
    

# model= nn.DataParallel(model)