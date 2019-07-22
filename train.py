# Imports here

import matplotlib.pyplot as plt

import torch
from torch import nn
import numpy as np
from torch import optim
import torch.nn.functional as F
import torchvision
from PIL import Image
import time
from torchvision import datasets, transforms, models
import argparse
import flower

parser = argparse.ArgumentParser()
parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
parser.add_argument('--save_dir', dest="save_dir", action="store", default="./checkpoint.pth")
parser.add_argument('--print_every', type=int, dest="print_every", action="store", default=30)
parser.add_argument('--epochs', type=int, dest="epochs", action="store", default=15)
parser.add_argument('--gpu',dest = "gpu", action="store", default="gpu")
parser.add_argument('--learning_rate', dest="learning_rate", action="store", default=0.01)
parser.add_argument('--arch', dest="arch", action="store", default="densenet121", type=str)
parser.add_argument('--hidden_units1',dest = "hidden_units1", action="store", default=500)                
parser.add_argument('--hidden_units2',dest = "hidden_units2", action="store", default=250)
parser.add_argument('--dropout', dest = "dropout", action="store", default = 0.5)

par = parser.parse_args()
flower_data = par.data_dir
checkpoint_path = par.save_dir
print_every = par.print_every
epochs = par.epochs
gpu = par.gpu
learning_rate = par.learning_rate
arch = par.arch
hidden_units1 = par.hidden_units1
hidden_units2 = par.hidden_units2
dropout = par.dropout

               
trainloader, validationloader, testloader, image_train_data = flower.load_data(flower_data)

model, criterion, optimizer, classifier = flower.nn_setup(arch, hidden_units1, hidden_units2, learning_rate,gpu,dropout)

flower.deep_learning(model, trainloader, epochs, print_every, criterion, optimizer,validationloader, gpu)
                    
flower.save_checkpoint(checkpoint_path, image_train_data,model,hidden_units1, hidden_units2, learning_rate )






                    