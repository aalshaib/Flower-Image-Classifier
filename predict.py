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
import json

parser = argparse.ArgumentParser()

parser.add_argument('path_img', default='/home/workspace/ImageClassifier/flowers/test/1/image_06760.jpg', nargs='*', action="store", type=str)
parser.add_argument('--cat_to_name', dest = "cat_to_name", action="store",default='/home/workspace/ImageClassifier/cat_to_name.json')
parser.add_argument('checkpoint', default='checkpoint.pth', nargs='*', action="store", type=str)
parser.add_argument('--top_k', default=5, dest="top_k", action="store", type=int)
parser.add_argument('data_dir', nargs='*', action="store", default="./flowers/")
parser.add_argument('--gpu',dest = "gpu", action="store", default="gpu")


par = parser.parse_args()
flower_data = par.data_dir
path_img = par.path_img
checkpoint = par.checkpoint
num_outputs= par.top_k
gpu = par.gpu
cat_to_name = par.cat_to_name


training_loader, validation_loader, test_loader,_ = flower.load_data(flower_data)

model= flower.load_checkpoint(checkpoint)

with open(cat_to_name, 'r') as json_file:
    cat_to_name = json.load(json_file)
    
prob, classes = flower.predict(path_img,model, num_outputs,gpu)

flower.test_data(test_loader,model,gpu)

index = 1
prob_val = np.array(prob[0][0])
    

name_val = [cat_to_name[index] for index in classes]
list = []
for i in name_val:
    list.append(i)

print("")
for i in range(num_outputs):
    print("{} with a probability of: {}".format(list[i],np.array(prob[0][i])))
   





