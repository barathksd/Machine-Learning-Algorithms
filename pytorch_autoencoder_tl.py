# -*- coding: utf-8 -*-
"""
Created on Thu Jun  3 11:15:47 2021

@author: Lenovo
"""


import os
import sys
import random
from datetime import datetime
import time
from PIL import Image

import numpy as np
#from scipy import ndimage
import cv2
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader

import torchvision
from torchvision import transforms

def setseed(n):
    torch.manual_seed(n)
    random.seed(n)
    
def get_resize(shape,inorder='hw',outorder='wh',minsize=256,div=32):
    
    if inorder == 'hw':
        h,w = shape
    else:
        w,h = shape
        
    if h>w:
        h = (minsize*h)//w
        if div>0:
            h = div*(h//div + (h%div+div//2)//div)
        w = minsize        
    else:
        w = (minsize*w)//h
        if div>0:
            w = div*(w//div + (w%div+div//2)//div)
        h = minsize
    
    if outorder == 'hw':
        return h,w
    return w,h
        
    
class LoadDataset(Dataset):
    
    def __init__(self,xdata,ydata,transform=None):
        super()
        if len(xdata) == 0 or len(xdata) != len(ydata):
            raise IndexError()
            
        self.xdata = xdata
        self.ydata = ydata
        self.transform = transform
        self.channels = len(xdata[0])
    
    def __getitem__(self,idx):
        
        x,y = self.xdata[idx], self.ydata[idx]
        xc = []
        seed = np.random.randint(0,1024)
        for i in range(self.channels):
            xc.append(self.apply_transform(self.xdata[idx][i],seed))
        y = self.apply_transform(y,seed)
         
        x = torch.stack([xc[i].squeeze() for i in range(self.channels)])
        
        return x,y
    
    def apply_transform(self,img,seed=1):
        setseed(seed)
        return self.transform(img)
    
    def __len__(self):
        return len(self.ydata)
    
        
rgbw = [0.3,0.59,0.11]
meanw = [0.485, 0.456, 0.406]
stdw = [0.229, 0.224, 0.225]
mean = np.round(np.dot(rgbw,meanw),3)
std = np.round(np.dot(rgbw,stdw),3)        

img = Image.open(r'C:\Users\Lenovo\Pictures'+'/Emilia.jpg').convert('L')
img = img.resize(get_resize(img.size,inorder='wh'))
channels = 4
n = 3
xlist = np.tile(np.array(img)[:,:,np.newaxis],channels)
l = [[Image.fromarray(xlist[:,:,i]) for i in range(channels)] for _ in range(n)]
xlist = l
ylist = [img for i in range(n)]

transform = transforms.Compose([transforms.RandomRotation(180),
             transforms.RandomResizedCrop((224,224)),
             transforms.ToTensor(),
             transforms.Normalize((mean,),(std,))])

dataset = LoadDataset(xlist, ylist, transform=transform)

traindl = DataLoader(dataset,batch_size=2,shuffle=True)
dataiter = iter(traindl)
x,y = next(dataiter)

# ------------------------


from torchvision import models
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

device = torch.device('cpu')
model = models.resnet18(pretrained=True)

input_w = model.conv1.weight
rgbw_tensor = torch.Tensor(rgbw).view(1,3,1,1)
input_w = input_w*rgbw_tensor
input_w = 3*input_w.sum(dim=1)/channels
w = torch.stack([input_w for _ in range(channels)],dim=1)
input_w = w

model.conv1 = nn.Conv2d(channels,64,kernel_size=(7,7), stride=(2,2), padding=(3,3),bias=False)
model.conv1.weight = torch.nn.Parameter(input_w)

for param in model.parameters():
    param.requires_grad = False
param = next(model.conv1.parameters())
param.requires_grad = True


activation = {}
def get_activation(name):
    def hook(model, input, output):
        activation[name] = output.detach()
    return hook

class Binary(torch.autograd.Function):

    @staticmethod
    def forward(ctx, input):
        return F.relu(Variable(input.sign())).data

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output

class Decoder(nn.Module):
    #expansion = 1

    def __init__(self, inchannels=512, outchannels=4, stride=1):
        super(Decoder, self).__init__()
        # 14*14*256
        self.transconv1 = nn.ConvTranspose2d(inchannels, inchannels//2, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv1 = nn.Conv2d(inchannels//2, inchannels//2, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn1 = nn.BatchNorm2d(inchannels//2)
        self.relu = nn.ReLU(inplace=True)
        # 28*28*128
        self.transconv2 = nn.ConvTranspose2d(inchannels//2, inchannels//4, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv2 = nn.Conv2d(inchannels//4, inchannels//4, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn2 = nn.BatchNorm2d(inchannels//4)
        # 56*56*64
        self.transconv3 = nn.ConvTranspose2d(inchannels//4, inchannels//8, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv3 = nn.Conv2d(inchannels//8, inchannels//8, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn3 = nn.BatchNorm2d(inchannels//8)
        # 112*112*32
        self.transconv4 = nn.ConvTranspose2d(inchannels//8, inchannels//16, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv4 = nn.Conv2d(inchannels//16, inchannels//16, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn4 = nn.BatchNorm2d(inchannels//16)
        # 224*224*4
        self.transconv5 = nn.ConvTranspose2d(inchannels//16, outchannels, kernel_size=3, stride=2, padding=1, output_padding=1)
        self.conv5 = nn.Conv2d(outchannels, outchannels, kernel_size=(3, 3), stride=stride, padding=(1, 1), bias=False)
        self.bn5 = nn.BatchNorm2d(outchannels)
        self.stride = stride

    def forward(self, x):
        x = self.transconv1(x)
        out = self.conv1(x)
        out = self.bn1(out)
        out += x
        out = self.relu(out)
        
        x = self.transconv2(out)
        out = self.conv2(x)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        
        x = self.transconv3(out)
        out = self.conv3(x)
        out = self.bn3(out)
        out += x
        out = self.relu(out)
        
        x = self.transconv4(out)
        out = self.conv4(x)
        out = self.bn4(out)
        out += x
        out = self.relu(out)
        
        x = self.transconv5(out)
        out = self.conv5(x)
        out = self.bn5(out)

        return out

class Autoencoder(nn.Module):
	def __init__(self,encoder,Binary,decoder):
		super(Autoencoder,self).__init__()
		self.encoder = encoder
		self.binary = Binary()
		self.decoder = decoder

	def forward(self,x):
		#x=Encoder(x)
		x = self.encoder(x)
		x = binary.apply(x)
		#print x
		#x,i2,i1 = self.binary(x)
		#x=Variable(x)
		x = self.decoder(x)
		return x

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ready to test
encoder = nn.Sequential(*(list(model.children())[:-2]))
decoder = Decoder(inchannels=512,outchannels=4)
autoencoder = Autoencoder(encoder,Binary,decoder)
autoencoder.to(device)
x = torch.ones((1,4,224,224))
traindl = DataLoader(dataset,batch_size=2,shuffle=True)
dataiter = iter(traindl)
x,y = next(dataiter)
x = x.to(device)
y = y.to(device)
x.to(device)
autoencoder.eval()
pred = autoencoder(x)


# Parameters of newly constructed modules have requires_grad=True by default
num_ftrs = model.fc.in_features
out_features = 2
model.fc = nn.Linear(num_ftrs, 2)




