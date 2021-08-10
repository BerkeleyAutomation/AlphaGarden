import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

import torch
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset
from torchsummary import summary
from torch.autograd import Function
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms.functional as TF

from torchvision import transforms
from PIL import Image
import pickle
from tqdm.notebook import tqdm
import random
from sklearn import metrics
from skimage import io, filters
import joblib
import json


class GradientReversalFn(Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        
        return x

    @staticmethod
    def backward(ctx, grad_output):
        output =  - ctx.alpha * grad_output

        return output, None


class Downsample(nn.Module):
    def __init__(self):
        super(Downsample,self).__init__()
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv1 = nn.Conv2d(3,64,kernel_size=3,padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        self.conv3 = nn.Conv2d(64,128,kernel_size=3,padding=1)
        self.bn3 = nn.BatchNorm2d(128)
        self.conv4 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.bn4 = nn.BatchNorm2d(128)
        
        self.conv5 = nn.Conv2d(128,256,kernel_size=3,padding=1)
        self.bn5 = nn.BatchNorm2d(256)
        self.conv6 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn6 = nn.BatchNorm2d(256)
        
        self.conv7 = nn.Conv2d(256,512,kernel_size=3,padding=1)
        self.bn7 = nn.BatchNorm2d(512)
        self.conv8 = nn.Conv2d(512,512,kernel_size=3,padding=1)
        self.bn8 = nn.BatchNorm2d(512)
        self.relu = nn.ReLU()
        
    
    def forward(self,x):
        x1 = self.relu(self.bn1(self.conv1(x)))
        x2 = self.relu(self.bn2(self.conv2(x1)))
        
        x3 = self.pool1(x2)
        x4 = self.relu(self.bn3(self.conv3(x3)))
        x5 = self.relu(self.bn4(self.conv4(x4)))
        
        x6 = self.pool1(x5)
        x7 = self.relu(self.bn5(self.conv5(x6)))
        x8 = self.relu(self.bn6(self.conv6(x7)))
        
        x9 = self.pool1(x8)
        x10 = self.relu(self.bn7(self.conv7(x9)))
        x11 = self.relu(self.bn8(self.conv8(x10)))
        
        return x2,x5,x8,x11
    

class Upsample(nn.Module):
    def __init__(self):
        super(Upsample,self).__init__()
        
        self.deconv2 = nn.ConvTranspose2d(512,256,kernel_size=2,stride=2)
        self.conv13 = nn.Conv2d(512,256,kernel_size=3,padding=1)
        self.bn13 = nn.BatchNorm2d(256)
        self.conv14 = nn.Conv2d(256,256,kernel_size=3,padding=1)
        self.bn14 = nn.BatchNorm2d(256)
        
        self.deconv3 = nn.ConvTranspose2d(256,128,kernel_size=2,stride=2)
        self.conv15 = nn.Conv2d(256,128,kernel_size=3,padding=1)
        self.bn15 = nn.BatchNorm2d(128)
        self.conv16 = nn.Conv2d(128,128,kernel_size=3,padding=1)
        self.bn16 = nn.BatchNorm2d(128)
        
        self.deconv4 = nn.ConvTranspose2d(128,64,kernel_size=2,stride=2)
        self.conv17 = nn.Conv2d(128,64,kernel_size=3,padding=1)
        self.bn17 = nn.BatchNorm2d(64)
        self.conv18 = nn.Conv2d(64,64,kernel_size=3,padding=1)
        self.bn18 = nn.BatchNorm2d(64)
        
        self.conv19 = nn.Conv2d(64,1,kernel_size=1)
            
        self.relu = nn.ReLU()
        self.classifier = nn.Sigmoid()
        
    
    def forward(self,x,x2,x5,x8):
        
        x19 = self.deconv2(x)  
        
        slice_f = x19.size()[-1]//2
        center = x8.size()[-1]//2
        s,e = center-slice_f,center+slice_f
        x20 = torch.cat((x8[:,:,s:e,s:e],(x19)),1)
        
        x21 = self.relu(self.bn13(self.conv13(x20)))
        x22 = self.relu(self.bn14(self.conv14(x21)))
        
        x23 = self.deconv3(x22)      
        
        slice_f = x23.size()[-1]//2
        center = x5.size()[-1]//2
        s,e = center-slice_f,center+slice_f
        
        x24 = torch.cat((x5[:,:,s:e,s:e],x23),1)
        x25 = self.relu(self.bn15(self.conv15(x24)))
        x26 = self.relu(self.bn16(self.conv16(x25)))
        
        x27 = self.deconv4(x26) 
        
        slice_f = x27.size()[-1]//2
        center = x2.size()[-1]//2
        s,e = center-slice_f,center+slice_f
        
        
        x28 = torch.cat((x2[:,:,s:e,s:e],(x27)),1)
        x29 = self.relu(self.bn17(self.conv17(x28)))
        x30 = self.relu(self.bn18(self.conv18(x29)))
        x31 = self.classifier(self.conv19(x30))
        
        return x31


class Adaptation(nn.Module):
    def __init__(self):
        super(Adaptation,self).__init__()
        
        self.pool1 = nn.MaxPool2d(kernel_size=2)
        
        self.conv_ad1 = nn.Conv2d(512,512,kernel_size=3)
        self.bn_ad1 = nn.BatchNorm2d(512)
        
        self.conv_ad2 = nn.Conv2d(512,256,kernel_size=3)
        self.bn_ad2 = nn.BatchNorm2d(256)
        
        self.conv_ad3 = nn.Conv2d(256,256,kernel_size=3)
        self.bn_ad3 = nn.BatchNorm2d(256)
        
        self.conv_ad4 = nn.Conv2d(256,1024,kernel_size=3,padding=1)
        self.bn_ad4 = nn.BatchNorm2d(1024)
        
        self.conv_ad5 = nn.Conv2d(1024,1,kernel_size=1)
        self.bn_ad5 = nn.BatchNorm2d(1)
        
        self.classifier = nn.Sigmoid()
        self.relu = nn.ReLU()

    
    def forward(self,x,grl_lambda=1):
        x_ad0 = GradientReversalFn.apply(x,grl_lambda)
        x_ad1 = self.pool1(self.relu(self.bn_ad1(self.conv_ad1(x_ad0))))
        x_ad2 = self.pool1(self.relu(self.bn_ad2(self.conv_ad2(x_ad1))))
        x_ad3 = self.pool1(self.relu(self.bn_ad3(self.conv_ad3(x_ad2))))
        x_ad4 = self.pool1(self.relu(self.bn_ad4(self.conv_ad4(x_ad3))))
        x_ad5 = self.classifier((self.conv_ad5(x_ad4)))
        
        return x_ad5


class CountEstimate(nn.Module):
    def __init__(self):
        super(CountEstimate,self).__init__()
        
        self.downsample = Downsample()
        self.upsample = Upsample()
        self.adapt = Adaptation()
        
    def forward(self,x,grl_lambda=1):
        x2,x5,x8,x11 = self.downsample(x)
        x2 = self.upsample(x11,x2,x5,x8)
        x3 = self.adapt(x11)
        
        return x2,x3
       