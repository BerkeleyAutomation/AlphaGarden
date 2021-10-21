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

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob


def get_data_from_pickle_dict(pkl_path):
    with open(pkl_path,'rb') as fp:
        d = pickle.load(fp)
    
    return d

def get_keys_from_pickle_dict(pkl_path):
    with open(pkl_path,'rb') as fp:
        d = pickle.load(fp)
    
    return list(d.keys())


class DataFromFolder(Dataset):
    def __init__(self,source_data_path,source_data_pkl_lbl,target_data_path,transform=None):
        self.source_data_path = source_data_path
        self.source_data_pkl_lbl = source_data_pkl_lbl
        self.target_data_path = target_data_path
        self.transform = transform
        
        self.source_data_lbl = get_data_from_pickle_dict(self.source_data_pkl_lbl)

        source_label = list(np.ones((len(self.source_data_path))))
        target_label = list(np.zeros((len(self.target_data_path))))
        
        source_ds_lbl = list(zip(self.source_data_path,source_label))
        target_ds_lbl = list(zip(self.target_data_path,target_label))
        
        self.merged_data = source_ds_lbl + target_ds_lbl
        random.shuffle(self.merged_data)      

        
    def __len__(self):
        return len(self.merged_data)
    
    def __getitem__(self,index):
        cur_data_id,is_source = self.merged_data[index]
        
        cur_key = cur_data_id.split('/')[-1][:-4]
        if is_source:
            x,y = Image.open(cur_data_id),self.source_data_lbl[cur_key]
        else:
            x,y = Image.open(cur_data_id), np.zeros((256,256))
        if self.transform != None:
            x = self.transform(x)

        return x,torch.Tensor(y).unsqueeze(0),is_source


class DataFromPickle(Dataset):
    def __init__(self,source_data_pkl,target_data_pkl,transform=None,source_ds=None,target_ds=None):
        self.source_data_pkl = source_data_pkl
        self.target_data_pkl = target_data_pkl
        self.transform = transform
        self.source_ds = source_ds
        self.target_ds = target_ds
        
        
        self.source_data = get_data_from_pickle_dict(self.source_data_pkl)
        self.target_data = get_data_from_pickle_dict(self.target_data_pkl) 
        
        if self.source_ds == None:
            self.source_ds = list(self.source_data.keys())
        
        if self.target_ds == None:
            self.target_ds = list(self.target_data.keys())
            
        source_label = list(np.ones((len(self.source_ds))))
        target_label = list(np.zeros((len(self.target_ds))))
        
        source_ds_lbl = list(zip(self.source_ds,source_label))
        target_ds_lbl = list(zip(self.target_ds,target_label))
        
        self.merged_data = source_ds_lbl + target_ds_lbl
        random.shuffle(self.merged_data)      

        
    def __len__(self):
        return len(self.merged_data)
    
    def __getitem__(self,index):
        cur_data_id,is_source = self.merged_data[index]
        
        if is_source:
            x,y = self.source_data[cur_data_id]
        else:
            x,y = self.target_data[cur_data_id], torch.zeros((256,256))
        
        if self.transform != None:
            x = self.transform(Image.fromarray(x))
            if is_source:
                y = self.transform(Image.fromarray(y))[0]
        # print(x.shape, y.shape, is_source)
        return x,torch.Tensor(y).unsqueeze(0),is_source.astype(np.float32)



class Data_baseline(Dataset):
    def __init__(self,data_pkl,ds_order=None,transform=None,is_train=True):
        self.data_pkl = data_pkl
        self.transform = transform
        self.is_train = is_train
        
        with open(data_pkl,'rb') as fp:
            self.ds = pickle.load(fp)                     
        
        self.ds_order = ds_order if np.all(ds_order != None)  else list(self.ds.keys())
        
    def __len__(self):
        return len(self.ds_order)
    
    def __getitem__(self,index):
        cur_data = self.ds_order[index]
        
        if self.is_train:
            x,y = self.ds[cur_data]
        else:
            x = self.ds[cur_data]
        
        if self.transform != None:
            x = self.transform(Image.fromarray((x*255).astype(np.uint8)))
        if not self.is_train: return x
        
        return x,torch.Tensor(y).unsqueeze(0)