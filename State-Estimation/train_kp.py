
import torch
from torch import nn,optim
from torch.utils.data import DataLoader,Dataset
from torchsummary import summary
from torch.autograd import Function
from torch.optim.lr_scheduler import StepLR
import torchvision.transforms.functional as TF
import torchvision

from torchvision import transforms
from PIL import Image
import pickle
from tqdm import tqdm
import random
from sklearn import metrics
from skimage import io, filters
import joblib
import json

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import glob

import data_loader as dl
import keypoint_model as model
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter

parser = ArgumentParser()
parser.add_argument("-s", "--source", help="source domain path")
parser.add_argument("-t", "--target",help="target domain path")
parser.add_argument("-o", "--write_weight",help="write weight to")
parser.add_argument("-e", "--epoch",help="number of epoch")
parser.add_argument("-w", "--weight",help="model weight")
parser.add_argument("-v", "--target_split",help="target train amt")


args = parser.parse_args()

SOURCE_DATA = args.source
TARGET_DATA = args.target
weight_path = args.write_weight
target_split = float(args.target_split) if 0< float(args.target_split) <= 1 else 0.81
print(target_split)
EPOCHS = int(args.epoch)
weight = args.weight


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
writer = SummaryWriter()
transf = transforms.Compose([transforms.ToTensor(), transforms.ToPILImage(), transforms.Resize((256,256)), transforms.ToTensor()])

SOURCE_TRAIN_SIZE = 0.8
TARGET_TRAIN_SIZE = target_split
TRAINING_BATCH = 8

source_data_keys = dl.get_keys_from_pickle_dict(SOURCE_DATA)
target_data_keys = dl.get_keys_from_pickle_dict(TARGET_DATA)
random.shuffle(source_data_keys)
random.shuffle(target_data_keys)

source_ds_size = len(source_data_keys)
target_ds_size = len(target_data_keys)

source_train_keys,source_valid_keys = source_data_keys[:int(source_ds_size*SOURCE_TRAIN_SIZE)],source_data_keys[int(source_ds_size*SOURCE_TRAIN_SIZE):]
target_train_keys,target_valid_keys = target_data_keys[:int(target_ds_size*TARGET_TRAIN_SIZE)],target_data_keys[int(target_ds_size*TARGET_TRAIN_SIZE):]

train_ds = dl.DataFromPickle(SOURCE_DATA,TARGET_DATA,source_ds=source_train_keys,target_ds=target_train_keys,transform=transf)
valid_ds = dl.DataFromPickle(SOURCE_DATA,TARGET_DATA,source_ds=source_valid_keys,target_ds=target_valid_keys,transform=transf)


train_ds_loader = DataLoader(train_ds,batch_size=TRAINING_BATCH,drop_last=True,shuffle=False)
valid_ds_loader = DataLoader(valid_ds,batch_size=8,drop_last=True,shuffle=False)


counter_model = model.CountEstimate()
if weight!=None:
	counter_model.load_state_dict(torch.load(weight))
counter_model.to(device)


criterion1 = nn.MSELoss(reduction='none')
criterion2 = nn.BCELoss()

learning_rate = 1e-3
optimizer = optim.Adam([
        {'params': counter_model.upsample.parameters(),'lr':1e-3},
        {'params': counter_model.downsample.parameters(),'lr':1e-3},
        {'params': counter_model.adapt.parameters(),'lr':1e-4},
    ],
    lr=learning_rate)


max_batches = len(train_ds)//TRAINING_BATCH


train_loss = []
valid_loss = []
cost_1z = []
cost_2z = []



ctr=0
for epoch in tqdm(range(EPOCHS)):
    counter_model.train()
    epoch_loss,steps = 0,0
    
    for x,y,z in train_ds_loader:
        
        x,y,z = x.to(device),y.to(device),z.to(device)
        
        optimizer.zero_grad()
        
        p = float(steps + epoch * max_batches) / (EPOCHS * max_batches)
        grl_lambda = 2. / (1. + np.exp(-10 * p)) - 1
        
        pred_density_map, is_source = counter_model(x,grl_lambda)
        
        cost1 = criterion1(pred_density_map,y).mean(axis=[-1,-2])
        cost1 = torch.dot(z,cost1.flatten())
        cost2 = criterion2(is_source.flatten(),z)
        
        cost = torch.log(cost1 + 1e-9) + cost2
        
        cost_1z.append(cost1)
        cost_2z.append(cost2)

        writer.add_scalar('Loss/cost1', torch.log(cost1 + 1e-9), epoch)
        writer.add_scalar('Loss/cost2', cost2, epoch)

        epoch_loss+=((cost1))
        steps+=1
        
        cost.backward()
        optimizer.step()

    cur_train_loss = epoch_loss/steps
    train_loss.append(cur_train_loss)

    prv_x = x
    img_grid0 = torchvision.utils.make_grid(prv_x)
    writer.add_image('input',img_grid0 ,epoch)

    pred_out,_ = counter_model(prv_x.to(device))
    pred_out = (pred_out-pred_out.min())/(pred_out.max()-pred_out.min()) 
    img_grid = torchvision.utils.make_grid(pred_out)
    writer.add_image('target prediction',img_grid,epoch)

    # if epoch == 0:
    # 	with open('imgz_for_plot.pkl','wb') as fp:
    # 		pickle.dump(x,fp)

    
    v_loss = 0 
    steps = 0
    with torch.no_grad():
        counter_model.eval()
        for x,y,z in valid_ds_loader:
            x,y,z = x.to(device), y.to(device),z.to(device)
            
            pred_density_map,is_source = counter_model(x)
            
            cost1 = criterion1(pred_density_map,y).mean(axis=[-1,-2])
            cost1 = torch.dot(z,cost1.flatten())
            
            cost = torch.log(cost1 + 1e-9) 
            
            v_loss+= cost
            steps+=1

    cur_val_loss = v_loss/steps
    valid_loss.append(cur_val_loss)

    writer.add_scalar('Loss/train', cur_train_loss, epoch)
    writer.add_scalar('Loss/valid', cur_val_loss, epoch)



torch.save(counter_model.state_dict(),weight_path)

