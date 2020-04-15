import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from alphagarden.Learning.constants import TrainingConstants
import numpy as np
from cv2 import cv2

torch.set_default_dtype(torch.float32)

class Net(nn.Module):
    def __init__(self, input_cc_mean, input_cc_std, input_raw_mean, input_raw_std, name='alpha_net'):
        super(Net, self).__init__()
        self.name = name
        self._device = TrainingConstants.DEVICE
        
        self.downsample_rate = 0.25
        input_cc_mean = np.transpose(input_cc_mean, (1, 2, 0))
        h, w, c = input_cc_mean.shape
        img = cv2.resize(input_cc_mean, (int(w * self.downsample_rate), int(h * self.downsample_rate)))
        input_cc_mean = np.transpose(img, (2, 0, 1))

        input_cc_std = np.transpose(input_cc_std, (1, 2, 0))
        h, w, c = input_cc_std.shape
        img = cv2.resize(input_cc_std, (int(w * self.downsample_rate), int(h * self.downsample_rate)))
        input_cc_std = np.transpose(img, (2, 0, 1))

        self.input_cc_mean = input_cc_mean
        self.input_cc_std = input_cc_std
        self.input_raw_mean = input_raw_mean
        self.input_raw_std = input_raw_std

        self.cc_conv1 = nn.Conv2d(in_channels=3, out_channels=8, stride=1, kernel_size=5, padding=2)
        self.cc_bn1 = nn.BatchNorm2d(8) 
        self.cc_pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0) 
        self.cc_conv2 = nn.Conv2d(8, 16, stride=1, kernel_size=3, padding=1)
        self.cc_bn2 = nn.BatchNorm2d(16)
        self.cc_pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.cc_fc = nn.Linear(31, 4) 

        self.raw_conv1 = nn.Conv2d(in_channels=13, out_channels=16, stride=1, kernel_size=5, padding=2)
        self.raw_bn1 = nn.BatchNorm2d(16)
        self.raw_pool1 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.raw_conv2 = nn.Conv2d(16, 32, stride=1, kernel_size=3, padding=1)
        self.raw_bn2 = nn.BatchNorm2d(32)
        self.raw_pool2 = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.raw_fc = nn.Linear(7, 4)
        
        self.fc = nn.Linear(TrainingConstants.FLAT_STATE_DIM, TrainingConstants.ACT_DIM)
                
    def forward(self, x):
        cc_sector = F.interpolate(x[0], scale_factor=self.downsample_rate)
        water_plants_health = x[1]
        global_cc = x[2]

        cc_normalized = (cc_sector - torch.tensor(self.input_cc_mean, dtype=torch.float32, device=self._device)) / torch.tensor(self.input_cc_std + 1e-10, dtype=torch.float32, device=self._device)
        cc = self.cc_conv1(cc_normalized)
        cc = self.cc_bn1(cc)
        cc = F.relu(cc)
        cc = self.cc_pool1(cc)
        cc = self.cc_conv2(cc)
        cc = self.cc_bn2(cc)
        cc = F.relu(cc)
        cc = self.cc_pool2(cc)
        cc = self.cc_fc(cc)

        water_plants_health_normalized = (water_plants_health - torch.tensor(self.input_raw_mean[1], dtype=torch.float32, device=self._device)) / torch.tensor(self.input_raw_std[1] + 1e-10, dtype=torch.float32, device=self._device)
        raw = self.raw_conv1(water_plants_health_normalized)
        raw = self.raw_bn1(raw)
        raw = F.relu(raw)
        raw = self.raw_pool1(raw)
        raw = self.raw_conv2(raw)
        raw = self.raw_bn2(raw)
        raw = F.relu(raw)
        raw = self.raw_pool2(raw)
        raw = self.raw_fc(raw)
        
        cc = cc.reshape((cc.shape[0], -1))
        raw = raw.reshape((raw.shape[0], -1))
        cc_and_raw = torch.cat((cc, raw), dim=1)
        global_cc_normalized = (global_cc - torch.tensor(self.input_raw_mean[0], dtype=torch.float32, device=self._device)) / torch.tensor(self.input_raw_std[0] + 1e-10, dtype=torch.float32, device=self._device)
        global_cc_normalized = global_cc_normalized.reshape((global_cc_normalized.shape[0], -1))
        state = torch.cat((cc_and_raw, global_cc_normalized), dim=1)
        state = F.relu(state)
        # print("state.size: ", state.size())
        action = self.fc(state)
        return action

    def save(self, dir_path, net_fname, net_label):
        net_path = os.path.join(dir_path, net_label + net_fname)
        moments_path = os.path.join(dir_path, 'moments')
        torch.save(self.state_dict(), net_path)
        if net_label == 'final':
            np.savez(moments_path, input_cc_mean=self.input_cc_mean, input_cc_std=self.input_cc_std, \
                input_raw_mean=self.input_raw_mean, input_raw_std=self.input_raw_std)

    def load(self, dir_path, net_fname):
        net_path = os.path.join(dir_path, net_fname)
        moments_path = os.path.join(dir_path, 'moments')
        state_dict = torch.load(net_path)
        self.load_state_dict(state_dict)
        moments = np.load(moments_path)
        self.input_cc_mean = moments['input_cc_mean']
        self.input_cc_std = moments['input_cc_std']
        self.input_raw_mean = moments['input_raw_mean']
        self.input_raw_std = moments['input_raw_std']
