import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from constants import TrainingConstants

torch.set_default_dtype(torch.float64)

class Net(nn.Module):
    def __init__(self, input_cc_mean, input_cc_std, input_raw_mean, input_raw_std, name='alpha_net'):
        super(Net, self).__init__()
        self.name = name
        self.input_cc_mean = input_cc_mean
        self.input_cc_std = input_cc_std
        self.input_raw_mean = input_raw_mean
        self.input_raw_std = input_raw_std

        self.cc_conv1 = nn.Conv2d(in_channels=3, out_channels=16, stride=1, kernel_size=5, padding=2)
        self.cc_bn1 = nn.BatchNorm2d(16) 
        self.cc_conv2 = nn.Conv2d(16, 32, stride=1, kernel_size=3, padding=1)
        self.cc_bn2 = nn.BatchNorm2d(32)
        self.cc_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.raw_conv1 = nn.Conv2d(in_channels=12, out_channels=32, stride=1, kernel_size=5, padding=2)
        self.raw_bn1 = nn.BatchNorm2d(32)
        self.raw_conv2 = nn.Conv2d(32, 64, stride=1, kernel_size=3, padding=1)
        self.raw_bn2 = nn.BatchNorm2d(64)
        self.raw_pool = torch.nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

        self.fc = nn.Linear(TrainingConstants.FLAT_STATE_DIM, TrainingConstants.ACT_DIM)

        ## the code from GLOMP for reference:
        # self.flatten = nn.Flatten()
        # self.fc1 = nn.Linear(DataConstants.INPUT_DIM, 8192)
        # self.fc2 = nn.Linear(8192, 4096)
        # self.fc3 = nn.Linear(4096, 4096)
        # self.fc4 = nn.Linear(4096, 2048)
        # self.fc5 = nn.Linear(2048, 2048)
        # self.fc6 = nn.Linear(2048, DataConstants.OUTPUT_DIM)

    def forward(self, x):
        print("FORWARD")
        cc_sector, water_and_plants, global_cc = x
        cc_normalized = (cc_sector - self.input_cc_mean) / self.input_cc_std
        cc = self.cc_conv1(cc_normalized)
        # print shape after each step
        cc = self.cc_bn1(cc)
        cc = F.relu(cc)
        cc = self.cc_conv2(cc)
        cc = self.cc_bn2(cc)
        cc = F.relu(cc)
        cc = self.cc_pool(cc)

        water_and_plants_normalized = (water_and_plants - self.input_raw_mean[1]) / self.input_raw_std[1]
        raw = self.raw_conv1(water_and_plants_normalized)
        raw = self.raw_bn1(raw)
        raw = F.relu(raw)
        raw = self.raw_conv2(raw)
        raw = self.raw_bn2(raw)
        raw = F.relu(raw)
        raw = self.raw_pool(raw)
        print("raw.size: ", raw.size())
        print("cc.size: ", cc.size())
        cc_and_raw = torch.cat((cc, raw), dim=1)
        global_cc_normalized = (global_cc - self.input_raw_mean[0]) / self.input_raw_std[0]
        state = torch.cat((cc_and_raw, global_cc_normalized), dim=1)
        state = F.relu(state)
        print("state.size: ", state.size())
        action = self.fc(state)

        ## the code from GLOMP for reference:
        # x = (x - torch.tensor(self.input_mean, dtype=torch.float32)) / torch.tensor(self.input_std, dtype=torch.float32)
        # x = self.flatten(x)
        # x = self.fc1(x)
        # x = F.relu(x)
        # x = self.fc2(x)
        # x = F.relu(x)
        # x = self.fc3(x)
        # x = F.relu(x)
        # x = self.fc4(x)
        # x = F.relu(x)
        # x = self.fc5(x)
        # x = F.relu(x)
        # x = self.fc6(x)
        # return x
        print("FORWARD-END")
        return action

    def save(self, dir_path, net_fname):
        net_path = os.path.join(dir_path, net_fname)
        moments_path = os.path.join(dir_path, 'moments.txt') ## TODO: save all moments you used
        torch.save(self.state_dict(), net_path)
        with open(moments_path, 'w+') as mfile:
            mfile.write(str(self.input_mean) + ', ' + str(self.input_std))

    def load(self, dir_path, net_fname):
        net_path = os.path.join(dir_path, net_fname)
        moments_path = os.path.join(dir_path, 'moments.txt')
        state_dict = torch.load(net_path)
        self.load_state_dict(state_dict)
        with open(moments_path) as mfile:
            moments = mfile.readlines()
            self.input_cc_mean = [float(val) for val in moments[0].split()[1:-1]] ## TODO: parse all moments
            self.input_cc_std = [float(val) for val in moments[1].split()[1:-1]]
            self.input_raw_mean = [float(val) for val in moments[0].split()[1:-1]]
            self.input_raw_std = [float(val) for val in moments[1].split()[1:-1]]
