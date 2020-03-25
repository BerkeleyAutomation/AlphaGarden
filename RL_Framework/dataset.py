import os
import numpy as np
from cv2 import cv2
from torch.utils.data.dataset import Dataset as TorchDataset
from constants import TrainingConstants

class Dataset(TorchDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir # Path to data directory

        self.input_cc_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if 'cc.png' in fname]) # we save all image file) 
        
        moments = np.load(data_dir + '/moments.npz')
        self.input_cc_mean, self.input_cc_std = moments['input_cc_mean'].astype(np.float32), moments['input_cc_std'].astype(np.float32)
        self.input_raw_mean, self.input_raw_std = (moments['input_raw_vec_mean'].astype(np.float32), \
            moments['input_raw_mean'].astype(np.float32)), (moments['input_raw_vec_std'].astype(np.float32), \
                moments['input_raw_std'].astype(np.float32))

    def __len__(self):
        return len(self.input_cc_fnames)

    def __getitem__(self, idx):
        input_cc_fname = self.input_cc_fnames[idx] # this is how we index the dataset
        sector_img = np.transpose(cv2.imread(input_cc_fname), (2, 0, 1)).astype(np.float32)
        tag = input_cc_fname[:input_cc_fname.rfind('_')] # this should extract only the hash from the cc file name
        input_raw_fname = '{}.npz'.format(tag) # find the raw data that corresponds with the cc by hash
        output = '{}_action.npy'.format(tag) # do the same thing for the output
        action = np.load(output)
        state = np.load(input_raw_fname)
        wph = np.transpose(np.dstack((state['plants'], state['water'], state['health'])), (2, 0, 1)).astype(np.float32)
        return (sector_img, wph, state['global_cc'].astype(np.float32), action)
