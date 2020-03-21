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
        self.input_cc_mean, self.input_cc_std = moments['input_cc_mean'], moments['input_cc_std']
        self.input_raw_mean, self.input_raw_std = (moments['input_raw_vec_mean'], moments['input_raw_mean']), (moments['input_raw_vec_std'], moments['input_raw_std'])

    def __len__(self):
        return len(self.input_cc_fnames)

    def __getitem__(self, idx):
        input_cc_fname = self.input_cc_fnames[idx] # this is how we index the dataset
        sector_img = np.transpose(cv2.imread(input_cc_fname), (2, 0, 1))
        tag = input_cc_fname[:input_cc_fname.rfind('_')] # this should extract only the hash from the cc file name
        input_raw_fname = '{}.npz'.format(tag) # find the raw data that corresponds with the cc by hash
        output = '{}_action.npy'.format(tag) # do the same thing for the output
        action = np.load(output)
        state = np.load(input_raw_fname)
        
        max_z = max(TrainingConstants.CC_IMG_DIMS[0], TrainingConstants.RAW_DIMS[0], TrainingConstants.GLOBAL_CC_DIMS[0])
        max_x = max(TrainingConstants.CC_IMG_DIMS[1], TrainingConstants.RAW_DIMS[1], TrainingConstants.GLOBAL_CC_DIMS[1])
        max_y = max(TrainingConstants.CC_IMG_DIMS[2], TrainingConstants.RAW_DIMS[2], TrainingConstants.GLOBAL_CC_DIMS[2])

        sector_img = np.pad(sector_img, ((0, max_z-TrainingConstants.CC_IMG_DIMS[0]), (0, max_x-TrainingConstants.CC_IMG_DIMS[1]), (0, max_y-TrainingConstants.CC_IMG_DIMS[2])), 'constant')
        raw = np.pad(np.transpose(np.dstack((state['plants'], state['water'])), (2, 0, 1)), ((0, max_z-TrainingConstants.RAW_DIMS[0]), (0, max_x-TrainingConstants.RAW_DIMS[1]), (0, max_y-TrainingConstants.RAW_DIMS[2])), 'constant')
        global_cc = np.pad(state['global_cc'].reshape(TrainingConstants.GLOBAL_CC_DIMS), ((0, max_z-TrainingConstants.GLOBAL_CC_DIMS[0]), (0, max_x-TrainingConstants.GLOBAL_CC_DIMS[1]), (0, max_y-TrainingConstants.GLOBAL_CC_DIMS[2])), 'constant')

        return (np.dstack((sector_img, raw, global_cc)), action)
