import os
import numpy as np
from cv2 import cv2
from torch.utils.data.dataset import Dataset as TorchDataset
from constants import TrainingConstants

class Dataset(TorchDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir # Path to data directory

        self.input_cc_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if 'cc.png' in fname]) # we save all image file)
        self.input_cc_fnames.sort()
        self.input_action_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if 'action.np' in fname])
        self.input_action_fnames.sort()
        self.input_raw_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if '.npz' in fname])
        self.input_raw_fnames.sort()
        self.input_cc_mean, self.input_cc_std = self._get_moments(self.input_cc_fnames, 'cc')
        self.input_raw_mean, self.input_raw_std = self._get_moments(self.input_raw_fnames, 'raw')

    def __len__(self):
        return len(self.input_cc_fnames)

    def _get_moments(self, input_fnames, data_type):
        inputs = []
        vec_inputs = []
        for input_fname in input_fnames:
            if data_type == 'raw':
                data = np.load(input_fname)
                plants = data['plants']
                water = data['water']
                global_cc = data['global_cc']
                # inputs.extend([plants[:,:,i] for i in range(plants.shape[2])])
                # inputs.append(water[:,:,0])
                inputs.append(np.transpose(np.dstack((plants, water)), (2, 0, 1)))
                vec_inputs.append(global_cc)
            else:
                image = np.transpose(cv2.imread(input_fname), (2, 0, 1))
                inputs.append(image)

        inputs = np.array(inputs)
        vec_inputs = np.array(vec_inputs)
        if data_type == 'raw':
            return (np.mean(vec_inputs, axis=0), np.mean(inputs, axis=0)), \
                (np.std(vec_inputs + 1e-10, axis=0), np.std(inputs + 1e-10, axis=0))
        else:
            return np.mean(inputs, axis=0), np.std(inputs + 1e-10, axis=0)

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