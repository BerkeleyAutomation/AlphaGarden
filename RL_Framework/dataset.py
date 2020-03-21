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
        self.input_action_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if 'action.npy' in fname])
        self.input_action_fnames.sort()
        self.input_raw_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if '.npz' in fname])
        self.input_raw_fnames.sort()
        self.input_cc_mean, self.input_cc_std = self._get_moments(self.input_cc_fnames, 'cc')
        self.input_raw_mean, self.input_raw_std = self._get_moments(self.input_raw_fnames, 'raw')

    def __len__(self):
        return len(self.input_cc_fnames)

    def _get_moments(self, input_fnames, data_type):
        count = float(len(input_fnames))
        i = 0
        input_mean = []
        input_sq_mean = []
        vec_mean = []
        vec_sq_mean = []
        for input_fname in input_fnames:
            i += 1
            if data_type == 'raw':
                data = np.load(input_fname)
                plants = data['plants']
                water = data['water']
                global_cc = data['global_cc']
                
                plants_and_water = np.array(np.transpose(np.dstack((plants, water)), (2, 0, 1)), dtype=np.float64)
                global_cc = np.array(global_cc, dtype=np.float64)
                pw_mean = np.divide(plants_and_water, count)
                pw2_mean = np.divide(plants_and_water**2, count)
                g_mean = np.divide(global_cc, count)
                g2_mean = np.divide(global_cc**2, count)

                if i == 1:
                    input_mean.append(pw_mean)
                    input_sq_mean.append(pw2_mean)
                    vec_mean.append(g_mean)
                    vec_sq_mean.append(g2_mean)
                else:
                    input_mean = np.stack((input_mean, pw_mean), axis=0)
                    input_sq_mean = np.stack((input_sq_mean, pw2_mean), axis=0)
                    vec_mean = np.stack((vec_mean, g_mean), axis=0)
                    vec_sq_mean = np.stack((vec_sq_mean, g2_mean), axis=0)

                input_mean = np.sum(input_mean, axis=0)
                input_sq_mean = np.sum(input_sq_mean, axis=0)
                vec_mean = np.sum(vec_mean, axis=0)
                vec_sq_mean = np.sum(vec_sq_mean, axis=0)
            else:
                image = np.array(np.transpose(cv2.imread(input_fname), (2, 0, 1)), dtype=np.float64)
                image_mean = np.divide(image, count)
                image2_mean = np.divide(image**2, count)
                if i == 1:
                    input_mean.append(image_mean)
                    input_sq_mean.append(image2_mean)
                else:
                    input_mean = np.stack((input_mean, image_mean), axis=0)
                    input_sq_mean = np.stack((input_sq_mean, image2_mean), axis=0)
                input_mean = np.sum(input_mean, axis=0)
                input_sq_mean = np.sum(input_sq_mean, axis=0)

        if data_type == 'raw':
            input_std = (input_sq_mean - (input_mean**2) + 1e-10)**0.5
            vec_std = (vec_sq_mean - (vec_mean**2) + 1e-10)**0.5
            return (vec_mean, input_mean), (vec_std, input_std)
        else:
            input_std = (input_sq_mean - (input_mean**2))**0.5
            return input_mean, input_std

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
