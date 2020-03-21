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
        count = 0
        input_sums = []
        input_sq_sums = []
        vec_sums = []
        vec_sq_sums = []
        for input_fname in input_fnames:
            count += 1
            if data_type == 'raw':
                data = np.load(input_fname)
                plants = data['plants']
                water = data['water']
                global_cc = data['global_cc']
                
                plants_and_water = np.array(np.transpose(np.dstack((plants, water)), (2, 0, 1)), dtype=np.float32)
                global_cc = np.array(global_cc, dtype=np.float32)

                if count == 1:
                    input_sums.append(plants_and_water)
                    input_sq_sums.append(plants_and_water**2)
                    vec_sums.append(global_cc)
                    vec_sq_sums.append(global_cc**2)
                else:
                    input_sums = np.stack((input_sums, plants_and_water), axis=0)
                    input_sq_sums = np.stack((input_sq_sums, plants_and_water**2), axis=0)
                    vec_sums = np.stack((vec_sums, global_cc), axis=0)
                    vec_sq_sums = np.stack((vec_sq_sums, global_cc**2), axis=0)

                input_sums = np.sum(input_sums, axis=0)
                input_sq_sums = np.sum(input_sq_sums, axis=0)
                vec_sums = np.sum(vec_sums, axis=0)
                vec_sq_sums = np.sum(vec_sq_sums, axis=0)
            else:
                image = np.array(np.transpose(cv2.imread(input_fname), (2, 0, 1)), dtype=np.float32)
                if count == 1:
                    input_sums.append(image)
                    input_sq_sums.append(image**2)
                else:
                    input_sums = np.stack((input_sums, image), axis=0)
                    input_sq_sums = np.stack((input_sq_sums, image**2), axis=0)
                input_sums = np.sum(input_sums, axis=0)
                input_sq_sums = np.sum(input_sq_sums, axis=0)

        if data_type == 'raw':
            input_mean = np.divide(input_sums, float(count))
            vec_mean = np.divide(vec_sums, float(count))
            input_sq_mean = np.divide(input_sq_sums, float(count))
            vec_sq_mean = np.divide(vec_sq_sums, float(count))
            input_std = (input_sq_mean - (input_mean**2) + 1e-10)**0.5
            vec_std = (vec_sq_mean - (vec_mean**2) + 1e-10)**0.5
            return (vec_mean, input_mean), (vec_std, input_std)
        else:
            input_mean = np.divide(input_sums, float(count))
            input_sq_mean = np.divide(input_sq_sums, float(count))
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
