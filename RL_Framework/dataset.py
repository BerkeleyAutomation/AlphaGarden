import os

import numpy as np
from torch.utils.data.dataset import Dataset as TorchDataset

class Dataset(TorchDataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir

        self.input_cc_fnames = ([fname for fname in os.listdir(self.data_dir) if 'cc' in fname]) # we save all image file)
        self.input_cc_fnames.sort()
        self.input_raw_fnames = ([fname for fname in os.listdir(self.data_dir) if 'cc' not in fname])
        self.input_raw_fnames.sort()
        self.input_cc_mean, self.input_cc_std = self._get_moments(self.input_cc_fnames) # TODO: update for both cc and raw inputs
        self.input_raw_mean, self.input_raw_std = self._get_moments(self.input_raw_fnames)  # TODO: update for both cc and raw inputs

    def __len__(self):
        return len(self.input_fnames)

    def _get_moments(self, input_fnames):
        inputs = []
        for input_fname in input_fnames:
        # TODO: read the entire dataset and calculate the mean and std over it. the resulting mean/std outputs should
        #       of the same dimensions as the input. For example if your dataset is made of (1,3) float arrays, the mean
        #       will be a (1,3) float array that holds the mean per cell

            with open(os.path.join(self.data_dir, input_fname), 'r') as fhandle:
        #         start_pos = np.asarray([float(e) for e in fhandle.readline()[:-1].split(' ')])[np.newaxis]
        #         end_pos = np.asarray([float(e) for e in fhandle.readline()[:-1].split(' ')])[np.newaxis]
        #     start_end_pos = np.concatenate([start_pos, end_pos])
        #     inputs.append(start_end_pos)
        inputs = np.array(inputs)
        return np.mean(inputs, axis=0), np.std(inputs + 1e-10, axis=0)


    def __getitem__(self, idx):
        input_cc_fname = self.input_fnames[idx] # this is how we index the dataset
        tag = input_cc_fname[:input_cc_fname.rfind('_')] # this should extract only the hash from the cc file name
        input_raw_fname = '{}.npz'.format(tag) # find the raw data that corresponds with the cc by hash
        output = '????' # TODO: do the same thing for the output I couldn't find where you saved it

        ## TODO: read the multiple inputs to a single variable, and the single output to a different single variable and return as a tuple
        # with open(os.path.join(self.data_dir, input_fname), 'r') as fhandle:
        #     start_pos = np.asarray([float(e) for e in fhandle.readline()[:-1].split(' ')])[np.newaxis]
        #     end_pos = np.asarray([float(e) for e in fhandle.readline()[:-1].split(' ')])[np.newaxis]
        # start_end_pos = np.concatenate([start_pos, end_pos])
        #
        # with open(os.path.join(self.data_dir, traj_fname), 'r') as fhandle:
        #     rows = []
        #     for line in fhandle:
        #         row = []
        #         tab_split = line[:-1].split('\t')
        #         for i in range(1, 4):
        #             row.append([float(e) for e in tab_split[i].split(' ')])
        #         rows.append(row)
        # traj = np.asarray(rows)
        #
        # traj_new = np.zeros((61, 3, 6))
        # traj_new[:traj.shape[0]] = traj
        # traj_new[traj.shape[0]:, 0] = traj[-1:, 0]
        # traj = traj_new

        return (start_end_pos, traj)