import os
import numpy as np
from cv2 import cv2
from constants import TrainingConstants

class Moments:
    def __init__(self, data_dir):
        self.data_dir = data_dir # Path to data directory

        self.input_cc_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if 'cc.png' in fname]) # we save all image file)
        self.input_cc_fnames.sort()
        self.input_raw_fnames = ([data_dir + '/' + fname for fname in os.listdir(self.data_dir) if '.npz' in fname])
        self.input_raw_fnames.sort()
        
        self.input_cc_mean, self.input_cc_std = self._get_moments(self.input_cc_fnames, 'cc')
        self.input_raw_vec_mean, self.input_raw_mean, self.input_raw_vec_std, self.input_raw_std = self._get_moments(self.input_raw_fnames, 'raw')
        
        moments_path = os.path.join(data_dir, 'moments')
        np.savez(moments_path, input_cc_mean=self.input_cc_mean, input_cc_std=self.input_cc_std, \
            input_raw_mean=self.input_raw_mean, input_raw_std=self.input_raw_std, \
                input_raw_vec_mean=self.input_raw_vec_mean, input_raw_vec_std=self.input_raw_vec_std)

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
                health = data['health']
                global_cc = data['global_cc']
                
                plants_water_health = np.array(np.transpose(np.dstack((plants, water, health)), (2, 0, 1)), dtype=np.float32)
                global_cc = np.array(global_cc, dtype=np.float32)
                pwh_mean = np.divide(plants_water_health, count)
                pwh2_mean = np.divide(plants_water_health**2, count)
                g_mean = np.divide(global_cc, count)
                g2_mean = np.divide(global_cc**2, count)

                if i == 1:
                    input_mean.append(pwh_mean)
                    input_sq_mean.append(pwh2_mean)
                    vec_mean.append(g_mean)
                    vec_sq_mean.append(g2_mean)
                else:
                    input_mean = np.stack((input_mean, pwh_mean), axis=0)
                    input_sq_mean = np.stack((input_sq_mean, pwh2_mean), axis=0)
                    vec_mean = np.stack((vec_mean, g_mean), axis=0)
                    vec_sq_mean = np.stack((vec_sq_mean, g2_mean), axis=0)

                input_mean = np.sum(input_mean, axis=0)
                input_sq_mean = np.sum(input_sq_mean, axis=0)
                vec_mean = np.sum(vec_mean, axis=0)
                vec_sq_mean = np.sum(vec_sq_mean, axis=0)
            else:
                image = np.array(np.transpose(cv2.imread(input_fname), (2, 0, 1)), dtype=np.float32)
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
            input_std = (input_sq_mean - (input_mean**2)).clip(min=0)**0.5
            vec_std = (vec_sq_mean - (vec_mean**2))**0.5
            return vec_mean, input_mean, vec_std, input_std
        else:
            input_std = (input_sq_mean - (input_mean**2)).clip(min=0)**0.5
            return input_mean, input_std

if __name__ == "__main__":
    Moments('Generated_data')