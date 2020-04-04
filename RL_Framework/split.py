import os
import numpy as np

class Split:
    def __init__(self, data_dir, id):
        self.data_dir = data_dir # Path to data directory

        self.input_action_fnames = [data_dir + '/' + fname for fname in os.listdir(self.data_dir) if '_action.npy' in fname]
        
        self.zero, self.one, self.two, self.three = self.split(self.input_action_fnames)
        
        moments_path = os.path.join(data_dir, '../split_' + str(id))
        print(moments_path)
        np.savez(moments_path, zero=self.zero, one=self.one, two=self.two, three=self.three)

    def split(self, input_fnames):
        zero = []
        one = []
        two = []
        three = []
        for input_fname in input_fnames:
            if np.load(input_fname) == 0:
                zero.append(input_fname)
            elif np.load(input_fname) == 1:
                one.append(input_fname)
            elif np.load(input_fname) == 2:
                two.append(input_fname)
            elif np.load(input_fname) == 3:
                three.append(input_fname)
        return zero, one, two, three

if __name__ == "__main__":
    Split('/Users/williamwong/Desktop/small', 1)