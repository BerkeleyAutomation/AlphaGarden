import numpy as np

pr = np.load('/Users/williamwong/Documents/Berkeley/Research/AlphaGarden/Test/326a84eaada2e165b304dae7da0b6e63_pr.npy')
full_obs = np.load('/Users/williamwong/Documents/Berkeley/Research/AlphaGarden/Test/326a84eaada2e165b304dae7da0b6e63.npz')
print(pr)
print(full_obs['plants'].shape)
print(full_obs['water'].shape)
print(full_obs['health'].shape)
print(full_obs['global_cc'].shape)
