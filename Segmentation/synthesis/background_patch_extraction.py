# !/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import random

from os import listdir
from os.path import join

def extract_background_patches(img_path, mask_path, patch_size=256, stride=10, target_rgb=(0, 0, 0), curr_index=0):
    """Takes strides in a row-major fashion across the image, and 
    extracts all (patch_size x patch_size) patches whose masked counterparts
    only contain other-type pixels (those that match all entries in target_rgb).
    """
    background_folder = "full_background_512"

    img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
    mask = cv2.cvtColor(cv2.imread(mask_path), cv2.COLOR_BGR2RGB)
    R = img.shape[0]
    C = img.shape[1]
    total_patch_pixels = patch_size ** 2

    for i in range(0, R - patch_size, stride):
        for j in range(0, C - patch_size, stride):
            curr_patch = img[i:i+patch_size, j:j+patch_size, :]
            curr_mask = mask[i:i+patch_size, j:j+patch_size, :]

            match_background = True
            for i in range(len(target_rgb)):
                match_background &= curr_mask[:, :, i] == target_rgb[i]
            match_count = np.count_nonzero(match_background)

            if match_count == total_patch_pixels:
                print('Found Full Background Patch!')
                plt.imsave("{}/full-background-{}.png".format(background_folder, curr_index), curr_patch)
                curr_index += 1
    return curr_index

def extract_background_patches_in_group(original_img_folder_path, patch_size=256, stride=10, target_rgb=(0, 0, 0)):
    """Extracts all matching full background patches from all the 
    images under the original_img_folder_path.
    """
    img_paths = []
    for f in listdir(original_img_folder_path):
        if f.endswith('jpg'):
            img_paths.append(join(original_img_folder_path, f))

    curr_index = 0
    for img_path in img_paths:
        mask_path = img_path[:-3] + "png"
        curr_index = extract_background_patches(img_path, mask_path, patch_size, stride, target_rgb, curr_index)

def reconstruct_full_backgrounds(background_path, background_size=512, patch_size=256, num_imgs=300):
    """Reconstructs a full background patch of dimension (background_size x background_size)
    from all the available patches. background_size must be divisible by the patch size used
    originally for extracting all matching patches.
    """
    full_background_folder = "reconstructed_full_background"
    background_patches = []
    num_patches = (background_size // patch_size) ** 2

    for f in listdir(background_path):
        if f.endswith('png'):
            background_patches.append(join(background_path, f))
    
    for i in range(num_imgs):
        initial_background = np.zeros((background_size, background_size, 3), dtype=np.uint8)
        selected_patches = random.sample(background_patches, num_patches)
        patch_i = 0
        for x in range(0, background_size, patch_size):
            for y in range(0, background_size, patch_size):
                print(x, y)
                selected_patch = cv2.cvtColor(cv2.imread(selected_patches[patch_i]), cv2.COLOR_BGR2RGB)
                initial_background[x:x+patch_size, y:y+patch_size, :] = selected_patch
                patch_i += 1
        plt.imsave("{}/reconstructed_backround-{}.png".format(full_background_folder, i), initial_background)
        

if __name__ == "__main__":
    IMG_FOLDER_PATH = "./original_images"
    FULL_BACKGROUND_PATH = "./full_background_512"
    NUM_BACKGROUNDS = 300

    PATCH_SIZE = 512
    BACKGROUND_SIZE = 512

    STRIDE = 1
    OTHER_TYPE_RGB = (0, 0, 0)
    extract_background_patches_in_group(IMG_FOLDER_PATH, PATCH_SIZE, STRIDE, OTHER_TYPE_RGB)
    reconstruct_full_backgrounds(FULL_BACKGROUND_PATH, BACKGROUND_SIZE, PATCH_SIZE, NUM_BACKGROUNDS)


