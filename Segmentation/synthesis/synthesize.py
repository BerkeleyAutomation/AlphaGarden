# !/usr/bin/env python3

import cv2
import numpy as np
import matplotlib.pyplot as plt
import json
import random
import os

def create_mask(leaf, leaf_type):
    """This method creates a mask over the leaf based on the provided encoding.
    The encoding determines the color of the mask (type-based).
    """
    leaf[np.where((leaf == [255, 255, 255]).all(axis=2))] = [0, 0, 0]
    gray_scaled = cv2.cvtColor(leaf, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray_scaled, 10, 255, cv2.THRESH_BINARY)
    rows, cols = mask.shape
    remasked = np.zeros((rows, cols, 3), dtype=np.uint8)
    remasked[mask[:, :] > 0, :] = ENCODING[str(leaf_type)]
    return remasked.astype(np.uint8)

def rotate_image(mat, angle):
    """Rotates an image (angle in degrees) about the center and expands image to avoid cropping
    """

    height, width = mat.shape[:2] # image shape has 3 dimensions
    image_center = (width/2, height/2) # getRotationMatrix2D needs coordinates in reverse order (width, height) compared to shape
    rotation_mat = cv2.getRotationMatrix2D(image_center, angle, 1.)

    # rotation calculates the cos and sin, taking absolutes of those.
    abs_cos = abs(rotation_mat[0,0]) 
    abs_sin = abs(rotation_mat[0,1])

    # find the new width and height bounds
    bound_w = int(height * abs_sin + width * abs_cos)
    bound_h = int(height * abs_cos + width * abs_sin)

    # subtract old image center (bringing image back to origo) and adding the new image center coordinates
    rotation_mat[0, 2] += bound_w/2 - image_center[0]
    rotation_mat[1, 2] += bound_h/2 - image_center[1]

    # rotate image with the new bounds and translated rotation matrix
    rotated_mat = cv2.warpAffine(mat, rotation_mat, (bound_w, bound_h))
    return rotated_mat

def build_randomized_layout(leaves_src, background_src, background_mask_src, augment_offset, background_dim, num_iters=15, seed=15):
    """This method builds a newly synthesized garden bed by randomly
    overlaying augmented single leaves on top of the original background.
    In the end, an updated garden bed along with its mask are returned.
    """
    leaves = [[np.copy(leaf_src[0]).astype(np.uint8), leaf_src[1]] for leaf_src in leaves_src]
    leaves_mask = [create_mask(leaf_src[0], leaf_src[1]) for leaf_src in leaves_src]

    max_rows, max_cols, _ = background_src.shape
    background = np.copy(background_src)
    background_mask = np.copy(background_mask_src)
    
    for i in random.sample(list(range(len(leaves))), num_iters):
        idx = i % len(leaves)
        leaf = np.copy(leaves[idx][0])
        leaf_mask = np.copy(leaves_mask[idx])

        # Randomly resize the leaf and its mask between 0.75x and 1.25x.
        dim_ratio = np.random.uniform(0.8, 1.2)
        leaf = cv2.resize(leaf, (0, 0), fx=dim_ratio, fy=dim_ratio)
        leaf_mask = cv2.resize(leaf_mask, (0, 0), fx=dim_ratio, fy=dim_ratio)

        # Randomly rotate the leaf and its mask between 0 and 360 degrees.
        rot = int(np.random.uniform(0, 360))
        leaf = rotate_image(leaf, rot)
        leaf_mask = rotate_image(leaf_mask, rot)

        rows, cols, channels = leaf.shape

        # Randomly picks a location to place the leaf from its top left corner.
        rot = np.random.randint(0, 360)
        d_row = int(np.random.uniform(0, max_rows - rows))
        d_col = int(np.random.uniform(0, max_cols - cols))

        # The specification of the ROI determines where the masked leaf will be placed.
        roi = background[d_row:rows + d_row, d_col:cols + d_col, :]
        roi_mask = background_mask[d_row:rows + d_row, d_col:cols + d_col, :]

        # Now create a mask of the leaf and its inverse mask 
        img2gray = cv2.cvtColor(leaf,cv2.COLOR_RGB2GRAY)
        ret, mask = cv2.threshold(img2gray, 10, 255, cv2.THRESH_BINARY)
        mask_inv = cv2.bitwise_not(mask)

        # Now black-out the area of leaf in ROI
        cropped_background = cv2.bitwise_and(roi,roi, mask=mask_inv)
        cropped_background_mask = cv2.bitwise_and(roi_mask, roi_mask, mask=mask_inv)

        # Take only region of leaf from leaf image.
        target = cv2.bitwise_and(leaf, leaf, mask=mask)
        target_mask = cv2.bitwise_and(leaf_mask, leaf_mask, mask=mask)

        # Put the leaf in ROI and modify the background image
        dst = cv2.add(cropped_background,target)
        background[d_row:rows + d_row, d_col:cols + d_col] = dst

        dst_mask = cv2.add(cropped_background_mask, target_mask)
        background_mask[d_row:rows + d_row, d_col:cols + d_col] = dst_mask
    
    # Restore the background and its mask back to the center patch of the canvas.
    background_center = background[augment_offset: augment_offset + background_dim, augment_offset: augment_offset + background_dim, :]
    background_mask_center = background_mask[augment_offset: augment_offset + background_dim, augment_offset: augment_offset + background_dim, :]
    return background_center, background_mask_center

if __name__ == "__main__":
    try:
        f = open("sample_config.json")
        config = json.load(f)
    except FileNotFoundError:
        print("Please specify a config json file.")
    except:
        print("Encountered errors while parsing json")
    else:
        generated_folder_name = "generated-images"
        generated_mask_folder_name = "generated-masks"
        original_folder_name = "original-images"
        original_mask_folder_name = "original-masks"

        # reconstructed_folder_name = "reconstructed_full_background"
        # reconstructed_backgrounds = [os.path.join(reconstructed_folder_name, f) for f in os.listdir(reconstructed_folder_name)]

        background_path = config["background"]
        background_mask_path = config["background_mask"]
        background_src = cv2.cvtColor(cv2.imread(background_path), cv2.COLOR_BGR2RGB)
        background_mask_src = cv2.cvtColor(cv2.imread(background_mask_path), cv2.COLOR_BGR2RGB)
        max_background_row, max_background_col, _ = background_src.shape

        leaves = config["leaves"]
        random.shuffle(leaves)
        leaves_src = [[cv2.cvtColor(cv2.imread(leaf[0]), cv2.COLOR_BGR2RGB), leaf[1]] for leaf in leaves]

        ENCODING = config["encodings"]
        num_leaves = config["iterations"]
        num_simulations = config["num_copies"]
        side_len = config["dim"]

        for folder_name in [generated_folder_name, generated_mask_folder_name, original_folder_name, original_mask_folder_name]:
            os.mkdir(folder_name)

        offset = 0

        for i in range(num_simulations):
            r = int(np.random.uniform(max_background_row // 2, max_background_row - side_len - 300))
            c = int(np.random.uniform(0, max_background_col - side_len))
            background_patch = background_src[r:r+side_len, c:c+side_len, :]
            background_mask_patch = background_mask_src[r:r+side_len, c:c+side_len, :]
            # reconstructed_background = random.choice(reconstructed_backgrounds)
            # background_patch =  cv2.cvtColor(cv2.imread(reconstructed_background), cv2.COLOR_BGR2RGB)
            # background_mask_patch = np.zeros((side_len, side_len, 3), dtype=np.uint8)

            background_r = background_patch.shape[0]
            background_c = background_patch.shape[1]

            # Augment the image range (to allow natural cropping of leaves and masks)
            # by placing patches in the center of a bigger canvas.
            augment_offset = 100
            background_canvas = np.zeros((2 * augment_offset + background_r, 2 * augment_offset + background_c, 3), dtype=np.uint8)
            background_mask_canvas = np.zeros((2 * augment_offset + background_r, 2 * augment_offset + background_c, 3), dtype=np.uint8)
            background_canvas[augment_offset: augment_offset + background_r, augment_offset: augment_offset + background_c, :] = \
                background_patch
            background_mask_canvas[augment_offset: augment_offset + background_r, augment_offset: augment_offset + background_c, :] = \
                background_mask_patch

            synthesized_background, synthesized_mask = build_randomized_layout(
                leaves_src, background_canvas, background_mask_canvas, augment_offset, side_len, num_leaves)
            plt.imsave("{}/{}-{}.jpg".format(original_folder_name, "original", i + offset), background_patch)
            plt.imsave("{}/{}-{}.png".format(original_mask_folder_name, "original", i + offset), background_mask_patch)
            plt.imsave("{}/{}-{}.jpg".format(generated_folder_name, "synthesized", i + offset), synthesized_background)
            plt.imsave("{}/{}-{}.png".format(generated_mask_folder_name, "synthesized", i + offset), synthesized_mask)


