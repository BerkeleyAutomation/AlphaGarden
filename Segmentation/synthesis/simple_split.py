import cv2
import matplotlib.pyplot as plt

width = 512
background = cv2.cvtColor(cv2.imread("original_0.png"), cv2.COLOR_BGR2RGB)
mask = cv2.cvtColor(cv2.imread("mask_0.png"), cv2.COLOR_BGR2RGB)
count = 21

R, C = background.shape[0], background.shape[1]
R0, C0 = mask.shape[0], mask.shape[1]

print(R, C, R0, C0)

for i in range(0, R - width, width):
    for j in range(0, C - width, width):
        background_patch = background[i:i+width, j:j+width, :]
        mask_patch = mask[i:i+width, j:j+width, :]
        plt.imsave("original_patches/original_{}.jpg".format(count), background_patch)
        plt.imsave("original_patches/original_{}.png".format(count), mask_patch)
        count += 1

