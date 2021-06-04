import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import cv2
from skimage.transform import resize
from skimage.io import imread, imshow, concatenate_images,imsave

def correct_image(im_src, one, two, three, four):
  size = (3478,1630,3) #change this or just take img size
  im_dst = np.zeros(size, np.uint8)
  pts_dst = np.array(
   [
    [0,0],
    [size[0] - 1, 0],
    [size[0] - 1, size[1] -1],
    [0, size[1] - 1 ]
    ], dtype=float
  )
  pts_src = np.array(
   [
    [one[0],one[1]],
    [two[0], two[1]],
    [three[0], three[1]],
    [four[0], four[1]]
    ], dtype=float
  )
  h, status = cv2.findHomography(pts_src, pts_dst)
  im_dst = cv2.warpPerspective(im_src, h, size[0:2])
  return im_dst
