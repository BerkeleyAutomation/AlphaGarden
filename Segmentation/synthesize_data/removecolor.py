import cv2
import matplotlib.pyplot as plt
import numpy as np

width = 512


images = ["original_0", "original_1"]
colors = [(0, 0, 255)]

for image in images:
    im = cv2.imread(image + ".png")
    for color in colors:
        indices = np.where(np.all(im == color))
        print(indices[0])
        for x in range(im.shape[0]):
            for y in range(im.shape[1]):
                if all(im[x][y] == color):
                    im[x][y] = (0, 0, 0)
    print("reached")
    im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
    plt.imsave('./sample.png', im)
