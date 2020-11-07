from skimage.io import imread, imshow, concatenate_images,imsave
from tqdm import tqdm_notebook, tnrange
import os
import numpy as np

# TRAIN_PATH = './images/train'
LOAD_PATH = './submask/'


def saveSplit(ids,chunk_num):
    ### Adjust Split Folder###
    savepath = "./mergemask/"
    ### Adjust photo ###
    
    for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        img = imread(LOAD_PATH + '/' + id_ + '.png')
        im_height = img.shape[0]
        im_width = img.shape[1]
        break

    print(im_height,im_width)

    cate,catefull = category(ids)
    print(cate)
    for k in range(len(cate)):
        m = 0
        merge = np.zeros((1,im_height*4 ,im_width*4,4))
        print(merge.shape)
        for i in range(4):
            for j in range(4):
                img = imread(LOAD_PATH + '/' + cate[k]+str(m).zfill(3) + '.png')
                merge[0,i*im_height:(i+1)*im_height,j*im_width:(j+1)*im_width] = img
                m += 1
        name = cate[k] + 'merge' #here INC
        imsave(savepath + name + ".png", merge[0])

def category(ids):
    cate = []
    catefull = []
    for _, id_ in tqdm_notebook(enumerate(ids), total=len(ids)):
        if id_[:-3] not in cate:
            cate.append(id_[:-3])
            catefull.append([])
        ind = cate.index(id_[:-3])
        catefull[ind].append(id_)
    return cate,catefull


leaf_ids = set([f_name[:-4] for f_name in os.listdir(LOAD_PATH)])
saveSplit(leaf_ids,chunk_num=4)

print("split over")