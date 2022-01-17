import os

import numpy as np
from PIL import Image, ImageOps

from paddleseg.datasets import Dataset
from paddleseg.utils.download import download_file_and_uncompress
from paddleseg.utils import seg_env
from paddleseg.cvlibs import manager
from paddleseg.transforms import Compose
import paddleseg.transforms.functional as F
import paddleseg.transforms as T

@manager.DATASETS.add_component
class AG_DS(Dataset):
    NUM_CLASSES = 150

    def __init__(self, transforms, dataset_root, label_root, mode='train', edge=False, relevant_dates = None):
        self.dataset_root = dataset_root
        self.transforms = Compose(transforms)
        mode = mode.lower()
        self.mode = mode
        self.file_list = list()
        self.num_classes = self.NUM_CLASSES
        self.ignore_index = 255
        self.edge = edge
        self.label_root = label_root

        if mode not in ['train', 'val']:
            raise ValueError(
                "`mode` should be one of ('train', 'val') in the AlphaGarden dataset, but got {}."
                .format(mode))

        if self.transforms is None:
            raise ValueError("`transforms` is necessary, but it is None.")
            
        if not (os.path.exists(self.dataset_root) and os.path.exists(self.label_root)):
            raise ValueError("Traning Dataset folder/s do not exist")
        
        
        if mode == 'train':
            img_dir = self.dataset_root
            label_dir = self.label_root
        elif mode == 'val':
            img_dir = self.dataset_root
            label_dir = self.label_root
        img_files = os.listdir(img_dir)
        label_files = [i.replace('.jpg', '.png') for i in img_files]
        raw_images = [i.replace('.jpg', '') for i in img_files]
        for i in range(len(img_files)):

            if relevant_dates is not None and img_files[i].replace(".jpg","") not in relevant_dates:
                continue
            img_path = os.path.join(img_dir, img_files[i])
            label_path = os.path.join(label_dir, label_files[i])
            if os.path.exists(img_path) and os.path.exists(label_path):
                self.file_list.append([img_path, label_path])
        self.raw_images = raw_images

    def __getitem__(self, idx):
        image_path, label_path = self.file_list[idx]
        if self.mode == 'val':
            im, _ = self.transforms(im=image_path)
            label = np.asarray(ImageOps.grayscale(Image.open(label_path)))
            # The class 0 is ignored. And it will equal to 255 after
            # subtracted 1, because the dtype of label is uint8.
            label = label - 1
            label = label[np.newaxis, :, :]
            return im, label
        else:
            im, label = self.transforms(im=image_path, label=np.asarray(ImageOps.grayscale(Image.open(label_path))))
            label = label - 1
            # Recover the ignore pixels adding by transform
            label[label == 254] = 255
            if self.edge:
                edge_mask = F.mask_to_binary_edge(
                    label, radius=2, num_classes=self.num_classes)
                return im, label, edge_mask
            else:
                return im, label

if __name__ == '__main__':
    transform = [
        T.ResizeStepScaling(0.5, 2.0, 0.25),
        T.RandomHorizontalFlip(),
        T.RandomPaddingCrop(crop_size=[480, 480]),
        T.RandomDistort(brightness_range=0.5,
                        contrast_range=0.5,
                        saturation_range=0.5),
        T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ]
    dataset_root = "/app/AlphaGarden/State-Estimation/out/cropped/"
    label_root = "/app/AlphaGarden/State-Estimation/out/post_process/"
    test = AG_DS(transform, dataset_root, label_root)