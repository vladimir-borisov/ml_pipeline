import cv2
import numpy as np
import pandas as pd
import torch
import os
import yaml
from addict import Dict
from torch.utils.data import Dataset
import albumentations as A

# read config file
with open('./configs/train_params.yaml') as config_file:
    config = yaml.load(config_file, Loader=yaml.FullLoader)

config = Dict(config)

def get_training_augmentation():
    train_transform = [

        A.OneOf(
            [
                A.Resize(height=config.image_height, width=config.image_width, interpolation=cv2.INTER_NEAREST),
                # A.RandomCrop(height = config.image_height, width = config.image_width)
            ],
            p=1.0,
        ),
        # A.Flip(p=0.5),
        # A.RandomRotate90(p = 0.5),
        # A.RandomContrast(limit = 0.3),
        # A.RandomGamma(gamma_limit = (50, 140)),
        # A.RandomBrightness(limit = 0.2)
    ]
    return A.Compose(train_transform)


def get_validation_augmentation():
    test_transform = [
        A.OneOf(
            [
                A.Resize(height=config.image_height, width=config.image_width, interpolation=cv2.INTER_NEAREST),
                # A.RandomCrop(height = config.image_height, width = config.image_width)
            ],
            p=1.0,
        ),
    ]
    return A.Compose(test_transform)


id2name = {
    0: 'dog',
    1: 'cat'
}

name2id = {
    'dog': 0,
    'cat': 1
}

class MyDataset(Dataset):

    def __init__(self, df: pd.DataFrame, image_folder_path: str = '', augmentations=None):

        self.image_folder_path = image_folder_path
        self.df = df
        self.augmentations = augmentations

    def __len__(self):
        return len(self.df)

    def __getitem__(self, ind):

        image_info = self.df.loc[ind]

        img_path = os.path.join(self.image_folder_path, image_info['image_name'])

        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if (self.augmentations is not None):
            augmented_image = self.augmentations(image=image)['image']
        else:
            augmented_image = image

        augmented_image = (augmented_image.transpose((2, 0, 1)) / 255.0).astype(np.float32)

        return {'image': torch.from_numpy(augmented_image),
                'class_name': image_info['class_name'],
                'class_id': image_info['class_id']}
