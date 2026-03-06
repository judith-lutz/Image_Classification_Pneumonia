
from torch.utils.data import Dataset
import os 
import pandas as pd
from torchvision.io import decode_image
from torch import nn
from torchvision.transforms.v2.functional import to_dtype

DATASET_NAME = 'chest-xray-pneumonia-balanced-dataset'

class CustomImageDataset(Dataset):
    def __init__(self, set_name, transform=None, target_transform=None, resolution=100):
        set_name = 'train'
        files = []
        labels = []
        for label in ['NORMAL', 'PNEUMONIA']:
            path_folder = "/".join(["../data/raw/", DATASET_NAME, set_name, label])
            list_files = os.listdir(path_folder)
            files =  files + list_files
            new_labels = [label] * len(list_files)
            labels = labels + new_labels
        self.set_name = set_name
        self.img_labels = labels
        self.img_dir = files
        self.transform = transform
        self.target_transform = target_transform
        self.resolution = resolution

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels[idx]
        if label == 'NORMAL':
            label_number = 0
        else:
            label_number = 1
        img_path = "/".join(["../data/raw/", DATASET_NAME, self.set_name, label, self.img_dir[idx]])
        #print(img_path)
        #img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = to_dtype(decode_image(img_path))[0,:,:]*1/255
        image = image[None, None, :,:]
        #print(image.shape)
        if self.transform:
            image = nn.functional.interpolate(input=image, size=(self.resolution,self.resolution))
        if self.target_transform:
            label_number = self.target_transform(label_number)
        return image[0,0,:,:], label_number
   
