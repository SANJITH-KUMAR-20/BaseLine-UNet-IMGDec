import cv2 as cv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import os
import pandas as pd
from PIL import Image

class IMD2020Dataset(torch.utils.data.Dataset):
    def __init__(self, csv_path):
        self.csv = pd.read_csv(csv_path)

        self.transforms_fake = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((400,400)),
            torchvision.transforms.Normalize(mean= 0, std = 1, inplace= True)
        ])

        self.transforms_mask = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor(),
            torchvision.transforms.Resize((400,400))
        ])
    
    def __len__(self):
        return len(self.csv)

    def __getitem__(self, idx):
        row = self.csv.iloc[idx]
        fake, mask = row['fakes'], row['masks']

        fake = cv.imread(fake)
        mask = cv.imread(mask)

        mask = cv.cvtColor(mask, cv.COLOR_BGR2GRAY)

        fake_t = self.transforms_fake(fake)
        mask_t = self.transforms_mask(mask)
        
        return fake_t, mask_t