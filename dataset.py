import torch
from torch.utils.data import Dataset
import numpy as np
import imageio
from kornia import create_meshgrid
import cv2
from einops import rearrange
from typing import Tuple

class ImageDataset(Dataset):
    def __init__(self, image_path: str, img_wh: Tuple[int, int], split: str):
        '''
        splite: 'train' or 'val'
        '''
        # rgb: [0,255] uint8
        # normalize rgb value to [0,1] float 
        image = imageio.imread(image_path)[..., :3]/255.
        image = cv2.resize(image, img_wh)
        '''
            create_meshgrid(H, W, normalize_coordinates, device, dtype)) generate a coordinate grid for an image
            normalize_coordinates: bool = True set the normalized coordinates to [-1, 1]
        ''' 
        self.uv = create_meshgrid(img_wh[1], img_wh[0], True)[0] # (1, 512, 512, 3)[0]->(512, 512, 3)
        # turn numpy array to tensor automaticlly
        self.rgb = torch.FloatTensor(image) # (512, 512, 3)
        
        '''
            how  to splite the data?
            we devide the whole image in to a set of 4*4 pixel arrays
            we set 1/4 of pixel to train_set and all pixel to be test_set in each pixel array
        '''
        if split == 'train':
            self.uv = self.uv[::2, ::2] # (256, 256, 2)
            self.rgb = self.rgb[::2, ::2] # (256, 256, 3)
            
        self.uv = rearrange(self.uv, 'h w c -> (h w) c')
        self.rgb = rearrange(self.rgb, 'h w c -> (h w) c')
    def __len__(self):
        return len(self.uv) 
    def __getitem__(self, idx:int):
        return {"uv": self.uv[idx], "rgb": self.rgb[idx]}