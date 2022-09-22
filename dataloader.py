import torch
import torch.utils.data as data
import numpy as np
import torchvision
import cv2
from utils import default_loader,disparity_loader

class myImageLoader(data.Dataset):
    def __init__(self, left, right, left_disparity, training, loader=default_loader, dploader=disparity_loader):

        self.left = left
        self.right = right
        self.disp_L = left_disparity
        self.loader = loader
        self.dploader = dploader
        self.training = training

    def __getitem__(self, index):
        left = self.left[index]
        right = self.right[index]
        disp_L = self.disp_L[index]

        th, tw = 256, 512
        dataL, scaleL = self.dploader(disp_L)
        dataL = np.ascontiguousarray(dataL, dtype=np.float32)

        left_img = np.array(self.loader(left))
        right_img = np.array(self.loader(right))
        left_img = cv2.resize(left_img,(tw,th),interpolation = cv2.INTER_CUBIC)
        right_img = cv2.resize(right_img,(tw,th),interpolation = cv2.INTER_CUBIC)
        dataL = torch.from_numpy(cv2.resize(dataL,(tw,th),interpolation = cv2.INTER_CUBIC))
        


        left_img = np.rint(255*((left_img-left_img.min())/(left_img.max()-left_img.min())))
        right_img = np.rint(255*((right_img-right_img.min())/(right_img.max()-right_img.min())))
        dataL = np.rint(255*((dataL-dataL.min())/(dataL.max()-dataL.min())))
        left_img = np.transpose(left_img,(2,0,1)).astype(float)
        right_img = np.transpose(right_img,(2,0,1)).astype(float)
        left_img = (torch.from_numpy(np.array(left_img)))
        right_img = (torch.from_numpy(np.array(right_img)))
        
        return left_img,right_img, dataL

    def __len__(self):
        return len(self.left)