import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader
import os
import cv2
from tqdm import tqdm,trange
from utils import get_coord_mat,bilinear_interpolate
from rpc import RPCModelParameterTorch
from torchvision import transforms
import random
import math
import time
import h5py
import json
   
    
def residual_average(arr, a):
    is_2d = arr.ndim == 2
    if is_2d:
        arr = arr[..., np.newaxis]
    H, W, C = arr.shape
    new_H = ((H + a - 1) // a) * a
    new_W = ((W + a - 1) // a) * a
    padded = np.pad(arr, ((0, new_H - H), (0, new_W - W), (0, 0)),constant_values=np.nan)
    reshaped = padded.reshape(new_H//a, a, new_W//a, a, C)
    output = np.nanmean(reshaped,axis=(1,3))
    if is_2d:
        output = output.squeeze(axis=-1)
    return output

def downsample(arr,ds):
    if ds <= 0:
        return arr
    H,W = arr.shape[:2]
    lines = np.arange(0,H - ds + 1,ds) + (ds - 1.) * 0.5
    samps = np.arange(0,W - ds + 1,ds) + (ds - 1.) * 0.5
    sample_idxs = np.stack(np.meshgrid(samps,lines,indexing='xy'),axis=-1).reshape(-1,2) # x,y
    arr_ds = bilinear_interpolate(arr,sample_idxs)
    arr_ds = arr_ds.reshape(len(lines),len(samps),-1).squeeze()
    return arr_ds



class PretrainDataset(Dataset):
    def __init__(self,root,dataset_num = None,iter_num = 100,batch_size = 1,downsample=16,input_size = 1024,mode='train'):
        super().__init__()
        self.root = root
        if mode == 'train':
            self.database = h5py.File(os.path.join(self.root,'train_data.h5'),'r')
        elif mode == 'test':
            self.database = h5py.File(os.path.join(self.root,'test_data.h5'),'r')
        else:
            raise ValueError("mode should be either train or test")

        if dataset_num is None:
            dataset_num = len(self.database.keys())
        self.dataset_num = dataset_num
        self.database_keys = list(self.database.keys())[:dataset_num]
        self.iter_num = iter_num
        self.DOWNSAMPLE=downsample
        self.img_size = self.database[self.database_keys[0]]['image_1'][:].shape[0]
        self.input_size = input_size
        self.batch_size = batch_size
        self.obj_bboxs = []

        # for key in tqdm(self.database_keys):
        #     obj = self.database[key]['obj'][:]
        #     self.obj_bboxs.append({
        #         'x_min':obj[:,:,0].min(),
        #         'x_max':obj[:,:,0].max(),
        #         'y_min':obj[:,:,1].min(),
        #         'y_max':obj[:,:,1].max(),
        #         'h_min':obj[:,:,2].min(),
        #         'h_max':obj[:,:,2].max(),
        #     })

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([transforms.ColorJitter(.4,.4,.4,.4)],p=.7),
                transforms.RandomInvert(p=.3),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

        # print("creating windows")
        rows = np.clip(np.random.randint(low=-self.input_size // 2,high=self.img_size - self.input_size // 2,size=(self.iter_num,self.dataset_num,self.batch_size,1)),0,self.img_size - self.input_size)
        cols = np.clip(np.random.randint(low=-self.input_size // 2,high=self.img_size - self.input_size // 2,size=(self.iter_num,self.dataset_num,self.batch_size,1)),0,self.img_size - self.input_size)
        self.windows = np.concatenate([
            np.concatenate([rows,cols],axis=-1),
            np.concatenate([self.img_size - rows - self.input_size,cols],axis=-1),
            np.concatenate([rows,self.img_size - cols - self.input_size],axis=-1),
            np.concatenate([self.img_size - rows - self.input_size,self.img_size - cols - self.input_size],axis=-1)
        ],axis=2)
    
    def __len__(self):
        return self.iter_num
    
    def __getitem__(self, index):
        imgs = []
        objs = []
        residuals = []

        for dataset_idx,key in enumerate(self.database_keys):
            windows = self.windows[index,dataset_idx]
            image_1_full = self.database[key]['image_1'][:]
            image_2_full = self.database[key]['image_2'][:]
            obj_full = self.database[key]['obj'][:]
            residual_1_full = self.database[key]['residual_1'][:]
            residual_2_full = self.database[key]['residual_2'][:]
            # local_full = get_coord_mat(self.img_size,self.img_size)
            image_1_full = np.stack([image_1_full] * 3,axis=-1)
            image_2_full = np.stack([image_2_full] * 3,axis=-1)

            imgs1 = torch.stack([self.transform(image_1_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size]) for tl in windows],dim=0)
            imgs2 = torch.stack([self.transform(image_2_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size]) for tl in windows],dim=0)

            obj = torch.from_numpy(np.stack([downsample(obj_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size],self.DOWNSAMPLE) for tl in windows],axis=0))
            # local = torch.from_numpy(np.stack([downsample(local_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size],self.DOWNSAMPLE) for tl in windows],axis=0))

            residual1 = torch.from_numpy(np.stack([residual_average(residual_1_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size],self.DOWNSAMPLE) for tl in windows],axis=0))
            residual2 = torch.from_numpy(np.stack([residual_average(residual_2_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size],self.DOWNSAMPLE) for tl in windows],axis=0))

            
            imgs.append({
                'v1':imgs1,
                'v2':imgs2
            })
            objs.append(obj)
            residuals.append({
                'v1':residual1,
                'v2':residual2
            })

            

        return imgs,objs,residuals
    

