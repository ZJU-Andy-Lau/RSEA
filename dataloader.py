import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset,DataLoader,Sampler
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
import torch.distributed as dist
   
    
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

def centerize_obj(obj:np.ndarray):
    x = obj[...,0]
    y = obj[...,1]
    h = obj[...,2]
    x = x - (x.max() + x.min()) * .5
    y = y - (y.max() + y.min()) * .5
    return np.stack([x,y,h],axis=-1)

def get_map_coef(target:np.ndarray,bins=1000,deg=20):
    src = np.linspace(0,1,bins)
    tgt = np.quantile(target,src)
    src = 2. * src - 1. # (0,1) -> (-1,1)
    coefs = np.polyfit(src,tgt,deg = deg)
    return coefs


class PretrainDataset(Dataset):
    def __init__(self,root,dataset_num = None,batch_size = 1,downsample=16,input_size = 1024,mode='train'):
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
        self.DOWNSAMPLE=downsample
        self.img_size = self.database[self.database_keys[0]]['image_1'][:].shape[0]
        self.input_size = input_size
        self.batch_size = batch_size
        self.obj_map_coefs = []
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        for key in tqdm(self.database_keys):
            obj = centerize_obj(self.database[key]['obj'][:])
            self.obj_map_coefs.append({
                'x':get_map_coef(obj[:,:,0]),
                'y':get_map_coef(obj[:,:,1]),
                'h':get_map_coef(obj[:,:,2])
            })

        if mode == 'train':
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.RandomApply([transforms.ColorJitter(.4,.4,.4,.4)],p=.7),
                # transforms.RandomInvert(p=.2),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])

    
    def __len__(self):
        return self.dataset_num
    
    def __getitem__(self, index):
        seed = index * self.world_size + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        rows = np.clip(np.random.randint(low=-self.input_size // 2,high=self.img_size - self.input_size // 2,size=(self.batch_size,1)),0,self.img_size - self.input_size)
        cols = np.clip(np.random.randint(low=-self.input_size // 2,high=self.img_size - self.input_size // 2,size=(self.batch_size,1)),0,self.img_size - self.input_size)
        windows = np.concatenate([rows,cols],axis=-1)

        key = self.database_keys[index]
        image_1_full = self.database[key]['image_1'][:]
        image_2_full = self.database[key]['image_2'][:]
        obj_full = centerize_obj(self.database[key]['obj'][:])
        residual_1_full = self.database[key]['residual_1'][:]
        residual_2_full = self.database[key]['residual_2'][:]
        image_1_full = np.stack([image_1_full] * 3,axis=-1)
        image_2_full = np.stack([image_2_full] * 3,axis=-1)


        # imgs1 = torch.from_numpy(np.stack([image_1_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size] for tl in windows],axis=0)).permute(0,3,1,2).to(torch.float32) # B,3,H,W
        # imgs2 = torch.from_numpy(np.stack([image_2_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size] for tl in windows],axis=0)).permute(0,3,1,2).to(torch.float32)
        # imgs1 = self.transform(imgs1)
        # imgs2 = self.transform(imgs2)
        imgs1 = torch.stack([self.transform(image_1_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size]) for tl in windows],dim=0)
        imgs2 = torch.stack([self.transform(image_2_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size]) for tl in windows],dim=0)


        obj = torch.from_numpy(np.stack([downsample(obj_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size],self.DOWNSAMPLE) for tl in windows],axis=0)).to(torch.float32)


        residual1 = np.stack([residual_average(residual_1_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size],self.DOWNSAMPLE) for tl in windows],axis=0)
        residual2 = np.stack([residual_average(residual_2_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size],self.DOWNSAMPLE) for tl in windows],axis=0)
        residual1[np.isnan(residual1)] = -1
        residual2[np.isnan(residual2)] = -1
        residual1 = torch.from_numpy(residual1)
        residual2 = torch.from_numpy(residual2)
        # t2 = time.perf_counter()

        # print(t1 - t0, t2 - t1)
        
        return imgs1,imgs2,obj,residual1,residual2,torch.tensor(index)


class ImageSampler(Sampler):
    """
    为所有rank提供相同的大图索引序列。
    确保在每个iteration，所有GPU都在处理同一张大图。
    """
    def __init__(self, data_source, shuffle=True):
        self.data_source = data_source
        self.shuffle = shuffle
        self.epoch = 0

    def __iter__(self):
        n = len(self.data_source)
        indices = list(range(n))
        
        if self.shuffle:
            # 使用epoch作为种子，确保每个epoch的shuffle顺序不同，
            # 但在所有进程中给定epoch的shuffle顺序是相同的。
            g = torch.Generator()
            g.manual_seed(self.epoch)
            indices = torch.randperm(n, generator=g).tolist()
            
        return iter(indices)

    def __len__(self):
        return len(self.data_source)

    def set_epoch(self, epoch):
        self.epoch = epoch