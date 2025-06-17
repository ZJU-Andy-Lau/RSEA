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

def k_to_ij(k, N):
    i = k // (N-1)
    m = k % (N-1)
    if m < i:
        j = m
    else:
        j = m + 1
    return i, j

def get_v1v2idx(img_nums,idx):
    sum = 0
    v1 = 0
    v_num = len(img_nums)
    for i,img_num in enumerate(img_nums):
        sum += img_num
        if sum >= idx:
            v1 = i
            break
    for i in range(v1):
        idx -= img_nums[i]
    v2 = int(idx * (v_num - 1) / img_nums[v1])
    if v2 >= v1:
        v2 += 1
    return v1,v2,idx
    
    
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
    

def load_overlay(path):
    # with open(txt_path,'r') as f:
    #     all_lines = f.readlines()
    # overlay = {}
    # for line in all_lines:
    #     line = line.split('\n')[0]
    #     names = line.split(' ')
    #     overlay[names[0]] = names[1:]
    # return overlay
    with open(path,'r') as f:
        overlay = json.load(f)
    return overlay


class PretrainDataset(Dataset):
    def __init__(self,root,dataset_num,view_num,downsample=16,mode='train'):
        super().__init__()
        self.root = root
        self.dataset_num = dataset_num
        self.view_num = view_num
        self.a_num = 1#self.view_num * (self.view_num - 1)
        self.datasets = []
        self.max_img_num = -1
        self.DOWNSAMPLE=downsample

        # self.mean = [0.4]
        # self.std = [0.25]
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

        for dataset_idx in range(self.dataset_num):
            img_nums = []
            total_img_num = 0
            for v in range(self.view_num):
                with h5py.File(os.path.join(os.path.join(root,f'd{dataset_idx}_v{v}'),'data.h5'),'r') as f:
                    img_num = len(f.keys())
                    img_nums.append(img_num)
                    total_img_num += img_num
            # img_num = len([i for i in os.listdir(os.path.join(root,f'd{dataset_idx}_v0')) if 'png' in i])
            self.max_img_num = max(self.max_img_num,total_img_num)
            dataset = {
                'img_nums':img_nums,
                'total_img_num':total_img_num,
                'rpcs':[],
                'datas':[None] * self.view_num,
                'extend':torch.from_numpy(np.load(os.path.join(root,f'd{dataset_idx}_extend.npy'))),
                'overlay':load_overlay(os.path.join(root,f'd{dataset_idx}_overlay.json'))
            }
            for v in range(self.view_num):
                rpc = RPCModelParameterTorch()
                rpc.load_from_file(os.path.join(root,f'd{dataset_idx}_v{v}',f'd{dataset_idx}_v{v}.rpc'))
                dataset['rpcs'].append(rpc)
            self.datasets.append(dataset)

    
    def __len__(self):
        return self.max_img_num * self.a_num
    
    def __getitem__(self, index):
        imgs = []
        objs = []
        locals = []
        residuals = []
        view_idxs = []

        for dataset_idx in range(self.dataset_num):
            dataset = self.datasets[dataset_idx]
            img_nums = dataset['img_nums']
            # idx = index % (img_num * self.a_num)
            # k = idx // img_num
            # idx = idx % img_num
            # v1,v2 = k_to_ij(k,self.view_num)
            idx = index % (dataset['total_img_num'])
            v1,v2,idx = get_v1v2idx(img_nums,idx)
            # print(v1,v2,idx,index)

            if self.datasets[dataset_idx]['datas'][v1] is None:
                self.datasets[dataset_idx]['datas'][v1] = h5py.File(os.path.join(os.path.join(self.root,f'd{dataset_idx}_v{v1}'),'data.h5'),'r')
            if self.datasets[dataset_idx]['datas'][v2] is None:
                self.datasets[dataset_idx]['datas'][v2] = h5py.File(os.path.join(os.path.join(self.root,f'd{dataset_idx}_v{v2}'),'data.h5'),'r')
            
            
            overlay = dataset['overlay'][str(v1)][str(v2)][str(idx)]
            v2_name = overlay[np.random.randint(len(overlay))]
            data_v1 = self.datasets[dataset_idx]['datas'][v1][str(idx)]
            data_v2 = self.datasets[dataset_idx]['datas'][v2][str(v2_name)]

            img1 = self.transform(data_v1['img'][:])
            img2 = self.transform(data_v2['img'][:])

            obj1 = torch.from_numpy(data_v1['obj'][:])
            obj2 = torch.from_numpy(data_v2['obj'][:])

            local1 = torch.from_numpy(data_v1['local'][:])
            local2 = torch.from_numpy(data_v2['local'][:])

            residual1 = torch.from_numpy(data_v1['res'][:])
            residual2 = torch.from_numpy(data_v2['res'][:]) 

            

            


            # # t1 = time.perf_counter()
            # path1 = os.path.join(self.root,f'd{dataset_idx}_v{v1}')
            # path2 = os.path.join(self.root,f'd{dataset_idx}_v{v2}')
            
            # img1 = cv2.imread(os.path.join(path1,f'{idx}.png'))
            # img2 = cv2.imread(os.path.join(path2,f'{v2_name}.png'))
            # # t2 = time.perf_counter()
            # img1 = self.transform(img1).to(torch.float32)
            # img2 = self.transform(img2).to(torch.float32)
            # # t3 = time.perf_counter()

            # obj1 = torch.from_numpy(np.load(os.path.join(path1,f'{idx}_obj.npy')).astype(np.float32))
            # obj2 = torch.from_numpy(np.load(os.path.join(path2,f'{v2_name}_obj.npy')).astype(np.float32))

            # local1 = torch.from_numpy(np.load(os.path.join(path1,f'{idx}_local.npy')))
            # local2 = torch.from_numpy(np.load(os.path.join(path2,f'{v2_name}_local.npy')))

            # residual1 = torch.from_numpy(np.load(os.path.join(path1,f"{idx}_res.npy")))
            # residual2 = torch.from_numpy(np.load(os.path.join(path2,f"{v2_name}_res.npy")))
            # t4 = time.perf_counter()
            # local1_downsampled = torch.from_numpy(downsample(local1,self.DOWNSAMPLE))
            # local2_downsampled = torch.from_numpy(downsample(local2,self.DOWNSAMPLE))
            # obj1_downsampled = torch.from_numpy(downsample(obj1,self.DOWNSAMPLE))
            # obj2_downsampled = torch.from_numpy(downsample(obj2,self.DOWNSAMPLE))
            # residual1_downsampled = torch.from_numpy(residual_average(residual1,self.DOWNSAMPLE))
            # residual2_downsampled = torch.from_numpy(residual_average(residual2,self.DOWNSAMPLE))
            # t5 = time.perf_counter()

            # print((t1 - t0) * 256 * 6,(t2 - t1) * 256 * 6,(t3 - t2) * 256 * 6,(t4 - t3) * 256 * 6,(t5 - t4) * 256 * 6,'\n')
            # exit()
            


            imgs.append({
                'v1':img1,
                'v2':img2
            })
            objs.append({
                'v1':obj1,
                'v2':obj2
            })
            locals.append({
                'v1':local1,
                'v2':local2
            })
            residuals.append({
                'v1':residual1,
                'v2':residual2
            })
            view_idxs.append({
                'v1':v1,
                'v2':v2
            })
            

        return imgs,locals,objs,residuals,view_idxs
    

