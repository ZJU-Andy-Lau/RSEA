import torch
import torch.nn as nn
import torch.nn.functional as F
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
from typing import List, Tuple

   
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

def sample_bilinear(A: torch.Tensor, idx: np.ndarray) -> torch.Tensor:
    B, H, W, _ = A.shape
    grid = torch.from_numpy(idx).to(device=A.device, dtype=A.dtype)
    A_permuted = A.permute(0, 3, 1, 2)
    normalized_grid = torch.zeros_like(grid)
    normalized_grid[..., 0] = 2.0 * grid[..., 1] / W - 1.0
    normalized_grid[..., 1] = 2.0 * grid[..., 0] / H - 1.0
    grid_reshaped = normalized_grid.unsqueeze(2)
    sampled_output = F.grid_sample(
        A_permuted, 
        grid_reshaped, 
        mode='bilinear', 
        padding_mode='border', 
        align_corners=True
    )

    result = sampled_output.permute(0, 2, 3, 1).squeeze(2)

    return result


def centerize_obj(obj:np.ndarray):
    x = obj[...,0]
    y = obj[...,1]
    h = obj[...,2]
    x = x - (x.max() + x.min()) * .5
    y = y - (y.max() + y.min()) * .5
    return np.stack([x,y,h],axis=-1)

def get_map_coef(target:np.ndarray,bins=1000,deg=20):
    extend_bins = int(bins * 0.1)
    src = np.linspace(0,1,bins)
    tgt = np.quantile(target,src)
    tgt = np.concatenate([2 * tgt[0] - tgt[:extend_bins][::-1],tgt,2 * tgt[-1] - tgt[-extend_bins:][::-1]],axis=0)
    src = np.linspace(-1,1,bins + 2 * extend_bins)
    coefs = np.polyfit(src,tgt,deg = deg)
    return coefs

def get_random_overlapping_crops(
    image_height: int,
    image_width: int,
    min_crop_side: int = 500,
) -> Tuple[Tuple[int, int, int], Tuple[int, int, int]]:
    max_crop_side = min(image_height, image_width)
    if min_crop_side > max_crop_side:
        raise ValueError("min_crop_side cannot be larger than the smallest image dimension.")
    s1 = np.random.randint(min_crop_side, max_crop_side + 1)
    s2 = np.random.randint(min_crop_side, max_crop_side + 1)
    
    s_small = min(s1, s2)
    s_o_min = math.ceil(s_small / 2.0)
    s_o_max = s_small
    s_o = np.random.randint(s_o_min, s_o_max + 1)

    x1 = np.clip(np.random.randint(-s1 // 2, image_width - s1 // 2),a_min=0,a_max=image_width - s1)
    y1 = np.clip(np.random.randint(-s1 // 2, image_height - s1 // 2),a_min=0,a_max=image_height - s1)

    dx_o1 = np.random.randint(0, s1 - s_o + 1)
    dy_o1 = np.random.randint(0, s1 - s_o + 1)
    x_o, y_o = x1 + dx_o1, y1 + dy_o1

    x2_min_ideal = x_o + s_o - s2
    x2_max_ideal = x_o
    y2_min_ideal = y_o + s_o - s2
    y2_max_ideal = y_o

    x2_min_final = max(0, x2_min_ideal)
    x2_max_final = min(image_width - s2, x2_max_ideal)
    y2_min_final = max(0, y2_min_ideal)
    y2_max_final = min(image_height - s2, y2_max_ideal)
    
    x2 = np.random.randint(x2_min_final, x2_max_final + 1)
    y2 = np.random.randint(y2_min_final, y2_max_final + 1)

    return (x1, y1, s1), (x2, y2, s2)


def process_image(
    img1_full: np.ndarray,
    img2_full: np.ndarray,
    obj_full:np.ndarray,
    residual1_full:np.ndarray,
    residual2_full:np.ndarray,
    K: int,
    min_crop_side: int = 500,
    output_size: int = 1024,
    downsample_ratio: int = 16
) -> Tuple[np.ndarray, np.ndarray, List[np.ndarray], List[np.ndarray], np.ndarray, np.ndarray]:
    H, W, _ = img1_full.shape
    
    imgs1 = np.zeros((K, output_size, output_size, 3), dtype=np.uint8)
    imgs2 = np.zeros((K, output_size, output_size, 3), dtype=np.uint8)

    obj1 = np.zeros((K, output_size, output_size, 3), dtype=np.float32)
    obj2 = np.zeros((K, output_size, output_size, 3), dtype=np.float32)

    residual1 = np.zeros((K, output_size, output_size), dtype=np.float32)
    residual2 = np.zeros((K, output_size, output_size), dtype=np.float32)

    corr_idxs1_list, corr_idxs2_list = [], []
    
    feature_map_size = output_size // downsample_ratio

    bboxes1 = []
    bboxes2 = []

    local = get_coord_mat(H,W).astype(np.float32)

    # 为K对图像生成数据
    for k in range(K):
        # 1. 获取一对有效的随机裁切框
        bbox1, bbox2 = get_random_overlapping_crops(H, W, min_crop_side)
        bboxes1.append(bbox1)
        bboxes2.append(bbox2)
        
        x1, y1, s1 = bbox1
        x2, y2, s2 = bbox2

        r1 = 1. * output_size / s1
        r2 = 1. * output_size / s2
        
        # 2. 裁切和缩放图像
        
        imgs1[k] = cv2.resize(img1_full[y1:y1+s1, x1:x1+s1, :], (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        imgs2[k] = cv2.resize(img2_full[y2:y2+s2, x2:x2+s2, :], (output_size, output_size), interpolation=cv2.INTER_LINEAR)

        obj1[k] = cv2.resize(obj_full[y1:y1+s1, x1:x1+s1, :], (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        obj2[k] = cv2.resize(obj_full[y2:y2+s2, x2:x2+s2, :], (output_size, output_size), interpolation=cv2.INTER_LINEAR)

        residual1[k] = cv2.resize(residual1_full[y1:y1+s1, x1:x1+s1], (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        residual2[k] = cv2.resize(residual2_full[y2:y2+s2, x2:x2+s2], (output_size, output_size), interpolation=cv2.INTER_NEAREST)
        
        local1 = cv2.resize(local[y1:y1+s1, x1:x1+s1, :], (output_size, output_size), interpolation=cv2.INTER_LINEAR)
        local2 = cv2.resize(local[y2:y2+s2, x2:x2+s2, :], (output_size, output_size), interpolation=cv2.INTER_LINEAR)

        residual_total = np.concatenate([residual1[k].reshape(-1),residual2[k].reshape(-1)],axis=0)
        local_total = np.concatenate([local1.reshape(-1,2),local2.reshape(-1,2)],axis=0)
        overlap_mask = (local_total[:,0] - y1 >= 0) & (local_total[:,0] - y2 >= 0) & (local_total[:,0] - y1 < s1) & (local_total[:,0] - y2 < s2) & \
                       (local_total[:,1] - x1 >= 0) & (local_total[:,1] - x2 >= 0) & (local_total[:,1] - x1 < s1) & (local_total[:,1] - x2 < s2)
        valid_mask = ~np.isnan(residual_total)
        residual_total = residual_total[overlap_mask & valid_mask]
        local_total = local_total[overlap_mask & valid_mask]

        sorted_idx = np.argsort(residual_total)
        local_total = local_total[sorted_idx]
        
        select_idx = np.array([],dtype=int)
        while(len(select_idx) < 10000):
            select_idx = np.concatenate([select_idx,np.arange(10000 - len(select_idx),dtype=int)[:len(local_total)]],axis=0,dtype=int)
        
        local_selected = local_total[select_idx[:10000]]

        # if dist.get_rank() == 0:
        #     test_idx1 = ((local_selected - np.array([y1,x1])) * r1).astype(int)
        #     test_idx2 = ((local_selected - np.array([y2,x2])) * r2).astype(int)
        #     test_obj1 = obj1[k][test_idx1[:,0],test_idx1[:,1]]
        #     test_obj2 = obj2[k][test_idx2[:,0],test_idx2[:,1]]
        #     dis_obj = np.linalg.norm(test_obj1 - test_obj2,axis=-1)
        #     print(f"dis_obj: \t {dis_obj.min()} \t {dis_obj.max()} \t {dis_obj.mean()} \t {np.median(dis_obj)}")

        corr_idxs1_list.append(np.clip((local_selected - np.array([y1,x1])) * r1 / downsample_ratio, a_min=0.,a_max=(output_size - 1) / downsample_ratio))
        corr_idxs2_list.append(np.clip((local_selected - np.array([y2,x2])) * r2 / downsample_ratio, a_min=0.,a_max=(output_size - 1) / downsample_ratio))

    corr_idxs1 = np.stack(corr_idxs1_list).astype(np.float32)
    corr_idxs2 = np.stack(corr_idxs2_list).astype(np.float32)

    return imgs1, imgs2, obj1, obj2, residual1, residual2, corr_idxs1, corr_idxs2

class PretrainDataset(Dataset):
    def __init__(self,root,dataset_idxs = None,batch_size = 1,downsample=16,input_size = 1024,mode='train'):
        super().__init__()
        self.root = root
        if mode == 'train':
            self.database = h5py.File(os.path.join(self.root,'train_data.h5'),'r')
        elif mode == 'test':
            self.database = h5py.File(os.path.join(self.root,'test_data.h5'),'r')
        else:
            raise ValueError("mode should be either train or test")

        # if dataset_num is None or dataset_num <= 0:
        #     dataset_num = len(self.database.keys())
        if dataset_idxs is None:
            self.dataset_num = len(self.database.keys())
        else:
            self.dataset_num = len(dataset_idxs)
        self.database_keys = [list(self.database.keys())[i] for i in dataset_idxs]
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
                transforms.RandomInvert(p=.2),
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

        key = self.database_keys[index]
        image_1_full = self.database[key]['image_1'][:]
        image_2_full = self.database[key]['image_2'][:]
        obj_full = centerize_obj(self.database[key]['obj'][:])
        residual_1_full = self.database[key]['residual_1'][:]
        residual_2_full = self.database[key]['residual_2'][:]
        image_1_full = np.stack([image_1_full] * 3,axis=-1)
        image_2_full = np.stack([image_2_full] * 3,axis=-1)

        imgs1, imgs2, obj1, obj2, residual1, residual2, overlaps_1, overlaps_2 = \
            process_image(img1_full=image_1_full,
                          img2_full=image_2_full,
                          obj_full=obj_full,
                          residual1_full=residual_1_full,
                          residual2_full=residual_2_full,
                          K=self.batch_size,
                          min_crop_side=500,
                          output_size=self.input_size,
                          downsample_ratio=self.DOWNSAMPLE)


        # imgs1 = torch.from_numpy(np.stack([image_1_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size] for tl in windows],axis=0)).permute(0,3,1,2).to(torch.float32) # B,3,H,W
        # imgs2 = torch.from_numpy(np.stack([image_2_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size] for tl in windows],axis=0)).permute(0,3,1,2).to(torch.float32)
        # imgs1 = self.transform(imgs1)
        # imgs2 = self.transform(imgs2)
        imgs1 = torch.stack([self.transform(img) for img in imgs1],dim=0)
        imgs2 = torch.stack([self.transform(img) for img in imgs2],dim=0)


        obj1 = torch.from_numpy(np.stack([downsample(obj,self.DOWNSAMPLE) for obj in obj1],axis=0)).to(torch.float32)
        obj2 = torch.from_numpy(np.stack([downsample(obj,self.DOWNSAMPLE) for obj in obj2],axis=0)).to(torch.float32)

        test_obj1 = sample_bilinear(obj1,overlaps_1)
        test_obj2 = sample_bilinear(obj2,overlaps_2)
        obj_dis = torch.norm(test_obj1[:,:2] - test_obj2[:,:2],dim=-1).reshape(-1)
        print(f"dis:{obj_dis.min().item()} \t {obj_dis.max().item()} \t {obj_dis.mean().item()} \t {obj_dis.median().item()}\n")

        residual1 = np.stack([residual_average(residual,self.DOWNSAMPLE) for residual in residual1],axis=0)
        residual2 = np.stack([residual_average(residual,self.DOWNSAMPLE) for residual in residual2],axis=0)
        residual1[np.isnan(residual1)] = -1
        residual2[np.isnan(residual2)] = -1
        residual1 = torch.from_numpy(residual1)
        residual2 = torch.from_numpy(residual2)

        overlaps_1 = torch.from_numpy(overlaps_1)
        overlaps_2 = torch.from_numpy(overlaps_2)
        # t2 = time.perf_counter()

        # print(t1 - t0, t2 - t1)
        
        return imgs1,imgs2,obj1,obj2,residual1,residual2,overlaps_1,overlaps_2,torch.tensor(index)


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