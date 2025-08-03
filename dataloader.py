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

   
def residual_average(arr:np.ndarray, a:int) -> np.ndarray:
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
    """
    Generates two random, overlapping, axis-aligned square crop boxes.
    """
    max_crop_side = min(image_height, image_width)
    if min_crop_side > max_crop_side:
        raise ValueError("min_crop_side cannot be larger than the smallest image dimension.")
    
    s1 = np.random.randint(min_crop_side, max_crop_side + 1)
    s2 = np.random.randint(min_crop_side, max_crop_side + 1)
    
    s_small = min(s1, s2)
    s_o_min = math.ceil(s_small / 2.0)
    s_o_max = s_small
    s_o = np.random.randint(s_o_min, s_o_max + 1)

    x1 = np.clip(np.random.randint(-s1 // 2, image_width - s1 // 2), a_min=0, a_max=image_width - s1)
    y1 = np.clip(np.random.randint(-s1 // 2, image_height - s1 // 2), a_min=0, a_max=image_height - s1)

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
    
    if x2_min_final > x2_max_final or y2_min_final > y2_max_final:
        return get_random_overlapping_crops(image_height, image_width, min_crop_side)

    x2 = np.random.randint(x2_min_final, x2_max_final + 1)
    y2 = np.random.randint(y2_min_final, y2_max_final + 1)

    return (x1, y1, s1), (x2, y2, s2)


def get_affine_transform_from_box(
    center_x: float, 
    center_y: float, 
    side: float, 
    angle_deg: float, 
    output_size: int
) -> np.ndarray:
    """
    Calculates the 2x3 affine transform matrix to warp a rotated box to a new image.
    """
    dst_points = np.array([
        [0, 0],
        [output_size - 1, 0],
        [0, output_size - 1]
    ], dtype=np.float32)

    half_side = side / 2.0
    src_corners = np.array([
        [-half_side, -half_side],
        [half_side, -half_side],
        [-half_side, half_side]
    ])

    angle_rad = np.deg2rad(angle_deg)
    cos_a, sin_a = np.cos(angle_rad), np.sin(angle_rad)
    rotation_matrix = np.array([[cos_a, -sin_a], [sin_a, cos_a]])
    rotated_corners = src_corners @ rotation_matrix.T
    src_points = rotated_corners + np.array([center_x, center_y])
    
    return cv2.getAffineTransform(src_points.astype(np.float32), dst_points)

# -----------------------------------------------------------------------------
# Main Processing Function (Modified and Fixed)
# -----------------------------------------------------------------------------

def process_image(
    img1_full: np.ndarray,
    img2_full: np.ndarray,
    obj_full: np.ndarray,
    residual1_full: np.ndarray,
    residual2_full: np.ndarray,
    K: int,
    min_crop_side: int = 500,
    output_size: int = 1024,
    downsample_ratio: int = 16,
    p_rotate: float = 0.5,
    max_angle_deg: float = 45.0
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Processes full-size images to generate K pairs of overlapping crops,
    with a chance of applying rotation. This version fixes black borders and correspondence calculation.
    """
    H, W, _ = img1_full.shape
    
    imgs1 = np.zeros((K, output_size, output_size, 3), dtype=np.uint8)
    imgs2 = np.zeros((K, output_size, output_size, 3), dtype=np.uint8)
    obj1 = np.zeros((K, output_size, output_size, obj_full.shape[2]), dtype=np.float32)
    obj2 = np.zeros((K, output_size, output_size, obj_full.shape[2]), dtype=np.float32)
    residual1 = np.zeros((K, output_size, output_size), dtype=np.float32)
    residual2 = np.zeros((K, output_size, output_size), dtype=np.float32)

    corr_idxs1_list, corr_idxs2_list = [], []
    local = get_coord_mat(H, W)

    for k in range(K):
        if np.random.rand() < p_rotate:
            # --- FINAL, CORRECTED ROTATED CROP PATH ---
            max_side = min(H, W)
            
            # 1. Generate first box by finding a valid (s, angle) pair first
            while True:
                s1 = np.random.randint(min_crop_side, max_side + 1)
                angle1 = np.random.uniform(-max_angle_deg, max_angle_deg)
                angle1_rad = np.deg2rad(angle1)
                
                # Calculate the half-dimension of the axis-aligned bounding box of the rotated square
                half_dim1 = (s1 / 2.0) * (abs(np.cos(angle1_rad)) + abs(np.sin(angle1_rad)))
                
                # Check if this box can geometrically fit in the image at all
                if 2 * half_dim1 <= W and 2 * half_dim1 <= H:
                    break # Found a valid size and angle

            # Now that we have a valid s1 and angle1, we can safely find a center
            c1_x = np.random.uniform(half_dim1, W - half_dim1)
            c1_y = np.random.uniform(half_dim1, H - half_dim1)

            # 2. Generate second box, also ensuring it's valid
            while True:
                s2 = np.random.randint(min_crop_side, max_side + 1)
                angle2 = np.random.uniform(-max_angle_deg, max_angle_deg)
                angle2_rad = np.deg2rad(angle2)
                half_dim2 = (s2 / 2.0) * (abs(np.cos(angle2_rad)) + abs(np.sin(angle2_rad)))
                if 2 * half_dim2 <= W and 2 * half_dim2 <= H:
                    break
            
            # Propose and clip the center for the second box to ensure it's also safe
            # and likely overlaps with the first one
            max_offset = min(s1, s2) * 0.5 # Increase potential overlap
            offset_x = np.random.uniform(-max_offset, max_offset)
            offset_y = np.random.uniform(-max_offset, max_offset)
            c2_x_prop = c1_x + offset_x
            c2_y_prop = c1_y + offset_y
            
            c2_x = np.clip(c2_x_prop, half_dim2, W - half_dim2)
            c2_y = np.clip(c2_y_prop, half_dim2, H - half_dim2)
            
            # Get affine matrices from the safe boxes
            M1 = get_affine_transform_from_box(c1_x, c1_y, s1, angle1, output_size)
            M2 = get_affine_transform_from_box(c2_x, c2_y, s2, angle2, output_size)

            dsize = (output_size, output_size)
            imgs1[k] = cv2.warpAffine(img1_full, M1, dsize, flags=cv2.INTER_LINEAR, borderValue=0)
            imgs2[k] = cv2.warpAffine(img2_full, M2, dsize, flags=cv2.INTER_LINEAR, borderValue=0)
            obj1[k] = cv2.warpAffine(obj_full, M1, dsize, flags=cv2.INTER_LINEAR, borderValue=0)
            obj2[k] = cv2.warpAffine(obj_full, M2, dsize, flags=cv2.INTER_LINEAR, borderValue=0)
            residual1[k] = cv2.warpAffine(residual1_full, M1, dsize, flags=cv2.INTER_NEAREST, borderValue=np.nan)
            residual2[k] = cv2.warpAffine(residual2_full, M2, dsize, flags=cv2.INTER_NEAREST, borderValue=np.nan)
            local1_k = cv2.warpAffine(local, M1, dsize, flags=cv2.INTER_NEAREST, borderValue=-1)
            local2_k = cv2.warpAffine(local, M2, dsize, flags=cv2.INTER_NEAREST, borderValue=-1)
        else:
            # --- AXIS-ALIGNED CROP PATH (Original Logic) ---
            bbox1, bbox2 = get_random_overlapping_crops(H, W, min_crop_side)
            x1, y1, s1 = bbox1
            x2, y2, s2 = bbox2
            
            dsize = (output_size, output_size)
            imgs1[k] = cv2.resize(img1_full[y1:y1+s1, x1:x1+s1, :], dsize, interpolation=cv2.INTER_LINEAR)
            imgs2[k] = cv2.resize(img2_full[y2:y2+s2, x2:x2+s2, :], dsize, interpolation=cv2.INTER_LINEAR)
            obj1[k] = cv2.resize(obj_full[y1:y1+s1, x1:x1+s1, :], dsize, interpolation=cv2.INTER_LINEAR)
            obj2[k] = cv2.resize(obj_full[y2:y2+s2, x2:x2+s2, :], dsize, interpolation=cv2.INTER_LINEAR)
            residual1[k] = cv2.resize(residual1_full[y1:y1+s1, x1:x1+s1], dsize, interpolation=cv2.INTER_NEAREST)
            residual2[k] = cv2.resize(residual2_full[y2:y2+s2, x2:x2+s2], dsize, interpolation=cv2.INTER_NEAREST)
            local1_k = cv2.resize(local[y1:y1+s1, x1:x1+s1, :], dsize, interpolation=cv2.INTER_NEAREST)
            local2_k = cv2.resize(local[y2:y2+s2, x2:x2+s2, :], dsize, interpolation=cv2.INTER_NEAREST)

        # --- ROBUST CORRESPONDENCE CALCULATION (Unchanged) ---
        map1 = {}
        valid_mask1 = local1_k[:, :, 0] != -1
        for y_crop, x_crop in np.argwhere(valid_mask1):
            y_orig, x_orig = local1_k[y_crop, x_crop]
            key = (int(y_orig), int(x_orig))
            if key not in map1: map1[key] = []
            map1[key].append((y_crop, x_crop))

        map2 = {}
        valid_mask2 = local2_k[:, :, 0] != -1
        for y_crop, x_crop in np.argwhere(valid_mask2):
            y_orig, x_orig = local2_k[y_crop, x_crop]
            key = (int(y_orig), int(x_orig))
            if key not in map2: map2[key] = []
            map2[key].append((y_crop, x_crop))

        common_coords_set = set(map1.keys()) & set(map2.keys())

        if not common_coords_set:
            corr_idxs1_list.append(np.zeros((10000, 2), dtype=np.float32))
            corr_idxs2_list.append(np.zeros((10000, 2), dtype=np.float32))
            continue

        corr_list1, corr_list2, residual_list = [], [], []
        for key in common_coords_set:
            y_crop1, x_crop1 = map1[key][0]
            y_crop2, x_crop2 = map2[key][0]
            res_val = residual1[k, y_crop1, x_crop1]
            if not np.isnan(res_val):
                corr_list1.append([y_crop1, x_crop1])
                corr_list2.append([y_crop2, x_crop2])
                residual_list.append(res_val)

        if not corr_list1:
            corr_idxs1_list.append(np.zeros((10000, 2), dtype=np.float32))
            corr_idxs2_list.append(np.zeros((10000, 2), dtype=np.float32))
            continue

        coords1_2d = np.array(corr_list1)
        coords2_2d = np.array(corr_list2)
        residual_common = np.array(residual_list)

        sorted_idx = np.argsort(residual_common)
        coords1_2d = coords1_2d[sorted_idx]
        coords2_2d = coords2_2d[sorted_idx]
        
        num_points = len(coords1_2d)
        sample_indices = np.random.choice(np.arange(num_points), size=10000, replace=True)
        
        selected_coords1 = coords1_2d[sample_indices]
        selected_coords2 = coords2_2d[sample_indices]
        
        corr1 = selected_coords1.astype(np.float32) / downsample_ratio
        corr2 = selected_coords2.astype(np.float32) / downsample_ratio

        corr_idxs1_list.append(np.clip(corr1, a_min=0., a_max=(output_size - 1) / downsample_ratio))
        corr_idxs2_list.append(np.clip(corr2, a_min=0., a_max=(output_size - 1) / downsample_ratio))

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
        self.img_size = self.database[self.database_keys[0]]['images']['image_0'][:].shape[0]
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
                transforms.RandomApply([transforms.ColorJitter(.4,.4,.4,.1)],p=.7),
                transforms.RandomInvert(p=.2),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        else:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
                ])
        
        self.clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    
    def __len__(self):
        return self.dataset_num
    
    def __getitem__(self, index):
        seed = index * self.world_size + self.rank
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        key = self.database_keys[index]
        img_num = len(self.database[key]['images'])
        idx1,idx2 = np.random.choice(img_num,2)

        image_1_full = self.database[key]['images'][f"image_{idx1}"][:]
        image_2_full = self.database[key]['images'][f"image_{idx2}"][:]
        obj_full = centerize_obj(self.database[key]['obj'][:])
        residual_1_full = self.database[key]['residuals'][f"residual_{idx1}"][:]
        residual_2_full = self.database[key]['residuals'][f"residual_{idx1}"][:]
        image_1_full = self.clahe.apply(image_1_full)
        image_2_full = self.clahe.apply(image_2_full)
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
                          downsample_ratio=self.DOWNSAMPLE,
                          p_rotate=0.5,
                          max_angle_deg=180.0)
        print(f"img:{imgs1.shape}\t{imgs1.min()}\t{imgs1.max()}")
        print(f"obj:{obj1.shape}\t{obj1.min()}\t{obj1.max()}")
        print(f"res:{residual1.shape}\t{np.nanmean(residual1)}\t{(~np.isnan(residual1)).sum()}")
        print(f"ovl:{overlaps_1.shape}\t{overlaps_1[0][:10]}")


        # imgs1 = torch.from_numpy(np.stack([image_1_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size] for tl in windows],axis=0)).permute(0,3,1,2).to(torch.float32) # B,3,H,W
        # imgs2 = torch.from_numpy(np.stack([image_2_full[tl[0]:tl[0] + self.input_size,tl[1]:tl[1] + self.input_size] for tl in windows],axis=0)).permute(0,3,1,2).to(torch.float32)
        # imgs1 = self.transform(imgs1)
        # imgs2 = self.transform(imgs2)
        imgs1 = torch.stack([self.transform(img) for img in imgs1],dim=0)
        imgs2 = torch.stack([self.transform(img) for img in imgs2],dim=0)


        obj1 = torch.from_numpy(np.stack([downsample(obj,self.DOWNSAMPLE) for obj in obj1],axis=0)).to(torch.float32)
        obj2 = torch.from_numpy(np.stack([downsample(obj,self.DOWNSAMPLE) for obj in obj2],axis=0)).to(torch.float32)

        # test_obj1 = sample_bilinear(obj1,overlaps_1)
        # test_obj2 = sample_bilinear(obj2,overlaps_2)
        # obj_dis = torch.norm(test_obj1[:,:2] - test_obj2[:,:2],dim=-1).reshape(-1)
        # print(f"dis:{obj_dis.min().item()} \t {obj_dis.max().item()} \t {obj_dis.mean().item()} \t {obj_dis.median().item()}\n")

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