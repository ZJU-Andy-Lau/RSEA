import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model_new import Encoder,Decoder
import os
import cv2
from datetime import datetime,timedelta
import time
from utils import get_coord_mat,project_mercator,mercator2lonlat,downsample,bilinear_interpolate,apply_polynomial,get_map_coef

from rpc import RPCModelParameterTorch
from tqdm import tqdm,trange
from scheduler import MultiStageOneCycleLR
from torch.optim import AdamW,lr_scheduler
from criterion_1012 import CriterionTrainGrid
import torch.nn.functional as F
from orthorectify import orthorectify_image
import rasterio
from scipy.interpolate import RegularGridInterpolator
from copy import deepcopy
from torchvision import transforms
import kornia.augmentation as K
from matplotlib import pyplot as plt
import random
from typing import List,Dict

class Element():
    """
    Element类是Grid与RSImage之间的桥梁。
    它代表某一张特定影像在某一个特定Grid内的影像数据单元。
    主要职责是：裁切数据、提取特征，并为训练准备好格式化的数据Buffer。
    """
    def __init__(self,options,encoder:Encoder,img_raw:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch,id:int,output_path:str,top_left_linesamp:np.ndarray = None,local_raw:np.ndarray = None,device:str = None,verbose:int = 0):
        # --- 1. 初始化基本属性 ---
        self.options = options
        self.id = id
        self.verbose = verbose
        self.device = device if device is not None else 'cuda'
        
        self.img_raw = img_raw
        cv2.imwrite(os.path.join(output_path,f'img_{id}.png'),img_raw)
        
        if local_raw is None:
            self.local_raw = get_coord_mat(self.img_raw.shape[0],self.img_raw.shape[1])
            self.local_raw += top_left_linesamp if top_left_linesamp is not None else np.array([0.,0.])
        else:
            self.local_raw = local_raw
            
        self.dem = dem
        self.rpc = rpc
        self.H,self.W = self.img_raw.shape[:2]
        
        # --- 训练和验证的图像变换流程 ---
        self.train_transform = nn.Sequential(
            K.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=.3),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
            K.RandomInvert(p=0.1),
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            ),
        )
        self.val_transform = nn.Sequential(
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            ),
        )
        
        self.encoder = encoder.eval()
        self.output_path = output_path
        
        self.train_crop_imgs_NHWC, self.train_crop_locals_NHW2, self.train_crop_dems_NHW = self.__crop_img__(crop_size=self.options.crop_size)
        self.val_crop_imgs_NHWC, self.val_crop_locals_NHW2, self.val_crop_dems_NHW = self.__crop_validation_img__(crop_size=self.options.crop_size)

        self.SAMPLE_FACTOR = self.options.sample_factor
        self.buffer = self.__extract_and_unfold_patches__(self.train_crop_imgs_NHWC, self.train_crop_locals_NHW2, self.train_crop_dems_NHW, is_training=True)
        self.validation_buffer = self.__extract_and_unfold_patches__(self.val_crop_imgs_NHWC, self.val_crop_locals_NHW2, self.val_crop_dems_NHW, is_training=False)
        
        self._log(f"=========================== Element {self.id} 初始化完成 ===========================")
    
    def _log(self, *args, **kwargs):
        if self.verbose:
            print(f"[Element {self.id}]:", *args, **kwargs)

    def __crop_img__(self, crop_size=1024, rotation_angle=10.):
        self._log("正在为训练集裁切窗口...")
        H, W = self.img_raw.shape[:2]
        
        crop_imgs, crop_locals, crop_dems = [], [], []

        if H > crop_size and W > crop_size:
            y_starts = np.linspace(0, H - crop_size, self.options.crop_num_h, dtype=int)
            x_starts = np.linspace(0, W - crop_size, self.options.crop_num_w, dtype=int)

            for row in y_starts:
                for col in x_starts:
                    crop_imgs.append(self.img_raw[row:row + crop_size, col:col + crop_size])
                    crop_locals.append(self.local_raw[row:row + crop_size, col:col + crop_size])
                    crop_dems.append(self.dem[row:row + crop_size, col:col + crop_size])

        n_uniform = len(crop_imgs)
        n_random = n_uniform
        if n_random > 0:
            half_diag = int(np.sqrt(2) * crop_size / 2) + 1
            if H > 2 * half_diag and W > 2 * half_diag:
                safe_top, safe_left = half_diag, half_diag
                safe_bottom, safe_right = H - half_diag, W - half_diag
                center_y = np.random.randint(safe_top, safe_bottom, n_random)
                center_x = np.random.randint(safe_left, safe_right, n_random)
                angles = np.random.uniform(-rotation_angle, rotation_angle, n_random)

                for cy, cx, angle in zip(center_y, center_x, angles):
                    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
                    flags = cv2.INTER_LINEAR
                    
                    rotated_img = cv2.warpAffine(self.img_raw, M, (W, H), flags=flags, borderMode=cv2.BORDER_REFLECT_101)
                    rotated_local = cv2.warpAffine(self.local_raw, M, (W, H), flags=flags, borderMode=cv2.BORDER_REFLECT_101)
                    rotated_dem = cv2.warpAffine(self.dem, M, (W, H), flags=flags, borderMode=cv2.BORDER_REFLECT_101)
                    
                    tl_x, tl_y = cx - crop_size // 2, cy - crop_size // 2
                    
                    crop_imgs.append(rotated_img[tl_y:tl_y + crop_size, tl_x:tl_x + crop_size])
                    crop_locals.append(rotated_local[tl_y:tl_y + crop_size, tl_x:tl_x + crop_size])
                    crop_dems.append(rotated_dem[tl_y:tl_y + crop_size, tl_x:tl_x + crop_size])

        if not crop_imgs:
            return np.array([]), np.array([]), np.array([])

        return np.stack(crop_imgs), np.stack(crop_locals), np.stack(crop_dems)

    def __crop_validation_img__(self, crop_size=1024, num_val_windows=16):
        self._log(f"正在为验证集裁切 {num_val_windows} 个窗口...")
        H, W = self.img_raw.shape[:2]
        
        crop_imgs, crop_locals, crop_dems = [], [], []
        
        if H > crop_size and W > crop_size:
            for _ in range(num_val_windows):
                row = np.random.randint(0, H - crop_size)
                col = np.random.randint(0, W - crop_size)
                crop_imgs.append(self.img_raw[row:row + crop_size, col:col + crop_size])
                crop_locals.append(self.local_raw[row:row + crop_size, col:col + crop_size])
                crop_dems.append(self.dem[row:row + crop_size, col:col + crop_size])

        if not crop_imgs:
            return np.array([]), np.array([]), np.array([])
        
        return np.stack(crop_imgs), np.stack(crop_locals), np.stack(crop_dems)
    
    def __extract_and_unfold_patches__(self, crop_imgs_nhwc, crop_locals_nhw2, crop_dems_nhw, is_training: bool) -> Dict[str, torch.Tensor]:
        """
        [核心重构] 提取特征图，并立即使用unfold将其转换为patch格式的数据集。
        """
        if crop_imgs_nhwc.size == 0:
            return {}

        self._log(f"正在提取并构建 {crop_imgs_nhwc.shape[0]} 个窗口的Patch数据集... (模式: {'训练' if is_training else '验证'})")
        
        # --- 1. 特征提取 (与之前相同) ---
        imgs_nchw = torch.from_numpy(crop_imgs_nhwc).permute(0, 3, 1, 2).float() / 255.0
        
        transform_to_use = self.train_transform if is_training else self.val_transform
        transform_to_use.to(self.device)

        with torch.no_grad():
            batch_num = int(np.ceil(imgs_nchw.shape[0] / self.options.encoder_batch_size))
            imgs_nchw_aug = []
            for b in range(batch_num):
                batch = imgs_nchw[b * self.options.encoder_batch_size : (b+1) * self.options.encoder_batch_size].to(self.device)
                imgs_nchw_aug.append(transform_to_use(batch))
            imgs_nchw_aug = torch.cat(imgs_nchw_aug, dim=0)

        locals_nhw2 = torch.from_numpy(crop_locals_nhw2)
        locals_nhw2_down = downsample(locals_nhw2, self.SAMPLE_FACTOR, mode='avg')
        
        dems_nhw = torch.from_numpy(crop_dems_nhw)
        dems_nhw_down = downsample(dems_nhw, self.SAMPLE_FACTOR, mode='avg')

        self.encoder.to(self.device)
        features_list, confs_list = [], []
        with torch.no_grad():
            for b in range(batch_num):
                batch_imgs = imgs_nchw_aug[b * self.options.encoder_batch_size : (b+1) * self.options.encoder_batch_size]
                feat, conf = self.encoder(batch_imgs)
                features_list.append(feat)
                confs_list.append(conf)

        features_bdhw = torch.cat(features_list, dim=0)
        confs_b1hw = torch.cat(confs_list, dim=0)

        B, h, w, _ = locals_nhw2_down.shape
        lats, lons = self.rpc.RPC_PHOTO2OBJ(
            locals_nhw2_down[..., 1].flatten().to(self.device), 
            locals_nhw2_down[..., 0].flatten().to(self.device), 
            dems_nhw_down.flatten().to(self.device)
        )
        xy = project_mercator(torch.stack([lats, lons], dim=-1))[:, [1, 0]]
        
        objs_bhw3 = torch.cat([xy, dems_nhw_down.flatten().to(self.device).unsqueeze(-1)], dim=-1).reshape(B, h, w, 3)

        feature_maps = {
            'features': features_bdhw,
            'confs': confs_b1hw,
            'locals': locals_nhw2_down.to(self.device).permute(0, 3, 1, 2), # NCHW
            'objs': objs_bhw3.permute(0, 3, 1, 2) # NCHW
        }

        # --- 2. [核心新增] 使用unfold将特征图转换为Patch ---
        patch_h, patch_w, patch_stride = 16, 16, 8
        patch_buffer = {}
        
        for key, tensor in feature_maps.items():
            B, C, H, W = tensor.shape
            unfolded = F.unfold(tensor, kernel_size=(patch_h, patch_w), stride=patch_stride)
            
            num_patches_per_img = unfolded.shape[-1]
            
            # (B, C*k*k, L) -> (B, L, C, k, k) -> (B*L, C, k, k)
            unfolded = unfolded.permute(0, 2, 1).reshape(B * num_patches_per_img, C, patch_h, patch_w)
            patch_buffer[key] = unfolded
        
        self._log(f"Patch数据集构建完成, 共 {len(patch_buffer['features'])} 个Patches。")
        return patch_buffer

    def clear_buffer(self):
        if hasattr(self, 'buffer') and self.buffer:
            del self.buffer
        if hasattr(self, 'validation_buffer') and self.validation_buffer:
            del self.validation_buffer
        self.buffer = None
        self.validation_buffer = None

    def to_device(self,device):
        self.device = device
        self.rpc.to_gpu(device)
        self.train_transform.to(device)
        self.val_transform.to(device)
        
        if hasattr(self, 'buffer') and self.buffer:
            for key in self.buffer:
                self.buffer[key] = self.buffer[key].to(device)
        
        if hasattr(self, 'validation_buffer') and self.validation_buffer:
            for key in self.validation_buffer:
                self.validation_buffer[key] = self.validation_buffer[key].to(device)

