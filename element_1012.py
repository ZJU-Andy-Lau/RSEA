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
    def __init__(self,options,encoder:Encoder,img_raw:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch,id:int,output_path:str,top_left_linesamp:np.ndarray = None,local_raw:np.ndarray = None,device:str = None,verbose:int = 0):
        self.options = options
        self.id = id
        self.verbose = verbose
        if device is None:
            self.device = 'cuda'
        else:
            self.device = device
        self.img_raw = img_raw 
        cv2.imwrite(os.path.join(output_path,f'img_{id}.png'),img_raw)
        if top_left_linesamp is None:
            self.top_left_linesamp = np.array([0.,0.])
        else:
            self.top_left_linesamp = top_left_linesamp
        if local_raw is None:
            self.local_raw = get_coord_mat(self.img_raw.shape[0],self.img_raw.shape[1])
            self.local_raw += self.top_left_linesamp
        else:
            self.local_raw = local_raw
            self.top_left_linesamp = local_raw[0,0]
        self.dem = dem
        self.rpc = rpc
        self.H,self.W = self.img_raw.shape[:2]

        self.transform = nn.Sequential(
            K.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
                p=.3,
            ),
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
            K.RandomInvert(p=0.1),
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            ),
        )
        
        self.encoder = encoder.eval()
        self.mapper = Decoder(in_channels=self.encoder.output_channels,block_num=options.mapper_blocks_num)
        self.output_path = output_path
        
        self.rpc.to_gpu(self.device)
        self.encoder.to(self.device)
        self.mapper.to(self.device)
        
        self.crop_imgs_NHWC,self.crop_locals_NHW2,self.crop_dems_NHW = self.__crop_img__(options.crop_size)
        self.crop_img_num = len(self.crop_imgs_NHWC)
        self.SAMPLE_FACTOR = self.options.sample_factor
        self.buffer, self.point_base = self.__extract_features__()
        
        self._log(f"===========================Element {self.id} Initiated===========================")
        self._log(f"img size:{img_raw.shape}")
        self._log(f"top_left_linesamp:{top_left_linesamp}")
        self._log(f"Generated {self.buffer['features'].shape[0]} windows for training.")
        self._log("=================================================================================")
    
    def _log(self, *args, **kwargs):
        if self.verbose:
            print("[Element]:", *args, **kwargs)

    def __crop_img__(self, crop_size=1024, random_ratio=1., rotation_angle=10.):
        self._log("cropping image with new strategy")
        H, W = self.img_raw.shape[:2]
        
        crop_imgs = []
        crop_locals = []
        crop_dems = []

        # Part 1: 高效的均匀裁切，保证最少窗口覆盖全图
        self._log("--- Performing uniform cropping")
        num_steps_h = int(np.ceil(H / crop_size)) if H > crop_size else 1
        num_steps_w = int(np.ceil(W / crop_size)) if W > crop_size else 1
        y_starts = np.linspace(0, H - crop_size, num_steps_h, dtype=int)
        x_starts = np.linspace(0, W - crop_size, num_steps_w, dtype=int)
        for row in y_starts:
            for col in x_starts:
                crop_imgs.append(self.img_raw[row:row + crop_size, col:col + crop_size])
                crop_locals.append(self.local_raw[row:row + crop_size, col:col + crop_size])
                crop_dems.append(self.dem[row:row + crop_size, col:col + crop_size])
        n_uniform = len(crop_imgs)
        self._log(f"--- Generated {n_uniform} uniform crops")

        # Part 2: 高效且安全的随机旋转裁切
        n_random = int(n_uniform * random_ratio)
        if n_random > 0:
            self._log(f"--- Performing {n_random} random rotated cropping")
            half_diag = int(np.sqrt(2) * crop_size / 2) + 1
            safe_top, safe_left = half_diag, half_diag
            safe_bottom, safe_right = H - half_diag, W - half_diag
            if safe_bottom > safe_top and safe_right > safe_left:
                center_y = np.random.randint(safe_top, safe_bottom, n_random)
                center_x = np.random.randint(safe_left, safe_right, n_random)
                angles = np.random.uniform(-rotation_angle, rotation_angle, n_random)
                for cy, cx, angle in zip(center_y, center_x, angles):
                    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
                    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
                    rotated_img = cv2.warpAffine(self.img_raw, M, (W, H), flags=flags)
                    rotated_local = cv2.warpAffine(self.local_raw, M, (W, H), flags=flags)
                    rotated_dem = cv2.warpAffine(self.dem, M, (W, H), flags=flags)
                    tl_x, tl_y = cx - crop_size // 2, cy - crop_size // 2
                    crop_imgs.append(rotated_img[tl_y:tl_y + crop_size, tl_x:tl_x + crop_size])
                    crop_locals.append(rotated_local[tl_y:tl_y + crop_size, tl_x:tl_x + crop_size])
                    crop_dems.append(rotated_dem[tl_y:tl_y + crop_size, tl_x:tl_x + crop_size])

        crop_imgs = np.stack(crop_imgs)
        crop_locals = np.stack(crop_locals)
        crop_dems = np.stack(crop_dems)
        return crop_imgs, crop_locals, crop_dems

    @torch.no_grad()
    def __extract_features__(self):
        self._log("Extracting features")
        start_time = time.perf_counter()
        self._log("---Transform input images")
        
        imgs_NCHW = torch.from_numpy(self.crop_imgs_NHWC).permute(0,3,1,2).float() / 255.0
        self.transform = self.transform.to(self.device)
        
        transformed_imgs_list = []
        batch_num_transform = int(np.ceil(imgs_NCHW.shape[0] / self.options.batch_size))
        for b in range(batch_num_transform):
             transformed_imgs_list.append(self.transform(imgs_NCHW[b * self.options.batch_size : (b+1) * self.options.batch_size].to(self.device)))
        imgs_NCHW = torch.cat(transformed_imgs_list, dim=0)

        locals_NHW2 = torch.from_numpy(self.crop_locals_NHW2)
        self._log("---Downsample locals")
        locals_Nhw2 = downsample(locals_NHW2,self.SAMPLE_FACTOR,use_cuda=True,show_detail=bool(self.verbose),mode='avg',device=self.device)
        dems_NHW = torch.from_numpy(self.crop_dems_NHW)
        self._log("---Downsample DEM")
        dems_Nhw = downsample(dems_NHW,self.SAMPLE_FACTOR,use_cuda=True,show_detail=bool(self.verbose),mode='avg',device=self.device)
        
        self.encoder = self.encoder.eval().to(self.device)
        
        batch_num_extract = int(np.ceil(self.crop_img_num / self.options.batch_size))
        features_list, confs_list, locals_list, dems_list = [], [], [], []
        
        pbar_title = "---Extracting Features"
        pbar = tqdm(total=batch_num_extract, desc=pbar_title) if self.verbose > 0 else range(batch_num_extract)

        for batch_idx in pbar:
            batch_imgs = imgs_NCHW[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size]
            feat_bdhw, conf_b1hw = self.encoder(batch_imgs)
            features_list.append(feat_bdhw.cpu())
            confs_list.append(conf_b1hw.cpu())
            locals_list.append(locals_Nhw2[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].cpu())
            dems_list.append(dems_Nhw[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].cpu())
            if self.verbose > 0: pbar.update(1)
        if self.verbose > 0: pbar.close()

        features_BDhw = torch.cat(features_list, dim=0)
        confs_B1hw = torch.cat(confs_list, dim=0)
        locals_Bhw2 = torch.cat(locals_list, dim=0)
        dems_Bhw = torch.cat(dems_list, dim=0)

        self._log("---Calculating geographic coordinates for buffer")
        B, h, w, _ = locals_Bhw2.shape
        lats, lons = self.rpc.RPC_PHOTO2OBJ(locals_Bhw2[..., 1].flatten(), locals_Bhw2[..., 0].flatten(), dems_Bhw.flatten())
        xy = project_mercator(torch.stack([lats, lons], dim=-1))[:, [1, 0]]
        objs_Bhw3 = torch.cat([xy, dems_Bhw.flatten().unsqueeze(-1)], dim=-1).reshape(B, h, w, 3)

        buffer = {
            'features': features_BDhw,
            'confs': confs_B1hw,
            'locals': locals_Bhw2,
            'objs': objs_Bhw3
        }
        
        self._log(f"Extract features done in {time.perf_counter() - start_time:.2f} seconds")
        return buffer, None

    def clear_buffer(self):
        del self.buffer
        self.buffer = None
    
    def to_device(self,device):
        self.device = device
        self.encoder.to(device)
        self.mapper.to(device)
        self.rpc.to_gpu(device)
        if hasattr(self, 'buffer') and self.buffer is not None:
            for key in self.buffer.keys():
                self.buffer[key] = self.buffer[key].to(device)
