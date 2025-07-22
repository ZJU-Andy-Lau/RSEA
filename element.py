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
from criterion import CriterionTrainOneImg,CriterionTrainElement,CriterionTrainGrid
import torch.nn.functional as F
from orthorectify import orthorectify_image
import rasterio
from scipy.interpolate import RegularGridInterpolator
from copy import deepcopy
from torchvision import transforms
import kornia.augmentation as K
from torch_kdtree import build_kd_tree
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
        self.img_raw = img_raw # cv2.imread(options.img_path,cv2.IMREAD_GRAYSCALE)
        self._log(img_raw.shape,dem.shape,local_raw.shape)
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
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.RandomApply([transforms.ColorJitter(.4,.4,.4,.4)],p=.7),
        #     transforms.RandomInvert(p=.3),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #     ])
        self.transform = nn.Sequential(
            K.ColorJitter(
                brightness=0.4,
                contrast=0.4,
                saturation=0.4,
                hue=0.4,
                p=0.7
            ),
            K.RandomInvert(p=0.3),
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            )
        )
        # self.encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(options.encoder_path)).items()})
        self.encoder = encoder
        self.encoder.eval()
        self.mapper = Decoder(in_channels=self.encoder.output_channels,block_num=options.mapper_blocks_num)
        self.use_gpu = options.use_gpu
        self.output_path = output_path
        
        
        if self.use_gpu:
            self.rpc.to_gpu(self.device)
            self.encoder.to(self.device)
            self.mapper.to(self.device)
        
        if options.crop_step > 0:
            crop_step = options.crop_step
        else:
            crop_step = int(np.sqrt((self.H - options.crop_size) * (self.W - options.crop_size) / 150.))

        self.crop_imgs_NHWC,self.crop_locals_NHW2,self.crop_dems_NHW = self.__crop_img__(options.crop_size,crop_step)
        self.crop_img_num = len(self.crop_imgs_NHWC)
        self.SAMPLE_FACTOR = self.encoder.SAMPLE_FACTOR
        self.buffer,self.kd_tree = self.__extract_features__()
        self.map_coeffs = self.__calculate_map_coeffs__()
        self.ransac_threshold = self.options.ransac_threshold
        self.batch_num = int(np.ceil(self.patch_num / self.options.patches_per_batch))
        self.vis_patches_idx = torch.randperm(self.patch_num)[:100000]
        self.vis_patches_locs = []

        self.correspondences = {
            "locals":[],
            "targets":[]
        }

        self.af_trans = np.array([
            [1.,0.,0.],
            [0.,1.,0.]
        ])
        self.applied_trans = np.array([
            [1.,0.,0.],
            [0.,1.,0.]
        ])

        self._log(f"===========================Element {self.id} Initiated===========================")
        self._log(f"img size:{img_raw.shape}")
        self._log(f"top_left_linesamp:{top_left_linesamp}")
        self._log("=================================================================================")
    
    def _log(self, *args, **kwargs):
        if self.verbose:
            self._log("[Element]:", *args, **kwargs)

    def __crop_img__(self,crop_size = 256,step = 256 // 2,random_ratio = 1.):

        self._log("cropping image")
        H, W = self.img_raw.shape[:2]
        cut_number = 0
        row_num = 0
        col_num = 0
        crop_imgs = []
        crop_locals = []
        crop_dems = []

        if self.verbose > 0:
            pbar = tqdm(total=int((H - crop_size ) / step + 1) * int((W - crop_size) / step + 1))

        for row in range(0, H - crop_size, step):
            for col in range(0, W - crop_size, step):
                if row + crop_size + step > H:
                    if col + crop_size > W:
                        row_start,row_end,col_start,col_end = H - crop_size, H, W - crop_size, W
                    else:
                        row_start,row_end,col_start,col_end = H - crop_size, H, col, col+crop_size
                else:
                    if col + crop_size + step > W:
                        row_start,row_end,col_start,col_end = row, row+crop_size, W - crop_size, W
                    else:
                        row_start,row_end,col_start,col_end = row, row+crop_size, col, col+crop_size
                if row_num % 2 == 1:
                    col_start,col_end = W - col_end,W - col_start
                if col_num % 2 == 1:
                    row_start,row_end = H - row_end,H - row_start

                crop_imgs.append(self.img_raw[row_start:row_end,col_start:col_end])
                crop_locals.append(self.local_raw[row_start:row_end,col_start:col_end])
                crop_dems.append(self.dem[row_start:row_end,col_start:col_end])

                cut_number += 1
                col_num += 1
                if self.verbose > 0:
                    pbar.update(1)
            row_num += 1
            col_num -= 1

        random_num = int(cut_number * random_ratio)

        for i in range(random_num):
            col = np.random.randint(0,W - crop_size)
            row = np.random.randint(0,H - crop_size)
            crop_imgs.append(self.img_raw[row:row + crop_size,col:col + crop_size])
            crop_locals.append(self.local_raw[row:row + crop_size,col:col + crop_size])
            crop_dems.append(self.dem[row:row + crop_size,col:col + crop_size])
        
        crop_imgs = np.stack(crop_imgs)
        crop_locals = np.stack(crop_locals)
        crop_dems = np.stack(crop_dems)

        return crop_imgs,crop_locals,crop_dems

    @torch.no_grad()
    def __extract_features__(self) -> Dict[str,torch.Tensor]:
        
        self._log("Extracting features")
        
        start_time = time.perf_counter()

        self._log("---Transform input images")
        # if self.verbose > 0:
        #     imgs_NHW = torch.stack([self.transform(img) for img in tqdm(self.crop_imgs_NHW)]) # N,H,W
        # else:
        #     imgs_NHW = torch.stack([self.transform(img) for img in self.crop_imgs_NHW])
        imgs_NCHW = torch.from_numpy(self.crop_imgs_NHWC).permute(0,3,1,2)
        imgs_NCHW = imgs_NCHW.float() / 255.0
        imgs_NCHW = imgs_NCHW.to(self.device)
        self.transform = self.transform.to(self.device)
        with torch.no_grad():
            batch_num = int(np.ceil(imgs_NCHW.shape[0] / self.options.batch_size))
            imgs_NCHW = [self.transform(imgs_NCHW[b * self.options.batch_size : (b+1) * self.options.batch_size]) for b in range(batch_num)]
            imgs_NCHW = torch.concatenate(imgs_NCHW,dim=0)

        locals_NHW2= torch.from_numpy(self.crop_locals_NHW2)
        self._log("---Downsample locals")
        locals_Nhw2 = downsample(locals_NHW2,self.encoder.SAMPLE_FACTOR,use_cuda=True,show_detail=bool(self.verbose),mode='avg')
        dems_NHW = torch.from_numpy(self.crop_dems_NHW)
        self._log("---Downsample DEM")
        dems_Nhw = downsample(dems_NHW,self.encoder.SAMPLE_FACTOR,use_cuda=True,show_detail=bool(self.verbose),mode='avg')

        total_patch_num = locals_Nhw2.shape[0] * locals_Nhw2.shape[1] * locals_Nhw2.shape[2]
        select_ratio = min(1. * self.options.max_buffer_size / total_patch_num,1.)
        self._log("select_ratio:",select_ratio)
        # avg = nn.AvgPool2d(self.SAMPLE_FACTOR,self.SAMPLE_FACTOR)
        self.encoder.eval().to(self.device)

        # if self.use_gpu:
        #     imgs_NHW = imgs_NHW.to(self.device)
        #     locals_Nhw2 = locals_Nhw2.to(self.device)
        #     dems_Nhw = dems_Nhw.to(self.device)
        
        batch_num = int(np.ceil(self.crop_img_num / self.options.batch_size))

        features_PD = []
        confs_P1 = []
        locals_P2 = []
        dems_P1 = []
        
        

        if self.verbose > 0:
            pbar = tqdm(total=batch_num)
        for batch_idx in range(batch_num):
            batch_imgs = imgs_NCHW[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].to(self.device)
            batch_locals = locals_Nhw2[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].to(self.device).flatten(0,2)
            batch_dems = dems_Nhw[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].to(self.device).flatten(0,2)

            feat,conf = self.encoder(batch_imgs) # B,D,H,W

            feat = feat.permute(0,2,3,1).flatten(0,2)
            conf = conf.squeeze().flatten(0,2)
            valid_mask = conf > self.options.conf_threshold
            select_idxs = torch.randperm(valid_mask.sum())[:int(select_ratio * len(conf))]  
            

            features_PD.append(feat[valid_mask][select_idxs])
            confs_P1.append(conf[valid_mask][select_idxs])
            locals_P2.append(batch_locals[valid_mask][select_idxs])
            dems_P1.append(batch_dems[valid_mask][select_idxs])
            if self.verbose > 0:
                pbar.update(1)


        features_PD = torch.cat(features_PD,dim=0)
        confs_P1 = torch.cat(confs_P1,dim=0)
        locals_P2 = torch.cat(locals_P2,dim=0)
        dems_P1 = torch.cat(dems_P1,dim=0)
        lats,lons = self.rpc.RPC_PHOTO2OBJ(locals_P2[:,1],locals_P2[:,0],dems_P1)
        xy = project_mercator(torch.stack([lats,lons],dim=-1))[:,[1,0]]
        objs_P3 = torch.cat([xy,dems_P1.unsqueeze(-1)],dim=-1)
        
        buffer = {
            'features':features_PD,
            'locals':locals_P2,
            'confs':confs_P1,
            'objs':objs_P3
        }
        self.patch_num = len(features_PD)

        kd_tree = build_kd_tree(locals_P2,device=self.device)

        self._log(f"Extract features done in {time.perf_counter() - start_time} seconds \t {self.patch_num} patches in total")
        
        return buffer,kd_tree
    
    def __calculate_map_coeffs__(self):
        map_coeffs = {
            'x':get_map_coef(self.buffer['objs'][:,0].cpu().numpy()),
            'y':get_map_coef(self.buffer['objs'][:,1].cpu().numpy()),
            'z':get_map_coef(self.buffer['objs'][:,2].cpu().numpy())
        }
        return map_coeffs

    def warp_by_poly(self,raw,coefs):
        x = apply_polynomial(raw[:,0],coefs['x'])
        y = apply_polynomial(raw[:,1],coefs['y'])
        h = apply_polynomial(raw[:,2],coefs['h'])
        warped = torch.stack([x,y,h],dim=-1)
        return warped

    def clear_transfrom(self):
        self.af_trans = np.array([
            [1.,0.,0.],
            [0.,1.,0.]
        ])
        
    def train_mapper(self):
        patches_per_batch = self.options.patches_per_batch // 4 * 4
        optimizer = AdamW(self.mapper.parameters(),lr=self.options.element_train_lr_max)
        scheduler = MultiStageOneCycleLR(optimizer = optimizer,
                                        max_lr = self.options.element_train_lr_max,
                                        min_lr= self.options.element_train_lr_min,
                                        n_epochs_per_stage = self.options.element_training_iters,
                                        steps_per_epoch = 1,
                                        pct_start = self.options.element_warmup_iters / self.options.element_training_iters,
                                        summit_hold = self.options.element_summit_hold_iters / self.options.element_training_iters,
                                        #gamma = self.options.lr_decay_per_100_epochs ** (1. / 100.),
                                        cooldown = 0.0
                                        )
        criterion = CriterionTrainElement()
        
        self.mapper.train()
        if self.use_gpu:
            self.mapper.to(self.device)

        min_photo_loss = 1e9
        best_mapper_state_dict = None

        self._log("Training Mapper Start")
        start_time = time.perf_counter()

        patch_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.patch_feature_channels,self.patch_num * 5,1)),dim=1).to(self.buffer['features'].device)
        global_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.global_feature_channels,self.patch_num * 5,1)),dim=1).to(self.buffer['features'].device)
        patch_noise_amp = torch.rand(1,1,self.patch_num * 5,1,device=patch_noise_buffer.device,dtype=patch_noise_buffer.dtype) * .1 + .1
        global_noise_amp = .5
        patch_noise_buffer = patch_noise_buffer * patch_noise_amp
        global_noise_buffer = global_noise_buffer * global_noise_amp

        total_loss = 0
        total_loss_obj = 0
        total_loss_height = 0
        total_loss_photo = 0
        total_reg = 0
        count = 0

        if self.verbose > 0:
            pbar = tqdm(total=self.options.element_training_iters)

        for iter_idx in range(self.options.element_training_iters):
            self.mapper.train()
            optimizer.zero_grad()
            noise_idx = torch.randperm(self.patch_num * 5)[:self.options.patches_per_batch]
            sample_linesamps = torch.stack([torch.clip(torch.randint(int(self.top_left_linesamp[0]) - 5,int(self.top_left_linesamp[0]) + self.H + 5,(patches_per_batch // 4,)),
                                                           min=int(self.top_left_linesamp[0]),max=int(self.top_left_linesamp[0]) + self.H - 1),
                                                torch.clip(torch.randint(int(self.top_left_linesamp[1]) - 5,int(self.top_left_linesamp[1]) + self.W + 5,(patches_per_batch // 4,)),
                                                           min=int(self.top_left_linesamp[1]),max=int(self.top_left_linesamp[1]) + self.W - 1)],
                                                dim=-1).to(dtype=self.buffer['locals'].dtype,device=self.buffer['locals'].device)
            sample_linesamps = torch.concatenate([sample_linesamps,
                                                    torch.stack([2 * int(self.top_left_linesamp[0]) + self.H - 1 - sample_linesamps[:,0],2 * int(self.top_left_linesamp[1]) + self.W - 1 - sample_linesamps[:,1]],dim=-1),
                                                    torch.stack([2 * int(self.top_left_linesamp[0]) + self.H - 1 - sample_linesamps[:,0],sample_linesamps[:,1]],dim=-1),
                                                    torch.stack([sample_linesamps[:,0],2 * int(self.top_left_linesamp[1]) + self.W - 1 - sample_linesamps[:,1]],dim=-1)],
                                                    dim=0)
            dists,idxs = self.kd_tree.query(sample_linesamps,nr_nns_searches=4)
            valid_mask = dists.max(dim=1).values < 256
            dists = 1. / (dists[valid_mask] + 1e-6)
            idxs = idxs[valid_mask]
            # noise_idx = noise_idx[valid_mask]
            dists = dists / torch.mean(dists,dim=-1,keepdim=True)
            features_pD = self.buffer['features'][idxs].contiguous()
            confs_p1 = self.buffer['confs'][idxs].contiguous()
            objs_p3 = self.buffer['objs'][idxs].contiguous()
            features_pD = features_pD * dists.unsqueeze(-1)
            confs_p1 = confs_p1 * dists
            objs_p3 = objs_p3 * dists.unsqueeze(-1)
            features_pD = torch.mean(features_pD,dim=1).to(torch.float32)
            confs_p1 = torch.mean(confs_p1,dim=1).to(torch.float32)
            objs_p3 = torch.mean(objs_p3,dim=1).to(torch.float32)
            locals_p2 = sample_linesamps[valid_mask]

            features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
            patch_feature_noise = patch_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:].contiguous()
            global_feature_noise = global_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:].contiguous()
            features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] = F.normalize(features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] + patch_feature_noise,dim=1)
            features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] = F.normalize(features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] + global_feature_noise,dim=1)
            
            output_13p1 = self.mapper(features_1Dp1)
            output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
            xyh_pred_p3 = self.warp_by_poly(output_p3,self.map_coeffs)
            
            loss,loss_obj,loss_height,loss_photo,loss_bias,loss_reg = criterion(iter_idx,self.options.element_training_iters,xyh_pred_p3,confs_p1,locals_p2,objs_p3,self.rpc)
            loss.backward()
            
            total_loss += loss.item()
            total_loss_obj += loss_obj.item()
            total_loss_photo += loss_photo.item()
            total_loss_height += loss_height.item()
            total_reg += loss_reg
            count += 1

            if self.verbose > 0:
                pbar.update(1)
                pbar.set_postfix({
                    'obj':f'{loss_obj.item():.2f}',
                    'photo':f'{loss_photo.item():.2f}',
                    'reg':f'{loss_reg:.2f}',
                    'min':f'{min_photo_loss:.2f}'
                })

            optimizer.step()
            scheduler.step()
            if (iter_idx + 1) % 10 == 0:
                total_loss /= count
                total_loss_obj /= count
                total_loss_height /= count
                total_loss_photo /= count
                total_reg /= count
                
                # cost_time = int(time.perf_counter() - start_time)
                # self._log(f"\n ============= iter:{iter_idx + 1} \t total_loss:{total_loss:.2f} \t total_loss_obj:{total_loss_obj:.2f} \t total_loss_photo:{total_loss_photo:.2f} \t total_loss_real:{total_loss_photo_real:.2f} \t total_loss_height:{total_loss_height:.2f} \t total_loss_reg:{total_reg:.2f} \t time:{cost_time}s \n")
                if total_loss_photo < min_photo_loss:
                    min_photo_loss = total_loss_photo
                    best_mapper_state_dict = self.mapper.state_dict()
                elif total_loss_photo > min_photo_loss * 2:
                    self.mapper.load_state_dict(best_mapper_state_dict)

                total_loss = 0
                total_loss_obj = 0
                total_loss_height = 0
                total_loss_photo = 0
                total_reg = 0
                count = 0
        
        self.mapper.load_state_dict(best_mapper_state_dict)
        self.ransac_threshold = min_photo_loss + 5
        self._log(f"Training Mapper Done in {time.perf_counter() - start_time} seconds")

    
    def finetune_mapper(self):
        patches_per_batch = self.options.patches_per_batch // 4 * 4
        optimizer = AdamW(self.mapper.parameters(),lr=self.options.element_finetune_lr_max)
        scheduler = MultiStageOneCycleLR(optimizer = optimizer,
                                        max_lr = self.options.element_finetune_lr_max,
                                        min_lr= self.options.element_finetune_lr_min,
                                        n_epochs_per_stage = self.options.element_finetune_iters,
                                        steps_per_epoch = 1,
                                        pct_start = self.options.finetune_warmup_iters / self.options.element_finetune_iters,
                                        summit_hold = self.options.finetune_summit_hold_iters / self.options.element_finetune_iters,
                                        #gamma = self.options.lr_decay_per_100_epochs ** (1. / 100.),
                                        cooldown = 0.0
                                        )
        criterion = CriterionTrainElement()
        
        self.mapper.train()
        if self.use_gpu:
            self.mapper.to(self.device)

        min_photo_loss = 1e9
        best_mapper_state_dict = None

        self._log("Finetune Mapper Start")

        start_time = time.perf_counter()

        patch_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.patch_feature_channels,self.patch_num * 5,1)),dim=1).to(self.buffer['features'].device)
        global_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.global_feature_channels,self.patch_num * 5,1)),dim=1).to(self.buffer['features'].device)
        patch_noise_amp = torch.rand(1,1,self.patch_num * 5,1,device=patch_noise_buffer.device,dtype=patch_noise_buffer.dtype) * .1 + .1
        global_noise_amp = .5
        patch_noise_buffer = patch_noise_buffer * patch_noise_amp
        global_noise_buffer = global_noise_buffer * global_noise_amp

        total_loss = 0
        total_loss_obj = 0
        total_loss_height = 0
        total_loss_photo = 0
        total_reg = 0
        count = 0

        if self.verbose > 0:
            pbar = tqdm(total=self.options.element_training_iters)

        for iter_idx in range(self.options.element_training_iters):
            self.mapper.train()
            optimizer.zero_grad()
            noise_idx = torch.randperm(self.patch_num * 5)[:self.options.patches_per_batch]
            sample_linesamps = torch.stack([torch.clip(torch.randint(int(self.top_left_linesamp[0]) - 5,int(self.top_left_linesamp[0]) + self.H + 5,(patches_per_batch // 4,)),
                                                           min=int(self.top_left_linesamp[0]),max=int(self.top_left_linesamp[0]) + self.H - 1),
                                                torch.clip(torch.randint(int(self.top_left_linesamp[1]) - 5,int(self.top_left_linesamp[1]) + self.W + 5,(patches_per_batch // 4,)),
                                                           min=int(self.top_left_linesamp[1]),max=int(self.top_left_linesamp[1]) + self.W - 1)],
                                                dim=-1).to(dtype=self.buffer['locals'].dtype,device=self.buffer['locals'].device)
            sample_linesamps = torch.concatenate([sample_linesamps,
                                                    torch.stack([2 * int(self.top_left_linesamp[0]) + self.H - 1 - sample_linesamps[:,0],2 * int(self.top_left_linesamp[1]) + self.W - 1 - sample_linesamps[:,1]],dim=-1),
                                                    torch.stack([2 * int(self.top_left_linesamp[0]) + self.H - 1 - sample_linesamps[:,0],sample_linesamps[:,1]],dim=-1),
                                                    torch.stack([sample_linesamps[:,0],2 * int(self.top_left_linesamp[1]) + self.W - 1 - sample_linesamps[:,1]],dim=-1)],
                                                    dim=0)
            dists,idxs = self.kd_tree.query(sample_linesamps,nr_nns_searches=4)
            valid_mask = dists.max(dim=1).values < 256
            dists = 1. / (dists[valid_mask] + 1e-6)
            idxs = idxs[valid_mask]
            # noise_idx = noise_idx[valid_mask]
            dists = dists / torch.mean(dists,dim=-1,keepdim=True)
            features_pD = self.buffer['features'][idxs].contiguous()
            confs_p1 = self.buffer['confs'][idxs].contiguous()
            objs_p3 = self.buffer['objs'][idxs].contiguous()
            features_pD = features_pD * dists.unsqueeze(-1)
            confs_p1 = confs_p1 * dists
            objs_p3 = objs_p3 * dists
            features_pD = torch.mean(features_pD,dim=1).to(torch.float32)
            confs_p1 = torch.mean(confs_p1,dim=1).to(torch.float32)
            objs_p3 = torch.mean(objs_p3,dim=1).to(torch.float32)
            locals_p2 = sample_linesamps[valid_mask]

            features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
            patch_feature_noise = patch_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:].contiguous()
            global_feature_noise = global_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:].contiguous()
            features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] = F.normalize(features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] + patch_feature_noise,dim=1)
            features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] = F.normalize(features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] + global_feature_noise,dim=1)
            
            output_13p1 = self.mapper(features_1Dp1)
            output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
            xyh_pred_p3 = self.warp_by_poly(output_p3,self.map_coeffs)
            
            loss,loss_obj,loss_height,loss_photo,loss_bias,loss_reg = criterion(iter_idx,self.options.element_training_iters,xyh_pred_p3,confs_p1,locals_p2,objs_p3,self.rpc)
            loss.backward()
            
            total_loss += loss.item()
            total_loss_obj += loss_obj.item()
            total_loss_photo += loss_photo.item()
            total_loss_height += loss_height.item()
            total_reg += loss_reg
            count += 1

            if self.verbose > 0:
                pbar.update(1)
                pbar.set_postfix({
                    'obj':f'{loss_obj.item():.2f}',
                    'photo':f'{loss_photo.item():.2f}',
                    'reg':f'{loss_reg:.2f}',
                    'min':f'{min_photo_loss:.2f}'
                })

            optimizer.step()
            scheduler.step()
            if (iter_idx + 1) % 10 == 0:
                total_loss /= count
                total_loss_obj /= count
                total_loss_height /= count
                total_loss_photo /= count
                total_reg /= count
                
                # cost_time = int(time.perf_counter() - start_time)
                # self._log(f"\n ============= iter:{iter_idx + 1} \t total_loss:{total_loss:.2f} \t total_loss_obj:{total_loss_obj:.2f} \t total_loss_photo:{total_loss_photo:.2f} \t total_loss_real:{total_loss_photo_real:.2f} \t total_loss_height:{total_loss_height:.2f} \t total_loss_reg:{total_reg:.2f} \t time:{cost_time}s \n")
                if total_loss_photo < min_photo_loss:
                    min_photo_loss = total_loss_photo
                    best_mapper_state_dict = self.mapper.state_dict()
                elif total_loss_photo > min_photo_loss * 2:
                    self.mapper.load_state_dict(best_mapper_state_dict)

                total_loss = 0
                total_loss_obj = 0
                total_loss_height = 0
                total_loss_photo = 0
                total_reg = 0
                count = 0
        
        self.mapper.load_state_dict(best_mapper_state_dict)
        self.ransac_threshold = min_photo_loss + 5
        self._log(f"Finetune Mapper Done in {time.perf_counter() - start_time} seconds")

    def save_mapper(self):
        state_dict = {
            'mapper':self.mapper.state_dict(),
            'map_coeffs_x':torch.from_numpy(self.map_coeffs['x']),
            'map_coeffs_y':torch.from_numpy(self.map_coeffs['y']),
            'map_coeffs_h':torch.from_numpy(self.map_coeffs['h']),
        }
        torch.save(state_dict,os.path.join(self.output_path,'element.pth'))
    
    def load_mapper(self,path):
        state_dict = torch.load(path)
        self.mapper.load_state_dict(state_dict['mapper'])
        self.map_coeffs = {
            'x':state_dict['map_coeffs_x'].cpu().numpy(),
            'y':state_dict['map_coeffs_y'].cpu().numpy(),
            'h':state_dict['map_coeffs_h'].cpu().numpy(),
        }


    def __calculate_transform__(self,locals:np.ndarray,targets:np.ndarray) -> np.ndarray:
        '''
        locals: (line,samp)
        targets: (line,samp)
        '''
        dis = np.linalg.norm(locals + (np.mean(targets,axis=0)[None] - np.mean(locals,axis=0)[None]) - targets,axis=-1)
        self._log("mean_dis:",dis.mean())
        dis_valid_idx = dis < dis.mean() + dis.std()
        locals = locals[dis_valid_idx]
        targets = targets[dis_valid_idx]
        

        threshold = np.mean(np.linalg.norm(locals + (np.mean(targets,axis=0)[None] - np.mean(locals,axis=0)[None]) - targets,axis=-1))
        affine_matrix,inliers = cv2.estimateAffine2D(locals,targets,ransacReprojThreshold=threshold,maxIters=10000)
        inliers = inliers.reshape(-1).astype(bool)
        inlier_num = inliers.sum()
        self._log(f'{inlier_num} / {len(locals)}')

        return affine_matrix
    
    def __average_transforms__(self,transforms:np.ndarray) -> np.ndarray:

        avg_transform = np.mean(transforms,axis=0)
        
        return avg_transform
    
    def __compose_transforms__(self,transforms:List[np.ndarray]) -> np.ndarray:
        res = None
        for transform in transforms:
            if res is None:
                res = transform
                continue
            A_1 = res[:2,:2]
            A_2 = transform[:2,:2]
            t_1 = res[:2,2]
            t_2 = transform[:2,2]
            res = np.hstack([A_2 @ A_1, (A_2 @ t_1 + t_2).reshape(2,1)])
        return res


    @torch.no_grad()
    def get_transform(self,elements:List['Element']):
        start_time = time.perf_counter()
        for element in elements:
            if element.id == self.id:
                continue
            mapper = element.mapper
            mapper.eval()
            xyh_preds = []
            locals = []
            confs = []

            block_line_num = int(np.ceil(self.H / 256.))
            block_samp_num = int(np.ceil(self.W / 256.))
            br_line = self.top_left_linesamp[0] + self.H
            br_samp = self.top_left_linesamp[1] + self.W
            
            for block_line in range(block_line_num):
                for block_samp in range(block_samp_num):
                    tl_line = self.top_left_linesamp[0] + block_line * 256 + .5
                    tl_samp = self.top_left_linesamp[1] + block_samp * 256 + .5
                    sample_linesamps = torch.from_numpy(np.stack(np.meshgrid(np.arange(tl_line,min(tl_line + 256,br_line)),np.arange(tl_samp,min(tl_samp + 256,br_samp)),indexing='ij')
                                                                       ,axis=-1)).to(dtype=self.buffer['locals'].dtype,device=self.buffer['locals'].device).reshape(-1,2)
                    dists,idxs = self.kd_tree.query(sample_linesamps,nr_nns_searches=4)
                    valid_mask = (dists.max(dim=1).values < 64) 
                    if valid_mask.sum() == 0:
                        continue
                    dists = 1. / (dists[valid_mask] + 1e-6)
                    idxs = idxs[valid_mask]
                    dists = dists / torch.mean(dists,dim=-1,keepdim=True)
                    features_pD = self.buffer['features'][idxs].contiguous()
                    confs_p1 = self.buffer['confs'][idxs].contiguous()
                    dems_p1 = self.buffer['dems'][idxs].contiguous()
                    features_pD = features_pD * dists.unsqueeze(-1)
                    confs_p1 = confs_p1 * dists
                    dems_p1 = dems_p1 * dists
                    features_pD = torch.mean(features_pD,dim=1).to(torch.float32)
                    confs_p1 = torch.mean(confs_p1,dim=1).to(torch.float32)
                    dems_p1 = torch.mean(dems_p1,dim=1).to(torch.float32)
                    locals_p2 = sample_linesamps[valid_mask]

                    features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
                    output_13p1 = mapper(features_1Dp1)
                    output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
                    xyh_pred_p3 = self.warp_by_poly(output_p3,element.map_coeffs)
                    xyh_preds.append(xyh_pred_p3)
                    # yxh_pred_p3 = warp_by_extend(output_p3,element.extend)
                    # yxh_preds.append(yxh_pred_p3)
                    locals.append(locals_p2)
                    confs.append(confs_p1)


            # for batch_idx in range(self.batch_num):
            #     features_pD = self.buffer['features'][batch_idx * self.options.patches_per_batch : (batch_idx + 1) * self.options.patches_per_batch]
            #     locals_p2 = self.buffer['locals'][batch_idx * self.options.patches_per_batch : (batch_idx + 1) * self.options.patches_per_batch]
            #     confs_p1 = self.buffer['confs'][batch_idx * self.options.patches_per_batch : (batch_idx + 1) * self.options.patches_per_batch]
            #     # dems_p1 = self.buffer['dems'][batch_idx * self.options.patches_per_batch : (batch_idx + 1) * self.options.patches_per_batch]

            #     features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
            #     output_13p1 = mapper(features_1Dp1)
            #     output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
            #     yxh_pred_p3 = warp_by_extend(output_p3,element.extend)
            #     yxh_preds.append(yxh_pred_p3)
            #     locals.append(locals_p2)
            #     confs.append(confs_p1)

            xyh_pred_P3 = torch.cat(xyh_preds,dim=0)
            latlon_pred_P2 = mercator2lonlat(xyh_pred_P3[:,[1,0]])
            dems_pred_P1 = xyh_pred_P3[:,2]
            linesamp_pred_P2 = torch.stack(self.rpc.RPC_OBJ2PHOTO(latlon_pred_P2[:,0],latlon_pred_P2[:,1],dems_pred_P1),dim=-1).cpu().numpy()[:,[1,0]]
            locals_P2 = torch.cat(locals,dim=0).cpu().numpy()
            confs_P1 = torch.cat(confs,dim=0).cpu().numpy()

            margin_mask = (locals_P2[:,0] < self.top_left_linesamp[0] + self.H * 0.1) | (locals_P2[:,0] > self.top_left_linesamp[0] + self.H * 0.9) | (locals_P2[:,1] < self.top_left_linesamp[1] + self.W * 0.1) | (locals_P2[:,1] > self.top_left_linesamp[1] + self.W * 0.9)
            conf_mask = confs_P1 > self.options.conf_threshold
            valid_mask = (~margin_mask) & conf_mask
            self.correspondences['locals'].append(locals_P2[valid_mask])
            self.correspondences['targets'].append(linesamp_pred_P2[valid_mask])
            # transform = self.__calculate_transform__(locals_P2[~margin_mask],linesamp_pred_P2[~margin_mask],confs_P1[~margin_mask])
            # transforms.append(transform)
        locals = np.concatenate(self.correspondences['locals'])
        targets = np.concatenate(self.correspondences['targets'])
        new_trans = self.__calculate_transform__(locals,targets)

        #============================
        output_img = self.img_raw
        for local in locals:
            cv2.circle(output_img,(int(local[1] - self.top_left_linesamp[1]),int(local[0] - self.top_left_linesamp[0])),1,(0,255,0),-1)
        cv2.imwrite(f'./datasets/Rsea/d0_v0/vis_{self.id}.png',output_img)
        #============================
        
        self.af_trans = self.__average_transforms__(np.stack([new_trans,self.af_trans],axis=0))
        self.correspondences = {
            "locals":[],
            "targets":[]
        }
        self._log(f"Calculate transform of element {self.id} done in {time.perf_counter() - start_time} seconds")
        self._log(f"Origin Transform of element {self.id}:")
        self._log(self.af_trans)

    def update_obj(self):
        heights = self.buffer['objs'][:,2]
        locals = self.buffer['locals']
        lats,lons = self.rpc.RPC_PHOTO2OBJ(locals[:,1],locals[:,0],heights)
        xy = project_mercator(torch.stack([lats,lons],dim=-1))[:,[1,0]]
        self.buffer['objs'][:,:2] = xy


    def apply_transform(self):
        self.applied_trans = self.__compose_transforms__([self.applied_trans,self.af_trans])
        self._log(f"New Transform for element {self.id}: \n {self.af_trans}")
        self._log(f"Composed Transform for element {self.id}: \n {self.applied_trans}")
        self.rpc.Update_Adjust(torch.from_numpy(self.applied_trans))
        self.update_obj()
        self.__calculate_map_coeffs__()

    def output_ortho(self,output_path):
        orthorectify_image(self.img_raw,self.dem,self.rpc,output_path)
    
    def clear_buffer(self):
        del self.buffer
        del self.kd_tree
        self.buffer = None
        self.kd_tree = None

    def to_device(self,device):
        self.device = device
        self.encoder.to(device)
        self.mapper.to(device)
        self.rpc.to_gpu(device)
        if not self.buffer is None:
            for key in self.buffer.keys():
                self.buffer[key].to(device)
        