from ast import mod
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
from torch_kdtree import build_kd_tree
from matplotlib import pyplot as plt
import random
from typing import List,Dict

from rs_image import RSImage
from element import Element

class Grid():
    def __init__(self,options,encoder:Encoder,output_path:str,diag:np.ndarray = None,grid_path:str = None):

        self.options = options
        self.encoder = encoder
        self.mapper = Decoder(in_channels=self.encoder.output_channels,block_num=options.mapper_blocks_num)
        if diag is None and grid_path is None:
            raise ValueError("Grid loaded error: Neither diag nor grid path is given")
        if grid_path is None :
            self.diag = diag #[[x,y],[x,y]]
            self.map_coeffs = {
                'x':np.array([.6 * np.abs(diag[0,0] - diag[1,0]), .5 * (diag[0,0] + diag[1,0])]),
                'y':np.array([.6 * np.abs(diag[0,1] - diag[1,1]), .5 * (diag[0,1] + diag[1,1])]),
                'h':None
            }
        else:
            self.load_grid(grid_path)
        self.border = np.array([self.diag[:,0].min(),self.diag[:,1].min(),self.diag[:,0].max(),self.diag[:,1].max()]) #[min_x,min_y,max_x,max_y]
        print(f"\n Grid range: x: {self.border[0]:.2f} ~ {self.border[2]:.2f} \t y: {self.border[1]:.2f} ~ {self.border[3]:.2f}\n")
        self.output_path = output_path
        self.elements:List[Element] = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.SAMPLE_FACTOR = 16
        self.pred_resolution = .7
        self.vis_points_latlon = None
    
    def get_overlap_image(self,img:RSImage,mode='bbox'):
        corner_samplines = img.xy_to_sampline(np.array([self.diag[0],[self.diag[1,0],self.diag[0,1]],self.diag[1],[self.diag[0,0],self.diag[1,1]]])) # tl,tr,br,bl
        if mode == 'bbox':
            top = max(min(corner_samplines[0,1],corner_samplines[1,1]),0)
            bottom = min(max(corner_samplines[2,1],corner_samplines[3,1]),img.H-1)
            left = max(min(corner_samplines[0,0],corner_samplines[3,0]),0)
            right = min(max(corner_samplines[1,0],corner_samplines[2,0]),img.W-1)
            img_raw = img.get_image_by_sampline(np.array([left,top]),np.array([right,bottom]))
            dem = img.get_dem_by_sampline(np.array([left,top]),np.array([right,bottom]))
            return img_raw,dem,np.array([top,left]),np.array([bottom,right])
        elif mode == 'interpolate':
            img_raw,local_hw2 = img.resample_image_by_sampline(corner_samplines,
                                                            (int((self.border[2] - self.border[0]) / self.pred_resolution),
                                                            int((self.border[3] - self.border[1]) / self.pred_resolution)),
                                                            need_local=True)
            
            dem = img.resample_dem_by_sampline(corner_samplines,
                                                (int((self.border[2] - self.border[0]) / self.pred_resolution),
                                                 int((self.border[3] - self.border[1]) / self.pred_resolution)))

            return img_raw,dem,local_hw2
        else:
            raise ValueError("mode should either be 'bbox' or 'interpolate'")

    def get_height_map_coeffs(self):
        heights = []
        for element in self.elements:
            heights.append(element.buffer['objs'][:,2])
        heights = torch.concatenate(heights).cpu().numpy()
        self.map_coeffs['h'] = get_map_coef(heights)


    def add_img(self,img:RSImage,output_path:str,mapper_path:str = None,id:int = None):
        """
        添加训练数据
        """
        output_path = os.path.join(output_path,f'element_{id}')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        img_raw,dem,local_hw2 = self.get_overlap_image(img,mode='interpolate')

        new_element = Element(options = self.options,
                              encoder = self.encoder,
                              img_raw = img_raw,
                              dem = dem,
                              rpc = img.rpc,
                              id = len(self.elements) if id is None else id,
                              output_path = output_path,
                              local_raw = local_hw2)
        if mapper_path:
            new_element.load_mapper(mapper_path)

        self.elements.append(new_element)
        self.get_height_map_coeffs()
    
    def train_elements(self,save = True):
        for element in self.elements:
            element.train_mapper()
            if save:
                element.save_mapper()
    

    def adjust_elements(self,iter_num = 1):
        # for iter_idx in range(iter_num):
        #     # self.vis_match(iter_idx)
        #     for element in self.elements:
        #         element.get_transform(self.elements)
        #     self.centerize_transforms(self.elements)
        #     for element in self.elements:
        #         element.apply_transform()
        #     for element in self.elements:
        #         element.finetune_mapper()

        for element in self.elements:
            element.get_transform(self.elements)

        self.centerize_transforms(self.elements)

        for element in self.elements:
            element.apply_transform()

        # self.vis_match(iter_num)

    def __average_transforms__(self,transforms:np.ndarray) -> np.ndarray:
 
        avg_transform = np.mean(transforms,axis=0)
        
        return avg_transform

    def __inverse_transform__(self,transform:np.ndarray) -> np.ndarray:
        A = transform[:2,:2]
        t = transform[:2,2]
        A_inv = np.linalg.inv(A)
        t_inv = -A_inv @ t
        return np.hstack([A_inv,t_inv.reshape(2,1)])
    
    def centerize_transforms(self,elements:List[Element]):
        transforms = np.stack([element.af_trans for element in elements],axis=0)
        trans_avg = self.__average_transforms__(transforms)
        trans_avg_inv = self.__inverse_transform__(trans_avg)
        for element in elements:
            A = element.af_trans[:2,:2]
            t = element.af_trans[:2,2]
            new_A = A @ trans_avg_inv[:2,:2]
            new_t = t + A @ trans_avg_inv[:2,2]
            new_trans = np.hstack([new_A,new_t.reshape(2,1)])
            element.af_trans = new_trans        

    def warp_by_poly(self,raw,coefs):
        x = apply_polynomial(raw[:,0],coefs['x'])
        y = apply_polynomial(raw[:,1],coefs['y'])
        h = apply_polynomial(raw[:,2],coefs['h'])
        warped = torch.stack([x,y,h],dim=-1)
        return warped

    def train_mapper(self):
        start_time = time.perf_counter()
        max_patch_num = max(*[element.patch_num for element in self.elements],0)
        patches_per_batch = self.options.patches_per_batch // 4 * 4
        optimizer = AdamW(self.mapper.parameters(),lr=self.options.grid_train_lr_max)
        scheduler = MultiStageOneCycleLR(optimizer = optimizer,
                                        max_lr = self.options.grid_train_lr_max,
                                        min_lr = self.options.grid_train_lr_min,
                                        n_epochs_per_stage = self.options.grid_training_iters,
                                        steps_per_epoch = 1,
                                        pct_start = self.options.grid_warmup_iters / self.options.grid_training_iters,
                                        summit_hold = self.options.grid_summit_hold_iters / self.options.grid_training_iters,
                                        #gamma = self.options.lr_decay_per_100_epochs ** (1. / 100.),
                                        cooldown = 0.0
                                        )
        criterion = CriterionTrainGrid()
        self.mapper.train()
        if self.options.use_gpu:
            self.mapper.cuda()

        min_photo_loss = 1e8

        patch_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.patch_feature_channels,max_patch_num * 5,1)),dim=1).to(self.elements[0].buffer['features'].device)
        global_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.global_feature_channels,max_patch_num * 5,1)),dim=1).to(self.elements[0].buffer['features'].device)
        patch_noise_amp = torch.rand(1,1,max_patch_num * 5,1,device=patch_noise_buffer.device,dtype=patch_noise_buffer.dtype) * .1 + .1
        global_noise_amp = .5 
        patch_noise_buffer = patch_noise_buffer * patch_noise_amp
        global_noise_buffer = global_noise_buffer * global_noise_amp


        total_loss = 0
        total_loss_obj = 0
        total_loss_height = 0
        total_loss_photo = 0
        total_reg = 0
        count = 0
        no_update_count = 0
        
        pbar = tqdm(total=self.options.grid_training_iters * len(self.elements))

        for iter_idx in range(self.options.grid_training_iters):
            noise_idx = torch.randperm(max_patch_num * 5)[:patches_per_batch]
            optimizer.zero_grad()
            for element in self.elements:
                

                sample_linesamps = torch.stack([torch.clip(torch.randint(int(element.top_left_linesamp[0]) - 5,int(element.top_left_linesamp[0]) + element.H + 5,(patches_per_batch // 4,)),
                                                           min=int(element.top_left_linesamp[0]),max=int(element.top_left_linesamp[0]) + element.H - 1),
                                                torch.clip(torch.randint(int(element.top_left_linesamp[1]) - 5,int(element.top_left_linesamp[1]) + element.W + 5,(patches_per_batch // 4,)),
                                                           min=int(element.top_left_linesamp[1]),max=int(element.top_left_linesamp[1]) + element.W - 1)],
                                                dim=-1).to(dtype=element.buffer['locals'].dtype,device=element.buffer['locals'].device)
                sample_linesamps = torch.concatenate([sample_linesamps,
                                                      torch.stack([2 * int(element.top_left_linesamp[0]) + element.H - 1 - sample_linesamps[:,0],2 * int(element.top_left_linesamp[1]) + element.W - 1 - sample_linesamps[:,1]],dim=-1),
                                                      torch.stack([2 * int(element.top_left_linesamp[0]) + element.H - 1 - sample_linesamps[:,0],sample_linesamps[:,1]],dim=-1),
                                                      torch.stack([sample_linesamps[:,0],2 * int(element.top_left_linesamp[1]) + element.W - 1 - sample_linesamps[:,1]],dim=-1)],
                                                      dim=0)
                dists,idxs = element.kd_tree.query(sample_linesamps,nr_nns_searches=4)
                valid_mask = dists.max(dim=1).values < 256
                dists = 1. / (dists[valid_mask] + 1e-6)
                idxs = idxs[valid_mask]
                dists = dists / torch.mean(dists,dim=-1,keepdim=True)
                features_pD = element.buffer['features'][idxs].contiguous()
                confs_p1 = element.buffer['confs'][idxs].contiguous()
                objs_p3 = element.buffer['objs'][idxs].contiguous()
                locals_p2 = sample_linesamps[valid_mask]
                features_pD = features_pD * dists.unsqueeze(-1)
                confs_p1 = confs_p1 * dists
                objs_p3 = objs_p3 * dists.unsqueeze(-1)
                features_pD = torch.mean(features_pD,dim=1).to(torch.float32)
                confs_p1 = torch.mean(confs_p1,dim=1).to(torch.float32)
                objs_p3 = torch.mean(objs_p3,dim=1).to(torch.float32)

                # 筛出在grid的border范围内的，范围外的不参与学习
                inside_border_mask = (objs_p3[:,0] >= self.border[0]) & (objs_p3[:,0] <= self.border[2]) & (objs_p3[:,1] >= self.border[1]) & (objs_p3[:,1] <= self.border[3])
                features_pD = features_pD[inside_border_mask]
                confs_p1 = confs_p1[inside_border_mask]
                objs_p3 = objs_p3[inside_border_mask]
                locals_p2 = locals_p2[inside_border_mask]

                features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
                patch_feature_noise = patch_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:][:,:,inside_border_mask,:].contiguous()
                global_feature_noise = global_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:][:,:,inside_border_mask,:].contiguous()
                features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] = F.normalize(features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] + patch_feature_noise,dim=1)
                features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] = F.normalize(features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] + global_feature_noise,dim=1)
            
                # global_feature_noise = F.normalize(torch.normal(mean=0,std=1,size=(1,self.encoder.global_feat_channels,features_1Dp1.shape[-2],1)),dim=1).to(features_1Dp1.device) * 0.5
                # features_1Dp1[:,-self.encoder.global_feat_channels:,:,:] += global_feature_noise
                
                output_13p1 = self.mapper(features_1Dp1)
                output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
                xyh_pred_p3 = self.warp_by_poly(output_p3,self.map_coeffs)
                
                loss,loss_obj,loss_height,loss_photo,loss_bias,loss_reg = criterion(iter_idx,self.options.element_training_iters,xyh_pred_p3,confs_p1,locals_p2,objs_p3,element.rpc)
                loss.backward()

                total_loss += loss.item()
                total_loss_obj += loss_obj.item()
                total_loss_photo += loss_photo.item()
                total_loss_height += loss_height.item()
                total_reg += loss_reg
                count += 1

                # print(f"iter:{iter_idx + 1} \t img:{element.id}/{len(self.elements)} \t loss:{loss.item():.2f} \t l_obj:{loss_obj.item():.2f} \t l_photo:{loss_photo.item():.2f} \t l_real:{loss_photo_real.item():.2f} \t l_height:{loss_height.item():.2f} \t bias:{loss_bias:.2f} \t reg:{loss_reg:.2f} \t lr:{scheduler.get_last_lr()[0]:.7f}")
                pbar.update(1)
                pbar.set_postfix({
                    'lr':f'{scheduler.get_last_lr()[0]:.2e}',
                    'obj':f'{loss_obj.item():.2f}',
                    'photo':f'{loss_photo.item():.2f}',
                    'h':f'{loss_height.item():.2f}',
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
                # print(f"\n ============= iter:{iter_idx + 1} \t total_loss:{total_loss:.2f} \t total_loss_obj:{total_loss_obj:.2f} \t total_loss_photo:{total_loss_photo:.2f} \t total_loss_real:{total_loss_photo_real:.2f} \t total_loss_height:{total_loss_height:.2f} \t total_loss_reg:{total_reg:.2f} \t time:{cost_time}s \n")
                if total_loss_photo < min_photo_loss:
                    min_photo_loss = total_loss_photo
                    best_mapper_state_dict = self.mapper.state_dict()
                    no_update_count = 0
                else:
                    no_update_count += 1
                
                if no_update_count >= 50:
                    self.mapper.load_state_dict(best_mapper_state_dict)
                    scheduler.cool_down()
                    no_update_count = -1e9 #防止重复启动

                total_loss = 0
                total_loss_obj = 0
                total_loss_height = 0
                total_loss_photo = 0
                total_reg = 0
                count = 0
            
        self.mapper.load_state_dict(best_mapper_state_dict)
        # torch.save(best_mapper_state_dict,os.path.join(self.output_path,'grid_mapper.pth'))
        self.save_grid()
        for element in self.elements:
            element.clear_buffer()
        self.elements = None

    def save_grid(self):
        state_dict = {
            'mapper':self.mapper.state_dict(),
            'diag':torch.from_numpy(self.diag),
            'map_coeffs_x':torch.from_numpy(self.map_coeffs['x']),
            'map_coeffs_y':torch.from_numpy(self.map_coeffs['y']),
            'map_coeffs_h':torch.from_numpy(self.map_coeffs['h']),
        }
        torch.save(state_dict,os.path.join(self.output_path,'grid_data.pth'))

    def load_grid(self,path:str):
        state_dict = torch.load(os.path.join(path,'grid_data.pth'))
        name = os.path.basename(path)
        self.mapper.load_state_dict(state_dict['mapper'])
        self.diag = state_dict['diag'].cpu().numpy()
        self.map_coeffs = {
            'x':state_dict['map_coeffs_x'].cpu().numpy(),
            'y':state_dict['map_coeffs_y'].cpu().numpy(),
            'h':state_dict['map_coeffs_h'].cpu().numpy(),
        }
        print(f"Grid '{name} loaded succesfully'")
    
    @torch.no_grad()
    def vis_match(self,idx=None):
        if self.vis_points_latlon is None:
            points_num = 10000
            range = int(min(self.elements[0].H,self.elements[0].W) * 0.75)
            points_linesamp_0 = torch.rand(points_num,2) * range - range // 2 + torch.tensor([self.elements[0].H // 2,self.elements[0].W // 2])
            points_linesamp_0 = points_linesamp_0.to(int)
            self.vis_dem = self.elements[0].dem[points_linesamp_0[:,0],points_linesamp_0[:,1]]
            self.vis_points_latlon = torch.stack(self.elements[0].rpc.RPC_PHOTO2OBJ(points_linesamp_0[:,1],points_linesamp_0[:,0],self.vis_dem ),dim=-1)
        for element in self.elements:
            points_sampline = torch.stack(element.rpc.RPC_OBJ2PHOTO(self.vis_points_latlon[:,0],self.vis_points_latlon[:,1],self.vis_dem ),dim=-1).cpu().numpy()
            img_vis = deepcopy(element.img_raw)
            # img_vis = np.stack([img_vis,img_vis,img_vis],axis=-1)
            # print(element.img_raw.shape)
            # print(img_vis.shape)
            for p in points_sampline:
                cv2.circle(img_vis,np.round(p).astype(int),1,(0,255,0),thickness=-1)
            # print(1)
            # print(cv2.imwrite(os.path.join(element.output_path,f'match_vis.png' if idx is None else f'match_vis_{idx}.png'),img_vis))
            cv2.imwrite(os.path.join(element.output_path,f'match_vis.png' if idx is None else f'match_vis_{idx}.png'),img_vis)

        
    def __crop_img__(self,img,crop_size,step,local = None,random_ratio = 1.):

        print("cropping image")
        H, W = img.shape[:2]
        if local is None:
            local = get_coord_mat(H,W)

        if local.shape[:2] != img.shape[:2]:
            raise ValueError(f"img shape {img.shape} does not match local shape {local.shape}")
        cut_number = 0
        row_num = 0
        col_num = 0
        crop_imgs = []
        crop_locals = []

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

                crop_imgs.append(img[row_start:row_end,col_start:col_end])
                crop_locals.append(local[row_start:row_end,col_start:col_end])

                cut_number += 1
                pbar.update(1)
                col_num += 1
            row_num += 1
            col_num -= 1

        random_num = int(cut_number * random_ratio)

        for i in range(random_num):
            col = np.random.randint(0,W - crop_size)
            row = np.random.randint(0,H - crop_size)
            crop_imgs.append(img[row:row + crop_size,col:col + crop_size])
            crop_locals.append(local[row:row + crop_size,col:col + crop_size])
        
        crop_imgs = np.stack(crop_imgs)
        crop_locals = np.stack(crop_locals)

        return crop_imgs,crop_locals

    @torch.no_grad()
    def pred_xyh(self,img_raw:np.ndarray,local_hw2:np.ndarray) -> Dict[str,np.ndarray]:
        """
        return: {"xy_P2","h_P1","locals_P2","confs_P1"} np.ndarray
        """
        H,W = img_raw.shape[:2]
        self.encoder.eval().cuda()
        self.mapper.eval().cuda()

        if self.options.crop_step > 0:
            crop_step = self.options.crop_step
        else:
            crop_step = int(np.sqrt((H - self.options.crop_size) * (W - self.options.crop_size) / 150.))
        crop_imgs_NHW,crop_locals_NHW2 = self.__crop_img__(img_raw,self.options.crop_size,crop_step,local=local_hw2)
        print("Tranforming Images")
        imgs_NHW = torch.stack([self.transform(img) for img in tqdm(crop_imgs_NHW)]) # N,H,W
        locals_NHW2= torch.from_numpy(crop_locals_NHW2)
        locals_Nhw2 = downsample(locals_NHW2,self.encoder.SAMPLE_FACTOR)
        total_patch_num = locals_Nhw2.shape[0] * locals_Nhw2.shape[1] * locals_Nhw2.shape[2]
        select_ratio = min(1. * self.options.max_buffer_size / total_patch_num,1.)

        batch_num = int(np.ceil(len(crop_imgs_NHW) / self.options.batch_size))
        features_PD = []
        confs_P1 = []
        locals_P2 = []

        print("Extracting Features")
        for batch_idx in trange(batch_num):
            batch_imgs = imgs_NHW[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].cuda()
            batch_locals = locals_Nhw2[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].cuda().flatten(0,2)
            feat,conf = self.encoder(batch_imgs)
            # features_NDhw.append(feat)
            # confs_Nhw.append(conf)
            feat = feat.permute(0,2,3,1).flatten(0,2)
            conf = conf.squeeze().flatten(0,2)
            valid_mask = conf > self.options.conf_threshold
            select_idxs = torch.randperm(valid_mask.sum())[:int(select_ratio * len(conf))]

            features_PD.append(feat[valid_mask][select_idxs])
            confs_P1.append(conf[valid_mask][select_idxs])
            locals_P2.append(batch_locals[valid_mask][select_idxs])

        features_PD = torch.cat(features_PD,dim=0)
        confs_P1 = torch.cat(confs_P1,dim=0)
        locals_P2 = torch.cat(locals_P2,dim=0)

        patches_per_batch = self.options.patches_per_batch
        batch_num = int(np.ceil(features_PD.shape[0] / patches_per_batch))
        print("Predicting Geographic Coordinates")
        xyh_preds = []
        for batch_idx in trange(batch_num):
            features_1Dp1 = features_PD[batch_idx * patches_per_batch : (batch_idx + 1) * patches_per_batch].permute(1,0)[None,:,:,None]
            output_13p1 = self.mapper(features_1Dp1)
            output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
            xyh_pred_p3 = self.warp_by_poly(output_p3,self.map_coeffs)
            xyh_preds.append(xyh_pred_p3)
        
        xyh_P3 = torch.concatenate(xyh_preds,dim=0)


        # kd_tree = build_kd_tree(locals_P2,device='cuda')

        # yxh_preds = []
        # locals = []
        # confs = []

        # block_line_num = int(np.ceil(H / 256.))
        # block_samp_num = int(np.ceil(W / 256.))
        
        # print("Decoding")
        # pbar = tqdm(total=block_line_num * block_samp_num)
        # for block_line in range(block_line_num):
        #     for block_samp in range(block_samp_num):
        #         tl_line = block_line * 256 + .5
        #         tl_samp = block_samp * 256 + .5
        #         sample_linesamps = torch.from_numpy(np.stack(np.meshgrid(np.arange(tl_line,min(tl_line + 256,H)),np.arange(tl_samp,min(tl_samp + 256,W)),indexing='ij')
        #                                                             ,axis=-1)).to(dtype=locals_P2.dtype,device=locals_P2.device).reshape(-1,2)
        #         dists,idxs = kd_tree.query(sample_linesamps,nr_nns_searches=4)
        #         valid_mask = (dists.max(dim=1).values < 64) 
        #         if valid_mask.sum() == 0:
        #             continue
        #         dists = 1. / (dists[valid_mask] + 1e-6)
        #         idxs = idxs[valid_mask]
        #         dists = dists / torch.mean(dists,dim=-1,keepdim=True)
        #         features_pD = features_PD[idxs].contiguous()
        #         confs_p1 = confs_P1[idxs].contiguous()
        #         features_pD = features_pD * dists.unsqueeze(-1)
        #         confs_p1 = confs_p1 * dists
        #         features_pD = torch.mean(features_pD,dim=1).to(torch.float32)
        #         confs_p1 = torch.mean(confs_p1,dim=1).to(torch.float32)
        #         locals_p2 = sample_linesamps[valid_mask]

        #         features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
        #         output_13p1 = self.mapper(features_1Dp1)
        #         output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
        #         yxh_pred_p3 = warp_by_extend(output_p3,self.extend)
        #         yxh_preds.append(yxh_pred_p3)
        #         locals.append(locals_p2)
        #         confs.append(confs_p1)
                
        #         pbar.update(1)

        # yxh_P3 = torch.cat(yxh_preds,dim=0)
        # locals_P2 = torch.cat(locals,dim=0)
        # confs_P1 = torch.cat(confs,dim=0)


        # pred_raw_N3hw = self.mapper(features_NDhw)
        # pred_raw_P3 = pred_raw_N3hw.permute(0,2,3,1).flatten(0,2)
        # yxh_P3 = warp_by_extend(pred_raw_P3,self.extend)
        
        res = {
            'xy_P2':xyh_P3[:,:2].cpu().numpy(),
            'h_P1':xyh_P3[:,2].cpu().numpy(),
            'locals_P2':locals_P2.cpu().numpy(),
            'confs_P1':confs_P1.cpu().numpy(),
        }
        crop_imgs_NHW = None
        crop_locals_NHW2 = None
        imgs_NHW = None

        return res
    
