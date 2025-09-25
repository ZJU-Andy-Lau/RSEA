from pyexpat import features
import stat

from scipy import cluster
from utils import Status
import warnings
import scheduler
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
from utils import get_coord_mat,project_mercator,mercator2lonlat,downsample,bilinear_interpolate,apply_polynomial,get_map_coef,visualize_subset_points

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
from matplotlib import pyplot as plt
import random
from typing import List,Dict
from pykeops.torch import LazyTensor

from rs_image import RSImage
from element import Element
from block import Block

def redirect_output(output_path:str,info:str):
    with open(output_path,'a') as f:
        f.write(info)



class Grid():
    STATES = Status
    def __init__(self,options,encoder:Encoder,output_path:str,diag:np.ndarray = None,grid_path:str = None,device:str = None):

        self.options = options
        self.encoder = encoder
        self.status = self.STATES.NOT_INIT
        if diag is None and grid_path is None:
            raise ValueError("Grid loaded error: Neither diag nor grid path is given")
        self.options.mapper_input_channel = self.encoder.output_channels
        
        if grid_path is None :
            self.diag = diag #[[x,y],[x,y]]
            # self.map_coeffs = {
            #     'x':np.array([.6 * np.abs(diag[0,0] - diag[1,0]), .5 * (diag[0,0] + diag[1,0])]),
            #     'y':np.array([.6 * np.abs(diag[0,1] - diag[1,1]), .5 * (diag[0,1] + diag[1,1])]),
            #     'h':None
            # }
            # self.mapper = Decoder(in_channels=self.encoder.output_channels,digit_num=options.digit_num,block_num=options.mapper_blocks_num)
            # self.optimizer = AdamW(self.mapper.parameters(),lr=self.options.grid_train_lr_max)
            # self.scheduler = MultiStageOneCycleLR(optimizer=self.optimizer,
            #                                     total_steps=self.options.grid_training_iters,
            #                                     warmup_ratio=self.options.grid_warmup_iters / self.options.grid_training_iters,
            #                                     cooldown_ratio=self.options.grid_cooldown_iters / self.options.grid_training_iters)
            # self.train_iter_idx = 0
            self.blocks = self.__devide_blocks__(self.options.block_size)

        else:
            self.load_grid(grid_path)
        self.border = np.array([self.diag[:,0].min(),self.diag[:,1].min(),self.diag[:,0].max(),self.diag[:,1].max()])#[min_x,min_y,max_x,max_y]
        self.output_path = output_path
        self.elements:List[Element] = []
        self.transform = nn.Sequential(
            #for-swt
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            )
            #for-dino
            # K.Normalize(
            #     mean=torch.tensor([0.430, 0.411, 0.296]), 
            #     std=torch.tensor([0.213, 0.156, 0.143])
            # )
        ).eval()
        self.train_data = []
        self.SAMPLE_FACTOR = 16
        self.pred_resolution = .7
        self.vis_points_latlon = None
        if device is None:
            self.device = 'cuda'
        else:
            self.device = device
    
    def to_device(self,device):
        self.device = device
        self.encoder.to(device)
        for block in self.blocks:
            block.mapper.to(device)
        for element in self.elements:
            element.to_device(device)
    
    def __devide_blocks__(self,block_size) -> list[Block]:
        h,w = np.abs(self.diag[1,1] - self.diag[0,1]), np.abs(self.diag[1,0] - self.diag[0,0])
        # h_pix,w_pix = int(h / self.pred_resolution),int(w / self.pred_resolution)
        h_block_num = int(np.ceil(h / block_size))
        w_block_num = int(np.ceil(w / block_size))
        h_shrink_ratio = h / (h_block_num * block_size)
        w_shrink_ratio = w / (w_block_num * block_size)
        y_tls = self.diag[0,1] - (np.arange(0,(h_block_num - 1) * block_size + 1,block_size) * h_shrink_ratio) 
        x_tls = self.diag[0,0] + (np.arange(0,(w_block_num - 1) * block_size + 1,block_size) * w_shrink_ratio)
        x_tls,y_tls = np.meshgrid(x_tls,y_tls,indexing='xy')
        x_tls,y_tls = x_tls.ravel(),y_tls.ravel()

        diags = np.stack([
            np.stack([x_tls,y_tls],axis=-1),
            np.stack([x_tls + block_size,y_tls - block_size],axis=-1)
        ],axis=1)

        # print(f"diags:{diags.astype(int)}")
        
        blocks = []
        for diag in diags:
            map_coeffs = {
                'x':np.array([.6 * np.abs(diag[0,0] - diag[1,0]), .5 * (diag[0,0] + diag[1,0])]),
                'y':np.array([.6 * np.abs(diag[0,1] - diag[1,1]), .5 * (diag[0,1] + diag[1,1])]),
                'h':None
            }
            diag_ratio = np.array([
                [np.abs(diag[0,1] - self.diag[0,1]) / h , np.abs(diag[0,0] - self.diag[0,0]) / w],
                [np.abs(diag[1,1] - self.diag[0,1]) / h , np.abs(diag[1,0] - self.diag[0,0]) / w]
            ])
            # print(f"diag:{diag} \n diag_ratio:{diag_ratio} \n==============================\n")
            block = Block(self.options,diag,diag_ratio,map_coeffs)
            blocks.append(block)
        return blocks

        
    def update_task_state(self,task_info,update_info):
        state = task_info['state'][task_info['id']]
        task_info['state'][task_info['id']] = {**state,**update_info}

    def fprint(self,info:str):
        output_path = os.path.join(self.output_path,'log.txt')
        info += '\n'
        redirect_output(output_path,info)

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
                                                            (int((self.border[3] - self.border[1]) / self.pred_resolution),
                                                            int((self.border[2] - self.border[0]) / self.pred_resolution)),
                                                            need_local=True)
            
            dem = img.resample_dem_by_sampline(corner_samplines,
                                                (int((self.border[3] - self.border[1]) / self.pred_resolution),
                                                int((self.border[2] - self.border[0]) / self.pred_resolution)))

            return img_raw,dem,local_hw2
        else:
            raise ValueError("mode should either be 'bbox' or 'interpolate'")

    def get_height_map_coeffs(self):
        heights = []
        xys = []
        for element in self.elements:
            heights.append(element.buffer['objs'][:,2])
            xys.append(element.buffer['objs'][:,:2])
        heights = torch.concatenate(heights).cpu().numpy()
        xys = torch.concatenate(xys).cpu().numpy()
        for block in self.blocks:
            mask = (xys[:,0] >= block.diag[0,0]) & (xys[:,1] <= block.diag[0,1]) & (xys[:,0] < block.diag[1,0]) & (xys[:,1] > block.diag[1,1])
            height = heights[mask]
            block.map_coeffs['h'] = get_map_coef(height)
        # self.map_coeffs['h'] = get_map_coef(heights)


    def add_img(self,img:RSImage):
        """
        添加训练数据
        """
        img_raw,dem,local_hw2 = self.get_overlap_image(img,mode='interpolate')
        self.train_data.append({
            'img':img_raw,
            'dem':dem,
            'local':local_hw2,
            'rpc':img.rpc
        })
    
    def create_elements(self,output_path:str = None,task_info = None,clear = False):
        if output_path is None:
            output_path = self.output_path
        
        if not task_info is None:
            self.update_task_state(task_info,{
                'status':f"Grid {task_info['id']}:提取特征",
                'total':len(self.train_data)
            })
        if clear:
            self.elements:List[Element] = []
        for idx,data in enumerate(self.train_data):
            id = len(self.elements)
            path = os.path.join(output_path,f'element_{id}')
            os.makedirs(path,exist_ok=True)
            new_element = Element(options = self.options,
                                encoder = self.encoder,
                                img_raw = data['img'],
                                dem = data['dem'],
                                rpc = data['rpc'],
                                id = id,
                                output_path = path,
                                local_raw = data['local'],
                                device=self.device,
                                verbose=1 if task_info is None else 0)

            self.elements.append(new_element)
            if not task_info is None:
                self.update_task_state(task_info,{'progress':idx+1})

        # self.get_height_map_coeffs()
    
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

    def train(self,task_info = None):
        self.get_height_map_coeffs()
        for block_idx in range(len(self.blocks)):
            self.train_mapper(block_idx,task_info)
        for element in self.elements:
            element.clear_buffer()
        self.elements = None
        if not task_info is None:
            self.update_task_state(task_info,{
                'status':f"Grid {task_info['id']}:训练完成"
            })

    def train_mapper(self,block_idx:int,task_info = None,save_checkpoint = True):
        block = self.blocks[block_idx]
        mapper = block.mapper

        max_patch_num = max(*[element.patch_num for element in self.elements],0)
        patches_per_batch = self.options.patches_per_batch // 4 * 4
        
        optimizer = AdamW(mapper.parameters(),lr=self.options.grid_train_lr_max)
        scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                            total_steps=self.options.grid_training_iters,
                                            warmup_ratio=self.options.grid_warmup_iters / self.options.grid_training_iters,
                                            cooldown_ratio=self.options.grid_cooldown_iters / self.options.grid_training_iters)
        optimizer = optimizer
        scheduler = scheduler
        criterion = CriterionTrainGrid()
        bce = nn.BCELoss()
        mapper.train()
        # if self.options.use_gpu:
        mapper.to(self.device)

        min_photo_loss = 1e8

        patch_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.output_channels,max_patch_num * 5,1)),dim=1).to(self.elements[0].buffer['features'].device)
        patch_noise_amp = torch.rand(1,1,max_patch_num * 5,1,device=patch_noise_buffer.device,dtype=patch_noise_buffer.dtype) * .1
        patch_noise_buffer = patch_noise_buffer * patch_noise_amp

        vis_flag = 0

        total_loss = 0
        total_loss_dist = 0
        total_loss_obj = 0
        total_loss_height = 0
        total_loss_photo = 0
        # total_reg = 0
        count = 0
        progress = 0
        no_update_count = 0
        early_stop_iter = -1
        last_mapper_state_dict = None
        # pbar = tqdm(total=self.options.grid_training_iters * len(self.elements))
        if not task_info is None:
            self.update_task_state(task_info,{
                'status':f"Grid {task_info['id']}:Block {block_idx + 1} / {len(self.blocks)} 训练",
                'total':self.options.grid_training_iters * len(self.elements)
            })
        else:
            pbar = tqdm(total=self.options.grid_training_iters * len(self.elements))

        for iter_idx in range(self.options.grid_training_iters):
            noise_idx = torch.randperm(max_patch_num * 5)[:patches_per_batch * 2]
            optimizer.zero_grad()
            for element in self.elements:
                # if iter_idx % 2 != 0:

                block_tl_linesamp = (block.diag_ratio[0] * element.img_raw.shape[:2]).astype(int)
                block_br_linesamp = (block.diag_ratio[1] * element.img_raw.shape[:2]).astype(int)
                linesamp_min,linesamp_max = element.local_raw[block_tl_linesamp[0],block_tl_linesamp[1]],element.local_raw[block_br_linesamp[0] - 1,block_br_linesamp[1] - 1]
                sample_linesamps = torch.stack([torch.rand((patches_per_batch // 4,)) * (linesamp_max[0] - linesamp_min[0]) + linesamp_min[0],
                                                torch.rand((patches_per_batch // 4,)) * (linesamp_max[1] - linesamp_min[1]) + linesamp_min[1]],
                                                dim=-1).to(dtype=element.buffer['locals'].dtype,device=element.buffer['locals'].device)
                sample_linesamps = torch.concatenate([sample_linesamps,
                                                    torch.stack([linesamp_max[0] + linesamp_min[0] - sample_linesamps[:,0],linesamp_max[1] + linesamp_min[1] - sample_linesamps[:,1]],dim=-1),
                                                    torch.stack([linesamp_max[0] + linesamp_min[0] - sample_linesamps[:,0],sample_linesamps[:,1]],dim=-1),
                                                    torch.stack([sample_linesamps[:,0],linesamp_max[1] + linesamp_min[1] - sample_linesamps[:,1]],dim=-1)],
                                                    dim=0)

                dists,idxs = element.query_point_base(sample_linesamps,k=self.options.nearest_neighbor_num) # n,3
                # torch.cuda.synchronize()
                valid_mask = (dists.max(dim=1).values < 256) & (dists.min(dim=1).values < 16)
                if valid_mask.sum() == 0:
                    continue
                dists = dists[valid_mask]
                idxs = idxs[valid_mask]
                
                dists_ratio = dists / torch.sum(dists,dim=1,keepdim=True) # n,3
                reverse_dists_ratio = 1. / dists_ratio
                reverse_dists_ratio = reverse_dists_ratio / torch.sum(reverse_dists_ratio,dim=1,keepdim=True)
                
                min_dist_idxs = torch.argmin(dists,dim=1)
                min_dist_idxs = idxs[torch.arange(len(min_dist_idxs),device=idxs.device),min_dist_idxs] # n
                
                features_p3D = element.buffer['features'][idxs].contiguous()
                confs_p3 = element.buffer['confs'][idxs].contiguous()
                objs_p33 = element.buffer['objs'][idxs].contiguous()
                locals_p32 = element.buffer['locals'][idxs].contiguous()

                features_sample_pD = torch.sum(features_p3D * reverse_dists_ratio.unsqueeze(-1),dim=1).to(torch.float32)
                confs_sample_p1 = torch.sum(confs_p3 * reverse_dists_ratio,dim=1).to(torch.float32)
                objs_sample_p3 = torch.sum(objs_p33 * reverse_dists_ratio.unsqueeze(-1),dim=1).to(torch.float32)
                locals_sample_p2 = torch.sum(locals_p32 * reverse_dists_ratio.unsqueeze(-1),dim=1).to(torch.float32)

                features_anchor_pD = element.buffer['features'][min_dist_idxs].to(torch.float32)
                confs_anchor_p1 = element.buffer['confs'][min_dist_idxs].to(torch.float32)
                objs_anchor_p3 = element.buffer['objs'][min_dist_idxs].to(torch.float32)
                locals_anchor_p2 = element.buffer['locals'][min_dist_idxs].to(torch.float32)

                inside_border_mask = (objs_sample_p3[:,0] >= block.border[0]) & (objs_sample_p3[:,0] <= block.border[2]) & (objs_sample_p3[:,1] >= block.border[1]) & (objs_sample_p3[:,1] <= block.border[3]) & \
                                     (objs_anchor_p3[:,0] >= block.border[0]) & (objs_anchor_p3[:,0] <= block.border[2]) & (objs_anchor_p3[:,1] >= block.border[1]) & (objs_anchor_p3[:,1] <= block.border[3])
                features_sample_pD = features_sample_pD[inside_border_mask]
                confs_sample_p1 = confs_sample_p1[inside_border_mask]
                objs_sample_p3 = objs_sample_p3[inside_border_mask]
                locals_sample_p2 = locals_sample_p2[inside_border_mask]

                features_anchor_pD = features_anchor_pD[inside_border_mask]
                confs_anchor_p1 = confs_anchor_p1[inside_border_mask]
                objs_anchor_p3 = objs_anchor_p3[inside_border_mask]
                locals_anchor_p2 = locals_anchor_p2[inside_border_mask]

                feature_dis = torch.norm(features_sample_pD - features_anchor_pD,dim=1)


                    
                # else:
                #     sample_idxs = torch.randperm(len(element.buffer['features']))[:patches_per_batch]
                #     features_pD = element.buffer['features'][sample_idxs].contiguous()
                #     confs_p1 = element.buffer['confs'][sample_idxs].contiguous()
                #     objs_p3 = element.buffer['objs'][sample_idxs].contiguous()
                #     locals_p2 = element.buffer['locals'][sample_idxs].contiguous()
                #     valid_mask = torch.full((patches_per_batch,),True,dtype=bool)

                
                # 筛出在grid的border范围内的，范围外的不参与学习
                

                if vis_flag < 1:
                    visualize_subset_points(locals_sample_p2.cpu().numpy(),locals_anchor_p2.cpu().numpy(),os.path.join(self.output_path,f'knn_vis_{block_idx}_{vis_flag}.png'),point_radius=2)
                    vis_flag += 1

                patch_num = inside_border_mask.sum()
                features_sample_1Dp1 = features_sample_pD.permute(1,0)[None,:,:,None]
                features_anchor_1Dp1 = features_anchor_pD.permute(1,0)[None,:,:,None]
                feature_sample_noise = patch_noise_buffer[:,:,noise_idx[:patches_per_batch],:][:,:,valid_mask,:][:,:,inside_border_mask,:].contiguous()
                feature_anchor_noise = patch_noise_buffer[:,:,noise_idx[patches_per_batch:],:][:,:,valid_mask,:][:,:,inside_border_mask,:].contiguous()
                #for-swt
                features_sample_1Dp1 = F.normalize(features_sample_1Dp1 + feature_sample_noise,dim=1)
                features_anchor_1Dp1 = F.normalize(features_anchor_1Dp1 + feature_anchor_noise,dim=1)
                #for-dino
                # features_sample_1Dp1 = features_sample_1Dp1 + feature_sample_noise
                # features_anchor_1Dp1 = features_anchor_1Dp1 + feature_anchor_noise
                #===================生成负样本特征=====================

                negative_sample_idxs = torch.randperm(len(element.buffer['features']))[:3 * patch_num] # 3p,D
                negative_features = element.buffer['features'][negative_sample_idxs].reshape(patch_num,3,-1) # p,3,D
                negative_locals = element.buffer['locals'][negative_sample_idxs].reshape(patch_num,3,-1) # p,3,2
                negative_avg_feature = torch.mean(negative_features,dim=1) # p,D
                negative_avg_local = torch.mean(negative_locals,dim=1) # p,2
                dis = torch.mean(torch.norm(negative_avg_local[:,None] - negative_locals,dim=-1),dim=1) # p
                negative_noise_amp =  100. / dis
                negative_noise = patch_noise_buffer[0,:,torch.randperm(max_patch_num * 5)[:patch_num],0].permute(1,0) # p,D
                #for-swt
                negative_avg_feature = F.normalize(negative_avg_feature + negative_noise * negative_noise_amp[:,None],dim=1)
                #for-dino
                # negative_avg_feature = negative_avg_feature + negative_noise * negative_noise_amp[:,None]

                negative_feature_1Dp1 = negative_avg_feature.permute(1,0)[None,:,:,None]

                #=====================================================

                output_sample_16p1,valid_score_sample = mapper(features_sample_1Dp1)
                output_anchor_16p1,valid_score_anchor = mapper(features_anchor_1Dp1)
                valid_score_positive = (valid_score_sample + valid_score_anchor) / 2.
                valid_score_nagetive = mapper.forward_valid(negative_feature_1Dp1)
                
                output_sample_p6 = output_sample_16p1.permute(0,2,3,1).flatten(0,2)
                output_anchor_p6 = output_anchor_16p1.permute(0,2,3,1).flatten(0,2)
                mu_xyh_sample_p3 = self.warp_by_poly(output_sample_p6[:,:3],block.map_coeffs)
                mu_xyh_anchor_p3 = self.warp_by_poly(output_anchor_p6[:,:3],block.map_coeffs)
                log_sigma_xyh_sample_p3 = output_sample_p6[:,3:]
                log_sigma_xyh_anchor_p3 = output_anchor_p6[:,3:]

                loss,loss_distribution,loss_obj,loss_height,loss_photo,loss_dis,sigma_avg = criterion(iter_idx,
                                                                                            self.options.grid_training_iters,
                                                                                            feature_dis,
                                                                                            [mu_xyh_sample_p3,mu_xyh_anchor_p3],
                                                                                            [log_sigma_xyh_sample_p3,log_sigma_xyh_anchor_p3],
                                                                                            [confs_sample_p1,confs_anchor_p1],
                                                                                            [locals_sample_p2,locals_anchor_p2],
                                                                                            [objs_sample_p3,objs_anchor_p3],
                                                                                            element.rpc) #,loss_bias,
                
                valid_pred = torch.concatenate([valid_score_positive.reshape(-1),valid_score_nagetive.reshape(-1)],dim=0)
                valid_label = torch.concatenate([torch.full((patch_num,),1.),torch.full((patch_num,),0.)],dim=0).to(valid_pred.device) # positive,negative
                loss_valid = bce(valid_pred,valid_label) * 100.

                loss = loss + loss_valid
                loss.backward()

                total_loss += loss.item()
                total_loss_dist += loss_distribution.item()
                total_loss_obj += loss_obj.item()
                total_loss_photo += loss_photo.item()
                total_loss_height += loss_height.item()
                # total_reg += loss_reg
                count += 1
                progress += 1 
                info = {
                        'i':f'{progress}',
                        'lr':f'{scheduler.get_last_lr()[0]:.2e}',
                        'd':f'{loss_dis.item():.2f}', 
                        's':f'{sigma_avg:.2f}',
                        'o':f'{loss_obj.item():.2f}',
                        'p':f'{loss_photo.item():.2f}',
                        'h':f'{loss_height.item():.2f}',
                        # 'r':f'{loss_reg:.2f}',
                        'v':f'{loss_valid:.2f}',
                        'min':f'{min_photo_loss:.2f}'
                    }
                if not task_info is None:
                    self.update_task_state(task_info,{
                        'progress':progress,
                        'info':info
                    })
                else:
                    pbar.update(1)
                    pbar.set_postfix(info)
            optimizer.step()

            scheduler.step()

            if loss_photo > min_photo_loss * 10.:
                mapper.load_state_dict(best_mapper_state_dict['model'])
                optimizer.load_state_dict(best_mapper_state_dict['optimizer'])
                if no_update_count > 0:
                    scheduler.trigger_cooldown()
                    no_update_count = -1e9 #防止重复启动
                    early_stop_iter = iter_idx + self.options.grid_cooldown_iters


            if (iter_idx + 1) % 10 == 0:
                total_loss /= count
                total_loss_dist /= count
                total_loss_obj /= count
                total_loss_height /= count
                total_loss_photo /= count
                # total_reg /= count
                
                # cost_time = int(time.perf_counter() - start_time)
                # print(f"\n ============= iter:{iter_idx + 1} \t total_loss:{total_loss:.2f} \t total_loss_obj:{total_loss_obj:.2f} \t total_loss_photo:{total_loss_photo:.2f} \t total_loss_real:{total_loss_photo_real:.2f} \t total_loss_height:{total_loss_height:.2f} \t total_loss_reg:{total_reg:.2f} \t time:{cost_time}s \n")
                if total_loss_photo < min_photo_loss:
                    min_photo_loss = total_loss_photo
                    no_update_count = 0
                    if last_mapper_state_dict is None:
                        best_mapper_state_dict = {
                            'model':deepcopy(mapper.state_dict()),
                            'optimizer':deepcopy(optimizer.state_dict())
                        }
                    else:
                        best_mapper_state_dict = last_mapper_state_dict
                else:
                    no_update_count += 1
                
                if no_update_count >= 200 or (no_update_count > 0 and total_loss_photo > min_photo_loss * 10.):
                    mapper.load_state_dict(best_mapper_state_dict['model'])
                    optimizer.load_state_dict(best_mapper_state_dict['optimizer'])
                    scheduler.trigger_cooldown()
                    no_update_count = -1e9 #防止重复启动
                    early_stop_iter = iter_idx + self.options.grid_cooldown_iters

                last_mapper_state_dict = {
                        'model':deepcopy(mapper.state_dict()),
                        'optimizer':deepcopy(optimizer.state_dict())
                    }

                # if save_checkpoint:
                #     self.save_grid()
                total_loss = 0
                total_loss_dist = 0
                total_loss_obj = 0
                total_loss_height = 0
                total_loss_photo = 0
                # total_reg = 0
                count = 0

            if early_stop_iter > 0 and iter_idx >= early_stop_iter:
                break
        # if early_stop_iter > 0:
        #     print("early stopped")
        mapper.load_state_dict(best_mapper_state_dict['model'])
        if min_photo_loss < 25.:
            block.status = self.STATES.WELL_TRAINED
        else:
            block.status = self.STATES.BAD_TRAINED
        # torch.save(best_mapper_state_dict,os.path.join(self.output_path,'grid_mapper.pth'))
        self.save_grid()
        
    
    def finetune_mapper(self,task_info = None,save_checkpoint = True):
        max_patch_num = max(*[element.patch_num for element in self.elements],0)
        patches_per_batch = self.options.patches_per_batch // 4 * 4
        self.optimizer = AdamW(self.mapper.parameters(),lr=self.options.grid_finetune_lr_max)
        self.scheduler = MultiStageOneCycleLR(optimizer=self.optimizer,
                                            total_steps=self.options.grid_finetune_iters,
                                            warmup_ratio=self.options.grid_finetune_warmup_iters / self.options.grid_finetune_iters,
                                            cooldown_ratio=self.options.grid_finetune_cooldown_iters / self.options.grid_finetune_iters)
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = CriterionTrainGrid()
        bce = nn.BCELoss()
        self.mapper.train()
        # if self.options.use_gpu:
        self.mapper.to(self.device)

        min_photo_loss = 1e8

        patch_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.output_channels,max_patch_num * 5,1)),dim=1).to(self.elements[0].buffer['features'].device)
        patch_noise_amp = torch.rand(1,1,max_patch_num * 5,1,device=patch_noise_buffer.device,dtype=patch_noise_buffer.dtype) * .1 + .1
        patch_noise_buffer = patch_noise_buffer * patch_noise_amp

        vis_flag = True

        total_loss = 0
        total_loss_dist = 0
        total_loss_obj = 0
        total_loss_height = 0
        total_loss_photo = 0
        count = 0
        no_update_count = 0
        early_stop_iter = -1
        last_mapper_state_dict = None
        progress = self.train_iter_idx * len(self.elements)
        if not task_info is None:
            self.update_task_state(task_info,{
                'status':f"Grid {task_info['id']}:Decoder训练",
                'total':self.options.grid_finetune_iters * len(self.elements)
            })
        else:
            pbar = tqdm(total=self.options.grid_finetune_iters * len(self.elements))
            pbar.update(progress)
        for self.train_iter_idx in range(self.train_iter_idx,self.options.grid_finetune_iters):
            iter_idx = self.train_iter_idx
            noise_idx = torch.randperm(max_patch_num * 5)[:patches_per_batch]
            optimizer.zero_grad()
            for element in self.elements:
                if iter_idx % 2 != 0:
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
                    dists,idxs = element.query_point_base(sample_linesamps,k=self.options.nearest_neighbor_num) # n,3
                    valid_mask = dists.max(dim=1).values < 64
                    if valid_mask.sum() == 0:
                        continue
                    dists_ratio = dists[valid_mask] / torch.sum(dists[valid_mask],dim=1,keepdim=True) # n,3
                    reverse_dists_ratio = 1. / dists_ratio
                    reverse_dists_ratio = reverse_dists_ratio / torch.sum(reverse_dists_ratio,dim=1,keepdim=True)
                    idxs = idxs[valid_mask]
                    features_p3D = element.buffer['features'][idxs].contiguous()
                    confs_p3 = element.buffer['confs'][idxs].contiguous()
                    objs_p33 = element.buffer['objs'][idxs].contiguous()
                    locals_p32 = element.buffer['locals'][idxs].contiguous()

                    features_pD = torch.sum(features_p3D * reverse_dists_ratio.unsqueeze(-1),dim=1).to(torch.float32)
                    confs_p1 = torch.sum(confs_p3 * reverse_dists_ratio,dim=1).to(torch.float32)
                    objs_p3 = torch.sum(objs_p33 * reverse_dists_ratio.unsqueeze(-1),dim=1).to(torch.float32)
                    locals_p2 = torch.sum(locals_p32 * reverse_dists_ratio.unsqueeze(-1),dim=1).to(torch.float32)

                else:
                    sample_idxs = torch.randperm(len(element.buffer['features']))[:patches_per_batch]
                    features_pD = element.buffer['features'][sample_idxs].contiguous()
                    confs_p1 = element.buffer['confs'][sample_idxs].contiguous()
                    objs_p3 = element.buffer['objs'][sample_idxs].contiguous()
                    locals_p2 = element.buffer['locals'][sample_idxs].contiguous()
                    valid_mask = torch.full((patches_per_batch,),True,dtype=bool)

                
                # 筛出在grid的border范围内的，范围外的不参与学习
                inside_border_mask = (objs_p3[:,0] >= self.border[0]) & (objs_p3[:,0] <= self.border[2]) & (objs_p3[:,1] >= self.border[1]) & (objs_p3[:,1] <= self.border[3])
                features_pD = features_pD[inside_border_mask]
                confs_p1 = confs_p1[inside_border_mask]
                objs_p3 = objs_p3[inside_border_mask]
                locals_p2 = locals_p2[inside_border_mask]

                patch_num = confs_p1.shape[0]
                features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
                patch_feature_noise = patch_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:][:,:,inside_border_mask,:].contiguous()
                #for-swt
                # features_1Dp1 = F.normalize(features_1Dp1 + patch_feature_noise,dim=1)
                #for-dino
                features_1Dp1 = features_1Dp1 + patch_feature_noise
                
                #===================生成负样本特征=====================

                negative_sample_idxs = torch.randperm(len(element.buffer['features']))[:3 * patch_num] # 3p,D
                negative_features = element.buffer['features'][negative_sample_idxs].reshape(patch_num,3,-1) # p,3,D
                negative_locals = element.buffer['locals'][negative_sample_idxs].reshape(patch_num,3,-1) # p,3,2
                negative_avg_feature = torch.mean(negative_features,dim=1) # p,D
                negative_avg_local = torch.mean(negative_locals,dim=1) # p,2
                dis = torch.mean(torch.norm(negative_avg_local[:,None] - negative_locals,dim=-1),dim=1) # p
                negative_noise_amp =  100. / dis
                negative_noise = F.normalize(torch.normal(mean=0.,std=1.,size=negative_avg_feature.shape,dtype=negative_avg_feature.dtype),dim=1).to(negative_avg_feature.device) # p,D
                #for-swt
                # negative_avg_feature = F.normalize(negative_avg_feature + negative_noise * negative_noise_amp[:,None],dim=1)
                #for-dino
                negative_avg_feature = negative_avg_feature + negative_noise * negative_noise_amp[:,None]

                negative_feature_1Dp1 = negative_avg_feature.permute(1,0)[None,:,:,None]

                #=====================================================

                output_16p1,valid_score_positive = self.mapper(features_1Dp1)
                valid_score_nagetive = self.mapper.forward_valid(negative_feature_1Dp1)
                
                output_p6 = output_16p1.permute(0,2,3,1).flatten(0,2)
                mu_xyh_p3 = self.warp_by_poly(output_p6[:,:3],self.map_coeffs)
                log_sigma_xyh_p3 = output_p6[:,3:]

                loss,loss_distribution,loss_obj,loss_height,loss_photo,sigma_avg = criterion(iter_idx,
                                                                                            self.options.grid_finetune_iters,
                                                                                            mu_xyh_p3,
                                                                                            log_sigma_xyh_p3,
                                                                                            confs_p1,
                                                                                            locals_p2,
                                                                                            objs_p3,
                                                                                            element.rpc) #,loss_bias,loss_reg
                
                valid_pred = torch.concatenate([valid_score_positive.reshape(-1),valid_score_nagetive.reshape(-1)],dim=0)
                valid_label = torch.concatenate([torch.full((patch_num,),1.),torch.full((patch_num,),0.)],dim=0).to(valid_pred.device) # positive,negative
                loss_valid = bce(valid_pred,valid_label) * 100.

                loss = loss + loss_valid
                loss.backward()

                total_loss += loss.item()
                total_loss_dist += loss_distribution.item()
                total_loss_obj += loss_obj.item()
                total_loss_photo += loss_photo.item()
                total_loss_height += loss_height.item()
                # total_reg += loss_reg
                count += 1
                progress += 1 
                info = {
                        'i':f'{progress}',
                        'lr':f'{scheduler.get_last_lr()[0]:.2e}',
                        'd':f'{loss_distribution.item():.2f}', 
                        's':f'{sigma_avg:.2f}',
                        'o':f'{loss_obj.item():.2f}',
                        'p':f'{loss_photo.item():.2f}',
                        'h':f'{loss_height.item():.2f}',
                        # 'r':f'{loss_reg:.2f}',
                        'v':f'{loss_valid:.2f}',
                        'min':f'{min_photo_loss:.2f}'
                    }
                if not task_info is None:
                    self.update_task_state(task_info,{
                        'progress':progress,
                        'info':info
                    })
                else:
                    pbar.update(1)
                    pbar.set_postfix(info)
            optimizer.step()

            scheduler.step()

            if loss_photo > min_photo_loss * 10.:
                self.mapper.load_state_dict(best_mapper_state_dict['model'])
                optimizer.load_state_dict(best_mapper_state_dict['optimizer'])
                if no_update_count > 0:
                    scheduler.trigger_cooldown()
                    no_update_count = -1e9 #防止重复启动
                    early_stop_iter = iter_idx + self.options.grid_finetune_cooldown_iters


            if (iter_idx + 1) % 10 == 0:
                total_loss /= count
                total_loss_dist /= count
                total_loss_obj /= count
                total_loss_height /= count
                total_loss_photo /= count
                # total_reg /= count
                
                # cost_time = int(time.perf_counter() - start_time)
                # print(f"\n ============= iter:{iter_idx + 1} \t total_loss:{total_loss:.2f} \t total_loss_obj:{total_loss_obj:.2f} \t total_loss_photo:{total_loss_photo:.2f} \t total_loss_real:{total_loss_photo_real:.2f} \t total_loss_height:{total_loss_height:.2f} \t total_loss_reg:{total_reg:.2f} \t time:{cost_time}s \n")
                if total_loss_photo < min_photo_loss:
                    min_photo_loss = total_loss_photo
                    no_update_count = 0
                    if last_mapper_state_dict is None:
                        best_mapper_state_dict = {
                            'model':deepcopy(self.mapper.state_dict()),
                            'optimizer':deepcopy(optimizer.state_dict())
                        }
                    else:
                        best_mapper_state_dict = last_mapper_state_dict
                else:
                    no_update_count += 1
                
                if no_update_count >= 200 or (no_update_count > 0 and total_loss_photo > min_photo_loss * 10.):
                    self.mapper.load_state_dict(best_mapper_state_dict['model'])
                    optimizer.load_state_dict(best_mapper_state_dict['optimizer'])
                    scheduler.trigger_cooldown()
                    no_update_count = -1e9 #防止重复启动
                    early_stop_iter = iter_idx + self.options.grid_cooldown_iters

                last_mapper_state_dict = {
                        'model':deepcopy(self.mapper.state_dict()),
                        'optimizer':deepcopy(optimizer.state_dict())
                    }

                if save_checkpoint:
                    self.save_grid()
                total_loss = 0
                total_loss_dist = 0
                total_loss_obj = 0
                total_loss_height = 0
                total_loss_photo = 0
                # total_reg = 0
                count = 0

            if early_stop_iter > 0 and iter_idx >= early_stop_iter:
                break
        # if early_stop_iter > 0:
        #     print("early stopped")
        self.mapper.load_state_dict(best_mapper_state_dict['model'])
        if min_photo_loss < 25.:
            self.status = self.STATES.WELL_TRAINED
        else:
            self.status = self.STATES.BAD_TRAINED
        # torch.save(best_mapper_state_dict,os.path.join(self.output_path,'grid_mapper.pth'))
        self.save_grid()
        for element in self.elements:
            element.clear_buffer()
        self.elements = None
        if not task_info is None:
            self.update_task_state(task_info,{
                'status':f"Grid {task_info['id']}:训练完成"
            })

    def save_grid(self):
        state_dict = {
            'diag':torch.from_numpy(self.diag),
            'mapper_blocks_num':self.options.mapper_blocks_num,
            'block_num':len(self.blocks),
            **{f'block_{block_idx}':block.get_block_state_dict() for block_idx,block in enumerate(self.blocks)}
        }
        torch.save(state_dict,os.path.join(self.output_path,'grid_data.pth'))

    def load_grid(self,path:str):
        state_dict = torch.load(os.path.join(path,'grid_data.pth'))
        name = os.path.basename(path)
        self.options.mapper_blocks_num = state_dict['mapper_blocks_num']
        self.diag = state_dict['diag'].numpy()
        self.blocks = []
        for block_idx in range(state_dict['block_num']):
            block_state_dict = state_dict[f'block_{block_idx}']
            block_diag = block_state_dict['diag'].numpy()
            block_diag_ratio = block_state_dict['diag_ratio'].numpy()
            block_map_coeffs = {
                'x':block_state_dict['map_coeffs_x'].numpy(),
                'y':block_state_dict['map_coeffs_y'].numpy(),
                'h':block_state_dict['map_coeffs_h'].numpy(),
            }
            block = Block(self.options,block_diag,block_diag_ratio,block_map_coeffs)
            block.mapper.load_state_dict(block_state_dict['mapper'])
            block.status = block_state_dict['status']
            self.blocks.append(block)        
        
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

        
    def __crop_img__(self,img,crop_size,local = None,random_ratio = 1.,size_ratios = [1.],expect_num = 64,step = None):

        print("cropping image")
        H, W = img.shape[:2]
        if local is None:
            local = get_coord_mat(H,W)
        
        index = get_coord_mat(H,W)

        flex_step = False
        if step is None:
            flex_step = True

        if local.shape[:2] != img.shape[:2]:
            raise ValueError(f"img shape {img.shape} does not match local shape {local.shape}")
        
        
        cut_number = 0
        row_num = 0
        col_num = 0
        crop_imgs = []
        crop_locals = []
        crop_indexs = []

        if not step is None and step <= 0 :
            img = cv2.resize(img,(crop_size + self.encoder.SAMPLE_FACTOR,crop_size + self.encoder.SAMPLE_FACTOR))
            local = cv2.resize(local,(crop_size + self.encoder.SAMPLE_FACTOR,crop_size + self.encoder.SAMPLE_FACTOR))
            index = cv2.resize(index,(crop_size + self.encoder.SAMPLE_FACTOR,crop_size + self.encoder.SAMPLE_FACTOR))
            for i in range(self.encoder.SAMPLE_FACTOR):
                for j in range(self.encoder.SAMPLE_FACTOR):
                    crop_imgs.append(img[i:crop_size + i,j:crop_size + j])
                    crop_locals.append(local[i:crop_size + i,j:crop_size + j])
                    crop_indexs.append(index[i:crop_size + i,j:crop_size + j])
            crop_imgs = np.stack(crop_imgs)
            crop_locals = np.stack(crop_locals)
            crop_indexs = np.stack(crop_indexs)

            return crop_imgs,crop_locals,crop_indexs
            

        for ratio in size_ratios:
            raw_size = int(crop_size * ratio)
            if flex_step:
                step = int(np.sqrt((H - raw_size) * (W - raw_size) / expect_num))
            
            rows = np.arange(0,H - raw_size + 1,step)
            cols = np.arange(0,W - raw_size + 1,step)

            pbar = tqdm(total=len(rows) * len(cols))

            for row in rows:
                for col in cols:
                    if row + raw_size + step > H:
                        if col + raw_size > W:
                            row_start,row_end,col_start,col_end = H - raw_size, H, W - raw_size, W
                        else:
                            row_start,row_end,col_start,col_end = H - raw_size, H, col, col+raw_size
                    else:
                        if col + raw_size + step > W:
                            row_start,row_end,col_start,col_end = row, row+raw_size, W - raw_size, W
                        else:
                            row_start,row_end,col_start,col_end = row, row+raw_size, col, col+raw_size
                    if row_num % 2 == 1:
                        col_start,col_end = W - col_end,W - col_start
                    if col_num % 2 == 1:
                        row_start,row_end = H - row_end,H - row_start

                    img_crop = cv2.resize(img[row_start:row_end,col_start:col_end],(crop_size,crop_size), interpolation=cv2.INTER_LINEAR)
                    local_crop = cv2.resize(local[row_start:row_end,col_start:col_end],(crop_size,crop_size), interpolation=cv2.INTER_LINEAR)
                    index_crop = cv2.resize(index[row_start:row_end,col_start:col_end],(crop_size,crop_size), interpolation=cv2.INTER_LINEAR)

                    crop_imgs.append(img_crop)
                    crop_locals.append(local_crop)
                    crop_indexs.append(index_crop)

                    cut_number += 1
                    pbar.update(1)
                    col_num += 1
                row_num += 1
                col_num -= 1

        if cut_number > 1:
            random_num = int(cut_number * random_ratio)

            for i in range(random_num):
                col = np.random.randint(0,W - crop_size)
                row = np.random.randint(0,H - crop_size)
                crop_imgs.append(img[row:row + crop_size,col:col + crop_size])
                crop_locals.append(local[row:row + crop_size,col:col + crop_size])
                crop_indexs.append(index[row:row + crop_size,col:col + crop_size])
            
        crop_imgs = np.stack(crop_imgs)
        crop_locals = np.stack(crop_locals)
        crop_indexs = np.stack(crop_indexs)

        return crop_imgs,crop_locals,crop_indexs

    @torch.no_grad()
    def pred_xyh(self,img_raw:np.ndarray,local_hw2:np.ndarray) -> Dict[str,np.ndarray]:
        """
        return: {"xy_P2","h_P1","locals_P2","confs_P1"} np.ndarray
        """
        H,W = img_raw.shape[:2]
        self.encoder.eval().to(self.device)

        crop_imgs_NHWC,crop_locals_NHW2,crop_indexs_NHW2 = self.__crop_img__(img = img_raw,
                                                                            crop_size = self.options.crop_size,
                                                                            expect_num = 32,
                                                                            size_ratios = [1.],
                                                                            random_ratio = 1.,
                                                                            local=local_hw2)
        print("Tranforming Images")
        imgs_NCHW = torch.from_numpy(crop_imgs_NHWC).permute(0,3,1,2)
        imgs_NCHW = imgs_NCHW.float() / 255.0
        imgs_NCHW = imgs_NCHW.to(self.device)
        self.transform = self.transform.to(self.device)
        batch_num = int(np.ceil(imgs_NCHW.shape[0] / self.options.batch_size))
        imgs_NCHW = [self.transform(imgs_NCHW[b * self.options.batch_size : (b+1) * self.options.batch_size]) for b in trange(batch_num)]
        imgs_NCHW = torch.concatenate(imgs_NCHW,dim=0)
        locals_NHW2= torch.from_numpy(crop_locals_NHW2)
        locals_Nhw2 = downsample(locals_NHW2,self.encoder.SAMPLE_FACTOR,use_cuda=True,mode='avg',device=self.device)
        indexs_NHW2 = torch.from_numpy(crop_indexs_NHW2)
        indexs_Nhw2 = downsample(indexs_NHW2,self.encoder.SAMPLE_FACTOR,use_cuda=True,mode='avg',device=self.device)
        total_patch_num = locals_Nhw2.shape[0] * locals_Nhw2.shape[1] * locals_Nhw2.shape[2]
        select_ratio = min(1. * self.options.max_buffer_size / total_patch_num,1.)

        batch_num = int(np.ceil(len(crop_imgs_NHWC) / self.options.batch_size))
        features_PD = []
        confs_P1 = []
        locals_P2 = []
        indexs_P2 = []

        print("Extracting Features")
        for batch_idx in trange(batch_num):
            batch_imgs = imgs_NCHW[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].to(self.device)
            batch_locals = locals_Nhw2[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size,1:-1,1:-1].to(self.device).flatten(0,2)
            batch_indexs = indexs_Nhw2[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size,1:-1,1:-1].to(self.device).flatten(0,2)
            feat,conf = self.encoder(batch_imgs)
            feat = feat[:,:,1:-1,1:-1]
            conf = conf[:,:,1:-1,1:-1]
            # features_NDhw.append(feat)
            # confs_Nhw.append(conf)
            feat = feat.permute(0,2,3,1).flatten(0,2)
            conf = conf.permute(0,2,3,1).flatten(0,3)
            valid_mask = conf > self.options.conf_threshold
            select_idxs = torch.randperm(valid_mask.sum())[:int(select_ratio * len(conf))]

            features_PD.append(feat[valid_mask][select_idxs])
            confs_P1.append(conf[valid_mask][select_idxs])
            locals_P2.append(batch_locals[valid_mask][select_idxs])
            indexs_P2.append(batch_indexs[valid_mask][select_idxs])

        features_PD = torch.cat(features_PD,dim=0)
        confs_P1 = torch.cat(confs_P1,dim=0)
        locals_P2 = torch.cat(locals_P2,dim=0)
        indexs_P2 = torch.cat(indexs_P2,dim=0)

        patches_per_batch = self.options.patches_per_batch
        batch_num = int(np.ceil(features_PD.shape[0] / patches_per_batch))



        print("Predicting Geographic Coordinates")
        mu_xyh_preds = []
        sigma_xyh_preds = []
        valid_scores = []
        linesamps_gt = []

        def find_cluster(points: torch.Tensor, m: int, k: float) -> List[int]:
            """
            在一个 (N, 2) 的点集中，随机寻找一个大小为 m 的簇。
            簇的定义是：簇内任意两点之间的欧氏距离都小于 k。

            Args:
                points (torch.Tensor): 一个形状为 (N, 2) 的 Tensor，代表 N 个点的二维坐标。
                m (int): 期望的簇中点的数量。
                k (float): 簇内点对之间的最大距离阈值。

            Returns:
                Optional[List[int]]: 一个包含 m 个点索引的列表，代表找到的簇。
                                    如果找不到满足条件的簇，则返回 None。
            """
            # 获取点的总数
            n = points.shape[0]

            # --- 处理边界情况 ---
            if m > n:
                print("错误：期望的簇大小 m 大于点的总数 N。")
                return None
            if m <= 1:
                # 如果簇大小为 0 或 1，直接返回前 m 个点的索引
                return list(range(m))

            # --- 核心算法 ---
            
            # 1. 计算所有点对之间的距离矩阵，方便快速查找
            dist_matrix = torch.cdist(points, points)

            # 2. 生成一个随机打乱的索引序列，用于遍历起始点
            indices = torch.randperm(n).tolist()

            # 3. 遍历每个点，尝试将其作为“种子点”来构建一个簇
            for i in indices:
                cluster = [i]
                
                # 4. 找到所有与种子点 i 距离小于 k 的点，作为候选点
                # .nonzero() 返回满足条件的索引, .view(-1) 确保它是一维的
                potential_neighbors_idx = (dist_matrix[i] < k).nonzero().view(-1).tolist()
                
                # 从候选列表中移除种子点自身
                candidates = [idx for idx in potential_neighbors_idx if idx != i]
                random.shuffle(candidates) # 随机打乱候选点，增加找到的簇的随机性

                # 5. 尝试从候选点中添加点到簇中，直到簇的大小达到 m
                for candidate_idx in candidates:
                    # 如果簇已经足够大，就跳出循环
                    if len(cluster) == m:
                        break

                    # 检查当前候选点与簇中已有的所有点是否都“兼容”
                    # (即它们之间的距离都小于 k)
                    is_compatible = True
                    for point_in_cluster_idx in cluster:
                        if dist_matrix[candidate_idx, point_in_cluster_idx] >= k:
                            is_compatible = False
                            break
                    
                    # 如果兼容，则将该候选点加入簇
                    if is_compatible:
                        cluster.append(candidate_idx)
                
                # 6. 如果成功找到了一个大小为 m 的簇，就返回结果
                if len(cluster) == m:
                    return sorted(cluster) # 返回排序后的索引列表，方便查看

            # 7. 如果遍历完所有点都找不到满足条件的簇，则返回 None
            return None

        for block_idx,block in enumerate(tqdm(self.blocks)):
            block.mapper.eval().to(self.device)
            line_min,line_max,samp_min,samp_max = block.diag_ratio[0,0] * H, block.diag_ratio[1,0] * H, block.diag_ratio[0,1] * W, block.diag_ratio[1,1] * W
            inside_block_mask = (indexs_P2[:,0] >= line_min) & (indexs_P2[:,1] >= samp_min) & (indexs_P2[:,0] <= line_max) & (indexs_P2[:,1] <= samp_max)
            features_1Dp1 = features_PD[inside_block_mask].permute(1,0)[None,:,:,None]
            output_16p1,valid_score = block.mapper(features_1Dp1)
            output_p6 = output_16p1.permute(0,2,3,1).flatten(0,2)
            mu_xyh_p3 = self.warp_by_poly(output_p6[:,:3],block.map_coeffs)
            sigma_xyh_p3 = torch.exp(output_p6[:,3:])
            valid_score_p1 = valid_score.reshape(-1)

            indexs_inside = indexs_P2[inside_block_mask]
            cluster_idxs = find_cluster(indexs_inside,k=2.,m=3)
            feature_dis = torch.cdist(features_PD[inside_block_mask][cluster_idxs],features_PD[inside_block_mask][cluster_idxs])
            feature_length = torch.norm(features_PD[inside_block_mask][cluster_idxs],dim=1)
            mu_dis = torch.cdist(mu_xyh_p3[cluster_idxs],mu_xyh_p3[cluster_idxs])
            local_dis = torch.cdist(indexs_inside[cluster_idxs],indexs_inside[cluster_idxs])
            print(f"block {block_idx + 1}: feature distance:\n{feature_dis}\nfeature length:\n{feature_length}\nmu distance:\n{mu_dis}\nlocal distance:\n{local_dis}")

            mu_xyh_preds.append(mu_xyh_p3)
            sigma_xyh_preds.append(sigma_xyh_p3)
            valid_scores.append(valid_score_p1)
            linesamps_gt.append(locals_P2[inside_block_mask])

            visualize_subset_points(indexs_P2[inside_block_mask].cpu().numpy(),indexs_P2[torch.randperm(len(indexs_P2))[:10000]].reshape(-1,2).cpu().numpy(),os.path.join(self.output_path,f'block_{block_idx + 1}_pred_points_local.png'),point_radius=2)

        # for batch_idx in trange(batch_num):
        #     features_1Dp1 = features_PD[batch_idx * patches_per_batch : (batch_idx + 1) * patches_per_batch].permute(1,0)[None,:,:,None]
        #     output_16p1,valid_score = self.mapper(features_1Dp1)
        #     output_p6 = output_16p1.permute(0,2,3,1).flatten(0,2)
        #     mu_xyh_p3 = self.warp_by_poly(output_p6[:,:3],self.map_coeffs)
        #     sigma_xyh_p3 = torch.exp(output_p6[:,3:])
        #     valid_score_p1 = valid_score.reshape(-1)

        #     if mu_xyh_p3.shape[0] != sigma_xyh_p3.shape[0] or mu_xyh_p3.shape[0] != valid_score_p1.shape[0]:
        #         print(mu_xyh_p3.shape,sigma_xyh_p3.shape,valid_score_p1.shape)
        #         raise ValueError("shape doesn't match")

        #     mu_xyh_preds.append(mu_xyh_p3)
        #     sigma_xyh_preds.append(sigma_xyh_p3)
        #     valid_scores.append(valid_score_p1)
        
        mu_xyh_P3 = torch.concatenate(mu_xyh_preds,dim=0)
        sigma_xyh_P3 = torch.concatenate(sigma_xyh_preds,dim=0)
        valid_scores_P1 = torch.concatenate(valid_scores,dim=0)
        linesamps_gt_P2 = torch.concatenate(linesamps_gt,dim=0) 
       
        res = {
            'mu_xyh_P3':mu_xyh_P3,
            'sigma_xyh_P3':sigma_xyh_P3,
            'locals_P2':linesamps_gt_P2,
            'confs_P1':confs_P1,
            'valid_score_P1':valid_scores_P1
        }
        crop_imgs_NHWC = None
        crop_locals_NHW2 = None
        imgs_NCHW = None

        return res
    
    @torch.no_grad()
    def pred_dense_xyh(self,img_raw:np.ndarray,local_hw2:np.ndarray) -> Dict[str,np.ndarray]:
        H,W = img_raw.shape[:2]
        self.encoder = self.encoder.eval().to(self.device)
        self.mapper = self.mapper.eval().to(self.device)

        crop_imgs_NHWC,crop_locals_NHW2 = self.__crop_img__(img=img_raw,
                                                            crop_size=self.options.crop_size,
                                                            expect_num=64,
                                                            random_ratio=0,
                                                            size_ratios=[.8,1.,1.25],
                                                            local=local_hw2)
        print("Tranforming Images")
        imgs_NCHW = torch.from_numpy(crop_imgs_NHWC).permute(0,3,1,2)
        imgs_NCHW = imgs_NCHW.float() / 255.0
        imgs_NCHW = imgs_NCHW.to(self.device)
        self.transform = self.transform.to(self.device)
        with torch.no_grad():
            batch_num = int(np.ceil(imgs_NCHW.shape[0] / self.options.batch_size))
            imgs_NCHW = [self.transform(imgs_NCHW[b * self.options.batch_size : (b+1) * self.options.batch_size]) for b in trange(batch_num)]
            imgs_NCHW = torch.concatenate(imgs_NCHW,dim=0)
        locals_NHW2= torch.from_numpy(crop_locals_NHW2)
        locals_Nhw2 = downsample(locals_NHW2,self.encoder.SAMPLE_FACTOR,use_cuda=True,mode='avg',device=self.device)
        total_patch_num = locals_Nhw2.shape[0] * locals_Nhw2.shape[1] * locals_Nhw2.shape[2]
        select_ratio = min(1. * self.options.max_buffer_size / total_patch_num,1.)

        batch_num = int(np.ceil(len(crop_imgs_NHWC) / self.options.batch_size))
        features_PD = []
        confs_P1 = []
        locals_P2 = []

        print("Extracting Features")
        for batch_idx in trange(batch_num):
            batch_imgs = imgs_NCHW[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].to(self.device)
            batch_locals = locals_Nhw2[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].to(self.device).flatten(0,2)
            feat,conf = self.encoder(batch_imgs)
            # features_NDhw.append(feat)
            # confs_Nhw.append(conf)
            feat = feat.permute(0,2,3,1).flatten(0,2)
            conf = conf.permute(0,2,3,1).flatten(0,3)
            valid_mask = conf > self.options.conf_threshold
            select_idxs = torch.randperm(valid_mask.sum())[:int(select_ratio * len(conf))]

            features_PD.append(feat[valid_mask][select_idxs])
            confs_P1.append(conf[valid_mask][select_idxs])
            locals_P2.append(batch_locals[valid_mask][select_idxs])

        features_PD = torch.cat(features_PD,dim=0)
        confs_P1 = torch.cat(confs_P1,dim=0)
        locals_P2 = torch.cat(locals_P2,dim=0)

        print("Building Point Base")
        points_base = LazyTensor(locals_P2.unsqueeze(0))

        def query_point_base(query_points:torch.Tensor,k=3):
            """
            query_points: (N,2)
            """
            query = LazyTensor(query_points.unsqueeze(1))
            dist_ij:LazyTensor = ((query - points_base) ** 2).sum(-1)
            dists,idxs = dist_ij.Kmin_argKmin(K=k, dim=1)
            return dists,idxs

        patches_per_batch = self.options.patches_per_batch
        
        margin = self.encoder.SAMPLE_FACTOR // 2
        lines = np.arange(margin,H - margin,1)
        samps = np.arange(margin,W - margin,1)
        lines,samps = np.meshgrid(lines,samps,indexing='ij')
        linesamps = np.stack([lines.ravel(),samps.ravel()],axis=-1)
        linesamps = torch.from_numpy(linesamps).to(device=features_PD.device,dtype=torch.float32)

        batch_num = int(np.ceil(len(linesamps) / patches_per_batch))

        print("Predicting Dense Geographic Coordinates")
        mu_xyh_preds = []
        sigma_xyh_preds = []
        valid_scores = []
        confs_total = []
        locals_total = []
        for batch_idx in trange(batch_num):
            sample_linesamps = linesamps[batch_idx * patches_per_batch : (batch_idx + 1) * patches_per_batch]
            dists,idxs = query_point_base(sample_linesamps,k=self.options.nearest_neighbor_num)
            # torch.cuda.synchronize()
            valid_mask = dists.max(dim=1).values < 256
            if valid_mask.sum() == 0:
                continue
            # break
            dists = 1. / (dists[valid_mask] + 1e-6)
            idxs = idxs[valid_mask]
            dists = dists / torch.mean(dists,dim=-1,keepdim=True)
            features_pD = features_PD[idxs].contiguous()
            confs_p1 = confs_P1[idxs].contiguous()
            locals_p2 = sample_linesamps[valid_mask]
            features_pD = features_pD * dists.unsqueeze(-1)
            confs_p1 = confs_p1 * dists
            features_pD = torch.mean(features_pD,dim=1).to(torch.float32)
            confs_p1 = torch.mean(confs_p1,dim=1).to(torch.float32)

            features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
            output_16p1,valid_score = self.mapper(features_1Dp1)
            output_p6 = output_16p1.permute(0,2,3,1).flatten(0,2)
            mu_xyh_p3 = self.warp_by_poly(output_p6[:,:3],self.map_coeffs)
            sigma_xyh_p3 = torch.exp(output_p6[:,3:])
            valid_score_p1 = valid_score.reshape(-1)

            if mu_xyh_p3.shape[0] != sigma_xyh_p3.shape[0] or mu_xyh_p3.shape[0] != valid_score_p1.shape[0]:
                print(mu_xyh_p3.shape,sigma_xyh_p3.shape,valid_score_p1.shape)
                raise ValueError("shape doesn't match")
            
            mu_xyh_preds.append(mu_xyh_p3)
            sigma_xyh_preds.append(sigma_xyh_p3)
            valid_scores.append(valid_score_p1)
            confs_total.append(confs_p1)
            locals_total.append(locals_p2)

        mu_xyh_P3 = torch.concatenate(mu_xyh_preds,dim=0)
        sigma_xyh_P3 = torch.concatenate(sigma_xyh_preds,dim=0)
        valid_scores_P1 = torch.concatenate(valid_scores,dim=0)
        confs_total = torch.concatenate(confs_total,dim=0)
        locals_total = torch.concatenate(locals_total,dim=0)
        
       
        res = {
            'mu_xyh_P3':mu_xyh_P3,
            'sigma_xyh_P3':sigma_xyh_P3,
            'locals_P2':locals_total,
            'confs_P1':confs_total,
            'valid_score_P1':valid_scores_P1
        }
        crop_imgs_NHWC = None
        crop_locals_NHW2 = None
        imgs_NCHW = None

        return res