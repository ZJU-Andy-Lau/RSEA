from enum import Enum
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
from matplotlib import pyplot as plt
import random
from typing import List,Dict
from pykeops.torch import LazyTensor

from rs_image import RSImage
from element import Element

def redirect_output(output_path:str,info:str):
    with open(output_path,'a') as f:
        f.write(info)

class GridStatus(Enum):
    NOT_INIT = 0
    WELL_TRAINED = 1
    BAD_TRAINED = 2

class Grid():
    STATES = GridStatus
    def __init__(self,options,encoder:Encoder,output_path:str,diag:np.ndarray = None,grid_path:str = None,device:str = None):

        self.options = options
        self.encoder = encoder
        self.status = self.STATES.NOT_INIT
        if diag is None and grid_path is None:
            raise ValueError("Grid loaded error: Neither diag nor grid path is given")
        if grid_path is None :
            self.diag = diag #[[x,y],[x,y]]
            self.map_coeffs = {
                'x':np.array([.6 * np.abs(diag[0,0] - diag[1,0]), .5 * (diag[0,0] + diag[1,0])]),
                'y':np.array([.6 * np.abs(diag[0,1] - diag[1,1]), .5 * (diag[0,1] + diag[1,1])]),
                'h':None
            }
            if options.use_global_feature:
                self.mapper = Decoder(in_channels=self.encoder.patch_feature_channels + self.encoder.global_feature_channels,block_num=options.mapper_blocks_num)
            else:
                self.mapper = Decoder(in_channels=self.encoder.patch_feature_channels,block_num=options.mapper_blocks_num)
            self.optimizer = AdamW(self.mapper.parameters(),lr=self.options.grid_train_lr_max)
            self.scheduler = MultiStageOneCycleLR(optimizer=self.optimizer,
                                                total_steps=self.options.grid_training_iters,
                                                warmup_ratio=self.options.grid_warmup_iters / self.options.grid_training_iters,
                                                cooldown_ratio=self.options.grid_cooldown_iters / self.options.grid_training_iters)
            self.train_iter_idx = 0
        else:
            self.load_grid(grid_path)
        self.border = np.array([self.diag[:,0].min(),self.diag[:,1].min(),self.diag[:,0].max(),self.diag[:,1].max()]) #[min_x,min_y,max_x,max_y]
        # print(f"\n Grid range: x: {self.border[0]:.2f} ~ {self.border[2]:.2f} \t y: {self.border[1]:.2f} ~ {self.border[3]:.2f}\n")
        self.output_path = output_path
        self.elements:List[Element] = []
        # self.transform = transforms.Compose([
        #     transforms.ToTensor(),
        #     transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        #     ])
        self.transform = nn.Sequential(
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            )
        )
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
        self.mapper.to(device)
        for element in self.elements:
            element.to_device(device)
        
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
    
    def create_elements(self,output_path:str = None,task_info = None):
        if output_path is None:
            output_path = self.output_path
        
        if not task_info is None:
            self.update_task_state(task_info,{
                'status':f"Grid {task_info['id']}:提取特征",
                'total':len(self.train_data)
            })
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

    def train_mapper(self,task_info = None,save_checkpoint = True):
        max_patch_num = max(*[element.patch_num for element in self.elements],0)
        patches_per_batch = self.options.patches_per_batch // 4 * 4
        # optimizer = AdamW(self.mapper.parameters(),lr=self.options.grid_train_lr_max)
        # scheduler = MultiStageOneCycleLR(optimizer = optimizer,
        #                                 max_lr = self.options.grid_train_lr_max,
        #                                 min_lr = self.options.grid_train_lr_min,
        #                                 n_epochs_per_stage = self.options.grid_training_iters,
        #                                 steps_per_epoch = 1,
        #                                 pct_start = self.options.grid_warmup_iters / self.options.grid_training_iters,
        #                                 summit_hold = self.options.grid_summit_hold_iters / self.options.grid_training_iters,
        #                                 #gamma = self.options.lr_decay_per_100_epochs ** (1. / 100.),
        #                                 cooldown = 0.0
        #                                 )
        # scheduler = MultiStageOneCycleLR(optimizer=optimizer,
        #                                  total_steps=self.options.grid_training_iters,
        #                                  warmup_ratio=self.options.grid_warmup_iters / self.options.grid_training_iters,
        #                                  cooldown_ratio=self.options.grid_cooldown_iters / self.options.grid_training_iters)
        optimizer = self.optimizer
        scheduler = self.scheduler
        criterion = CriterionTrainGrid()
        bce = nn.BCELoss()
        self.mapper.train()
        # if self.options.use_gpu:
        self.mapper.to(self.device)

        min_photo_loss = 1e8

        patch_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.patch_feature_channels,max_patch_num * 5,1)),dim=1).to(self.elements[0].buffer['features'].device)
        global_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.global_feature_channels,max_patch_num * 5,1)),dim=1).to(self.elements[0].buffer['features'].device)
        patch_noise_amp = torch.rand(1,1,max_patch_num * 5,1,device=patch_noise_buffer.device,dtype=patch_noise_buffer.dtype) * .1 + .1
        global_noise_amp = .5 
        patch_noise_buffer = patch_noise_buffer * patch_noise_amp
        global_noise_buffer = global_noise_buffer * global_noise_amp


        total_loss = 0
        total_loss_dist = 0
        total_loss_obj = 0
        total_loss_height = 0
        total_loss_photo = 0
        # total_reg = 0
        count = 0
        no_update_count = 0
        early_stop_iter = -1
        last_mapper_state_dict = None
        # pbar = tqdm(total=self.options.grid_training_iters * len(self.elements))
        progress = self.train_iter_idx * len(self.elements)
        if not task_info is None:
            self.update_task_state(task_info,{
                'status':f"Grid {task_info['id']}:Decoder训练",
                'total':self.options.grid_training_iters * len(self.elements)
            })
        else:
            pbar = tqdm(total=self.options.grid_training_iters * len(self.elements))
            pbar.update(progress)
        for self.train_iter_idx in range(self.train_iter_idx,self.options.grid_training_iters):
            iter_idx = self.train_iter_idx
            noise_idx = torch.randperm(max_patch_num * 5)[:patches_per_batch]
            optimizer.zero_grad()
            for element in self.elements:
                if iter_idx % 3 != 0:
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
                    # torch.cuda.synchronize()
                    # dists,idxs = element.kd_tree.query(sample_linesamps,nr_nns_searches=3)
                    try:
                        dists,idxs = element.query_point_base(sample_linesamps,k=self.options.nearest_neighbor_num)
                    except Exception as e:
                        self.fprint(f'{e}')
                    # torch.cuda.synchronize()
                    valid_mask = dists.max(dim=1).values < 256
                    # break
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
                features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] = F.normalize(features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] + patch_feature_noise,dim=1)

                if self.options.use_global_feature:
                    global_feature_noise = global_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:][:,:,inside_border_mask,:].contiguous()
                    features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] = F.normalize(features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] + global_feature_noise,dim=1)

                # global_feature_noise = F.normalize(torch.normal(mean=0,std=1,size=(1,self.encoder.global_feat_channels,features_1Dp1.shape[-2],1)),dim=1).to(features_1Dp1.device) * 0.5
                # features_1Dp1[:,-self.encoder.global_feat_channels:,:,:] += global_feature_noise
                
                #===================生成负样本特征=====================

                negative_sample_idxs = torch.randperm(len(element.buffer['features']))[:3 * patch_num] # 3p,D
                negative_features = element.buffer['features'][negative_sample_idxs].reshape(patch_num,3,-1) # p,3,D
                negative_locals = element.buffer['locals'][negative_sample_idxs].reshape(patch_num,3,-1) # p,3,2
                negative_avg_feature = torch.mean(negative_features,dim=1) # p,D
                negative_avg_local = torch.mean(negative_locals,dim=1) # p,2
                dis = torch.mean(torch.norm(negative_avg_local[:,None] - negative_locals,dim=-1),dim=1) # p
                negative_noise_amp =  100. / dis
                negative_noise = F.normalize(torch.normal(mean=0.,std=1.,size=negative_avg_feature.shape,dtype=negative_avg_feature.dtype),dim=1).to(negative_avg_feature.device) # p,D
                negative_avg_feature = F.normalize(negative_avg_feature + negative_noise * negative_noise_amp[:,None],dim=1)
                negative_feature_1Dp1 = negative_avg_feature.permute(1,0)[None,:,:,None]

                #=====================================================

                features_input_1Dq1 = torch.concatenate([features_1Dp1,negative_feature_1Dp1],dim=2)
                output_16q1,valid_score_11q1 = self.mapper(features_input_1Dq1)
                output_16p1 = output_16q1[:,:,:patch_num]
                valid_score_positive,valid_score_nagetive = valid_score_11q1[:,:,:patch_num],valid_score_11q1[:,:,patch_num:]
                # output_16p1,valid_score_positive = self.mapper(features_1Dp1)
                # _,valid_score_nagetive = self.mapper(negative_feature_1Dp1)
                
                output_p6 = output_16p1.permute(0,2,3,1).flatten(0,2)
                mu_xyh_p3 = self.warp_by_poly(output_p6[:,:3],self.map_coeffs)
                log_sigma_xyh_p3 = output_p6[:,3:]

                loss,loss_distribution,loss_obj,loss_height,loss_photo,sigma_avg = criterion(iter_idx,
                                                                                            self.options.grid_training_iters,
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
        if min_photo_loss < 15.:
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
            'mapper':self.mapper.state_dict(),
            'optimizer':self.optimizer.state_dict(),
            'scheduler':self.scheduler.state_dict(),
            'train_iter_idx':self.train_iter_idx,
            'diag':torch.from_numpy(self.diag),
            'map_coeffs_x':torch.from_numpy(self.map_coeffs['x']),
            'map_coeffs_y':torch.from_numpy(self.map_coeffs['y']),
            'map_coeffs_h':torch.from_numpy(self.map_coeffs['h']),
            'use_global_feature':self.options.use_global_feature,
            'num_blocks':self.options.mapper_blocks_num,
            'status':self.status
        }
        torch.save(state_dict,os.path.join(self.output_path,'grid_data.pth'))

    def load_grid(self,path:str):
        state_dict = torch.load(os.path.join(path,'grid_data.pth'))
        name = os.path.basename(path)
        self.options.use_global_feature = state_dict['use_global_feature']
        self.options.mapper_blocks_num = state_dict['num_blocks']
        if self.options.use_global_feature:
            self.mapper = Decoder(in_channels=self.encoder.patch_feature_channels + self.encoder.global_feature_channels,block_num=self.options.mapper_blocks_num)
        else:
            self.mapper = Decoder(in_channels=self.encoder.patch_feature_channels,block_num=self.options.mapper_blocks_num)
        self.mapper.load_state_dict(state_dict['mapper'])
        self.optimizer = AdamW(self.mapper.parameters(),lr=self.options.grid_train_lr_max)
        self.scheduler = MultiStageOneCycleLR(optimizer=self.optimizer,
                                            total_steps=self.options.grid_training_iters,
                                            warmup_ratio=self.options.grid_warmup_iters / self.options.grid_training_iters,
                                            cooldown_ratio=self.options.grid_cooldown_iters / self.options.grid_training_iters)
        self.optimizer.load_state_dict(state_dict['optimizer'])
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.train_iter_idx = state_dict['train_iter_idx']
        self.diag = state_dict['diag'].cpu().numpy()
        self.map_coeffs = {
            'x':state_dict['map_coeffs_x'].cpu().numpy(),
            'y':state_dict['map_coeffs_y'].cpu().numpy(),
            'h':state_dict['map_coeffs_h'].cpu().numpy(),
        }
        self.status = state_dict['status']
        
        
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

        if local.shape[:2] != img.shape[:2]:
            raise ValueError(f"img shape {img.shape} does not match local shape {local.shape}")
        
        
        cut_number = 0
        row_num = 0
        col_num = 0
        crop_imgs = []
        crop_locals = []

        if not step is None and step <= 0 :
            img = cv2.resize(img,(crop_size + self.encoder.SAMPLE_FACTOR,crop_size + self.encoder.SAMPLE_FACTOR))
            local = cv2.resize(local,(crop_size + self.encoder.SAMPLE_FACTOR,crop_size + self.encoder.SAMPLE_FACTOR))
            for i in range(self.encoder.SAMPLE_FACTOR):
                for j in range(self.encoder.SAMPLE_FACTOR):
                    crop_imgs.append(img[i:crop_size + i,j:crop_size + j])
                    crop_locals.append(local[i:crop_size + i,j:crop_size + j])
            crop_imgs = np.stack(crop_imgs)
            crop_locals = np.stack(crop_locals)

            return crop_imgs,crop_locals
            

        for ratio in size_ratios:
            raw_size = int(crop_size * ratio)
            if step is None:
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

                    crop_imgs.append(img_crop)
                    crop_locals.append(local_crop)

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
            
        crop_imgs = np.stack(crop_imgs)
        crop_locals = np.stack(crop_locals)

        return crop_imgs,crop_locals

    @torch.no_grad()
    def pred_xyh(self,img_raw:np.ndarray,local_hw2:np.ndarray) -> Dict[str,np.ndarray]:
        """
        return: {"xy_P2","h_P1","locals_P2","confs_P1"} np.ndarray
        """
        H,W = img_raw.shape[:2]
        self.encoder.eval().to(self.device)
        self.mapper.eval().to(self.device)

        if self.options.crop_step > 0:
            crop_step = self.options.crop_step
        else:
            crop_step = min(int(np.sqrt((H - self.options.crop_size) * (W - self.options.crop_size) / 64.)),self.options.crop_size)
        crop_imgs_NHWC,crop_locals_NHW2 = self.__crop_img__(img_raw,self.options.crop_size,crop_step,local=local_hw2)
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
            # if not self.options.use_global_feature:
            #     feat = feat[:,:self.encoder.patch_feature_channels]
            conf = conf.permute(0,2,3,1).flatten(0,3)
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
        mu_xyh_preds = []
        sigma_xyh_preds = []
        valid_scores = []
        for batch_idx in trange(batch_num):
            features_1Dp1 = features_PD[batch_idx * patches_per_batch : (batch_idx + 1) * patches_per_batch].permute(1,0)[None,:,:,None]
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
        
        mu_xyh_P3 = torch.concatenate(mu_xyh_preds,dim=0)
        sigma_xyh_P3 = torch.concatenate(sigma_xyh_preds,dim=0)
        valid_scores_P1 = torch.concatenate(valid_scores,dim=0)
       
        res = {
            'mu_xyh_P3':mu_xyh_P3,
            'sigma_xyh_P3':sigma_xyh_P3,
            'locals_P2':locals_P2,
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
        self.encoder.eval().to(self.device)
        self.mapper.eval().to(self.device)

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
            # if not self.options.use_global_feature:
            #     feat = feat[:,:self.encoder.patch_feature_channels]
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

        patches_per_batch = self.options.patches_per_batch
        
        margin = self.encoder.SAMPLE_FACTOR // 2
        lines = np.arange(margin,H - margin,1)
        samps = np.arange(margin,W - margin,1)
        lines,samps = np.meshgrid(lines,samps,indexing='ij')
        linesamps = np.stack([lines.ravel(),samps.ravel()],axis=-1)

        batch_num = int(np.ceil(len(linesamps) / patches_per_batch))

        print("Predicting Dense Geographic Coordinates")
        mu_xyh_preds = []
        sigma_xyh_preds = []
        valid_scores = []
        confs_total = []
        locals_total = []
        for batch_idx in trange(batch_num):
            sample_linesamps = linesamps[batch_idx * patches_per_batch : (batch_idx + 1) * patches_per_batch]
            try:
                dists,idxs = points_base(sample_linesamps,k=self.options.nearest_neighbor_num)
            except Exception as e:
                self.fprint(f'{e}')
            # torch.cuda.synchronize()
            valid_mask = dists.max(dim=1).values < 256
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

            features_1Dp1 = features_PD.permute(1,0)[None,:,:,None]
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