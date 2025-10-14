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
from torch.optim import AdamW,lr_scheduler
from scheduler import MultiStageOneCycleLR
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
from typing import List,Dict, Tuple

from rs_image import RSImage
from element_1012 import Element
from block import Block

def redirect_output(output_path:str,info:str):
    # 将打印信息重定向到日志文件
    with open(output_path,'a') as f:
        f.write(info)

class Grid():
    """
    Grid类是模型训练和坐标预测的实际执行者。
    它代表地球表面的一个矩形区域，并管理所有与该区域重叠的影像数据（Elements）。
    它将自身划分为更小的Blocks，并为每个Block训练一个独立的坐标回归模型（mapper）。
    """
    STATES = Status
    def __init__(self,options,encoder:Encoder,output_path:str,diag:np.ndarray = None,grid_path:str = None,device:str = None):
        # --- 1. 初始化基本属性 ---
        self.options = options
        self.encoder = encoder
        self.status = self.STATES.NOT_INIT
        if diag is None and grid_path is None:
            raise ValueError("Grid初始化错误: 必须提供diag或grid_path")
        self.options.mapper_input_channel = self.encoder.output_channels
        
        # 根据是新建还是加载来初始化Grid
        if grid_path is None :
            self.diag = diag # [[x_min, y_max], [x_max, y_min]] in Mercator
            self.blocks = self.__devide_blocks__(self.options.block_size)
        else:
            self.load_grid(grid_path)
            
        self.border = np.array([self.diag[:,0].min(),self.diag[:,1].min(),self.diag[:,0].max(),self.diag[:,1].max()])
        self.output_path = output_path
        self.elements:List[Element] = []
        self.transform = nn.Sequential(
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            )
        ).eval()
        self.train_data = []
        self.SAMPLE_FACTOR = options.sample_factor
        self.pred_resolution = .7
        self.vis_points_latlon = None
        self.device = device if device is not None else 'cuda'
    
    def to_device(self,device):
        """将Grid及其所有子组件移动到指定设备"""
        self.device = device
        self.encoder.to(device)
        for block in self.blocks:
            block.mapper.to(device)
        for element in self.elements:
            element.to_device(device)
    
    def __devide_blocks__(self,block_size) -> list[Block]:
        """将Grid的地理区域划分为更小的、固定大小的Blocks"""
        h,w = np.abs(self.diag[1,1] - self.diag[0,1]), np.abs(self.diag[1,0] - self.diag[0,0])
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
        blocks = []
        for diag in diags:
            map_coeffs = {
                'x':np.array([diag[:,0].min(),diag[:,0].max()]),
                'y':np.array([diag[:,1].min(),diag[:,1].max()]),
                'h':None,
                'h_min':None,
                'h_max':None
            }
            diag_ratio = np.array([
                [np.abs(diag[0,1] - self.diag[0,1]) / h , np.abs(diag[0,0] - self.diag[0,0]) / w],
                [np.abs(diag[1,1] - self.diag[0,1]) / h , np.abs(diag[1,0] - self.diag[0,0]) / w]
            ])
            block = Block(self.options,diag,diag_ratio,map_coeffs)
            blocks.append(block)
        return blocks
        
    def update_task_state(self,task_info,update_info):
        # 更新多进程任务的状态，用于在主进程中显示进度
        state = task_info['state'][task_info['id']]
        task_info['state'][task_info['id']] = {**state,**update_info}

    def fprint(self,info:str):
        # 打印信息到日志文件
        output_path = os.path.join(self.output_path,'log.txt')
        info += '\n'
        redirect_output(output_path,info)

    def get_overlap_image(self,img:RSImage,mode='bbox'):
        """根据Grid的地理范围，从一张大图中获取重叠区域的影像数据"""
        corner_samplines = img.xy_to_sampline(np.array([self.diag[0],[self.diag[1,0],self.diag[0,1]],self.diag[1],[self.diag[0,0],self.diag[1,1]]]))
        if mode == 'bbox':
            top = max(min(corner_samplines[0,1],corner_samplines[1,1]),0)
            bottom = min(max(corner_samplines[2,1],corner_samplines[3,1]),img.H-1)
            left = max(min(corner_samplines[0,0],corner_samplines[3,0]),0)
            right = min(max(corner_samplines[1,0],corner_samplines[2,0]),img.W-1)
            img_raw = img.get_image_by_sampline(np.array([left,top]),np.array([right,bottom]))
            dem = img.get_dem_by_sampline(np.array([left,top]),np.array([right,bottom]))
            return img_raw,dem,np.array([top,left]),np.array([right,bottom])
        elif mode == 'interpolate':
            # 'interpolate'模式会将四边形区域重采样为矩形
            target_h = int((self.border[3] - self.border[1]) / self.pred_resolution)
            target_w = int((self.border[2] - self.border[0]) / self.pred_resolution)
            
            # 确保目标尺寸不为0
            if target_h <= 0 or target_w <= 0:
                return None, None, None

            img_raw,local_hw2 = img.resample_image_by_sampline(corner_samplines, (target_h, target_w), need_local=True)
            dem = img.resample_dem_by_sampline(corner_samplines, (target_h, target_w))
            return img_raw,dem,local_hw2
        else:
            raise ValueError("mode should either be 'bbox' or 'interpolate'")

    def get_height_map_coeffs(self):
        """
        根据所有Element Buffer中的数据，为每个Block估计高度多项式系数，并计算真实高程范围 (h_min, h_max)。
        """
        # 确保有elements
        if not self.elements:
            self.fprint("警告: 没有任何Element，无法计算高程系数。")
            return

        # 收集所有patch的中心点坐标和高程
        all_centers_list = []
        all_heights_list = []
        for el in self.elements:
            if el.buffer and 'objs' in el.buffer and el.buffer['objs'].numel() > 0:
                obj_patches = el.buffer['objs']
                centers = F.avg_pool2d(obj_patches, kernel_size=(16, 16)).squeeze(-1).squeeze(-1)
                all_centers_list.append(centers)
                all_heights_list.append(obj_patches[:, 2, ...].flatten())

        if not all_centers_list:
            self.fprint("警告: 没有任何Element Buffer包含高度信息，无法计算高程系数。")
            return
        
        all_centers = torch.cat(all_centers_list)
        all_heights = torch.cat(all_heights_list)

        global_h_min = all_heights.min().item()
        global_h_max = all_heights.max().item()

        for block in self.blocks:
            mask = (all_centers[:, 0] >= block.border[0]) & (all_centers[:, 0] < block.border[2]) & \
                   (all_centers[:, 1] >= block.border[1]) & (all_centers[:, 1] < block.border[3])

            points_in_block = mask.sum().item()
            
            if points_in_block > 10: 
                heights_in_block = all_heights[mask]
                block.map_coeffs['h'] = get_map_coef(heights_in_block.cpu().numpy())
                block.map_coeffs['h_min'] = heights_in_block.min().item()
                block.map_coeffs['h_max'] = heights_in_block.max().item()
            else:
                self.fprint(f"警告: Block {self.blocks.index(block)} 内只有 {points_in_block} 个数据点，使用全局高程统计。")
                block.map_coeffs['h'] = get_map_coef(all_heights.cpu().numpy())
                block.map_coeffs['h_min'] = global_h_min
                block.map_coeffs['h_max'] = global_h_max


    def add_img(self,img:RSImage):
        """添加一张用于训练的影像"""
        img_raw,dem,local_hw2 = self.get_overlap_image(img,mode='interpolate')
        if img_raw is None:
            print(f"警告: 影像 {img.id} 与当前Grid无有效重叠，跳过。")
            return
        self.train_data.append({
            'img':img_raw,
            'dem':dem,
            'local':local_hw2,
            'rpc':img.rpc
        })
    
    def create_elements(self,output_path:str = None,task_info = None,clear = False):
        """为所有添加的影像数据创建Element实例"""
        if output_path is None: output_path = self.output_path
        if not task_info is None:
            self.update_task_state(task_info, {'status':f"Grid {task_info['id']}:提取特征", 'total':len(self.train_data)})
        if clear: self.elements:List[Element] = []
        for idx,data in enumerate(self.train_data):
            id = len(self.elements)
            path = os.path.join(output_path,f'element_{id}')
            os.makedirs(path,exist_ok=True)
            new_element = Element(options=self.options, encoder=self.encoder, img_raw=data['img'], dem=data['dem'], rpc=data['rpc'], id=id, output_path=path, local_raw=data['local'], device=self.device, verbose=1 if task_info is None else 0)
            self.elements.append(new_element)
            if not task_info is None: self.update_task_state(task_info,{'progress':idx+1})

    def visualize_block_assignment(self):
        """可视化函数：绘制Block划分及数据点分布"""
        self.fprint("正在可视化Block的数据分配情况...")
        
        all_points_list = []
        for element in self.elements:
            if not element.buffer or element.buffer['objs'].numel() == 0: continue
            points = element.buffer['objs'].permute(0, 2, 3, 1).reshape(-1, 3)
            all_points_list.append(points)
        
        if not all_points_list:
            self.fprint("没有可用于可视化的数据点。")
            return

        all_points = torch.cat(all_points_list, dim=0).cpu().numpy()

        sample_size = min(50000, len(all_points))
        sampled_points = all_points[np.random.choice(len(all_points), sample_size, replace=False)]

        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        colors = plt.cm.get_cmap('hsv', len(self.blocks))

        for i, block in enumerate(self.blocks):
            min_x, max_x = block.diag[0, 0], block.diag[1, 0]
            min_y, max_y = block.diag[1, 1], block.diag[0, 1]
            rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                 linewidth=2, edgecolor=colors(i), facecolor='none', label=f'Block {i}')
            ax.add_patch(rect)
            mask = (sampled_points[:, 0] >= min_x) & (sampled_points[:, 0] < max_x) & \
                   (sampled_points[:, 1] >= min_y) & (sampled_points[:, 1] < max_y)
            block_points = sampled_points[mask]
            ax.scatter(block_points[:, 0], block_points[:, 1], color=colors(i), s=1, alpha=0.5)
        
        plt.title('Grid内Block划分及数据分布')
        plt.xlabel('墨卡托坐标X (m)')
        plt.ylabel('墨卡托坐标Y (m)')
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(self.output_path, 'block_assignment_visualization.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        self.fprint(f"Block数据分配可视化图像已保存至 {save_path}")

    def train(self,task_info = None):
        """Grid的训练总控函数"""
        self.to_device(self.device)
        self.get_height_map_coeffs()
        self.visualize_block_assignment()
        for block_idx in range(len(self.blocks)):
            if self.blocks[block_idx].map_coeffs.get('h_min') is None:
                self.fprint(f"错误: Block {block_idx} 未能成功计算高程范围，跳过训练。")
                continue
            self.train_mapper(block_idx,task_info)
        
        # 训练结束后清理内存
        for element in self.elements:
            element.clear_buffer()
        self.elements.clear()
        
        if not task_info is None:
            self.update_task_state(task_info, {'status':f"Grid {task_info['id']}:训练完成"})

    def warp_by_poly(self,raw,coefs):
        """核心函数：将mapper输出的原始值，通过多项式展开，转换为绝对地理坐标"""
        coefs_x = torch.from_numpy(coefs['x']).to(raw.device, dtype=raw.dtype)
        coefs_y = torch.from_numpy(coefs['y']).to(raw.device, dtype=raw.dtype)
        coefs_h = torch.from_numpy(coefs['h']).to(raw.device, dtype=raw.dtype)
        x = (raw[...,0] + 1.) * .5 * (coefs_x[1] - coefs_x[0]) + coefs_x[0]
        y = (raw[...,1] + 1.) * .5 * (coefs_y[1] - coefs_y[0]) + coefs_y[0]
        h_poly = raw[...,2]
        h = torch.zeros_like(h_poly)
        for i in range(len(coefs_h)):
            h += coefs_h[i] * (h_poly ** (len(coefs_h) - 1 - i))
        warped = torch.stack([x,y,h],dim=-1)
        return warped

    def _normalize_coords(self, coords_abs: torch.Tensor, block: Block) -> torch.Tensor:
        """
        [核心新增] 辅助函数：将绝对坐标根据Block的边界归一化到[-1, 1]范围。
        Args:
            coords_abs (torch.Tensor): 绝对地理坐标张量, shape [..., 3] (X, Y, H)
            block (Block): 目标Block对象
        Returns:
            torch.Tensor: 归一化后的坐标张量, shape [..., 3]
        """
        # --- 准备归一化参数 ---
        min_x, max_x = block.diag[0, 0], block.diag[1, 0]
        min_y, max_y = block.diag[1, 1], block.diag[0, 1]
        h_min = block.map_coeffs['h_min']
        h_max = block.map_coeffs['h_max']

        range_x = max_x - min_x
        range_y = max_y - min_y
        range_h = h_max - h_min

        # --- 归一化X, Y ---
        # 避免除以零
        norm_x = 2 * (coords_abs[..., 0] - min_x) / (range_x + 1e-8) - 1
        norm_y = 2 * (coords_abs[..., 1] - min_y) / (range_y + 1e-8) - 1
        
        # --- 归一化H (基于真实的物理范围) ---
        if range_h < 1e-6: # 处理地形平坦的特殊情况
            norm_h = torch.zeros_like(coords_abs[..., 2])
        else:
            norm_h = 2 * (coords_abs[..., 2] - h_min) / range_h - 1
            
        return torch.stack([norm_x, norm_y, norm_h], dim=-1)

    @torch.no_grad()
    def validate_and_visualize_block(self, mapper: nn.Module, block: Block, block_idx: int, total_iter_idx: int, val_patches: dict):
        """[核心修改] 在验证集上评估模型，使用固定的验证噪声"""
        if not val_patches or val_patches['features'].numel() == 0:
            return float('nan'), float('nan')

        mapper.eval()
        
        num_val_patches = len(val_patches['features'])
        val_batch_size = min(self.options.mapper_batch_size, num_val_patches)
        
        if val_batch_size == 0:
            mapper.train()
            return float('nan'), float('nan')
            
        sample_indices = torch.randperm(num_val_patches, device=self.device)[:val_batch_size]

        # [核心修改 V3]: 确保数据类型为float
        feature_batch = val_patches['features'][sample_indices].to(torch.float32)
        obj_batch = val_patches['objs'][sample_indices].to(torch.float32)

        # [核心修改 V3]: 生成Patch级统一平移噪声
        batch_size_val = obj_batch.shape[0]
        noise_per_patch_val = torch.randn(batch_size_val, 3, 1, 1, device=obj_batch.device, dtype=obj_batch.dtype)
        noise = noise_per_patch_val * self.options.validation_noise_std
        noisy_prior_absolute_val = obj_batch + noise
        
        noisy_prior_nhw3_val = noisy_prior_absolute_val.permute(0, 2, 3, 1)
        normalized_prior_nhw3 = self._normalize_coords(noisy_prior_nhw3_val, block)
        normalized_prior_n3hw = normalized_prior_nhw3.permute(0, 3, 1, 2)
        
        mapper_input = torch.cat([feature_batch, normalized_prior_n3hw], dim=1)
        
        output_raw, _ = mapper(mapper_input)
        
        output_raw_flat = output_raw.permute(0, 2, 3, 1)
        pred_mu_absolute = self.warp_by_poly(output_raw_flat[...,:3], block.map_coeffs).permute(0, 3, 1, 2)

        error_rmse = torch.sqrt(torch.sum((pred_mu_absolute[:, :2, ...] - obj_batch[:, :2, ...])**2, dim=1))
        val_rmse = error_rmse.mean().item()
        
        error_squared = (pred_mu_absolute - obj_batch) ** 2
        val_loss_obj = error_squared[:, :2, ...].sum(dim=1).mean().item()
        
        mapper.train()
        return val_rmse, np.sqrt(val_loss_obj)

    def train_mapper(self, block_idx: int, task_info=None, save_checkpoint=True):
        """ [核心重构 V3] 完整Epoch遍历，修正负采样，简化RPC传递, Patch级噪声 """
        
        # --- 1. 初始化 ---
        block = self.blocks[block_idx]
        mapper = block.mapper
        optimizer = AdamW(mapper.parameters(), lr=self.options.grid_train_lr_max)
        
        criterion = CriterionTrainGrid(
            consistency_weight=self.options.consistency_weight,
            affine_weight=self.options.affine_weight
        )
        bce_loss_fn = nn.BCEWithLogitsLoss()
        valid_score_weight = 10.0
        
        mapper.train()
        min_loss = 1e8
        best_mapper_state_dict = None
        
        # --- 2. 数据源准备 ---
        self.fprint(f"为Block {block_idx} 准备数据源...")
        all_element_patches = [elem.buffer for elem in self.elements]
        
        val_patches = {}
        for element in self.elements:
            if element.validation_buffer and element.validation_buffer['features'].numel() > 0:
                if not val_patches:
                    val_patches = {k: v.clone() for k, v in element.validation_buffer.items()}
                else:
                    for key in val_patches:
                        val_patches[key] = torch.cat([val_patches[key], element.validation_buffer[key]], dim=0)

        # --- 3. 即时索引构建 (Just-in-Time Indexing) ---
        self.fprint(f"为Block {block_idx} 即时构建训练与验证索引...")

        train_indices_for_block = {}
        neg_indices_pool_for_block = {} 

        for element_id, element_data in enumerate(all_element_patches):
            if not element_data or 'objs' not in element_data or element_data['objs'].numel() == 0:
                continue
            
            obj_patches = element_data['objs']
            centers = F.avg_pool2d(obj_patches, kernel_size=(16, 16)).squeeze(-1).squeeze(-1)
            
            pos_mask = (centers[:, 0] >= block.border[0]) & (centers[:, 0] < block.border[2]) & \
                       (centers[:, 1] >= block.border[1]) & (centers[:, 1] < block.border[3])
            
            valid_pos_indices = torch.where(pos_mask)[0]
            if valid_pos_indices.numel() > 0:
                train_indices_for_block[element_id] = valid_pos_indices.to(self.device)

            valid_neg_indices = torch.where(~pos_mask)[0]
            if valid_neg_indices.numel() > 0:
                neg_indices_pool_for_block[element_id] = valid_neg_indices.to(self.device)

        filtered_val_patches = {}
        if val_patches and val_patches.get('features') is not None and val_patches['features'].numel() > 0:
            val_obj_patches = val_patches['objs']
            val_centers = F.avg_pool2d(val_obj_patches, kernel_size=(16, 16)).squeeze(-1).squeeze(-1)
            
            val_pos_mask = (val_centers[:, 0] >= block.border[0]) & (val_centers[:, 0] < block.border[2]) & \
                           (val_centers[:, 1] >= block.border[1]) & (val_centers[:, 1] < block.border[3])
                           
            val_indices_for_block = torch.where(val_pos_mask)[0].to(self.device)

            if val_indices_for_block.numel() > 0:
                for key in val_patches:
                    filtered_val_patches[key] = val_patches[key][val_indices_for_block]

        if not train_indices_for_block:
            self.fprint(f"错误：未能为Block {block_idx} 提取任何训练Patch，跳过。")
            return
        
        # --- 4. 训练循环设置 ---
        all_batches = []
        for eid, indices in train_indices_for_block.items():
            for i in range(0, len(indices), self.options.mapper_batch_size):
                all_batches.append((eid, i))
        
        total_training_steps = len(all_batches) * self.options.num_epochs
        
        scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                     total_steps=total_training_steps,
                                     warmup_ratio=self.options.grid_warmup_epochs / self.options.num_epochs if self.options.num_epochs > 0 else 0,
                                     cooldown_ratio=self.options.grid_cooldown_epochs / self.options.num_epochs if self.options.num_epochs > 0 else 0)

        self.fprint(f"Block {block_idx} 索引构建完成. 每个Epoch包含 {len(all_batches)} 个批次. 总训练步数: {total_training_steps}")

        if task_info: 
            self.update_task_state(task_info, {'status':f"Grid {task_info['id']}:Block {block_idx + 1}/{len(self.blocks)} 训练", 'total':total_training_steps, 'progress': 0})
        else:
            pbar = tqdm(total=total_training_steps, desc=f"训练 Block {block_idx+1}")
        
        latest_val_loss_obj = float('nan')
        warmup_ratio = self.options.grid_warmup_epochs / self.options.num_epochs
        current_total_iter = 0

        # --- 5. Epoch-based 训练循环 ---
        for epoch in range(self.options.num_epochs):
            random.shuffle(all_batches) # 每个epoch都打乱批次顺序

            for element_id, i in all_batches:
                optimizer.zero_grad()
                
                indices = train_indices_for_block[element_id]
                element_patches = all_element_patches[element_id]
                current_rpc = self.elements[element_id].rpc
                
                indices_to_sample = indices[i : i + self.options.mapper_batch_size]
                
                feature_batch = element_patches['features'][indices_to_sample].to(torch.float32)
                obj_batch = element_patches['objs'][indices_to_sample].to(torch.float32)
                conf_batch = element_patches['confs'][indices_to_sample].to(torch.float32)
                local_batch = element_patches['locals'][indices_to_sample].to(torch.float32)

                # 正样本处理
                feature_pos = feature_batch.repeat(2, 1, 1, 1)
                obj_pos = obj_batch.repeat(2, 1, 1, 1)
                conf_pos = conf_batch.repeat(2, 1, 1, 1)
                local_pos = local_batch.repeat(2, 1, 1, 1)

                noise = torch.randn_like(feature_pos)
                proj = torch.sum(feature_pos * noise, dim=1, keepdim=True) / (torch.sum(feature_pos * feature_pos, dim=1, keepdim=True) + 1e-8) * feature_pos
                orthogonal_noise = F.normalize(noise - proj, p=2, dim=1)
                noisy_features = F.normalize(feature_pos + self.options.feature_noise_level * orthogonal_noise, p=2, dim=1)

                progress = min(1.,current_total_iter / (total_training_steps * warmup_ratio)) if total_training_steps > 0 else 0
                current_noise_std = self.options.prior_noise_min + (self.options.prior_noise_max - self.options.prior_noise_min) * progress
                
                # 生成Patch级统一平移噪声
                noise_per_patch_pos = torch.randn(obj_pos.shape[0], 3, 1, 1, device=obj_pos.device, dtype=obj_pos.dtype) * current_noise_std
                noisy_prior_patch = obj_pos + noise_per_patch_pos
                
                noisy_prior = noisy_prior_patch.permute(0, 2, 3, 1)
                normalized_prior = self._normalize_coords(noisy_prior, block).permute(0, 3, 1, 2)
                mapper_input_pos = torch.cat([noisy_features, normalized_prior], dim=1)
                
                output_raw_pos, valid_score_pos = mapper(mapper_input_pos)
                
                output_raw_flat = output_raw_pos.permute(0, 2, 3, 1)
                pred_mu_pos = self.warp_by_poly(output_raw_flat[...,:3], block.map_coeffs).permute(0, 3, 1, 2)
                pred_log_sigma_pos = output_raw_flat[...,3:].permute(0, 3, 1, 2)
                
                loss_regression, loss_details = criterion(epoch, self.options.num_epochs, pred_mu_pos, pred_log_sigma_pos, obj_pos, conf_pos, local_pos, current_rpc)
                
                # 负样本处理
                valid_score_neg = torch.empty(0, 1, 16, 16, device=self.device)
                neg_indices = neg_indices_pool_for_block.get(element_id)
                if neg_indices is not None and neg_indices.numel() > 0:
                    num_neg_patches = len(neg_indices)
                    neg_sample_size = min(len(feature_batch), num_neg_patches)
                    neg_perm = torch.randperm(num_neg_patches, device=self.device)[:neg_sample_size]
                    neg_indices_to_sample = neg_indices[neg_perm]

                    feature_neg = element_patches['features'][neg_indices_to_sample].to(torch.float32)

                    num_pos = len(obj_batch)
                    pos_perm = torch.randint(0, num_pos, (neg_sample_size,), device=self.device)
                    obj_for_neg = obj_batch[pos_perm]

                    noise_per_patch_neg = torch.randn(obj_for_neg.shape[0], 3, 1, 1, device=obj_for_neg.device, dtype=obj_for_neg.dtype) * current_noise_std
                    noisy_prior_patch_neg = obj_for_neg + noise_per_patch_neg
                    
                    noisy_prior_neg = noisy_prior_patch_neg.permute(0, 2, 3, 1)
                    normalized_prior_neg = self._normalize_coords(noisy_prior_neg, block).permute(0, 3, 1, 2)
                    mapper_input_neg = torch.cat([feature_neg, normalized_prior_neg], dim=1)
                    valid_score_neg = mapper.forward_valid(mapper_input_neg)

                valid_scores = torch.cat([valid_score_pos, valid_score_neg], dim=0)
                if valid_scores.numel() > 0:
                    labels = torch.cat([torch.ones_like(valid_score_pos), torch.zeros_like(valid_score_neg)], dim=0)
                    loss_valid = bce_loss_fn(valid_scores, labels)
                    total_loss_for_batch = loss_regression + loss_valid * valid_score_weight
                    loss_details['v'] = loss_valid.item()
                else:
                    total_loss_for_batch = loss_regression
                
                total_loss_for_batch.backward()
                optimizer.step()
                scheduler.step()
                
                if total_loss_for_batch.item() < min_loss:
                    min_loss = total_loss_for_batch.item()
                    best_mapper_state_dict = deepcopy(mapper.state_dict())
                
                info = { 'e': f'{epoch+1}', 'me': f'{self.options.num_epochs}', 'lr':f'{scheduler.get_last_lr()[0]:.2e}', 'loss': f'{total_loss_for_batch.item():.2f}', **{k: f'{v:.2f}' for k, v in loss_details.items()}, 'val_obj': f'{latest_val_loss_obj:.2f}'}
                current_total_iter += 1
                if task_info:
                    self.update_task_state(task_info, {'progress': current_total_iter, 'info': info})
                else:
                    pbar.update(1)
                    pbar.set_postfix(info)
            
            if (epoch + 1) % self.options.validation_epoch_interval == 0:
                val_rmse, val_loss_obj = self.validate_and_visualize_block(mapper, block, block_idx, current_total_iter, filtered_val_patches)
                if not np.isnan(val_rmse):
                    latest_val_loss_obj = val_loss_obj
        
        if not task_info: pbar.close()
        
        if best_mapper_state_dict is not None:
             mapper.load_state_dict(best_mapper_state_dict)
        block.status = self.STATES.WELL_TRAINED if min_loss < 25. else self.STATES.BAD_TRAINED
        self.save_grid()

    def save_grid(self):
        """保存Grid的状态，包括所有Blocks的模型权重"""
        state_dict = {
            'diag':torch.from_numpy(self.diag),
            'mapper_blocks_num':self.options.mapper_blocks_num,
            'block_num':len(self.blocks),
            **{f'block_{block_idx}':block.get_block_state_dict() for block_idx,block in enumerate(self.blocks)}
        }
        torch.save(state_dict,os.path.join(self.output_path,'grid_data.pth'))

    def load_grid(self,path:str):
        """[核心修改] 从文件加载Grid的状态，包含高程范围"""
        state_dict = torch.load(os.path.join(path,'grid_data.pth'), map_location='cpu')
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
                'h_min': block_state_dict.get('map_coeffs_h_min', torch.tensor(0.0)).item(),
                'h_max': block_state_dict.get('map_coeffs_h_max', torch.tensor(0.0)).item()
            }
            block = Block(self.options,block_diag,block_diag_ratio,block_map_coeffs)
            block.mapper.load_state_dict(block_state_dict['mapper'])
            block.status = block_state_dict['status']
            self.blocks.append(block)        
        print(f"Grid '{name}' 加载成功")

    @torch.no_grad()
    def _extract_full_features(self, img_raw: np.ndarray) -> torch.Tensor:
        """
        一个辅助函数，封装了从原始影像中提取全局特征的逻辑，专用于诊断。
        返回一个包含所有特征点的 "点云式" Tensor。
        """
        H, W = img_raw.shape[:2]
        self.encoder.eval().to(self.device)
        self.transform.to(self.device)

        # --- 一次性提取全图特征 ---
        self.fprint("正在为诊断影像提取全局特征...")
        crop_size = self.options.crop_size
        step = crop_size // 2
        y_starts = np.unique(np.append(np.arange(0, H - crop_size, step), H - crop_size)).astype(int)
        x_starts = np.unique(np.append(np.arange(0, W - crop_size, step), W - crop_size)).astype(int)

        all_features = []
        # 使用tqdm来显示进度
        for row in tqdm(y_starts, desc="为诊断提取特征"):
            batch_imgs = []
            for col in x_starts:
                img_crop = img_raw[row:row + crop_size, col:col + crop_size]
                img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float().div(255.0)
                batch_imgs.append(img_tensor)

            if not batch_imgs: continue

            batch_tensor = torch.stack(batch_imgs).to(self.device)
            batch_tensor = self.transform(batch_tensor)
            
            features_b, _ = self.encoder(batch_tensor)
            # 展平并收集特征
            all_features.append(features_b.permute(0, 2, 3, 1).reshape(-1, self.encoder.output_channels))
        
        if not all_features:
            return torch.empty(0, self.encoder.output_channels, device=self.device)
            
        return torch.cat(all_features, dim=0)

    @torch.no_grad()
    def pred_xyh(self, img_raw: np.ndarray, dem: np.ndarray, local_hw2: np.ndarray, rpc: RPCModelParameterTorch) -> Dict[str, torch.Tensor]:
        """
        [核心修改] 对新的影像进行密集地理坐标预测 (引入坐标先验)。
        采用“一次提取，按需分发”的高效策略。
        """
        if img_raw is None or img_raw.size == 0:
            return {}
            
        H, W = img_raw.shape[:2]
        self.encoder.eval().to(self.device)
        self.transform.to(self.device)

        # --- 1. [核心修改] 预计算全局坐标先验 ---
        self.fprint("正在为预测影像预计算全局坐标先验...")
        local_flat = local_hw2.reshape(-1, 2)
        dem_flat = dem.reshape(-1)
        lats, lons = rpc.RPC_PHOTO2OBJ(
            local_flat[:, 1],
            local_flat[:, 0],
            dem_flat
        )
        xy = project_mercator(torch.stack([lats, lons], dim=-1))[:, [1, 0]].to('cpu', dtype=torch.float32)
        prior_abs_flat = torch.cat([xy, torch.from_numpy(dem_flat).to(dtype=torch.float32).unsqueeze(-1)], dim=-1)
        prior_abs_hw3 = prior_abs_flat.reshape(H, W, 3)

        # --- 2. 一次性提取全图特征，构建一个“点云式”的全局Buffer ---
        self.fprint("正在为预测影像提取全局特征...")
        crop_size = self.options.crop_size
        num_h = 1 if H <= crop_size else self.options.crop_num_h
        num_w = 1 if W <= crop_size else self.options.crop_num_w
        y_starts = np.linspace(0, max(0, H - crop_size), num_h, dtype=int)
        x_starts = np.linspace(0, max(0, W - crop_size), num_w, dtype=int)

        full_buffer = {'features': [], 'locals': [], 'confs': [], 'priors': []}

        for row in tqdm(y_starts, desc="提取特征与先验"):
            for col in x_starts:
                img_crop = img_raw[row:row + crop_size, col:col + crop_size]
                local_crop = local_hw2[row:row + crop_size, col:col + crop_size]
                prior_crop = prior_abs_hw3[row:row + crop_size, col:col + crop_size, :]
                
                img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
                img_tensor = self.transform(img_tensor)
                
                features_b, confs_b = self.encoder(img_tensor)
                local_down = downsample(torch.from_numpy(local_crop).unsqueeze(0), self.SAMPLE_FACTOR).to(self.device)
                prior_down = downsample(prior_crop.unsqueeze(0), self.SAMPLE_FACTOR).to(self.device)

                full_buffer['features'].append(features_b.permute(0, 2, 3, 1).reshape(-1, self.encoder.output_channels))
                full_buffer['locals'].append(local_down.reshape(-1, 2))
                full_buffer['confs'].append(confs_b.permute(0, 2, 3, 1).reshape(-1))
                full_buffer['priors'].append(prior_down.reshape(-1, 3))

        
        full_buffer_features = torch.cat(full_buffer['features'], dim=0).to(dtype=torch.float32)
        full_buffer_locals = torch.cat(full_buffer['locals'], dim=0).to(dtype=torch.float32)
        full_buffer_confs = torch.cat(full_buffer['confs'], dim=0).to(dtype=torch.float32)
        full_buffer_priors = torch.cat(full_buffer['priors'], dim=0).to(dtype=torch.float32)
        self.fprint(f"全局特征与先验提取完成，共 {len(full_buffer_features)} 个唯一特征点。")

        # --- 3. 遍历所有Block，按需筛选并分发数据进行预测 ---
        all_results = {'mu_xyh_P3': [], 'sigma_xyh_P3': [], 'locals_P2': [], 'confs_P1': [], 'valid_score_P1': []}

        for block in tqdm(self.blocks, desc="分区预测"):
            block.mapper.eval().to(self.device)

            min_x, max_x = block.diag[0, 0], block.diag[1, 0]
            min_y, max_y = block.diag[1, 1], block.diag[0, 1]

            mask = (full_buffer_priors[:, 0] >= min_x) & (full_buffer_priors[:, 0] < max_x) & \
                   (full_buffer_priors[:, 1] >= min_y) & (full_buffer_priors[:, 1] < max_y)
            
            if not mask.any():
                continue

            block_features = full_buffer_features[mask]
            block_locals = full_buffer_locals[mask]
            block_confs = full_buffer_confs[mask]
            block_priors = full_buffer_priors[mask]

            num_points = len(block_features)
            batch_size = self.options.mapper_batch_size * 16 * 16 
            
            for i in range(0, num_points, batch_size):
                feature_batch = block_features[i:i+batch_size]
                prior_batch = block_priors[i:i+batch_size]

                # [核心修改] 归一化坐标先验并与特征拼接
                normalized_prior_batch = self._normalize_coords(prior_batch, block)
                feature_batch_img = feature_batch.unsqueeze(-1).unsqueeze(-1)
                normalized_prior_img = normalized_prior_batch.unsqueeze(-1).unsqueeze(-1)
                mapper_input = torch.cat([feature_batch_img, normalized_prior_img], dim=1)
                
                output, valid_score = block.mapper(mapper_input)
                
                output_flat = output.permute(0, 2, 3, 1).reshape(-1, 6)
                valid_score_flat = valid_score.permute(0, 2, 3, 1).reshape(-1)

                pred_mu_flat = self.warp_by_poly(output_flat[:, :3], block.map_coeffs)
                pred_sigma_flat = torch.exp(output_flat[:, 3:])

                all_results['mu_xyh_P3'].append(pred_mu_flat)
                all_results['sigma_xyh_P3'].append(pred_sigma_flat)
                all_results['locals_P2'].append(block_locals[i:i+batch_size])
                all_results['confs_P1'].append(block_confs[i:i+batch_size])
                all_results['valid_score_P1'].append(valid_score_flat)

        # --- 4. 整合所有结果 ---
        if not all_results['mu_xyh_P3']:
            return {k: torch.empty(0, v, device=self.device) for k, v in {'mu_xyh_P3': 3, 'sigma_xyh_P3': 3, 'locals_P2': 2, 'confs_P1': 1, 'valid_score_P1': 1}.items()}

        final_res = {
            'mu_xyh_P3': torch.cat(all_results['mu_xyh_P3'], dim=0),
            'sigma_xyh_P3': torch.cat(all_results['sigma_xyh_P3'], dim=0),
            'locals_P2': torch.cat(all_results['locals_P2'], dim=0),
            'confs_P1': torch.cat(all_results['confs_P1'], dim=0),
            'valid_score_P1': torch.cat(all_results['valid_score_P1'], dim=0)
        }

        return final_res

