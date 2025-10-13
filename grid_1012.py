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
        [核心修改] 根据所有Element Buffer中的数据，为每个Block估计高度多项式系数，并计算真实高程范围 (h_min, h_max)。
        """
        heights_list = [el.buffer['objs'][..., 2].flatten() for el in self.elements if el.buffer and 'objs' in el.buffer]
        if not heights_list: 
            self.fprint("警告: 没有任何Element Buffer包含高度信息，无法计算高程系数。")
            return
        
        heights_all = torch.cat(heights_list)
        xys_all = torch.cat([el.buffer['objs'][..., :2].reshape(-1, 2) for el in self.elements if el.buffer and 'objs' in el.buffer])

        global_h_min = heights_all.min().item()
        global_h_max = heights_all.max().item()

        for block in self.blocks:
            # 使用PyTorch进行高效的掩码操作
            mask = (xys_all[:,0] >= block.diag[0,0]) & (xys_all[:,1] <= block.diag[0,1]) & \
                   (xys_all[:,0] < block.diag[1,0]) & (xys_all[:,1] > block.diag[1,1])
            
            points_in_block = mask.sum().item()
            
            if points_in_block > 10: # 确保有足够的数据点
                heights_in_block = heights_all[mask]
                block.map_coeffs['h'] = get_map_coef(heights_in_block.cpu().numpy())
                block.map_coeffs['h_min'] = heights_in_block.min().item()
                block.map_coeffs['h_max'] = heights_in_block.max().item()
            else:
                # 如果块内数据太少，使用全局统计作为回退
                self.fprint(f"警告: Block {self.blocks.index(block)} 内只有 {points_in_block} 个数据点，使用全局高程统计。")
                block.map_coeffs['h'] = get_map_coef(heights_all.cpu().numpy())
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
            points = element.buffer['objs'].reshape(-1, 3)
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
            if self.blocks[block_idx].map_coeffs['h_min'] is None:
                self.fprint(f"错误: Block {block_idx} 未能成功计算高程范围，跳过训练。")
                continue
            self.train_mapper(block_idx,task_info)
        for element in self.elements:
            element.clear_buffer()
        self.elements = None
        if not task_info is None:
            self.update_task_state(task_info, {'status':f"Grid {task_info['id']}:训练完成"})

    def warp_by_poly(self,raw,coefs):
        """核心函数：将mapper输出的原始值，通过多项式展开，转换为绝对地理坐标"""
        coefs_x = torch.from_numpy(coefs['x']).to(raw.device, dtype=raw.dtype)
        coefs_y = torch.from_numpy(coefs['y']).to(raw.device, dtype=raw.dtype)
        coefs_h = torch.from_numpy(coefs['h']).to(raw.device, dtype=raw.dtype)
        x = (raw[:,0] + 1.) * .5 * (coefs_x[1] - coefs_x[0]) + coefs_x[0]
        y = (raw[:,1] + 1.) * .5 * (coefs_y[1] - coefs_y[0]) + coefs_y[0]
        h_poly = raw[:,2]
        h = torch.zeros_like(h_poly)
        for i in range(len(coefs_h)):
            h += coefs_h[i] * (h_poly ** (len(coefs_h) - 1 - i))
        warped = torch.stack([x,y,h],dim=1)
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
    def validate_and_visualize_block(self, mapper: nn.Module, block: Block, block_idx: int, iter_idx: int, val_patch_indices: List[Tuple[int, int, int, int]], criterion: nn.Module):
        """[核心修改] 在验证集上评估模型，对齐训练和验证任务"""
        if not val_patch_indices:
            return float('nan'), float('nan')

        mapper.eval()
        
        patch_h, patch_w = 16, 16
        val_batch_size = min(self.options.patches_per_batch // 2, len(val_patch_indices))
        
        all_features, all_objs = [], [],
        sample_indices = torch.randint(0, len(val_patch_indices), (val_batch_size,))
        for i in sample_indices:
            element_idx, window_idx, y, x = val_patch_indices[i]
            element = self.elements[element_idx]
            all_features.append(element.validation_buffer['features'][window_idx, :, y:y+patch_h, x:x+patch_w])
            all_objs.append(element.validation_buffer['objs'][window_idx, y:y+patch_h, x:x+patch_w, :])

        if not all_features:
            mapper.train()
            return float('nan'), float('nan')
        
        feature_batch = torch.stack(all_features).to(torch.float32)
        obj_batch = torch.stack(all_objs).permute(0, 3, 1, 2).to(torch.float32)

        # --- [核心修改] 为验证集的坐标先验注入同分布的噪声，确保任务一致性 ---
        noise = torch.randn_like(obj_batch) * self.options.prior_noise_std
        noisy_prior_absolute_val = obj_batch + noise
        
        noisy_prior_nhw3_val = noisy_prior_absolute_val.permute(0, 2, 3, 1)
        normalized_prior_nhw3 = self._normalize_coords(noisy_prior_nhw3_val, block)
        normalized_prior_n3hw = normalized_prior_nhw3.permute(0, 3, 1, 2)
        
        mapper_input = torch.cat([feature_batch, normalized_prior_n3hw], dim=1)
        
        output_raw, _ = mapper(mapper_input)
        pred_mu_absolute = self.warp_by_poly(output_raw[:, :3, :, :], block.map_coeffs)
        
        # 验证误差仍然用预测值和“无噪声的真实坐标”进行比较
        error_rmse = torch.sqrt(torch.sum((pred_mu_absolute[:, :2, ...] - obj_batch[:, :2, ...])**2, dim=1))
        val_rmse = error_rmse.mean().item()
        
        error_squared = (pred_mu_absolute - obj_batch) ** 2
        val_loss_obj = error_squared[:, :2, ...].sum(dim=1).mean().item()
        
        pred_coords_all = pred_mu_absolute.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        true_coords_all = obj_batch.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        
        # 可视化时，可以选择一个样本进行对比
        if val_batch_size > 0:
            patch_to_vis_idx = random.randint(0, val_batch_size - 1)
            pred_coords_single = pred_mu_absolute[patch_to_vis_idx].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()
            true_coords_single = obj_batch[patch_to_vis_idx].permute(1, 2, 0).reshape(-1, 3).cpu().numpy()

            plot_dir = os.path.join(self.output_path, f'block_{block_idx}_plots', 'validation')
            os.makedirs(plot_dir, exist_ok=True)
            
            # ... (可视化代码保持不变)
            plt.figure(figsize=(10, 10))
            plt.scatter(true_coords_all[:, 0], true_coords_all[:, 1], s=5, c='blue', alpha=0.6, label='真实坐标')
            plt.scatter(pred_coords_all[:, 0], pred_coords_all[:, 1], s=5, c='red', marker='x', alpha=0.6, label='预测坐标')
            plt.title(f'验证集总体散点图 - Block {block_idx}, 迭代 {iter_idx}')
            plt.xlabel('墨卡托坐标X (m)'); plt.ylabel('墨卡托坐标Y (m)'); plt.legend(); plt.grid(True)
            ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
            save_path_all = os.path.join(plot_dir, f'val_scatter_iter_{iter_idx}.png')
            plt.savefig(save_path_all, dpi=150)
            plt.close()

            plt.figure(figsize=(10, 10))
            plt.scatter(true_coords_single[:, 0], true_coords_single[:, 1], s=15, c='blue', alpha=0.8, label='真实 Patch 形状')
            plt.scatter(pred_coords_single[:, 0], pred_coords_single[:, 1], s=15, c='red', marker='x', alpha=0.8, label='预测 Patch 形状')
            plt.title(f'单个Patch形状对比 - Block {block_idx}, 迭代 {iter_idx}')
            plt.xlabel('墨卡托坐标X (m)'); plt.ylabel('墨卡托坐标Y (m)'); plt.legend(); plt.grid(True)
            ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
            save_path_single = os.path.join(plot_dir, f'val_single_patch_iter_{iter_idx}.png')
            plt.savefig(save_path_single, dpi=150)
            plt.close()
        
        mapper.train()
        return val_rmse, np.sqrt(val_loss_obj)

    def train_mapper(self,block_idx:int,task_info = None,save_checkpoint = True):
        """ 核心训练函数 """
        # --- 1. 初始化 ---
        block = self.blocks[block_idx]
        mapper = block.mapper
        optimizer = AdamW(mapper.parameters(),lr=self.options.grid_train_lr_max)
        scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                     total_steps=self.options.grid_training_iters,
                                     warmup_ratio=self.options.grid_warmup_iters / self.options.grid_training_iters,
                                     cooldown_ratio=self.options.grid_cooldown_iters / self.options.grid_training_iters)
        criterion = CriterionTrainGrid()
        
        mapper.train()
        min_loss = 1e8
        best_mapper_state_dict = None
        
        patch_h, patch_w = 16, 16
        patches_per_batch = self.options.patches_per_batch
        num_positive_samples = patches_per_batch // 2
        num_negative_samples = patches_per_batch - num_positive_samples
        
        # --- 2. [高效向量化] 构建精确到Patch级别的正/负/验证样本索引 ---
        if task_info: self.update_task_state(task_info, {'status': f"Grid {task_info['id']}:Block {block_idx + 1} 索引预处理", 'total': len(self.elements), 'progress': 0})
        self.fprint(f"为Block {block_idx} 构建精确的Patch索引...")
        
        positive_patches, negative_patches, val_patch_indices = [], [], []
        block_min_x, block_max_x = min(block.diag[:,0]),max(block.diag[:,0])
        block_min_y, block_max_y = min(block.diag[:,1]),max(block.diag[:,1])

        for element_idx, element in enumerate(self.elements):
            # --- [核心修改] 使用池化操作进行高效的“完全包含”判断 ---
            
            # 处理训练buffer
            if element.buffer and element.buffer['features'].numel() > 0:
                objs_tensor = element.buffer['objs'].permute(0, 3, 1, 2).to(torch.float32)
                x_coords = objs_tensor[:, 0:1, :, :]
                y_coords = objs_tensor[:, 1:2, :, :]

                patch_x_max = F.max_pool2d(x_coords, kernel_size=(patch_h, patch_w), stride=1)
                patch_y_max = F.max_pool2d(y_coords, kernel_size=(patch_h, patch_w), stride=1)
                patch_x_min = -F.max_pool2d(-x_coords, kernel_size=(patch_h, patch_w), stride=1)
                patch_y_min = -F.max_pool2d(-y_coords, kernel_size=(patch_h, patch_w), stride=1)
                
                is_positive_mask = (patch_x_min >= block_min_x) & (patch_x_max < block_max_x) & \
                                   (patch_y_min >= block_min_y) & (patch_y_max < block_max_y)
                
                pos_indices = torch.where(is_positive_mask)
                for i in range(len(pos_indices[0])):
                    positive_patches.append((element_idx, pos_indices[0][i].item(), pos_indices[1][i].item(), pos_indices[2][i].item()))
                
                neg_indices = torch.where(~is_positive_mask)
                for i in range(len(neg_indices[0])):
                    negative_patches.append((element_idx, neg_indices[0][i].item(), neg_indices[1][i].item(), neg_indices[2][i].item()))

            # 处理验证buffer
            if element.validation_buffer and element.validation_buffer['features'].numel() > 0:
                val_objs_tensor = element.validation_buffer['objs'].permute(0, 3, 1, 2).to(torch.float32)
                val_x_coords = val_objs_tensor[:, 0:1, :, :]
                val_y_coords = val_objs_tensor[:, 1:2, :, :]

                val_patch_x_max = F.max_pool2d(val_x_coords, kernel_size=(patch_h, patch_w), stride=1)
                val_patch_y_max = F.max_pool2d(val_y_coords, kernel_size=(patch_h, patch_w), stride=1)
                val_patch_x_min = -F.max_pool2d(-val_x_coords, kernel_size=(patch_h, patch_w), stride=1)
                val_patch_y_min = -F.max_pool2d(-val_y_coords, kernel_size=(patch_h, patch_w), stride=1)

                is_val_positive_mask = (val_patch_x_min >= block_min_x) & (val_patch_x_max < block_max_x) & \
                                       (val_patch_y_min >= block_min_y) & (val_patch_y_max < block_max_y)
                                       
                val_pos_indices = torch.where(is_val_positive_mask)
                for i in range(len(val_pos_indices[0])):
                    val_patch_indices.append((element_idx, val_pos_indices[0][i].item(), val_pos_indices[1][i].item(), val_pos_indices[2][i].item()))
            
            if task_info: self.update_task_state(task_info, {'progress': element_idx + 1})

        self.fprint(f"Block {block_idx} 索引构建完成: {len(positive_patches)} 个正样本, {len(negative_patches)} 个负样本, {len(val_patch_indices)} 个验证样本。")

        if not positive_patches:
            print(f"警告: Block {block_idx} 缺少完全位于内部的正样本，跳过训练。")
            return
        if not negative_patches: # 允许没有负样本的情况，但需要调整采样逻辑
             print(f"警告: Block {block_idx} 缺少负样本，将仅使用正样本进行训练。")
             num_positive_samples = patches_per_batch
             num_negative_samples = 0
        
        vis_interval, val_interval = 500, 500
        
        if task_info: self.update_task_state(task_info, {'status':f"Grid {task_info['id']}:Block {block_idx + 1}/{len(self.blocks)} 训练", 'total':self.options.grid_training_iters, 'progress': 0})
        else:
            pbar = tqdm(total=self.options.grid_training_iters, desc=f"训练 Block {block_idx+1}")
            
        latest_val_loss_obj = float('nan')
        
        # --- 3. 主训练循环 ---
        for iter_idx in range(self.options.grid_training_iters):
            optimizer.zero_grad()
            
            # --- 3a. 高效采样批次数据 ---
            all_features, all_objs, all_confs, all_locals, all_element_indices = [], [], [], [], []
            
            pos_sample_indices = torch.randint(0, len(positive_patches), (num_positive_samples,))
            for i in pos_sample_indices:
                element_idx, window_idx, y, x = positive_patches[i]
                element = self.elements[element_idx]
                all_features.append(element.buffer['features'][window_idx, :, y:y+patch_h, x:x+patch_w])
                all_objs.append(element.buffer['objs'][window_idx, y:y+patch_h, x:x+patch_w, :])
                all_confs.append(element.buffer['confs'][window_idx, :, y:y+patch_h, x:x+patch_w])
                all_locals.append(element.buffer['locals'][window_idx, y:y+patch_h, x:x+patch_w, :])
                all_element_indices.append(element_idx)

            if num_negative_samples > 0:
                neg_sample_indices = torch.randint(0, len(negative_patches), (num_negative_samples,))
                for i in neg_sample_indices:
                    element_idx, window_idx, y, x = negative_patches[i]
                    element = self.elements[element_idx]
                    all_features.append(element.buffer['features'][window_idx, :, y:y+patch_h, x:x+patch_w])
                    all_objs.append(element.buffer['objs'][window_idx, y:y+patch_h, x:x+patch_w, :])
                    all_confs.append(element.buffer['confs'][window_idx, :, y:y+patch_h, x:x+patch_w])
                    all_locals.append(element.buffer['locals'][window_idx, y:y+patch_h, x:x+patch_w, :])
                    all_element_indices.append(element_idx)
            
            feature_batch = torch.stack(all_features)
            obj_batch_absolute = torch.stack(all_objs).permute(0, 3, 1, 2)
            conf_batch = torch.stack(all_confs)
            local_batch = torch.stack(all_locals).permute(0, 3, 1, 2)

            # --- 显式转换所有批处理张量为 float32 ---
            feature_batch = feature_batch.to(torch.float32)
            obj_batch_absolute = obj_batch_absolute.to(torch.float32)
            conf_batch = conf_batch.to(torch.float32)
            local_batch = local_batch.to(torch.float32)
            
            positive_labels = torch.ones(num_positive_samples, 1, patch_h, patch_w, device=self.device)
            negative_labels = torch.zeros(num_negative_samples, 1, patch_h, patch_w, device=self.device)
            valid_labels = torch.cat([positive_labels, negative_labels], dim=0).to(torch.float32)
            
            # --- 3b. 生成并融合含噪坐标先验 ---
            noise = torch.randn_like(obj_batch_absolute) * self.options.prior_noise_std
            noisy_prior_absolute = obj_batch_absolute + noise
            noisy_prior_absolute_nhw3 = noisy_prior_absolute.permute(0, 2, 3, 1)
            normalized_prior_nhw3 = self._normalize_coords(noisy_prior_absolute_nhw3, block)
            normalized_prior_n3hw = normalized_prior_nhw3.permute(0, 3, 1, 2)
            mapper_input = torch.cat([feature_batch, normalized_prior_n3hw], dim=1)

            # --- 3c. 前向传播 ---
            output_raw, valid_score_batch = mapper(mapper_input)
            pred_mu_absolute = self.warp_by_poly(output_raw[:, :3, :, :], block.map_coeffs)
            pred_log_sigma_batch = output_raw[:, 3:, :, :]
            
            # --- 3d. 计算损失 ---
            loss, loss_details = criterion(iter_idx, self.options.grid_training_iters, pred_mu_absolute, pred_log_sigma_batch, obj_batch_absolute, conf_batch, local_batch, self.elements, all_element_indices, valid_score_batch, valid_labels, num_positive_samples)
            
            # --- 3e. 反向传播与优化 ---
            loss.backward()
            optimizer.step()
            scheduler.step()

            # --- 3f. 记录与日志 ---
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_mapper_state_dict = deepcopy(mapper.state_dict())
            
            info = { 'lr':f'{scheduler.get_last_lr()[0]:.2e}', 'loss': f'{loss.item():.2f}', **{k: f'{v:.2f}' for k, v in loss_details.items()}, 'val_obj': f'{latest_val_loss_obj:.2f}'}
            
            # --- 3g. 周期性验证 ---
            if (iter_idx + 1) % val_interval == 0:
                val_rmse, val_loss_obj = self.validate_and_visualize_block(mapper, block, block_idx, iter_idx + 1, val_patch_indices, criterion)
                if not np.isnan(val_rmse):
                    info['val_err'] = f'{val_rmse:.2f}'
                    latest_val_loss_obj = val_loss_obj
            
            # --- 3h. 周期性可视化训练过程 ---
            if (iter_idx + 1) % vis_interval == 0:
                true_coords_train = obj_batch_absolute[:num_positive_samples].permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
                pred_coords_train = pred_mu_absolute[:num_positive_samples].permute(0, 2, 3, 1).reshape(-1, 3).detach().cpu().numpy()
                
                plt.figure(figsize=(10, 10))
                plt.scatter(true_coords_train[:, 0], true_coords_train[:, 1], s=5, c='blue', alpha=0.6, label='真实坐标')
                plt.scatter(pred_coords_train[:, 0], pred_coords_train[:, 1], s=5, c='red', marker='x', alpha=0.6, label='预测坐标')
                plt.title(f'训练集散点图 - Block {block_idx}, 迭代 {iter_idx + 1}')
                plt.xlabel('墨卡托坐标X (m)'); plt.ylabel('墨卡托坐标Y (m)'); plt.legend(); plt.grid(True)
                ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
                plot_dir = os.path.join(self.output_path, f'block_{block_idx}_plots', 'training')
                os.makedirs(plot_dir, exist_ok=True)
                plt.savefig(os.path.join(plot_dir, f'train_scatter_iter_{iter_idx + 1}.png'), dpi=150)
                plt.close()

            if task_info:
                self.update_task_state(task_info, {'progress': iter_idx + 1, 'info': info})
            else:
                pbar.update(1)
                pbar.set_postfix(info)
                
        if not task_info: pbar.close()
        
        # --- 5. 结束训练，保存最佳模型 ---
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
            torch.from_numpy(local_flat[:, 1]).to(self.device, dtype=torch.float32),
            torch.from_numpy(local_flat[:, 0]).to(self.device, dtype=torch.float32),
            torch.from_numpy(dem_flat).to(self.device, dtype=torch.float32)
        )
        xy = project_mercator(torch.stack([lats, lons], dim=-1))[:, [1, 0]]
        prior_abs_flat = torch.cat([xy, torch.from_numpy(dem_flat).to(self.device, dtype=torch.float32).unsqueeze(-1)], dim=-1)
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
                local_down = downsample(torch.from_numpy(local_crop).unsqueeze(0), self.SAMPLE_FACTOR, device=self.device)
                prior_down = downsample(prior_crop.unsqueeze(0), self.SAMPLE_FACTOR, device=self.device)

                full_buffer['features'].append(features_b.permute(0, 2, 3, 1).reshape(-1, self.encoder.output_channels))
                full_buffer['locals'].append(local_down.reshape(-1, 2))
                full_buffer['confs'].append(confs_b.permute(0, 2, 3, 1).reshape(-1))
                full_buffer['priors'].append(prior_down.reshape(-1, 3))

        # 拼接列表成一个大的Tensor，并进行去重
        all_locals_flat = torch.cat(full_buffer['locals'], dim=0)
        _, unique_indices = np.unique(all_locals_flat.cpu().numpy(), axis=0, return_index=True)
        
        full_buffer_features = torch.cat(full_buffer['features'], dim=0)[unique_indices]
        full_buffer_locals = all_locals_flat[unique_indices]
        full_buffer_confs = torch.cat(full_buffer['confs'], dim=0)[unique_indices]
        full_buffer_priors = torch.cat(full_buffer['priors'], dim=0)[unique_indices]
        self.fprint(f"全局特征与先验提取完成，共 {len(full_buffer_features)} 个唯一特征点。")

        # --- 3. 遍历所有Block，按需筛选并分发数据进行预测 ---
        all_results = {'mu_xyh_P3': [], 'sigma_xyh_P3': [], 'locals_P2': [], 'confs_P1': [], 'valid_score_P1': []}

        for block in tqdm(self.blocks, desc="分区预测"):
            if block.status != self.STATES.WELL_TRAINED:
                continue
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
            batch_size = self.options.patches_per_batch * 16 * 16 
            
            for i in range(0, num_points, batch_size):
                feature_batch = block_features[i:i+batch_size]
                prior_batch = block_priors[i:i+batch_size]

                # [核心修改] 归一化坐标先验并与特征拼接
                normalized_prior_batch = self._normalize_coords(prior_batch, block)
                feature_batch_img = feature_batch.unsqueeze(-1).unsqueeze(-1)
                normalized_prior_img = normalized_prior_batch.unsqueeze(-1).unsqueeze(-1).permute(0, 3, 1, 2)
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

