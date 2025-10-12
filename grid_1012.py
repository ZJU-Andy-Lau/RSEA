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
                'h':None
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
            return img_raw,dem,np.array([top,left]),np.array([bottom,right])
        elif mode == 'interpolate':
            # 'interpolate'模式会将四边形区域重采样为矩形
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
        """根据所有Element Buffer中的数据，为每个Block估计高度多项式的系数"""
        heights = torch.cat([el.buffer['objs'][..., 2].flatten() for el in self.elements]).cpu().numpy()
        xys = torch.cat([el.buffer['objs'][..., :2].reshape(-1, 2) for el in self.elements]).cpu().numpy()
        
        for block in self.blocks:
            mask = (xys[:,0] >= block.diag[0,0]) & (xys[:,1] <= block.diag[0,1]) & \
                   (xys[:,0] < block.diag[1,0]) & (xys[:,1] > block.diag[1,1])
            if np.any(mask):
                block.map_coeffs['h'] = get_map_coef(heights[mask])
            else: # 如果Block内没有点，则用整个Grid的点来估计
                block.map_coeffs['h'] = get_map_coef(heights)

    def add_img(self,img:RSImage):
        """添加一张用于训练的影像"""
        img_raw,dem,local_hw2 = self.get_overlap_image(img,mode='interpolate')
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
        
        all_points = []
        for element in self.elements:
            if element.buffer['objs'].numel() == 0: continue
            points = element.buffer['objs'].reshape(-1, 3).cpu().numpy()
            all_points.append(points)
        
        if not all_points:
            self.fprint("没有可用于可视化的数据点。")
            return

        all_points = np.concatenate(all_points, axis=0)

        # 随机采样一部分点进行可视化，避免图像过于拥挤
        sample_size = min(50000, len(all_points))
        sampled_points = all_points[np.random.choice(len(all_points), sample_size, replace=False)]

        plt.figure(figsize=(12, 12))
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        
        colors = plt.cm.get_cmap('hsv', len(self.blocks))

        for i, block in enumerate(self.blocks):
            # 绘制Block的矩形边框
            min_x, max_x = block.diag[0, 0], block.diag[1, 0]
            min_y, max_y = block.diag[1, 1], block.diag[0, 1]
            rect = plt.Rectangle((min_x, min_y), max_x - min_x, max_y - min_y,
                                 linewidth=2, edgecolor=colors(i), facecolor='none', label=f'Block {i}')
            ax.add_patch(rect)

            # 筛选出落入当前Block内的数据点
            mask = (sampled_points[:, 0] >= min_x) & (sampled_points[:, 0] < max_x) & \
                   (sampled_points[:, 1] >= min_y) & (sampled_points[:, 1] < max_y)
            block_points = sampled_points[mask]

            # 用与边框相同的颜色绘制数据点
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
        self.get_height_map_coeffs()
        self.visualize_block_assignment()
        # 依次训练每个Block
        for block_idx in range(len(self.blocks)):
            self.train_mapper(block_idx,task_info)
        # 训练完成后清理内存
        for element in self.elements:
            element.clear_buffer()
        self.elements = None
        if not task_info is None:
            self.update_task_state(task_info, {'status':f"Grid {task_info['id']}:训练完成"})

    def warp_by_poly(self,raw,coefs):
        """
        核心函数：将mapper输出的原始值，通过多项式展开，转换为绝对地理坐标。
        这是您原始代码中的关键映射步骤。
        """
        # raw shape: (N, 3, ph, pw), coefs是numpy数组
        # 将numpy系数转换为tensor并移动到与raw相同的设备
        coefs_x = torch.from_numpy(coefs['x']).to(raw.device, dtype=raw.dtype)
        coefs_y = torch.from_numpy(coefs['y']).to(raw.device, dtype=raw.dtype)
        coefs_h = torch.from_numpy(coefs['h']).to(raw.device, dtype=raw.dtype)
        
        # 将范围在[-1, 1]的x, y通道，线性映射到当前Block的地理坐标范围
        x = (raw[:,0] + 1.) * .5 * (coefs_x[1] - coefs_x[0]) + coefs_x[0]
        y = (raw[:,1] + 1.) * .5 * (coefs_y[1] - coefs_y[0]) + coefs_y[0]
        
        # 将h通道的值作为输入，应用多项式变换
        h_poly = raw[:,2]
        h = torch.zeros_like(h_poly)
        for i in range(len(coefs_h)):
            h += coefs_h[i] * (h_poly ** (len(coefs_h) - 1 - i))

        # 将计算出的x, y, h拼接成最终的坐标预测
        warped = torch.stack([x,y,h],dim=1)
        return warped

    @torch.no_grad()
    def validate_and_visualize_block(self, mapper: nn.Module, block: Block, block_idx: int, iter_idx: int, val_indices: List[Tuple[int, int]]):
        """在验证集上评估模型，并生成散点图"""
        if not val_indices:
            return float('nan'), None, None

        mapper.eval() # 切换到评估模式
        
        patch_h, patch_w = 16, 16
        val_batch_size = self.options.patches_per_batch // 2

        all_features, all_objs = [], []

        # 从验证集中随机采样一个批次
        for _ in range(val_batch_size):
            element_idx, window_idx = random.choice(val_indices)
            element = self.elements[element_idx]
            if element.validation_buffer['features'].numel() == 0: continue
            _, _, H, W = element.validation_buffer['features'].shape
            y, x = random.randint(0, H - patch_h), random.randint(0, W - patch_w)
            
            all_features.append(element.validation_buffer['features'][window_idx, :, y:y+patch_h, x:x+patch_w])
            all_objs.append(element.validation_buffer['objs'][window_idx, y:y+patch_h, x:x+patch_w, :])

        if not all_features:
            mapper.train() # 切回训练模式
            return float('nan'), None, None
            
        feature_batch = torch.stack(all_features)
        obj_batch = torch.stack(all_objs).permute(0, 3, 1, 2)

        # 前向传播
        output_raw, _ = mapper(feature_batch)
        pred_mu_absolute = self.warp_by_poly(output_raw[:, :3, :, :], block.map_coeffs)

        # 计算XY平面上的均方根误差（RMSE）
        error = torch.sqrt(torch.sum((pred_mu_absolute[:, :2, ...] - obj_batch[:, :2, ...])**2, dim=1))
        val_rmse = error.mean().item()

        # 准备绘图数据
        pred_coords = pred_mu_absolute.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        true_coords = obj_batch.permute(0, 2, 3, 1).reshape(-1, 3).cpu().numpy()
        
        # 绘制散点图
        plt.figure(figsize=(10, 10))
        plt.scatter(true_coords[:, 0], true_coords[:, 1], s=5, c='blue', alpha=0.6, label='真实坐标')
        plt.scatter(pred_coords[:, 0], pred_coords[:, 1], s=5, c='red', marker='x', alpha=0.6, label='预测坐标')
        for i in range(len(true_coords)):
            plt.plot([true_coords[i, 0], pred_coords[i, 0]], [true_coords[i, 1], pred_coords[i, 1]], 'gray', linewidth=0.5, alpha=0.5)
        plt.title(f'验证集散点图 - Block {block_idx}, 迭代 {iter_idx}')
        plt.xlabel('墨卡托坐标X (m)'); plt.ylabel('墨卡托坐标Y (m)'); plt.legend(); plt.grid(True)
        ax = plt.gca(); ax.set_aspect('equal', adjustable='box')
        
        plot_dir = os.path.join(self.output_path, f'block_{block_idx}_plots', 'validation')
        os.makedirs(plot_dir, exist_ok=True)
        save_path = os.path.join(plot_dir, f'val_scatter_iter_{iter_idx}.png')
        plt.savefig(save_path, dpi=150)
        plt.close()
        
        mapper.train() # 切回训练模式
        return val_rmse, pred_coords, true_coords

    def train_mapper(self,block_idx:int,task_info = None,save_checkpoint = True):
        """为指定的Block训练mapper模型"""
        # --- 1. 初始化 ---
        block = self.blocks[block_idx]
        mapper = block.mapper
        optimizer = AdamW(mapper.parameters(),lr=self.options.grid_train_lr_max)
        # 使用OneCycleLR调度器以实现更快的收敛
        scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.options.grid_train_lr_max, total_steps=self.options.grid_training_iters, anneal_strategy='cos')
        criterion = CriterionTrainGrid()
        
        mapper.train().to(self.device)
        min_loss = 1e8
        best_mapper_state_dict = None
        
        patch_h, patch_w = 16, 16
        patches_per_batch = self.options.patches_per_batch
        num_positive_samples = patches_per_batch // 2
        num_negative_samples = patches_per_batch - num_positive_samples

        # --- 2. 准备数据 (效率优化) ---
        # 在训练开始前，一次性将所有需要用到的Element数据移动到GPU
        for el in self.elements:
            el.to_device(self.device)

        # --- 3. 划分正/负/验证样本索引 ---
        # 这个操作在CPU上完成，因为它是一次性的
        train_pos_indices, train_neg_indices, val_pos_indices = [], [], []
        
        for element_idx, element in enumerate(self.elements):
            # 处理训练buffer
            if element.buffer['features'].numel() > 0:
                window_centers = element.buffer['objs'].mean(dim=(1, 2)).cpu()
                in_mask = (window_centers[:, 0] >= block.diag[0, 0]) & (window_centers[:, 1] <= block.diag[0, 1]) & \
                          (window_centers[:, 0] < block.diag[1, 0]) & (window_centers[:, 1] > block.diag[1, 1])
                # 属于当前Block的为正样本
                for window_idx in torch.where(in_mask)[0]: train_pos_indices.append((element_idx, window_idx.item()))
                # 不属于的为负样本
                for window_idx in torch.where(~in_mask)[0]: train_neg_indices.append((element_idx, window_idx.item()))
            
            # 处理验证buffer
            if element.validation_buffer['features'].numel() > 0:
                val_window_centers = element.validation_buffer['objs'].mean(dim=(1, 2)).cpu()
                val_in_mask = (val_window_centers[:, 0] >= block.diag[0, 0]) & (val_window_centers[:, 1] <= block.diag[0, 1]) & \
                              (val_window_centers[:, 0] < block.diag[1, 0]) & (val_window_centers[:, 1] > block.diag[1, 1])
                for window_idx in torch.where(val_in_mask)[0]: val_pos_indices.append((element_idx, window_idx.item()))

        if not train_pos_indices or not train_neg_indices:
            print(f"警告: Block {block_idx} 缺少正样本或负样本，跳过训练。")
            return
        
        vis_interval, val_interval = 500, 500

        if not task_info is None:
            self.update_task_state(task_info, {'status':f"Grid {task_info['id']}:Block {block_idx + 1}/{len(self.blocks)} 训练", 'total':self.options.grid_training_iters})
        else:
            pbar = tqdm(total=self.options.grid_training_iters, desc=f"训练 Block {block_idx+1}")
            
        # --- 4. 主训练循环 ---
        for iter_idx in range(self.options.grid_training_iters):
            optimizer.zero_grad()
            
            # --- 4a. 高效采样批次数据 (在GPU上进行) ---
            all_features, all_objs, all_confs, all_locals = [], [], [], []
            
            # 采样正样本
            pos_sample_indices = torch.randint(0, len(train_pos_indices), (num_positive_samples,))
            for i in pos_sample_indices:
                element_idx, window_idx = train_pos_indices[i]
                element = self.elements[element_idx]
                _, _, H, W = element.buffer['features'].shape
                y, x = random.randint(0, H - patch_h), random.randint(0, W - patch_w)
                all_features.append(element.buffer['features'][window_idx, :, y:y+patch_h, x:x+patch_w])
                all_objs.append(element.buffer['objs'][window_idx, y:y+patch_h, x:x+patch_w, :])
                all_confs.append(element.buffer['confs'][window_idx, :, y:y+patch_h, x:x+patch_w])
                all_locals.append(element.buffer['locals'][window_idx, y:y+patch_h, x:x+patch_w, :])

            # 采样负样本
            neg_sample_indices = torch.randint(0, len(train_neg_indices), (num_negative_samples,))
            for i in neg_sample_indices:
                element_idx, window_idx = train_neg_indices[i]
                element = self.elements[element_idx]
                _, _, H, W = element.buffer['features'].shape
                y, x = random.randint(0, H - patch_h), random.randint(0, W - patch_w)
                all_features.append(element.buffer['features'][window_idx, :, y:y+patch_h, x:x+patch_w])
                all_objs.append(element.buffer['objs'][window_idx, y:y+patch_h, x:x+patch_w, :])
                all_confs.append(element.buffer['confs'][window_idx, :, y:y+patch_h, x:x+patch_w])
                all_locals.append(element.buffer['locals'][window_idx, y:y+patch_h, x:x+patch_w, :])
            
            feature_batch = torch.stack(all_features)
            obj_batch_absolute = torch.stack(all_objs).permute(0, 3, 1, 2)
            conf_batch = torch.stack(all_confs)
            local_batch = torch.stack(all_locals).permute(0, 3, 1, 2)
            
            # 准备valid_score的标签
            positive_labels = torch.ones(num_positive_samples, 1, patch_h, patch_w, device=self.device)
            negative_labels = torch.zeros(num_negative_samples, 1, patch_h, patch_w, device=self.device)
            valid_labels = torch.cat([positive_labels, negative_labels], dim=0)
            
            # --- 4b. 前向传播 ---
            output_raw, valid_score_batch = mapper(feature_batch)
            
            # 使用warp_by_poly计算绝对坐标预测
            pred_mu_absolute = self.warp_by_poly(output_raw[:, :3, :, :], block.map_coeffs)
            pred_log_sigma_batch = output_raw[:, 3:, :, :]
            
            # --- 4c. 计算损失 ---
            loss, loss_details = criterion(iter_idx, self.options.grid_training_iters, pred_mu_absolute, pred_log_sigma_batch, obj_batch_absolute, conf_batch, local_batch, element.rpc, valid_score_batch, valid_labels, num_positive_samples)
            
            # --- 4d. 反向传播与优化 ---
            loss.backward()
            optimizer.step()
            scheduler.step()

            # --- 4e. 记录与日志 ---
            if loss.item() < min_loss:
                min_loss = loss.item()
                best_mapper_state_dict = deepcopy(mapper.state_dict())
            
            # 格式化日志输出，保留两位小数
            info = { 'lr':f'{scheduler.get_last_lr()[0]:.2e}', 'loss': f'{loss.item():.2f}', **{k: f'{v:.2f}' for k, v in loss_details.items()} }
            
            # --- 4f. 周期性验证 ---
            if (iter_idx + 1) % val_interval == 0:
                val_rmse, _, _ = self.validate_and_visualize_block(mapper, block, block_idx, iter_idx + 1, val_pos_indices)
                if not np.isnan(val_rmse):
                    info['val_err'] = f'{val_rmse:.2f}m'
            
            # --- 4g. 周期性可视化训练过程 ---
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
        """从文件加载Grid的状态"""
        state_dict = torch.load(os.path.join(path,'grid_data.pth'), map_location=self.device)
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
        print(f"Grid '{name}' 加载成功")

    @torch.no_grad()
    def pred_xyh(self,img_raw:np.ndarray,local_hw2:np.ndarray) -> Dict[str,np.ndarray]:
        """对新的影像进行密集地理坐标预测"""
        H,W = img_raw.shape[:2]
        self.encoder.eval().to(self.device)
        self.transform.to(self.device)

        crop_size = self.options.crop_size
        step = crop_size // 2
        
        y_starts = np.unique(np.append(np.arange(0, H - crop_size, step), H - crop_size)).astype(int)
        x_starts = np.unique(np.append(np.arange(0, W - crop_size, step), W - crop_size)).astype(int)

        all_mu_xyh, all_sigma_xyh, all_locals, all_confs, all_valid_scores = [], [], [], [], []
        
        for row in tqdm(y_starts, desc="正在预测"):
            for col in x_starts:
                img_crop = img_raw[row:row + crop_size, col:col + crop_size]
                local_crop = local_hw2[row:row + crop_size, col:col + crop_size]

                img_tensor = torch.from_numpy(img_crop).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(self.device)
                img_tensor = self.transform(img_tensor)
                
                features_b, confs_b = self.encoder(img_tensor)
                local_down = downsample(torch.from_numpy(local_crop).unsqueeze(0), self.SAMPLE_FACTOR, device=self.device)
                
                crop_preds_mu, crop_preds_sigma, crop_valid_scores = [], [], []

                for block in self.blocks:
                    block.mapper.eval().to(self.device)
                    output, valid_score = block.mapper(features_b)
                    pred_mu = self.warp_by_poly(output[:, :3, ...], block.map_coeffs)
                    # 使用sigmoid将valid_score转换为(0,1)的概率，并用其加权预测结果
                    valid_prob = torch.sigmoid(valid_score)
                    crop_preds_mu.append(pred_mu * valid_prob)
                    crop_preds_sigma.append(torch.exp(output[:, 3:, ...]))
                    crop_valid_scores.append(valid_prob)
                
                # 对所有Block的预测进行加权平均
                sum_valid_scores = torch.stack(crop_valid_scores).sum(dim=0).clamp(min=1e-8)
                avg_pred_mu = torch.stack(crop_preds_mu).sum(dim=0) / sum_valid_scores
                avg_pred_sigma = torch.stack(crop_preds_sigma).mean(dim=0)
                avg_valid_score = sum_valid_scores / len(self.blocks)

                all_mu_xyh.append(avg_pred_mu.permute(0,2,3,1).reshape(-1, 3))
                all_sigma_xyh.append(avg_pred_sigma.permute(0,2,3,1).reshape(-1, 3))
                all_locals.append(local_down.reshape(-1, 2))
                all_confs.append(confs_b.permute(0,2,3,1).reshape(-1))
                all_valid_scores.append(avg_valid_score.permute(0,2,3,1).reshape(-1))

        mu_xyh_P3 = torch.cat(all_mu_xyh, dim=0)
        sigma_xyh_P3 = torch.cat(all_sigma_xyh, dim=0)
        locals_P2 = torch.cat(all_locals, dim=0)
        confs_P1 = torch.cat(all_confs, dim=0)
        valid_score_P1 = torch.cat(all_valid_scores, dim=0)
        
        # 仅保留valid_score > 0.5的可靠预测
        valid_mask = valid_score_P1 > 0.5
        
        # 对重叠区域的预测进行平均
        unique_locals, inverse_indices = torch.unique(torch.round(locals_P2[valid_mask]), dim=0, return_inverse=True)
        
        mu_xyh_filtered = mu_xyh_P3[valid_mask]
        
        mu_xyh_aggregated = torch.zeros((unique_locals.shape[0], 3), device=self.device, dtype=torch.float32)
        mu_xyh_aggregated.scatter_add_(0, inverse_indices.unsqueeze(1).expand(-1, 3), mu_xyh_filtered)
        
        counts = torch.zeros((unique_locals.shape[0],), device=self.device, dtype=torch.float32)
        counts.scatter_add_(0, inverse_indices, torch.ones_like(inverse_indices, dtype=torch.float32))
        
        mu_xyh_P3_unique = mu_xyh_aggregated / counts.unsqueeze(1).clamp(min=1)
        
        # 为其他值（方差、置信度等）找到对应的唯一索引
        _, unique_indices_for_others = np.unique(inverse_indices.cpu().numpy(), return_index=True)

        res = {
            'mu_xyh_P3': mu_xyh_P3_unique,
            'sigma_xyh_P3': sigma_xyh_P3[valid_mask][unique_indices_for_others],
            'locals_P2': unique_locals,
            'confs_P1': confs_P1[valid_mask][unique_indices_for_others],
            'valid_score_P1': valid_score_P1[valid_mask][unique_indices_for_others]
        }

        return res

