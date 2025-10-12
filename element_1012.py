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

from utils import get_coord_mat,project_mercator,mercator2lonlat,downsample,bilinear_interpolate,apply_polynomial,get_map_coef

from rpc import RPCModelParameterTorch

import kornia.augmentation as K
from tqdm import tqdm

from typing import List,Dict, Tuple

class Element():
    """
    Element类代表单张影像在一个特定Grid内的影像数据单元。
    它负责对这部分影像进行裁切、特征提取，并为后续的训练准备数据 Buffer。
    """
    def __init__(self,options,encoder:Encoder,img_raw:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch,id:int,output_path:str,top_left_linesamp:np.ndarray = None,local_raw:np.ndarray = None,device:str = None,verbose:int = 0):
        # --- 1. 初始化基本属性 ---
        self.options = options
        self.id = id
        self.verbose = verbose
        self.device = device if device is not None else 'cuda'

        # --- 2. 加载影像、DEM和RPC模型 ---
        self.img_raw = img_raw  # 原始影像数据 (H, W, 3)
        cv2.imwrite(os.path.join(output_path,f'img_{id}.png'),img_raw) # 保存一份原始影像用于检查
        
        # local_raw 存储了每个像素对应的原始影像中的行列号（linesamp坐标）
        if local_raw is None:
            # 如果没有提供，则根据影像尺寸自己生成
            self.local_raw = get_coord_mat(self.img_raw.shape[0],self.img_raw.shape[1])
            if top_left_linesamp is not None:
                self.local_raw += top_left_linesamp
        else:
            self.local_raw = local_raw
        
        self.dem = dem  # 数字高程模型 (H, W)
        self.rpc = rpc  # RPC相机模型
        self.H,self.W = self.img_raw.shape[:2] # 影像的高和宽

        # --- 3. 定义图像增强/预处理流程 ---
        self.transform = nn.Sequential(
            # 随机颜色抖动
            K.ColorJitter(
                brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1, p=.3,
            ),
            # 随机高斯模糊
            K.RandomGaussianBlur(kernel_size=(3, 3), sigma=(0.1, 2.0), p=0.2),
            # 随机颜色反转
            K.RandomInvert(p=0.1),
            # 标准化 (使用ImageNet的均值和标准差)
            K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            ),
        )
        
        # --- 4. 初始化模型 ---
        self.encoder = encoder.eval() # 特征提取器，设为评估模式
        self.mapper = Decoder(in_channels=self.encoder.output_channels,block_num=options.mapper_blocks_num) # 坐标回归器
        self.output_path = output_path
        
        # --- 5. 生成训练和验证数据集 ---
        # 裁切用于训练的窗口
        self.crop_imgs_train, self.crop_locals_train, self.crop_dems_train = self.__crop_training_img__(options.crop_size)
        # 裁切用于验证的窗口
        self.crop_imgs_val, self.crop_locals_val, self.crop_dems_val = self.__crop_validation_img__(options.crop_size)

        # --- 6. 提取特征并构建数据Buffer ---
        self.SAMPLE_FACTOR = self.options.sample_factor # 从配置中获取下采样因子
        self.buffer, self.validation_buffer = self.__extract_features__()
        
        self._log(f"===========================Element {self.id} 初始化完成===========================")
        self._log(f"影像尺寸: {img_raw.shape}")
        self._log(f"生成了 {self.buffer['features'].shape[0]} 个窗口用于训练。")
        self._log(f"生成了 {self.validation_buffer['features'].shape[0]} 个窗口用于验证。")
        self._log("=================================================================================")
    
    def _log(self, *args, **kwargs):
        # 日志打印函数
        if self.verbose:
            print(f"[Element {self.id}]:", *args, **kwargs)

    def __crop_training_img__(self, crop_size=1024, random_ratio=1., rotation_angle=10.):
        """
        为训练集裁切数据窗口。
        包含两部分：1. 高效的均匀裁切，保证用最少的窗口覆盖全图。2. 随机旋转裁切，增加数据多样性。
        """
        self._log("正在为训练集裁切数据窗口...")
        H, W = self.img_raw.shape[:2]
        
        crop_imgs, crop_locals, crop_dems = [], [], []

        # --- 第一部分: 高效的均匀裁切 ---
        self._log("--- 步骤1: 执行均匀裁切")
        # 计算在H和W方向上保证覆盖全图所需要的最少步数
        num_steps_h = int(np.ceil(H / crop_size)) if H > crop_size else 1
        num_steps_w = int(np.ceil(W / crop_size)) if W > crop_size else 1
        # 使用linspace生成均匀分布的起始坐标，确保边界也能被覆盖
        y_starts = np.linspace(0, H - crop_size, num_steps_h, dtype=int)
        x_starts = np.linspace(0, W - crop_size, num_steps_w, dtype=int)
        # 遍历所有起始坐标进行裁切
        for row in y_starts:
            for col in x_starts:
                crop_imgs.append(self.img_raw[row:row + crop_size, col:col + crop_size])
                crop_locals.append(self.local_raw[row:row + crop_size, col:col + crop_size])
                crop_dems.append(self.dem[row:row + crop_size, col:col + crop_size])
        n_uniform = len(crop_imgs)
        self._log(f"--- 完成了 {n_uniform} 个均匀裁切窗口")

        # --- 第二部分: 安全的随机旋转裁切 ---
        n_random = int(n_uniform * random_ratio)
        if n_random > 0:
            self._log(f"--- 步骤2: 执行 {n_random} 个随机旋转裁切")
            # 为保证旋转后的窗口完全在图像内，需要在一个“安全区域”内采样中心点
            # 这个安全区域的边界到原图边界的距离，至少是窗口对角线的一半
            half_diag = int(np.sqrt(2) * crop_size / 2) + 1
            safe_top, safe_left = half_diag, half_diag
            safe_bottom, safe_right = H - half_diag, W - half_diag
            if safe_bottom > safe_top and safe_right > safe_left:
                # 批量生成随机中心点和旋转角度，以提高效率
                center_y = np.random.randint(safe_top, safe_bottom, n_random)
                center_x = np.random.randint(safe_left, safe_right, n_random)
                angles = np.random.uniform(-rotation_angle, rotation_angle, n_random)
                for cy, cx, angle in zip(center_y, center_x, angles):
                    # 获取旋转矩阵
                    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
                    # 对影像、局部坐标和DEM应用同一个仿射变换
                    flags = cv2.INTER_LINEAR | cv2.WARP_INVERSE_MAP
                    rotated_img = cv2.warpAffine(self.img_raw, M, (W, H), flags=flags)
                    rotated_local = cv2.warpAffine(self.local_raw, M, (W, H), flags=flags)
                    rotated_dem = cv2.warpAffine(self.dem, M, (W, H), flags=flags)
                    # 从旋转后的大图中裁切出中心位置的窗口
                    tl_x, tl_y = cx - crop_size // 2, cy - crop_size // 2
                    crop_imgs.append(rotated_img[tl_y:tl_y + crop_size, tl_x:tl_x + crop_size])
                    crop_locals.append(rotated_local[tl_y:tl_y + crop_size, tl_x:tl_x + crop_size])
                    crop_dems.append(rotated_dem[tl_y:tl_y + crop_size, tl_x:tl_x + crop_size])

        return np.stack(crop_imgs), np.stack(crop_locals), np.stack(crop_dems)

    def __crop_validation_img__(self, crop_size=1024, val_count=32):
        """
        为验证集裁切数据窗口。
        采用稀疏的随机采样，与训练集不重叠，用于评估模型的泛化能力。
        """
        self._log("正在为验证集裁切数据窗口...")
        H, W = self.img_raw.shape[:2]
        crop_imgs, crop_locals, crop_dems = [], [], []

        if H < crop_size or W < crop_size:
            self._log("--- 影像尺寸过小，跳过验证集裁切。")
            return np.array([]), np.array([]), np.array([])

        # 在整个影像范围内随机生成不带旋转的验证窗口
        for _ in range(val_count):
            row = np.random.randint(0, H - crop_size + 1)
            col = np.random.randint(0, W - crop_size + 1)
            crop_imgs.append(self.img_raw[row:row + crop_size, col:col + crop_size])
            crop_locals.append(self.local_raw[row:row + crop_size, col:col + crop_size])
            crop_dems.append(self.dem[row:row + crop_size, col:col + crop_size])

        if not crop_imgs:
            return np.array([]), np.array([]), np.array([])
            
        return np.stack(crop_imgs), np.stack(crop_locals), np.stack(crop_dems)


    @torch.no_grad()
    def __extract_features_for_set__(self, crop_imgs_np, crop_locals_np, crop_dems_np) -> Dict[str, torch.Tensor]:
        """
        一个通用的特征提取函数，用于处理一个集合（训练集或验证集）的裁切窗口。
        """
        # 如果没有裁切窗口，返回空的字典
        if crop_imgs_np.size == 0:
            return {'features': torch.empty(0), 'confs': torch.empty(0), 'locals': torch.empty(0), 'objs': torch.empty(0)}

        self._log("--- 步骤1: 图像预处理与增强")
        # 将Numpy数组转换为Tensor, 调整维度顺序 (B, H, W, C) -> (B, C, H, W)，并归一化到[0,1]
        imgs_NCHW = torch.from_numpy(crop_imgs_np).permute(0,3,1,2).float() / 255.0
        
        # 分批次进行图像增强，防止显存爆炸
        transformed_imgs_list = []
        batch_num_transform = int(np.ceil(imgs_NCHW.shape[0] / self.options.batch_size))
        for b in range(batch_num_transform):
             # 将当前批次数据移到GPU并应用transform
             transformed_imgs_list.append(self.transform(imgs_NCHW[b * self.options.batch_size : (b+1) * self.options.batch_size].to(self.device)))
        imgs_NCHW = torch.cat(transformed_imgs_list, dim=0)

        # --- 步骤2: 对坐标和高程图进行下采样，以匹配特征图的分辨率 ---
        locals_NHW2 = torch.from_numpy(crop_locals_np)
        self._log("--- 步骤2a: 下采样局部坐标")
        locals_Nhw2 = downsample(locals_NHW2,self.SAMPLE_FACTOR,use_cuda=True,show_detail=bool(self.verbose),mode='avg',device=self.device)
        dems_NHW = torch.from_numpy(crop_dems_np)
        self._log("--- 步骤2b: 下采样DEM")
        dems_Nhw = downsample(dems_NHW,self.SAMPLE_FACTOR,use_cuda=True,show_detail=bool(self.verbose),mode='avg',device=self.device)
        
        # --- 步骤3: 批量提取特征 ---
        batch_num_extract = int(np.ceil(crop_imgs_np.shape[0] / self.options.batch_size))
        features_list, confs_list, locals_list, dems_list = [], [], [], []
        
        pbar_title = "--- 步骤3: 提取深度特征"
        pbar = tqdm(total=batch_num_extract, desc=pbar_title) if self.verbose > 0 else range(batch_num_extract)

        for batch_idx in pbar:
            batch_imgs = imgs_NCHW[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size]
            # 通过Encoder模型提取特征和置信度
            feat_bdhw, conf_b1hw = self.encoder(batch_imgs)
            # 将结果移回CPU存储，以节省GPU显存
            features_list.append(feat_bdhw.cpu())
            confs_list.append(conf_b1hw.cpu())
            # 收集对应批次的下采样坐标和DEM
            locals_list.append(locals_Nhw2[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].cpu())
            dems_list.append(dems_Nhw[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].cpu())
            if isinstance(pbar, tqdm): pbar.update(1)
        if isinstance(pbar, tqdm): pbar.close()

        # 将所有批次的结果拼接成一个大的四维张量
        features_BDhw = torch.cat(features_list, dim=0)
        confs_B1hw = torch.cat(confs_list, dim=0)
        locals_Bhw2 = torch.cat(locals_list, dim=0)
        dems_Bhw = torch.cat(dems_list, dim=0)

        # --- 步骤4: 计算每个特征点对应的地理坐标 ---
        self._log("--- 步骤4: 计算地理坐标 (objs)")
        B, h, w, _ = locals_Bhw2.shape
        
        # 为了使用GPU加速RPC计算，先把需要的数据展平并移到GPU
        locals_flat_samp = locals_Bhw2[..., 1].flatten().to(self.device)
        locals_flat_line = locals_Bhw2[..., 0].flatten().to(self.device)
        dems_flat = dems_Bhw.flatten().to(self.device)
        
        # 调用RPC模型进行从影像坐标到地理坐标的转换
        lats, lons = self.rpc.RPC_PHOTO2OBJ(locals_flat_samp, locals_flat_line, dems_flat)
        
        # 将经纬度坐标投影为墨卡托坐标
        xy = project_mercator(torch.stack([lats, lons], dim=-1))[:, [1, 0]]
        
        # 将地理坐标(X, Y, H)拼接起来，并恢复其原始的四维形状，最后移回CPU存储
        objs_Bhw3 = torch.cat([xy, dems_flat.unsqueeze(-1)], dim=-1).reshape(B, h, w, 3).cpu()

        # 返回包含所有提取信息的字典
        return {
            'features': features_BDhw,  # (B, D, h, w)
            'confs': confs_B1hw,      # (B, 1, h, w)
            'locals': locals_Bhw2,    # (B, h, w, 2)
            'objs': objs_Bhw3         # (B, h, w, 3)
        }

    @torch.no_grad()
    def __extract_features__(self) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        """
        主特征提取函数，负责调用通用提取函数来分别处理训练集和验证集。
        """
        # 确保所有模型都在正确的设备上
        self.to_device(self.device)
        
        self._log("开始为训练集提取特征...")
        training_buffer = self.__extract_features_for_set__(self.crop_imgs_train, self.crop_locals_train, self.crop_dems_train)
        
        self._log("开始为验证集提取特征...")
        validation_buffer = self.__extract_features_for_set__(self.crop_imgs_val, self.crop_locals_val, self.crop_dems_val)
        
        return training_buffer, validation_buffer

    def clear_buffer(self):
        """
        训练完成后，释放大的Buffer以节省内存。
        """
        del self.buffer
        self.buffer = None
        del self.validation_buffer
        self.validation_buffer = None
    
    def to_device(self,device):
        """
        将Element的所有组件（模型、数据Buffer）移动到指定的设备。
        """
        self.device = device
        self.encoder.to(device)
        self.mapper.to(device)
        self.rpc.to_gpu(device)
        self.transform.to(device) # 图像增强模块也需要移动
        
        # 如果Buffer已创建，也将其移动到设备
        if hasattr(self, 'buffer') and self.buffer is not None and self.buffer['features'].numel() > 0:
            for key in self.buffer.keys():
                self.buffer[key] = self.buffer[key].to(device)
        if hasattr(self, 'validation_buffer') and self.validation_buffer is not None and self.validation_buffer['features'].numel() > 0:
            for key in self.validation_buffer.keys():
                self.validation_buffer[key] = self.validation_buffer[key].to(device)

