import warnings

from huggingface_hub import StateDictSplit
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
from matplotlib import pyplot as plt
import random
from typing import List,Dict

class RSImage():
    def __init__(self,options,root:str,id:int,size_limit = 0):
        """
        root: path to folder which contains 'image.png','dem.npy','rpc.txt',
        id: index of this image
        """
        self.options = options
        self.root = root
        self.id = id
        # self.image = self.__load_image__(os.path.join(root,'image.tif'))
        self.image = cv2.imread(os.path.join(root,'image.png'))
        self.dem = np.load(os.path.join(root,'dem.npy'))
        if os.path.exists(os.path.join(root,'tie_points.txt')):
            self.tie_points = self.__load_tie_points__(os.path.join(root,'tie_points.txt'))
        else:
            self.tie_points = None

        if size_limit > 0:
            self.image = self.image[:size_limit,:size_limit]
            self.dem = self.dem[:size_limit,:size_limit]

        self.H,self.W = self.image.shape[:2]
        self.rpc = RPCModelParameterTorch()
        self.rpc.load_from_file(os.path.join(root,'rpc.txt'))
        if options.use_gpu:
            self.rpc.to_gpu()
        # if os.path.exists(os.path.join(root,'dem.tif')):
        #     self.dem = self.__sample_dem__(os.path.join(root,'dem.tif'))
        # else:
        #     self.dem = None
        
        self.corner_xys = self.__get_corner_xys__() #[tl,tr,bl,br] [x,y]

    def __load_image__(self,path) -> np.ndarray:
        print("Loading Image")
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            for band in range(data.shape[0]):
                # data[band] = (data[band] - data[band].min()) / (data[band].max() - data[band].min() + 1e-6)
                data[band] = (255. * data[band] / data[band].max())
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0)).squeeze()
        return data[:10000,:10000]

    def __load_tie_points__(self,path) -> np.ndarray:
        tie_points = np.loadtxt(path,dtype=int)
        if tie_points.ndim == 1:
            tie_points = tie_points.reshape(1,-1)
        elif tie_points.shape[1] != 2:
            print("tie points format error")
            return None
        return tie_points

            
    
    @torch.no_grad()
    def __get_corner_xys__(self):
        """
        return: [tl,tr,bl,br] [x,y] np.ndarray
        """
        latlons = torch.stack(self.rpc.RPC_PHOTO2OBJ([0.,self.W-1.,0.,self.W-1.],[0.,0.,self.H - 1.,self.H - 1.],[self.dem[0,0],self.dem[0,-1],self.dem[-1,0],self.dem[-1,-1]]),dim=-1)
        xys = project_mercator(latlons)
        return xys.cpu().numpy()[:,[1,0]] # y,x -> x,y

    @torch.no_grad()
    def __sample_dem__(self,dem_path:str, block_size:int = 5000, max_iter:int=5, tol:float=0.1) -> np.ndarray:
        """
        根据RPC模型迭代计算与遥感影像像素对应的DEM高程
        
        参数:
            dem_path (str): DEM文件路径
            max_iter (int): 最大迭代次数，默认5次
            tol (float): 收敛阈值（米），默认0.1米
        
        返回:
            np.ndarray: 高程数组，形状为[H,W]
        """
        # 读取DEM数据
        print("Loading DEM")
        with rasterio.open(dem_path) as dem_ds:
            dem = dem_ds.read(1)
            dem_transform = dem_ds.transform
            dem_nodata = dem_ds.nodatavals[0]
        print("Sampling DEM")
        # 处理无效值并转换为float
        dem = np.where(dem == dem_nodata, np.nan, dem).astype(np.float32)

        # 生成DEM网格坐标（中心点）
        dem_rows, dem_cols = dem.shape
        x_coords = []  # 经度坐标
        y_coords = []  # 纬度坐标
        
        # 获取DEM每个像素中心的经纬度
        for col in range(dem_cols):
            x, _ = rasterio.transform.xy(dem_transform, 0, col, offset='center')
            x_coords.append(x)
        for row in range(dem_rows):
            _, y = rasterio.transform.xy(dem_transform, row, 0, offset='center')
            y_coords.append(y)

        x_coords = np.array(x_coords,dtype=np.double)
        y_coords = np.array(y_coords,dtype=np.double)

        # 确保坐标单调递增（处理北向上DEM）
        if y_coords[0] < y_coords[1]:
            y_coords = y_coords[::-1]
            dem = dem[::-1, :]

        # 创建DEM插值器
        dem_interp = RegularGridInterpolator(
            (y_coords, x_coords),  # 纬度在前，经度在后
            dem,
            method='linear',
            bounds_error=False,
            fill_value=np.nan
        )

        dem_mean = np.nanmean(dem[~np.isnan(dem)])

        elevation = np.full((self.H, self.W), np.nan, dtype=np.float32)
        pbar = tqdm(total=int(np.ceil(self.H / block_size)) * int(np.ceil(self.W / block_size)))
    
        # 分块处理逻辑
        for y_offset in range(0, self.H, block_size):
            # 计算当前块的行数
            y_size = min(block_size, self.H - y_offset)
            
            for x_offset in range(0, self.W, block_size):
                # 计算当前块的列数
                x_size = min(block_size, self.W - x_offset)
                
                # 生成当前块的网格坐标
                cols, rows = np.meshgrid(
                    np.arange(x_offset, x_offset+x_size),
                    np.arange(y_offset, y_offset+y_size)
                )
                samp = cols.ravel()  # 列坐标
                line = rows.ravel()  # 行坐标

                # 初始化当前块的高程
                h = np.full(samp.size, dem_mean, dtype=np.float32)
                
                # 迭代计算（与原始方法相同）
                for _ in range(max_iter):
                    prev_h = h.copy()
                    lat, lon = self.rpc.RPC_PHOTO2OBJ(samp, line, h, 'numpy')
                    points = np.column_stack((lat, lon))
                    new_h = dem_interp(points)
                    valid = ~np.isnan(new_h)
                    h[valid] = new_h[valid]
                    if np.nanmean(np.abs(h - prev_h)) < tol:
                        break

                # 将结果写入对应位置
                elevation[y_offset:y_offset+y_size, x_offset:x_offset+x_size] = h.reshape(y_size, x_size)
                pbar.update(1)

        return elevation.reshape((self.H, self.W))
    
    @torch.no_grad()
    def dem_interp(self,sampline:np.ndarray):
        if sampline.ndim == 1:
            sampline = sampline[None]
        return bilinear_interpolate(self.dem,sampline)
    
    @torch.no_grad()
    def xy_to_sampline(self,xy:np.ndarray,max_iter = 100):
        if xy.ndim == 1:
            xy = xy[None]
        latlon = mercator2lonlat(xy[:,[1,0]])
        sampline = np.array([self.W,self.H],dtype=np.float32) * (xy - self.corner_xys[0]) / (self.corner_xys[3] - self.corner_xys[0])
        dem = self.dem_interp(sampline)
        invalid_mask = np.full(dem.shape,True,dtype=bool)
        for iter in range(max_iter):
            sampline_new = np.stack(self.rpc.RPC_OBJ2PHOTO(latlon[invalid_mask,0],latlon[invalid_mask,1],dem[invalid_mask],'numpy'),axis=-1)
            dis = np.linalg.norm(sampline_new - sampline[invalid_mask],axis=-1)
            sampline[invalid_mask] = sampline_new
            invalid_mask[invalid_mask] = dis > 1.
            if invalid_mask.sum() == 0:
                break
        return sampline.squeeze()

    @torch.no_grad()
    def get_image_by_sampline(self,tl_sampline:np.ndarray,br_sampline:np.ndarray,div_factor:int = 16):
        tl_sampline = np.array(tl_sampline)
        br_sampline = np.array(br_sampline)
        H = ((br_sampline[1] - tl_sampline[1]) // div_factor) * div_factor
        W = ((br_sampline[0] - tl_sampline[0]) // div_factor) * div_factor
        line_start = (br_sampline[1] - tl_sampline[1] - H) // 2 + tl_sampline[1]
        samp_start = (br_sampline[0] - tl_sampline[0] - W) // 2 + tl_sampline[0]
        tl_sampline = np.array([samp_start,line_start],dtype=int)
        br_sampline = np.array([samp_start + W,line_start + H],dtype=int)
        return self.image[tl_sampline[1]:br_sampline[1],tl_sampline[0]:br_sampline[0]]
    
    @torch.no_grad()
    def get_dem_by_sampline(self,tl_sampline:np.ndarray,br_sampline:np.ndarray,div_factor:int = 16):
        tl_sampline = np.array(tl_sampline)
        br_sampline = np.array(br_sampline)
        H = ((br_sampline[1] - tl_sampline[1]) // div_factor) * div_factor
        W = ((br_sampline[0] - tl_sampline[0]) // div_factor) * div_factor
        line_start = (br_sampline[1] - tl_sampline[1] - H) // 2 + tl_sampline[1]
        samp_start = (br_sampline[0] - tl_sampline[0] - W) // 2 + tl_sampline[0]
        tl_sampline = np.array([samp_start,line_start],dtype=int)
        br_sampline = np.array([samp_start + W,line_start + H],dtype=int)
        return self.dem[tl_sampline[1]:br_sampline[1],tl_sampline[0]:br_sampline[0]]

    @torch.no_grad()
    def get_image_by_xy(self,tlxy:np.ndarray,brxy:np.ndarray,div_factor:int = 16):
        """
        return: crop_img,tl_sampline,br_sampline
        """
        tlxy = np.array(tlxy)
        brxy = np.array(brxy)
        tl_sampline = self.xy_to_sampline(tlxy)
        br_sampline = self.xy_to_sampline(brxy)
        return self.get_image_by_sampline(tl_sampline,br_sampline),tl_sampline,br_sampline

    @torch.no_grad()
    def get_dem_by_xy(self,tlxy:np.ndarray,brxy:np.ndarray,div_factor:int = 16):
        """
        return: crop_dem,tl_sampline,br_sampline
        """
        tlxy = np.array(tlxy)
        brxy = np.array(brxy)
        tl_sampline = self.xy_to_sampline(tlxy)
        br_sampline = self.xy_to_sampline(brxy)
        return self.get_dem_by_sampline(tl_sampline,br_sampline),tl_sampline,br_sampline


