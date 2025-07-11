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
from utils import get_coord_mat,warp_by_extend,estimate_affine_ransac,project_mercator,mercator2lonlat,downsample,bilinear_interpolate

from rpc import RPCModelParameterTorch
from tqdm import tqdm,trange
from scheduler import MultiStageOneCycleLR
from torch.optim import AdamW,lr_scheduler
from criterion import CriterionTrainOneImg
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

cfg_base = {
            'input_channels':3,
            'patch_feat_channels':512,
            'global_feat_channels':256,
            'img_size':256,
            'window_size':8,
            'embed_dim':128,
            'depth':[2,2,18],
            'num_heads':[4,8,16],
            'drop_path_rate':.5,
            'unfreeze_backbone_modules':[]
        }
cfg_large = {
        'input_channels':3,
        'patch_feature_channels':512,
        'global_feature_channels':256,
        'img_size':1024,
        'window_size':16,
        'embed_dim':192,
        'depth':[2,2,18],
        'num_heads':[6, 12, 24],
        'drop_path_rate':.2,
        'pretrain_window_size':[12, 12, 12],
        'unfreeze_backbone_modules':[]
    }

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
        # tl_latlon = mercator2lonlat(tlxy[None,[1,0]])
        # br_latlon = mercator2lonlat(brxy[None,[1,0]])
        # tl_sampline = np.stack(self.rpc.RPC_OBJ2PHOTO(tl_latlon[:,0],tl_latlon[:,1],[self.dem_interp(tl_latlon[0])],'numpy'),axis=-1).squeeze().astype(int)
        # br_sampline = np.stack(self.rpc.RPC_OBJ2PHOTO(br_latlon[:,0],br_latlon[:,1],[self.dem_interp(br_latlon[0])],'numpy'),axis=-1).squeeze().astype(int)
        # H = ((br_sampline[1] - tl_sampline[1]) // div_factor) * div_factor
        # W = ((br_sampline[0] - tl_sampline[0]) // div_factor) * div_factor
        # line_start = (br_sampline[1] - tl_sampline[1] - H) // 2
        # samp_start = (br_sampline[0] - tl_sampline[0] - W) // 2
        # tl_sampline = np.array([samp_start,line_start],dtype=int)
        # br_sampline = np.array([samp_start + W,line_start + H],dtype=int)
        # return self.dem[tl_sampline[1]:br_sampline[1],tl_sampline[0]:br_sampline[0]],tl_sampline,br_sampline

class Element():
    def __init__(self,options,encoder:Encoder,img_raw:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch,id:int,output_path:str,top_left_linesamp:np.ndarray = np.array([0.,0.])):
        self.options = options
        self.id = id
        self.img_raw = img_raw # cv2.imread(options.img_path,cv2.IMREAD_GRAYSCALE)
        cv2.imwrite(os.path.join(output_path,f'img_{id}.png'),img_raw)
        self.local_raw = get_coord_mat(self.img_raw.shape[0],self.img_raw.shape[1])
        self.top_left_linesamp = top_left_linesamp
        self.local_raw += top_left_linesamp
        self.dem = dem
        self.rpc = rpc
        self.H,self.W = self.img_raw.shape[:2]
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.RandomApply([transforms.ColorJitter(.4,.4,.4,.4)],p=.7),
            transforms.RandomInvert(p=.3),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        # self.encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(options.encoder_path)).items()})
        self.encoder = encoder
        self.encoder.eval()
        self.mapper = Decoder(in_channels=self.encoder.output_channels,block_num=options.mapper_blocks_num)
        self.use_gpu = options.use_gpu
        self.output_path = output_path
        
        if self.use_gpu:
            self.rpc.to_gpu()
            self.encoder.cuda()
            self.mapper.cuda()
        
        if options.crop_step > 0:
            crop_step = options.crop_step
        else:
            crop_step = int(np.sqrt((self.H - options.crop_size) * (self.W - options.crop_size) / 150.))

        self.crop_imgs_NHW,self.crop_locals_NHW2,self.crop_dems_NHW = self.__crop_img__(options.crop_size,crop_step)
        self.crop_img_num = len(self.crop_imgs_NHW)
        self.SAMPLE_FACTOR = self.encoder.SAMPLE_FACTOR
        self.buffer,self.kd_tree = self.__extract_features__()
        self.extend = self.__calculate_extend__()
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

        print(f"===========================Element {self.id} Initiated===========================")
        print(f"img size:{img_raw.shape}")
        print(f"top_left_linesamp:{top_left_linesamp}")
        print(f"extend:{self.extend}")
        print("=================================================================================")

    def __crop_img__(self,crop_size = 256,step = 256 // 2,random_ratio = 1.):

        print("cropping image")
        H, W = self.img_raw.shape[:2]
        cut_number = 0
        row_num = 0
        col_num = 0
        crop_imgs = []
        crop_locals = []
        crop_dems = []

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
                pbar.update(1)
                col_num += 1
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
        
        print("Extracting features")
        
        start_time = time.perf_counter()

        imgs_NHW = torch.stack([self.transform(img) for img in self.crop_imgs_NHW]) # N,H,W
        locals_NHW2= torch.from_numpy(self.crop_locals_NHW2)
        locals_Nhw2 = downsample(locals_NHW2,self.encoder.SAMPLE_FACTOR)
        dems_NHW = torch.from_numpy(self.crop_dems_NHW)
        dems_Nhw = downsample(dems_NHW,self.encoder.SAMPLE_FACTOR)

        total_patch_num = locals_Nhw2.shape[0] * locals_Nhw2.shape[1] * locals_Nhw2.shape[2]
        select_ratio = min(1. * self.options.max_buffer_size / total_patch_num,1.)
        print("select_ratio:",select_ratio)
        # avg = nn.AvgPool2d(self.SAMPLE_FACTOR,self.SAMPLE_FACTOR)
        self.encoder.eval().cuda()

        # if self.use_gpu:
        #     imgs_NHW = imgs_NHW.cuda()
        #     locals_Nhw2 = locals_Nhw2.cuda()
        #     dems_Nhw = dems_Nhw.cuda()
        
        batch_num = int(np.ceil(self.crop_img_num / self.options.batch_size))

        features_PD = []
        confs_P1 = []
        locals_P2 = []
        dems_P1 = []
        
        


        for batch_idx in trange(batch_num):
            batch_imgs = imgs_NHW[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].cuda()
            batch_locals = locals_Nhw2[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].cuda().flatten(0,2)
            batch_dems = dems_Nhw[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size].cuda().flatten(0,2)


            # feat,conf = self.encoder(imgs_NHW[batch_idx * self.options.batch_size : (batch_idx+1) * self.options.batch_size]) # B,D,H,W
            feat,conf = self.encoder(batch_imgs) # B,D,H,W

            feat = feat.permute(0,2,3,1).flatten(0,2)
            conf = conf.squeeze().flatten(0,2)
            # threshold = torch.topk(input = conf, k = int(select_ratio * len(conf))).values[-1]
            valid_mask = conf > self.options.conf_threshold
            select_idxs = torch.randperm(valid_mask.sum())[:int(select_ratio * len(conf))]  
            

            features_PD.append(feat[valid_mask][select_idxs])
            confs_P1.append(conf[valid_mask][select_idxs])
            locals_P2.append(batch_locals[valid_mask][select_idxs])
            dems_P1.append(batch_dems[valid_mask][select_idxs])


        features_PD = torch.cat(features_PD,dim=0)
        confs_P1 = torch.cat(confs_P1,dim=0)
        locals_P2 = torch.cat(locals_P2,dim=0)
        dems_P1 = torch.cat(dems_P1,dim=0)

        #============================
        # output_img = self.img_raw
        # for idx,local in enumerate(locals_P2):
        #     if confs_P1[idx] > .5:
        #         color = (0,255,0)
        #     else:
        #         color = (255,0,0)
        #     cv2.circle(output_img,(int(local[1] - self.top_left_linesamp[1]),int(local[0] - self.top_left_linesamp[0])),1,color,-1)
        # cv2.imwrite(f'./datasets/Rsea/d0_v0/conf_{self.id}.png',output_img)
        # exit()
        #============================
        
        buffer = {
            'features':features_PD,
            'locals':locals_P2,
            'confs':confs_P1,
            'dems':dems_P1
        }
        self.patch_num = len(features_PD)

        kd_tree = build_kd_tree(locals_P2,device='cuda')

        print(f"Extract features done in {time.perf_counter() - start_time} seconds \t {self.patch_num} patches in total")
        
        return buffer,kd_tree
    
    
    def __calculate_extend__(self):

        def angle_between_vectors(v1, v2):
            v1_norm = np.linalg.norm(v1)
            v2_norm = np.linalg.norm(v2)
            cos_theta = np.dot(v1, v2) / (v1_norm * v2_norm)
            cos_theta = np.clip(cos_theta, -1.0, 1.0)  # 防止数值误差
            return np.arccos(cos_theta)

        def get_rotation_angle(A, B):
            diag_A = A[2] - A[0]
            diag_B = B[2] - B[0]
            
            # 计算对角线之间的角度
            return angle_between_vectors(diag_A, diag_B)

        def sort_vertices(rect):
            center = np.mean(rect, axis=0)
            sorted_rect = sorted(rect, key=lambda p: np.arctan2(p[1] - center[1], p[0] - center[0]))
            return np.array(sorted_rect)

        def find_matching_rectangles(rect1, rect2):
            rect1 = np.array(rect1)
            rect2 = np.array(rect2)
            
            sorted_rect1 = sort_vertices(rect1)
            sorted_rect2 = sort_vertices(rect2)

            angle1 = get_rotation_angle(sorted_rect1, sorted_rect2)

            sorted_rect2_180 = np.array([sorted_rect2[3], sorted_rect2[0], sorted_rect2[1], sorted_rect2[2]])
            angle2 = get_rotation_angle(sorted_rect1, sorted_rect2_180)
            
            if angle1 < angle2:
                return sorted_rect1, sorted_rect2
            else:
                return sorted_rect1, sorted_rect2_180
            
        H,W = self.img_raw.shape[:2]
        corner_lats,corner_lons = self.rpc.RPC_PHOTO2OBJ([self.top_left_linesamp[1],self.top_left_linesamp[1],self.top_left_linesamp[1] + W-1,self.top_left_linesamp[1] + W-1],
                                                         [self.top_left_linesamp[0],self.top_left_linesamp[0] + H-1,self.top_left_linesamp[0],self.top_left_linesamp[0] + H-1],
                                                         [self.dem[0,0],self.dem[-1,0],self.dem[0,-1],self.dem[-1,-1]])
        corner_xy = project_mercator(torch.stack([corner_lats,corner_lons],dim=-1))[:,[1,0]]
        rect = cv2.minAreaRect(corner_xy.cpu().numpy().astype(np.float32))
        box = cv2.boxPoints(rect).astype(np.float32)
        scale = np.array(rect[1]) / 2.
        local_rec = ((0,0),rect[1],0)
        local_box = cv2.boxPoints(local_rec).astype(np.float32)
        box,local_box = find_matching_rectangles(box,local_box)
        af_mat = cv2.getAffineTransform(local_box[:3],box[:3])
        
        dem_mean = np.mean(self.dem)
        dem_std = np.std(self.dem)
        dem_min = max(self.dem.min(),dem_mean - 2 * dem_std)
        dem_max = min(self.dem.max(),dem_mean + 2 * dem_std)
        extend = np.concatenate([af_mat.reshape(-1),scale,np.array([dem_min,dem_max])])
        
        extend = torch.from_numpy(extend)
        if self.use_gpu:
            extend = extend.cuda()

        return extend

    def clear_transfrom(self):
        self.af_trans = np.array([
            [1.,0.,0.],
            [0.,1.,0.]
        ])

    def train_mapper(self,output_vis = False):
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
        criterion = CriterionTrainOneImg()
        
        self.mapper.train()
        if self.use_gpu:
            self.mapper.cuda()

        min_photo_loss = 1e9
        best_mapper_state_dict = None

        print("Training Mapper Start")
        start_time = time.perf_counter()

        patch_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.patch_feature_channels,self.patch_num * 5,1)),dim=1).to(self.buffer['features'].device)
        global_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.global_feature_channels,self.patch_num * 5,1)),dim=1).to(self.buffer['features'].device)
        patch_noise_amp = torch.rand(1,1,self.patch_num * 5,1,device=patch_noise_buffer.device,dtype=patch_noise_buffer.dtype) * .1 + .1
        global_noise_amp = .5#torch.rand(1,1,self.patch_num * 5,1,device=global_noise_buffer.device,dtype=global_noise_buffer.dtype) * .2 + .5
        patch_noise_buffer = patch_noise_buffer * patch_noise_amp
        global_noise_buffer = global_noise_buffer * global_noise_amp

        total_loss = 0
        total_loss_obj = 0
        total_loss_height = 0
        total_loss_photo = 0
        total_loss_photo_real = 0
        total_reg = 0
        count = 0

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
            dems_p1 = self.buffer['dems'][idxs].contiguous()
            features_pD = features_pD * dists.unsqueeze(-1)
            confs_p1 = confs_p1 * dists
            dems_p1 = dems_p1 * dists
            features_pD = torch.mean(features_pD,dim=1).to(torch.float32)
            confs_p1 = torch.mean(confs_p1,dim=1).to(torch.float32)
            dems_p1 = torch.mean(dems_p1,dim=1).to(torch.float32)
            locals_p2 = sample_linesamps[valid_mask]

            features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
            patch_feature_noise = patch_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:].contiguous()
            global_feature_noise = global_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:].contiguous()
            features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] = F.normalize(features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] + patch_feature_noise,dim=1)
            features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] = F.normalize(features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] + global_feature_noise,dim=1)
            
            output_13p1 = self.mapper(features_1Dp1)
            output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
            yxh_pred_p3 = warp_by_extend(output_p3,self.extend)
            
            loss,loss_obj,loss_height,loss_photo,loss_photo_real,loss_bias,loss_reg = criterion(iter_idx,self.options.element_training_iters,yxh_pred_p3,confs_p1,locals_p2,dems_p1,self.rpc)
            loss.backward()
            
            total_loss += loss.item()
            total_loss_obj += loss_obj.item()
            total_loss_photo += loss_photo.item()
            total_loss_photo_real += loss_photo_real.item()
            total_loss_height += loss_height.item()
            total_reg += loss_reg
            count += 1

            # print(f"iter:{iter_idx + 1}\t loss:{loss.item():.2f} \t l_obj:{loss_obj.item():.2f} \t l_photo:{loss_photo.item():.2f} \t l_real:{loss_photo_real.item():.2f} \t l_height:{loss_height.item():.2f} \t bias:{loss_bias:.2f} \t reg:{loss_reg:.2f} \t lr:{scheduler.get_last_lr()[0]:.7f}")

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
                total_loss_photo_real /= count
                total_reg /= count
                
                # cost_time = int(time.perf_counter() - start_time)
                # print(f"\n ============= iter:{iter_idx + 1} \t total_loss:{total_loss:.2f} \t total_loss_obj:{total_loss_obj:.2f} \t total_loss_photo:{total_loss_photo:.2f} \t total_loss_real:{total_loss_photo_real:.2f} \t total_loss_height:{total_loss_height:.2f} \t total_loss_reg:{total_reg:.2f} \t time:{cost_time}s \n")
                if total_loss_photo < min_photo_loss:
                    min_photo_loss = total_loss_photo
                    best_mapper_state_dict = self.mapper.state_dict()
                elif total_loss_photo > min_photo_loss * 2:
                    self.mapper.load_state_dict(best_mapper_state_dict)

                total_loss = 0
                total_loss_obj = 0
                total_loss_height = 0
                total_loss_photo = 0
                total_loss_photo_real = 0
                total_reg = 0
                count = 0

                # if output_vis:
                #     self.get_vis_patches_loc()
        
        self.mapper.load_state_dict(best_mapper_state_dict)
        self.ransac_threshold = min_photo_loss + 5
        print(f"Training Mapper Done in {time.perf_counter() - start_time} seconds")
        # if output_vis:
        #     self.vis_patches_locs = np.stack(self.vis_patches_locs,axis=0)
        #     np.save(f"{self.output_path}/train_vis.npy",self.vis_patches_locs)
    
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
        criterion = CriterionTrainOneImg()
        
        self.mapper.train()
        if self.use_gpu:
            self.mapper.cuda()

        min_photo_loss = 1e9
        best_mapper_state_dict = None

        print("Finetune Mapper Start")
        start_time = time.perf_counter()

        patch_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.patch_feature_channels,self.patch_num * 5,1)),dim=1).to(self.buffer['features'].device)
        global_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.global_feature_channels,self.patch_num * 5,1)),dim=1).to(self.buffer['features'].device)
        patch_noise_amp = torch.rand(1,1,self.patch_num * 5,1,device=patch_noise_buffer.device,dtype=patch_noise_buffer.dtype) * .1 + .1
        global_noise_amp = .5#torch.rand(1,1,self.patch_num * 5,1,device=global_noise_buffer.device,dtype=global_noise_buffer.dtype) * .2 + .5
        patch_noise_buffer = patch_noise_buffer * patch_noise_amp
        global_noise_buffer = global_noise_buffer * global_noise_amp

        total_loss = 0
        total_loss_obj = 0
        total_loss_height = 0
        total_loss_photo = 0
        total_loss_photo_real = 0
        total_reg = 0
        count = 0

        pbar = tqdm(total=self.options.element_finetune_iters)

        for iter_idx in range(self.options.element_finetune_iters):
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
            dems_p1 = self.buffer['dems'][idxs].contiguous()
            features_pD = features_pD * dists.unsqueeze(-1)
            confs_p1 = confs_p1 * dists
            dems_p1 = dems_p1 * dists
            features_pD = torch.mean(features_pD,dim=1).to(torch.float32)
            confs_p1 = torch.mean(confs_p1,dim=1).to(torch.float32)
            dems_p1 = torch.mean(dems_p1,dim=1).to(torch.float32)
            locals_p2 = sample_linesamps[valid_mask]

            features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
            patch_feature_noise = patch_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:].contiguous()
            global_feature_noise = global_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:].contiguous()
            features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] = F.normalize(features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] + patch_feature_noise,dim=1)
            features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] = F.normalize(features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] + global_feature_noise,dim=1)
            
            output_13p1 = self.mapper(features_1Dp1)
            output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
            yxh_pred_p3 = warp_by_extend(output_p3,self.extend)
            
            loss,loss_obj,loss_height,loss_photo,loss_photo_real,loss_bias,loss_reg = criterion(iter_idx,self.options.element_training_iters,yxh_pred_p3,confs_p1,locals_p2,dems_p1,self.rpc)
            loss.backward()
            
            total_loss += loss.item()
            total_loss_obj += loss_obj.item()
            total_loss_photo += loss_photo.item()
            total_loss_photo_real += loss_photo_real.item()
            total_loss_height += loss_height.item()
            total_reg += loss_reg
            count += 1

            # print(f"iter:{iter_idx + 1}\t loss:{loss.item():.2f} \t l_obj:{loss_obj.item():.2f} \t l_photo:{loss_photo.item():.2f} \t l_real:{loss_photo_real.item():.2f} \t l_height:{loss_height.item():.2f} \t bias:{loss_bias:.2f} \t reg:{loss_reg:.2f} \t lr:{scheduler.get_last_lr()[0]:.7f}")

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
                total_loss_photo_real /= count
                total_reg /= count
                
                # cost_time = int(time.perf_counter() - start_time)
                # print(f"\n ============= iter:{iter_idx + 1} \t total_loss:{total_loss:.2f} \t total_loss_obj:{total_loss_obj:.2f} \t total_loss_photo:{total_loss_photo:.2f} \t total_loss_real:{total_loss_photo_real:.2f} \t total_loss_height:{total_loss_height:.2f} \t total_loss_reg:{total_reg:.2f} \t time:{cost_time}s \n")
                if total_loss_photo < min_photo_loss:
                    min_photo_loss = total_loss_photo
                    best_mapper_state_dict = self.mapper.state_dict()
                elif total_loss_photo > min_photo_loss * 2:
                    self.mapper.load_state_dict(best_mapper_state_dict)

                total_loss = 0
                total_loss_obj = 0
                total_loss_height = 0
                total_loss_photo = 0
                total_loss_photo_real = 0
                total_reg = 0
                count = 0
        
        self.mapper.load_state_dict(best_mapper_state_dict)
        self.ransac_threshold = min_photo_loss + 5
        
        self.mapper.load_state_dict(best_mapper_state_dict)
        self.ransac_threshold = min_photo_loss + 5.
        print(f"Finetune Mapper Done in {time.perf_counter() - start_time} seconds")

    def save_mapper(self):
        torch.save(self.mapper.state_dict(),os.path.join(self.output_path,'mapper.pth'))
    
    def load_mapper(self,path):
        self.mapper.load_state_dict(torch.load(path))

    @torch.no_grad()
    def get_vis_patches_loc(self):
        self.mapper.eval()
        features_pD = self.buffer['features'][self.vis_patches_idx].contiguous()
        confs_p1 = self.buffer['confs'][self.vis_patches_idx].contiguous()
        features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
        output_13p1 = self.mapper(features_1Dp1)
        output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
        yxh_pred_p3 = warp_by_extend(output_p3,self.extend)
        yxh_pred_p3[:,:2] -= yxh_pred_p3[:,:2].mean(dim=0)
        self.vis_patches_locs.append(yxh_pred_p3.cpu().numpy())

    @torch.no_grad()
    def get_error(self):
        def score_to_color(score):
            red = int((1 - score) * 255)
            green = int(score * 255)
            return (red, green, 0)
        def draw_points(img, points, values, radius=1):
            # 创建图像副本避免修改原图
            img_draw = img.copy()
            if len(img_draw.shape) == 2:
                img_draw = cv2.cvtColor(img_draw, cv2.COLOR_GRAY2BGR)
            elif img_draw.shape[2] == 4:
                img_draw = img_draw[:, :, :3]  # 去除alpha通道
            
            for (y, x), val in zip(points, values):
                color = score_to_color(val)
                ix = int(round(x))
                iy = int(round(y))
                if 0 <= ix < img_draw.shape[1] and 0 <= iy < img_draw.shape[0]:
                    cv2.circle(img_draw, (ix, iy), radius, 
                            color, -1, cv2.LINE_AA)
            return img_draw
        
        self.mapper.eval()
        features_pD = self.buffer['features']
        locals_p2 = self.buffer['locals']
        features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
        output_13p1 = self.mapper(features_1Dp1)
        output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
        yxh_pred_p3 = warp_by_extend(output_p3,self.extend)
        yx_pred_p2,h_pred_p1 = yxh_pred_p3[:,:2],yxh_pred_p3[:,2]
        latlon_pred_p2 = mercator2lonlat(yx_pred_p2)
        linesamp_pred_p2 = torch.stack(self.rpc.RPC_OBJ2PHOTO(latlon_pred_p2[:,0],latlon_pred_p2[:,1],h_pred_p1),dim=1)[:,[1,0]]
        error = torch.norm(linesamp_pred_p2 - locals_p2,dim=1)
        min_err = error.min()
        max_err = 2 * error.median() - min_err
        score = (max_err - error) / (max_err - min_err)
        score = torch.clip(score,0.,1.)
        img_draw = draw_points(self.img_raw,locals_p2.cpu().numpy(),score.cpu().numpy())
        cv2.imwrite('./error.png',img_draw[:,:,[2,1,0]])


    def __calculate_transform__(self,locals:np.ndarray,targets:np.ndarray) -> np.ndarray:
        '''
        locals: (line,samp)
        targets: (line,samp)
        '''
        dis = np.linalg.norm(locals + (np.mean(targets,axis=0)[None] - np.mean(locals,axis=0)[None]) - targets,axis=-1)
        print("mean_dis:",dis.mean())
        dis_valid_idx = dis < dis.mean() + dis.std()
        locals = locals[dis_valid_idx]
        targets = targets[dis_valid_idx]
        

        threshold = np.mean(np.linalg.norm(locals + (np.mean(targets,axis=0)[None] - np.mean(locals,axis=0)[None]) - targets,axis=-1))
        affine_matrix,inliers = cv2.estimateAffine2D(locals,targets,ransacReprojThreshold=threshold,maxIters=10000)
        inliers = inliers.reshape(-1).astype(bool)
        inlier_num = inliers.sum()
        print(f'{inlier_num} / {len(locals)}')

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
            yxh_preds = []
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
                    yxh_pred_p3 = warp_by_extend(output_p3,element.extend)
                    yxh_preds.append(yxh_pred_p3)
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

            yxh_pred_P3 = torch.cat(yxh_preds,dim=0)
            latlon_pred_P2 = mercator2lonlat(yxh_pred_P3[:,:2])
            dems_pred_P1 = yxh_pred_P3[:,2]
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
        print(f"Calculate transform of element {self.id} done in {time.perf_counter() - start_time} seconds")
        print(f"Origin Transform of element {self.id}:")
        print(self.af_trans)

    def apply_transform(self):
        self.applied_trans = self.__compose_transforms__([self.applied_trans,self.af_trans])
        print(f"New Transform for element {self.id}: \n {self.af_trans}")
        print(f"Composed Transform for element {self.id}: \n {self.applied_trans}")
        self.rpc.Update_Adjust(torch.from_numpy(self.applied_trans))
        self.extend = self.__calculate_extend__()

    def output_ortho(self,output_path):
        orthorectify_image(self.img_raw,self.dem,self.rpc,output_path)
    
    def clear_buffer(self):
        del self.buffer
        del self.kd_tree
        self.buffer = None
        self.kd_tree = None


class Grid():
    def __init__(self,options,extend:np.ndarray,diag:np.ndarray,encoder:Encoder,output_path:str):
        """
        extend:(cxx,cxy,cx0,cyx,cyy,cy0,sx,sy,dmin,dmax)
        """
        self.options = options
        self.extend = torch.from_numpy(extend)
        self.diag = diag #[[x,y],[x,y]]
        self.encoder = encoder
        self.output_path = output_path
        self.mapper = Decoder(in_channels=self.encoder.output_channels,block_num=options.mapper_blocks_num)
        self.elements:List[Element] = []
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
            ])
        self.SAMPLE_FACTOR = 16
        self.vis_points_latlon = None
    
    def get_overlap_image(self,img:RSImage,margin = np.array([0,0])):
        corner_samplines = img.xy_to_sampline(np.array([self.diag[0],[self.diag[1,0],self.diag[0,1]],self.diag[1],[self.diag[0,0],self.diag[1,1]]])) # tl,tr,br,bl
        top = max(corner_samplines[0,1],corner_samplines[1,1]) + margin[0]
        bottom = min(corner_samplines[2,1],corner_samplines[3,1]) - margin[0]
        left = max(corner_samplines[0,0],corner_samplines[3,0]) + margin[1]
        right = min(corner_samplines[1,0],corner_samplines[2,0]) - margin[1]
        img_raw = img.get_image_by_sampline(np.array([left,top]),np.array([right,bottom]))
        dem = img.get_dem_by_sampline(np.array([left,top]),np.array([right,bottom]))
        return img_raw,dem,np.array([top,left]),np.array([bottom,right])

    def add_img(self,img:RSImage,output_path:str,mapper_path:str = None,id:int = None):
        output_path = os.path.join(output_path,f'element_{id}')
        if not os.path.exists(output_path):
            os.mkdir(output_path)
        
        img_raw,dem,tl_linesamp,br_linesamp = self.get_overlap_image(img)

        new_element = Element(options = self.options,
                              encoder = self.encoder,
                              img_raw = img_raw,
                              dem = dem,
                              rpc = img.rpc,
                              id = len(self.elements) if id is None else id,
                              output_path = output_path,
                              top_left_linesamp=tl_linesamp)
        if mapper_path:
            new_element.load_mapper(mapper_path)

        self.elements.append(new_element)
    
    def valid_features(self,sample_num = 10000,batch_size = 10000):
        sample_idxs = torch.randperm(len(self.elements[0].buffer['locals']))[:sample_num]
        sample_locals = self.elements[0].buffer['locals'][sample_idxs]
        sample_dems = self.elements[0].buffer['dems'][sample_idxs]
        sample_latlons = torch.stack(self.elements[0].rpc.RPC_PHOTO2OBJ(sample_locals[:,1],sample_locals[:,0],sample_dems),dim=-1)
        sample_features = []
        for element in self.elements:
            sample_points = torch.stack(element.rpc.RPC_OBJ2PHOTO(sample_latlons[:,0],sample_latlons[:,1],sample_dems),dim=-1)[:,[1,0]]
            locals = element.buffer['locals']
            batch_num = int(np.ceil(len(locals) / batch_size))
            min_dis = torch.full((len(sample_points),),1e9,device=locals.device,dtype=torch.double )
            min_dis_idx = torch.full((len(sample_points),),-1,device=locals.device,dtype=int)
            for batch_idx in range(batch_num):
                dis = torch.cdist(sample_points,locals[batch_idx * batch_size : (batch_idx + 1) * batch_size])
                min_dis_batch,min_dis_idx_batch = torch.min(dis,dim=1)
                replace_mask = min_dis_batch < min_dis
                min_dis[replace_mask] = min_dis_batch[replace_mask]
                min_dis_idx[replace_mask] = min_dis_idx_batch[replace_mask] + batch_idx * batch_size
            sample_features.append(element.buffer['features'][min_dis_idx])
        for i in range(len(self.elements)-1):
            for j in range(i+1,len(self.elements)):
                feature1 = sample_features[i][:,:512]
                feature2 = sample_features[j][:,:512]
                simi = torch.sum(feature1 * feature2,dim=1).mean()
                print(f"simi {i} and {j} : {simi}")

    def valid_mappers(self,sample_num = 10000,batch_size = 10000):
        sample_idxs = torch.randperm(len(self.elements[0].buffer['locals']))[:sample_num]
        sample_locals = self.elements[0].buffer['locals'][sample_idxs]
        sample_dems = self.elements[0].buffer['dems'][sample_idxs]
        sample_latlons = torch.stack(self.elements[0].rpc.RPC_PHOTO2OBJ(sample_locals[:,1],sample_locals[:,0],sample_dems),dim=-1)
        sample_features = []
        for element in self.elements:
            sample_points = torch.stack(element.rpc.RPC_OBJ2PHOTO(sample_latlons[:,0],sample_latlons[:,1],sample_dems),dim=-1)[:,[1,0]]
            locals = element.buffer['locals']
            batch_num = int(np.ceil(len(locals) / batch_size))
            min_dis = torch.full((len(sample_points),),1e9,device=locals.device,dtype=torch.double )
            min_dis_idx = torch.full((len(sample_points),),-1,device=locals.device,dtype=int)
            for batch_idx in range(batch_num):
                dis = torch.cdist(sample_points,locals[batch_idx * batch_size : (batch_idx + 1) * batch_size])
                min_dis_batch,min_dis_idx_batch = torch.min(dis,dim=1)
                replace_mask = min_dis_batch < min_dis
                min_dis[replace_mask] = min_dis_batch[replace_mask]
                min_dis_idx[replace_mask] = min_dis_idx_batch[replace_mask] + batch_idx * batch_size
            sample_features.append(element.buffer['features'][min_dis_idx])
        for i in range(len(self.elements)-1):
            for j in range(i+1,len(self.elements)):
                feature1 = sample_features[i][:,:,None,None]
                feature2 = sample_features[j][:,:,None,None]
                mapper1 = self.elements[i].mapper
                mapper2 = self.elements[j].mapper
                mapper1.eval()
                mapper2.eval()
                output11 = mapper1(feature1).permute(0,2,3,1).flatten(0,2)
                output22 = mapper2(feature2).permute(0,2,3,1).flatten(0,2)
                output12 = mapper1(feature2).permute(0,2,3,1).flatten(0,2)
                output21 = mapper2(feature1).permute(0,2,3,1).flatten(0,2)
                pred11 = warp_by_extend(output11,self.elements[i].extend)[:,:2]
                pred22 = warp_by_extend(output22,self.elements[j].extend)[:,:2]
                pred12 = warp_by_extend(output12,self.elements[j].extend)[:,:2]
                pred21 = warp_by_extend(output21,self.elements[i].extend)[:,:2]
                dis1 = torch.norm(pred11 - pred12,dim=-1).mean()
                dis2 = torch.norm(pred11 - pred22,dim=-1).mean()
                dis3 = torch.norm(pred11 - pred21,dim=-1).mean()
                dis4 = torch.norm(pred22 - pred21,dim=-1).mean()
                dis5 = torch.norm(pred22 - pred12,dim=-1).mean()
                dis6 = torch.norm(pred12 - pred21,dim=-1).mean()
                print(f"dis {i} and {j} : {dis1} \t {dis2} \t {dis3} \t {dis4} \t {dis5} \t {dis6}")
        

    
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

    def train_mapper(self):
        start_time = time.perf_counter()
        max_patch_num = max(*[element.patch_num for element in self.elements])
        patches_per_batch = self.options.patches_per_batch // 4 * 4
        optimizer = AdamW(self.mapper.parameters(),lr=self.options.grid_train_lr_max)
        scheduler = MultiStageOneCycleLR(optimizer = optimizer,
                                            max_lr = self.options.grid_train_lr_max,
                                            min_lr= self.options.grid_train_lr_min,
                                            n_epochs_per_stage = self.options.grid_training_iters,
                                            steps_per_epoch = 1,
                                            pct_start = self.options.grid_warmup_iters / self.options.grid_training_iters,
                                            summit_hold = self.options.grid_summit_hold_iters / self.options.grid_training_iters,
                                            #gamma = self.options.lr_decay_per_100_epochs ** (1. / 100.),
                                            cooldown = 0.0
                                            )
        criterion = CriterionTrainOneImg()
        self.mapper.train()
        if self.options.use_gpu:
            self.mapper.cuda()

        min_photo_loss = 1e8

        patch_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.patch_feature_channels,max_patch_num * 5,1)),dim=1).to(self.elements[0].buffer['features'].device)
        global_noise_buffer = F.normalize(torch.normal(mean=0.,std=1.,size=(1,self.encoder.global_feature_channels,max_patch_num * 5,1)),dim=1).to(self.elements[0].buffer['features'].device)
        patch_noise_amp = torch.rand(1,1,max_patch_num * 5,1,device=patch_noise_buffer.device,dtype=patch_noise_buffer.dtype) * .1 + .1
        global_noise_amp = .5 #torch.rand(1,1,max_patch_num * 5,1,device=global_noise_buffer.device,dtype=global_noise_buffer.dtype) * .5 + .3
        patch_noise_buffer = patch_noise_buffer * patch_noise_amp
        global_noise_buffer = global_noise_buffer * global_noise_amp


        total_loss = 0
        total_loss_obj = 0
        total_loss_height = 0
        total_loss_photo = 0
        total_loss_photo_real = 0
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
                # error_idx = torch.argmax(dists.max(dim=1).values)
                # print(error_idx,dists[error_idx],idxs[error_idx])
                # print(sample_linesamps[error_idx])
                # print(element.buffer['locals'][idxs[error_idx]])
                valid_mask = dists.max(dim=1).values < 256
                dists = 1. / (dists[valid_mask] + 1e-6)
                idxs = idxs[valid_mask]
                dists = dists / torch.mean(dists,dim=-1,keepdim=True)
                features_pD = element.buffer['features'][idxs].contiguous()
                confs_p1 = element.buffer['confs'][idxs].contiguous()
                dems_p1 = element.buffer['dems'][idxs].contiguous()
                features_pD = features_pD * dists.unsqueeze(-1)
                confs_p1 = confs_p1 * dists
                dems_p1 = dems_p1 * dists
                features_pD = torch.mean(features_pD,dim=1).to(torch.float32)
                confs_p1 = torch.mean(confs_p1,dim=1).to(torch.float32)
                dems_p1 = torch.mean(dems_p1,dim=1).to(torch.float32)
                locals_p2 = sample_linesamps[valid_mask]

                features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
                patch_feature_noise = patch_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:].contiguous()
                global_feature_noise = global_noise_buffer[:,:,noise_idx,:][:,:,valid_mask,:].contiguous()
                features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] = F.normalize(features_1Dp1[:,:self.encoder.patch_feature_channels,:,:] + patch_feature_noise,dim=1)
                features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] = F.normalize(features_1Dp1[:,-self.encoder.global_feature_channels:,:,:] + global_feature_noise,dim=1)
            
                # global_feature_noise = F.normalize(torch.normal(mean=0,std=1,size=(1,self.encoder.global_feat_channels,features_1Dp1.shape[-2],1)),dim=1).to(features_1Dp1.device) * 0.5
                # features_1Dp1[:,-self.encoder.global_feat_channels:,:,:] += global_feature_noise
                
                output_13p1 = self.mapper(features_1Dp1)
                output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
                yxh_pred_p3 = warp_by_extend(output_p3,self.extend)
                
                loss,loss_obj,loss_height,loss_photo,loss_photo_real,loss_bias,loss_reg = criterion(iter_idx,self.options.grid_training_iters,yxh_pred_p3,confs_p1,locals_p2,dems_p1,element.rpc)
                loss.backward()

                total_loss += loss.item()
                total_loss_obj += loss_obj.item()
                total_loss_photo += loss_photo.item()
                total_loss_photo_real += loss_photo_real.item()
                total_loss_height += loss_height.item()
                total_reg += loss_reg
                count += 1

                # print(f"iter:{iter_idx + 1} \t img:{element.id}/{len(self.elements)} \t loss:{loss.item():.2f} \t l_obj:{loss_obj.item():.2f} \t l_photo:{loss_photo.item():.2f} \t l_real:{loss_photo_real.item():.2f} \t l_height:{loss_height.item():.2f} \t bias:{loss_bias:.2f} \t reg:{loss_reg:.2f} \t lr:{scheduler.get_last_lr()[0]:.7f}")
                pbar.update(1)
                pbar.set_postfix({
                    'lr':f'{scheduler.get_last_lr()[0]:.2e}',
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
                total_loss_photo_real /= count
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
                total_loss_photo_real = 0
                total_reg = 0
                count = 0
            
        self.mapper.load_state_dict(best_mapper_state_dict)
        torch.save(best_mapper_state_dict,os.path.join(self.output_path,'grid_mapper.pth'))
        for element in self.elements:
            element.clear_buffer()
        self.elements = None

    def load_mapper(self,path:str):
        self.mapper.load_state_dict(torch.load(path))
    
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

        
    def __crop_img__(self,img,crop_size,step,random_ratio = 1.):

        print("cropping image")
        H, W = img.shape[:2]
        local = get_coord_mat(H,W)
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
    def pred_xyh(self,img_raw:np.ndarray) -> Dict[str,np.ndarray]:
        """
        return: {"yx_P2","h_P1","locals_P2","confs_P1"} np.ndarray
        """
        H,W = img_raw.shape[:2]
        self.encoder.eval().cuda()
        self.mapper.eval().cuda()

        if self.options.crop_step > 0:
            crop_step = self.options.crop_step
        else:
            crop_step = int(np.sqrt((H - self.options.crop_size) * (W - self.options.crop_size) / 150.))
        crop_imgs_NHW,crop_locals_NHW2 = self.__crop_img__(img_raw,self.options.crop_size,crop_step)
        imgs_NHW = torch.stack([self.transform(img) for img in crop_imgs_NHW]) # N,H,W
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

        kd_tree = build_kd_tree(locals_P2,device='cuda')

        yxh_preds = []
        locals = []
        confs = []

        block_line_num = int(np.ceil(H / 256.))
        block_samp_num = int(np.ceil(W / 256.))
        
        print("Decoding")
        pbar = tqdm(total=block_line_num * block_samp_num)
        for block_line in range(block_line_num):
            for block_samp in range(block_samp_num):
                tl_line = block_line * 256 + .5
                tl_samp = block_samp * 256 + .5
                sample_linesamps = torch.from_numpy(np.stack(np.meshgrid(np.arange(tl_line,min(tl_line + 256,H)),np.arange(tl_samp,min(tl_samp + 256,W)),indexing='ij')
                                                                    ,axis=-1)).to(dtype=locals_P2.dtype,device=locals_P2.device).reshape(-1,2)
                dists,idxs = kd_tree.query(sample_linesamps,nr_nns_searches=4)
                valid_mask = (dists.max(dim=1).values < 64) 
                if valid_mask.sum() == 0:
                    continue
                dists = 1. / (dists[valid_mask] + 1e-6)
                idxs = idxs[valid_mask]
                dists = dists / torch.mean(dists,dim=-1,keepdim=True)
                features_pD = features_PD[idxs].contiguous()
                confs_p1 = confs_P1[idxs].contiguous()
                features_pD = features_pD * dists.unsqueeze(-1)
                confs_p1 = confs_p1 * dists
                features_pD = torch.mean(features_pD,dim=1).to(torch.float32)
                confs_p1 = torch.mean(confs_p1,dim=1).to(torch.float32)
                locals_p2 = sample_linesamps[valid_mask]

                features_1Dp1 = features_pD.permute(1,0)[None,:,:,None]
                output_13p1 = self.mapper(features_1Dp1)
                output_p3 = output_13p1.permute(0,2,3,1).flatten(0,2)
                yxh_pred_p3 = warp_by_extend(output_p3,self.extend)
                yxh_preds.append(yxh_pred_p3)
                locals.append(locals_p2)
                confs.append(confs_p1)
                
                pbar.update(1)

        yxh_P3 = torch.cat(yxh_preds,dim=0)
        locals_P2 = torch.cat(locals,dim=0)
        confs_P1 = torch.cat(confs,dim=0)


        # pred_raw_N3hw = self.mapper(features_NDhw)
        # pred_raw_P3 = pred_raw_N3hw.permute(0,2,3,1).flatten(0,2)
        # yxh_P3 = warp_by_extend(pred_raw_P3,self.extend)
        
        res = {
            'yx_P2':yxh_P3[:,:2].cpu().numpy(),
            'h_P1':yxh_P3[:,2].cpu().numpy(),
            'locals_P2':locals_P2.cpu().numpy(),
            'confs_P1':confs_P1.cpu().numpy(),
        }
        crop_imgs_NHW = None
        crop_locals_NHW2 = None
        imgs_NHW = None

        return res
    




class RSEA():
    def __init__(self,options):
        print("==============================options==============================")
        for k,v in vars(options).items():
            print(f"{k}:{v}")
        print("===================================================================")
        self.options = options
        random.seed(42)
        self.imgs:List[RSImage] = []
        self.grids:List[Grid] = []
        self.encoder = Encoder(cfg_large)
        self.encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(self.options.encoder_path).items()})
        self.encoder.eval()
        self.root = options.root
        
        if not os.path.isdir(self.root):
            raise ValueError("Output path is not a folder")
        if not os.path.exists(self.root):
            os.mkdir(self.root)
        # else:
        #     if len(os.listdir(self.root)) > 0:
        #         print("Output folder is not empty, a new folder is creating")
        #     self.root = f"{self.root}_{int(time.time())}"
        #     os.mkdir(self.root)
        
    def add_image(self,image_folder:str,size_limit = 0):
        """
        image_folder: path to folder which contains 'image.tif','dem.tif','rpc.txt'
        """
        img_id = len(self.imgs)
        print(f"===============================Adding image {img_id}===============================")
        new_image = RSImage(self.options,image_folder,img_id,size_limit = size_limit)
        self.imgs.append(new_image)
        print(f"===============================Add image {img_id} done===============================")
    
    def create_grids(self,grid_size:int = 1000,max_grid_num:int = -1):
        def find_grids(corners, grid_size):
            x_left = np.maximum(corners[:, 0, 0],corners[:, 2, 0]) 
            x_right = np.minimum(corners[:, 1, 0],corners[:, 3, 0])
            y_top = np.minimum(corners[:, 0, 1],corners[:, 1, 1]  ) 
            y_bottom = np.maximum(corners[:, 2, 1],corners[:, 3, 1]) 
            
            x_left_max = np.max(x_left)
            x_right_min = np.min(x_right)
            y_bottom_max = np.max(y_bottom)
            y_top_min = np.min(y_top)
            
            W = x_right_min - x_left_max
            H = y_top_min - y_bottom_max
            
            if W < grid_size or H < grid_size:
                raise ValueError("Overlap area too small")
            
            cols = int(W // grid_size)
            rows = int(H // grid_size)
            
            i_grid, j_grid = np.meshgrid(np.arange(cols), np.arange(rows), indexing='ij')
            i_flat = i_grid.ravel()
            j_flat = j_grid.ravel()
            
            x0 = x_left_max + i_flat * grid_size
            y0 = y_top_min - j_flat * grid_size
            x1 = x0 + grid_size
            y1 = y0 - grid_size
            
            diags = np.stack([
                np.stack([x0, y0], axis=1),
                np.stack([x1, y1], axis=1)
            ], axis=1)
            
            return diags

        corners = np.stack([image.corner_xys for image in self.imgs])
        grid_diags = find_grids(corners,grid_size) # M,2,2
        if max_grid_num > 0:
            grid_diags = grid_diags[:max_grid_num]
        
        print(f"{len(grid_diags)} grids is going to be created")

        for grid_idx,diag in enumerate(grid_diags):
            print(f"Creating grid {grid_idx}")
            tl,br = diag
            center = (tl + br) / 2.
            scale = np.abs((br - tl) / 2.)
            af_coef = np.array([1.,0.,center[0],0.,1.,center[1]])
            tlbr_samplines = np.stack([image.xy_to_sampline(np.stack([tl,br],axis=0)) for image in self.imgs]).astype(int)
            dem = np.concatenate([image.get_dem_by_sampline(tlbr_samplines[img_idx,0],tlbr_samplines[img_idx,1])[0].reshape(-1) for img_idx,image in enumerate(self.imgs)])
            dem_mean = np.mean(dem)
            dem_std = np.std(dem)
            dem_min = max(dem.min(),dem_mean - 2 * dem_std)
            dem_max = min(dem.max(),dem_mean + 2 * dem_std)
            extend = np.concatenate([af_coef,scale,np.array([dem_min,dem_max])])

            grid_output_path = os.path.join(self.root,f"grid_{grid_idx}")
            if not os.path.exists(grid_output_path):
                os.mkdir(grid_output_path)

            np.save(os.path.join(grid_output_path,'extend.npy'),extend)

            new_grid = Grid(self.options,extend,diag,self.encoder,grid_output_path)

            for img_idx,image in enumerate(self.imgs):
                # tl_sampline,br_sampline = tlbr_samplines[img_idx]
                # img_raw = image.get_image_by_sampline(tl_sampline,br_sampline)
                
                # dem_raw = image.get_dem_by_sampline(tl_sampline,br_sampline)
                # rpc = deepcopy(image.rpc)
                # rpc.LINE_OFF -= tl_sampline[1]
                # rpc.SAMP_OFF -= tl_sampline[0]
                new_grid.add_img(img = image,
                                 output_path = new_grid.output_path,
                                 id = image.id
                                #  mapper_path=os.path.join(new_grid.output_path,f'element_{image.id}','mapper.pth')
                                 )
            # new_grid.valid_features()
            # new_grid.train_elements()
            # new_grid.valid_mappers()
            # new_grid.adjust_elements()
            new_grid.train_mapper()
            self.grids.append(new_grid)
            # if grid_idx + 1 >= 12:
            #     break
                
        print(f"{len(self.grids)} grids created")
    
    def __overlap__(self,tl1:np.ndarray,tl2:np.ndarray,br1:np.ndarray,br2:np.ndarray):
        """
        return : [tl,br] [x,y] np.ndarray
        """
        print(tl1,tl2,br1,br2)
        if tl1[0] > br2[0] or tl1[1] < br2[1] or br1[0] < tl2[0] or br1[1] > tl2[1]:
            return None
        tl = np.array([max(tl1[0],tl2[0]),min(tl1[1],tl2[1])])
        br = np.array([min(br1[0],br2[0]),max(br1[1],br2[1])])
        return np.stack([tl,br],axis=0)

    def __calculate_transform__(self,locals:np.ndarray,targets:np.ndarray,confs:np.ndarray) -> np.ndarray:
        def estimate_affine_transform(src, dst):
            n = len(src)
            A = np.zeros((2*n, 6))
            b = np.zeros(2*n)
            
            for i in range(n):
                A[2*i] = [src[i, 0], src[i, 1], 0, 0, 1, 0]
                A[2*i+1] = [0, 0, src[i, 0], src[i, 1], 0, 1]
                b[2*i] = dst[i, 0]
                b[2*i+1] = dst[i, 1]
            
            # 使用最小二乘法求解
            x, _, _, _ = np.linalg.lstsq(A, b, rcond=None)
            
            # 构建变换矩阵 [a1, a2, b1; a3, a4, b2]
            transform_matrix = np.array([
                [x[0], x[1], x[4]],
                [x[2], x[3], x[5]]
            ])
            
            return transform_matrix
            
        total_num = len(confs)
        conf_valid_idx = confs > self.options.conf_threshold
        locals = locals[conf_valid_idx]
        targets = targets[conf_valid_idx]


        dis = np.linalg.norm(locals + (np.mean(targets,axis=0)[None] - np.mean(locals,axis=0)[None]) - targets,axis=-1)
        
        # print("mean_dis:",dis.mean())
        dis_valid_idx = dis < dis.mean() + dis.std()
        locals = locals[dis_valid_idx]
        targets = targets[dis_valid_idx]
        # offset = np.mean(targets,axis=0) - np.mean(locals,axis=0)
        

        threshold = np.mean(np.linalg.norm(locals + (np.mean(targets,axis=0)[None] - np.mean(locals,axis=0)[None]) - targets,axis=-1)) + 5.
        # print(np.mean(targets,axis=0),np.mean(locals,axis=0),np.mean(targets,axis=0) - np.mean(locals,axis=0))
        M,inliers = cv2.estimateAffine2D(locals,targets,ransacReprojThreshold=threshold,maxIters=10000)
        inliers = inliers.reshape(-1).astype(bool)
        locals = locals[inliers]
        targets = targets[inliers]
        dis = np.linalg.norm(locals + (np.mean(targets,axis=0)[None] - np.mean(locals,axis=0)[None]) - targets,axis=-1)
        inlier_num = inliers.sum()
        print("mean_dis:",dis.mean())

        
        affine_matrix = estimate_affine_transform(locals,targets)

        # offset = np.mean(targets,axis=0) - np.mean(locals,axis=0)
        print("M:")
        print(M)


        print(f"threshold:{threshold} \t {inlier_num}/{conf_valid_idx.sum()}/{total_num} \t {inlier_num / total_num}")

        # affine_matrix = np.array([
        #     [1.,0.,offset[0]],
        #     [0.,1.,offset[1]]
        # ])
        # print(f"{dis_valid_idx.sum()}/{conf_valid_idx.sum()}/{total_num}")
        return affine_matrix

    def load_grids(self,root = None):
        if root is None:
            root = self.root
        grid_paths = [i for i in os.listdir(root) if 'grid_' in i]
        for grid_path in grid_paths:
            extend = np.load(os.path.join(root,grid_path,'extend.npy'))
            center = np.array([extend[2],extend[5]])
            scale = extend[6:8]
            diag = np.stack([np.array([center[0] - scale[0],center[1] + scale[1]]),np.array([center[0] + scale[0],center[1] - scale[1]])],axis=0)
            print(center)
            print(scale)
            print(diag)
            new_grid = Grid(self.options,extend,diag,self.encoder,os.path.join(self.root,grid_path))
            new_grid.load_mapper(os.path.join(self.root,grid_path,'grid_mapper.pth'))
            self.grids.append(new_grid)
        print(f"{len(grid_paths)} grids loaded \t total {len(self.grids)} grids in RSEA now")
    

    def adjust(self,image_folders:List[str],options):

        # def check_error(check_points:np.ndarray,trans:np.ndarray):
        #     ones = np.ones((check_points.shape[0],1))
        #     trans_points = np.concatenate([check_points,ones],axis=-1)
        #     trans_points = np.matmul(trans_points,trans.T)
        #     dis = np.linalg.norm(trans_points - check_points,axis=-1)
        #     return dis
        
        def output_obj_vis(xyh:np.ndarray,output_path):
            mean = np.mean(xyh,axis=0)
            xyh -= mean
            np.savetxt(output_path,xyh,fmt='%.2f',delimiter=' ')
        
        adjust_images:List[RSImage] = []
        
        for image_id,image_folder in enumerate(image_folders):
            image = RSImage(self.options,image_folder,image_id)
            adjust_images.append(image)
        
        for img_idx,image in enumerate(adjust_images):
            all_locals = []
            all_preds = []
            all_confs = []
            all_xyh = []
            # orthorectify_image(image.image[:5000,:5000,0],image.dem[:5000,:5000],image.rpc,os.path.join(image.root,'dom.tif'))
            # continue
            for grid_idx,grid in enumerate(self.grids):
                print(f"processing grid {grid_idx}")
                overlap_diag = self.__overlap__(grid.diag[0],image.corner_xys[0],grid.diag[1],image.corner_xys[3])
                if overlap_diag is None :
                    continue
                img_raw,dem,tl_linesamp,br_linesamp = grid.get_overlap_image(image,margin=[64,64])
                print("tl:",tl_linesamp,"br:",br_linesamp)
                cv2.imwrite(os.path.join(grid.output_path,f'adjust_img_{img_idx}.png'),img_raw)
                
                pred_res = grid.pred_xyh(img_raw)

                xyh = np.concatenate([pred_res['yx_P2'][:,[1,0]],pred_res['h_P1'][:,None]],axis=-1)
                all_xyh.append(xyh)

                latlon_P2 = mercator2lonlat(pred_res['yx_P2'])
                locals_P2 = pred_res['locals_P2'] + tl_linesamp # (line,samp) + (line,samp)
                linesamp_pred_P2 = np.stack(image.rpc.RPC_OBJ2PHOTO(latlon_P2[:,0],latlon_P2[:,1],pred_res['h_P1'],'numpy'),axis=1)[:,[1,0]]
                all_locals.append(locals_P2)
                all_preds.append(linesamp_pred_P2)
                all_confs.append(pred_res['confs_P1'])
            all_locals = np.concatenate(all_locals,axis=0)
            all_preds = np.concatenate(all_preds,axis=0)
            all_confs = np.concatenate(all_confs,axis=0)
            all_xyh = np.concatenate(all_xyh,axis=0)
            transform = self.__calculate_transform__(all_locals,all_preds,all_confs)
            image.rpc.Update_Adjust(torch.from_numpy(transform))
            print(image.rpc.adjust_params)

            # output_obj_vis(all_xyh,output_path=os.path.join(image.root,'obj_vis.txt'))

            # check_points = np.stack(np.meshgrid(np.arange(0,image.H,10),np.arange(0,image.W,10),indexing='ij'),axis=-1).reshape(-1,2)
            # errors = check_error(check_points,transform)
            # errors = self.check_error()


            # info = f"error:\nmax:{errors.max()}\nmin:{errors.min()}\nmean:{errors.mean()}\nmedian:{np.median(errors)}\n<1px:{(errors < 1.).sum() * 1. / len(errors)}\n<3px:{(errors < 3.).sum() * 1. / len(errors)}\n<5px:{(errors < 5.).sum() * 1. / len(errors)}"
            # print("error:")
            # print("max:",errors.max())
            # print("min:",errors.min())
            # print("mean:",errors.mean())
            # print("median:",np.median(errors))
            # print("<1px:",(errors < 1.).sum() * 1. / len(errors))
            # print("<3px:",(errors < 3.).sum() * 1. / len(errors))
            # print("<5px:",(errors < 5.).sum() * 1. / len(errors))
            # print(info)


            # image.rpc.Merge_Adjust()
            orthorectify_image(image.image[:,:,0],image.dem,image.rpc,os.path.join(image.root,'dom.tif'))
            image.rpc.save_rpc_to_file(os.path.join(image.root,'rpc_corrected.txt'))
            # timestamp  = time.strftime("%Y%m%d%H%M%S")
            # with open(os.path.join(image.root,f'adjust_info_{timestamp}.txt'),'w') as f:
            #     for k,v in vars(options).items():
            #         info = info + f"{k}:{v}\n"
            #     f.write(info)

            return adjust_images

    def check_error(self,log_path,images:List[RSImage] = None):        
        def haversine_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
            R = 6371000 
            lat1 = coords1[:, 0]
            lon1 = coords1[:, 1]
            lat2 = coords2[:, 0]
            lon2 = coords2[:, 1]

            lat1_rad = np.radians(lat1)
            lon1_rad = np.radians(lon1)
            lat2_rad = np.radians(lat2)
            lon2_rad = np.radians(lon2)

            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = R * c
            
            return distance
        
        if images is None:
            images = self.imgs
        
        if not os.path.exists(log_path):
            df = pd.DataFrame(columns=['num_blocks','grid_size','median','mean','max','min','@0.5m','@1m','@3m'])
        else:
            df = pd.read_csv(log_path)

        error_flag = False
        for image in images:
            if image.tie_points is None:
                print(f"image {image.id} has no tie points")
                error_flag = True
        if error_flag:
            print("error check aborted")
            return
        
        coords = []
        distances = []
        for image in images:
            lines = image.tie_points[:,0]
            samps = image.tie_points[:,1]
            heights = image.dem[lines,samps]
            lats,lons = image.rpc.RPC_PHOTO2OBJ(samps,lines,heights,'numpy')
            coords.append(np.stack([lats,lons],axis=-1))
        n = len(coords)
        print(n)
        for i in range(n-1):
            for j in range(i+1,n):
                distances.append(haversine_distance(coords[i],coords[j]))
        
        distances = np.stack(distances,axis=-1).reshape(-1)

        log = {
            'num_blocks':self.options.mapper_blocks_num,
            'grid_size':self.options.grid_size,
            'median':np.median(distances),
            'mean':distances.mean(),
            'max':distances.max(),
            'min':distances.min(),
            '@0.5m':(distances < .5).sum() * 1. / len(distances),
            '@1m':(distances < 1.).sum() * 1. / len(distances),
            '@3m':(distances < 3.).sum() * 1. / len(distances)
        }

        df.loc[len(df)] = log
        df.to_csv(log_path,index=False)        

        return distances

        
            
        

    #TODO: debug!!!
    
            
        # def __calculate_warped_extend__(self):
    #     width = self.extend[2] - self.extend[0]
    #     height = self.extend[3] - self.extend[1]
    #     mid_x = (self.extend[0] + self.extend[2]) / 2.
    #     mid_y = (self.extend[1] + self.extend[3]) / 2.
    #     scale = np.array([width / 2.,height / 2.])
    #     af_mat = np.array([1.,0.,mid_x,
    #                        0.,1.,mid_y])

    #     dem = np.stack([element.dem for element in self.elements])        
    #     dem_mean = np.mean(dem)
    #     dem_std = np.std(dem)
    #     dem_min = max(dem.min(),dem_mean - 2 * dem_std)
    #     dem_max = min(dem.max(),dem_mean + 2 * dem_std)
    #     warped_extend = np.concatenate([af_mat.reshape(-1),scale,np.array([dem_min,dem_max])])
    #     return warped_extend


    

        
    
    