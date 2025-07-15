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
from torch_kdtree import build_kd_tree
from matplotlib import pyplot as plt
import random
from typing import List,Dict

from element import Element
from rs_image import RSImage
from grid import Grid

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
            grid_output_path = os.path.join(self.root,f"grid_{grid_idx}")
            os.makedirs(grid_output_path,exist_ok=True)

            new_grid = Grid(options = self.options,
                            encoder = self.encoder,
                            diag = diag,
                            output_path = grid_output_path)

            for img_idx,image in enumerate(self.imgs):
                new_grid.add_img(img = image,
                                 output_path = new_grid.output_path,
                                 id = image.id
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
        
    
    