import os
import logging

from sklearn import metrics

import grid
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
logging.basicConfig(level=logging.ERROR)
import time
from turtle import pos
import warnings
warnings.filterwarnings('ignore')
import argparse
import torch
import torch.nn as nn
import torch.multiprocessing as mp
import numpy as np
import pandas as pd
from model_new import Encoder,AffineFitter
import cv2
from utils import mercator2lonlat
import queue
from rpc import RPCModelParameterTorch
from tqdm import tqdm,trange
from rich.live import Live
from rich.table import Table
from rich.progress import Progress, BarColumn, TextColumn, TimeRemainingColumn, track
import torch.nn.functional as F
from orthorectify import orthorectify_image
from matplotlib import pyplot as plt
import random
from typing import List,Dict,Set

from rs_image import RSImage
from grid import Grid

cfg_base = {
            'input_channels':3,
            'output_channels':512,
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
        'output_channels':512,
        'img_size':1024,
        'window_size':16,
        'embed_dim':192,
        'depth':[2,2,18],
        'num_heads':[6, 12, 24],
        'drop_path_rate':.2,
        'pretrain_window_size':[12, 12, 12],
        'unfreeze_backbone_modules':[]
    }

def train_grid_worker(rank:int, task_queue, task_state, encoder_state_dict, imgs, options):
    device = torch.device(f'cuda:{rank}')
    while True:
        try:
            task_config = task_queue.get(timeout = 1)
            if task_config is None:
                break
        except queue.Empty:
            break

        task_id,diag,output_path = task_config
        task_state[task_id]['status'] = f"Grid {task_id} 状态：正在初始化"
        os.makedirs(output_path,exist_ok=True)
        encoder = Encoder(cfg_large,verbose=0)
        encoder.load_state_dict(encoder_state_dict)
        grid = Grid(options = options,
                    encoder = encoder,
                    diag = diag,
                    output_path = output_path,
                    device = device
                    )
        for img in imgs:
            grid.add_img(img = img)
        grid.to_device(device)
        grid.create_elements(task_info = {'state':task_state,'id':task_id})
        grid.train_mapper(task_info = {'state':task_state,'id':task_id},save_checkpoint=options.save_checkpoints)

def finetune_grid_worker(rank:int, task_queue, task_state, encoder_state_dict, imgs, options):
    device = torch.device(f'cuda:{rank}')
    while True:
        try:
            task_config = task_queue.get(timeout = 1)
            if task_config is None:
                break
        except queue.Empty:
            break

        task_id,grid_path,output_path = task_config
        task_state[task_id]['status'] = f"Grid {task_id} 状态：正在初始化"
        os.makedirs(output_path,exist_ok=True)
        encoder = Encoder(cfg_large,verbose=0)
        encoder.load_state_dict(encoder_state_dict)
        grid = Grid(options = options,
                    encoder = encoder,
                    grid_path = grid_path,
                    output_path = output_path,
                    device = device
                    )
        
        for img in imgs:
            grid.add_img(img = img)
        grid.to_device(device)
        grid.create_elements(task_info = {'state':task_state,'id':task_id})
        grid.finetune_mapper(task_info = {'state':task_state,'id':task_id},save_checkpoint=options.save_checkpoints)

def dict2str(dict):
    output = ""
    keys = dict.keys()
    if len(keys) == 0:
        return output
    for key in keys:
        output += f"{key}={dict[key]},"
    return output[:-1]

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
        self.encoder = Encoder(cfg_large,verbose=0)
        self.encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(self.options.encoder_path).items()})
        self.encoder.eval()
        self.root = options.root
        self.grid_root = os.path.join(self.root,"grids")
        os.makedirs(self.grid_root,exist_ok=True)
        
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
    

    def find_init_grids(self, corners, grid_size):
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
    
    def find_new_grids(self, corners: np.ndarray, exist_grids: np.ndarray, size: float) -> np.ndarray:
        """
        在一个由多个四边形定义的区域内，划分出尽可能多的、边长为size的、轴向的正方形网格，
        同时避开已有的网格区域。

        Args:
            corners (np.ndarray): 形状为 (N, 4, 2) 的数组，记录N个四边形的顶点。
                                    第二个维度的顺序为 [左上, 右上, 左下, 右下]。
            exist_grids (np.ndarray): 形状为 (M, 2, 2) 的数组，记录M个已有网格的
                                    [左上角, 右下角] 坐标。
            size (float): 新划分的正方形网格的边长。

        Returns:
            np.ndarray: 形状为 (K, 2, 2) 的数组，记录K个新生成的网格的
                        [左上角, 右下角] 坐标。
        """
        # --- 嵌套的几何计算辅助函数 ---

        def sign(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
            """
            计算一个点p1相对于由p2和p3定义的有向线段的位置。
            """
            return (p1[0] - p3[0]) * (p2[1] - p3[1]) - (p2[0] - p3[0]) * (p1[1] - p3[1])

        def is_point_in_triangle(point: np.ndarray, v1: np.ndarray, v2: np.ndarray, v3: np.ndarray) -> bool:
            """
            判断一个点是否在三角形内部（或边界上）。
            """
            d1 = sign(point, v1, v2)
            d2 = sign(point, v2, v3)
            d3 = sign(point, v3, v1)
            has_neg = (d1 < 0) or (d2 < 0) or (d3 < 0)
            has_pos = (d1 > 0) or (d2 > 0) or (d3 > 0)
            return not (has_neg and has_pos)

        def is_point_in_quad(point: np.ndarray, quad_corners: np.ndarray) -> bool:
            """
            判断一个点是否在一个四边形内部。
            """
            tl, tr, bl, br = quad_corners[0], quad_corners[1], quad_corners[2], quad_corners[3]
            in_triangle1 = is_point_in_triangle(point, tl, tr, br)
            in_triangle2 = is_point_in_triangle(point, tl, br, bl)
            return in_triangle1 or in_triangle2

        def is_point_in_union(point: np.ndarray, all_quads: np.ndarray) -> bool:
            """
            判断一个点是否在所有四边形的联合区域内。
            """
            for quad in all_quads:
                if is_point_in_quad(point, quad):
                    return True
            return False

        # --- 主函数逻辑开始 ---
        
        # 处理输入为空的边界情况
        if corners.shape[0] == 0:
            return np.empty((0, 2, 2))

        # 1. 计算所有四边形的总边界框，以确定搜索范围
        all_points = corners.reshape(-1, 2)
        min_x, min_y = np.min(all_points, axis=0)
        max_x, max_y = np.max(all_points, axis=0)

        valid_grids_list = []

        # 2. 在总边界框内生成候选网格并进行筛选
        for x in np.arange(min_x, max_x, size):
            for y in np.arange(min_y, max_y, size):
                
                cand_tl = np.array([x, y])
                cand_br = np.array([x + size, y + size])

                if cand_br[0] > max_x or cand_br[1] > max_y:
                    continue
                
                # 3. 排他检查：确保候选网格不与任何已有网格重叠
                is_excluded = False
                for exist_tl, exist_br in exist_grids:
                    separated = (
                        cand_br[0] <= exist_tl[0] or
                        cand_tl[0] >= exist_br[0] or
                        cand_br[1] <= exist_tl[1] or
                        cand_tl[1] >= exist_br[1]
                    )
                    if not separated:
                        is_excluded = True
                        break
                
                if is_excluded:
                    continue

                # 4. 包含检查：确保候选网格完全位于四边形联合区域内
                cand_corners = [
                    cand_tl,
                    np.array([cand_br[0], cand_tl[1]]),
                    np.array([cand_tl[0], cand_br[1]]),
                    cand_br
                ]
                
                is_fully_included = True
                for point in cand_corners:
                    if not is_point_in_union(point, corners):
                        is_fully_included = False
                        break
                
                # 5. 如果通过所有检查，则将其添加到结果列表中
                if is_fully_included:
                    valid_grids_list.append([cand_tl, cand_br])

        # 6. 将结果列表转换为Numpy数组并返回
        if not valid_grids_list:
            return np.empty((0, 2, 2))
        else:
            return np.array(valid_grids_list, dtype=np.float64)

    def create_grids(self,imgs, grid_diags:np.ndarray,max_grid_num:int = -1):
        # if self.options.resume_training:
        #     grid_names = os.listdir(self.grid_root)
        #     grid_names = sorted(grid_names, key=lambda s: int(s.split('_')[1]))
        #     grid_paths = [os.path.join(self.grid_root,i) for i in grid_names]
        #     grid_num = len(grid_paths)
        #     print(f"{len(grid_paths)} grids is going to resume creating")
        # else:
        # corners = np.stack([image.corner_xys for image in imgs])
        # grid_diags = self.find_init_grids(corners,grid_size) # M,2,2
        if max_grid_num > 0:
            grid_diags = grid_diags[:max_grid_num]
        grid_num = len(grid_diags)
        print(f"{len(grid_diags)} grids is going to be created")

        try:
            mp.set_start_method("spawn", force=True)
            gpu_num = torch.cuda.device_count()
            world_size = min(gpu_num,grid_num)
            manager = mp.Manager()
            task_queue = manager.Queue()
            task_states = manager.dict()

            for i in range(grid_num):
                task_id = i + 1
                # if self.options.resume_training:
                #     task_queue.put((task_id,grid_paths[i],os.path.join(self.grid_root,f"grid_{task_id}")))
                # else:
                task_queue.put((task_id,grid_diags[i],os.path.join(self.grid_root,f"grid_{task_id}")))
                task_states[task_id] = {
                    "status":f"Grid {task_id}:等待分配GPU",
                    "progress":0,
                    "total":1,
                    "info":{

                    }
                }
            for _ in range(world_size):
                task_queue.put(None)

            processes = []
            for rank in track(range(world_size), description="[bold green]正在启动工作进程..."):
                p = mp.Process(target=train_grid_worker,args=(rank, task_queue, task_states, self.encoder.state_dict(), imgs, self.options))
                p.start()
                processes.append(p)

            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=None,finished_style='green'),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                TextColumn("[bold yellow]{task.fields[metrics]}"),
                expand=True
            )
            task_progress_ids = [progress.add_task(f"{i+1}", total=1, metrics = "") for i in range(grid_num)]
            progress_table = Table.grid(expand=True)
            progress_table.add_row(progress)

            with Live(progress_table, refresh_per_second=50, screen=False, transient=False) as live:
                acitive_workers = world_size
                while acitive_workers > 0:
                    acitive_workers = 0
                    for p in processes:
                        if p.is_alive():
                            acitive_workers += 1
                    
                    for i in range(grid_num):
                        task_id = i + 1
                        state = task_states[task_id]
                        progress.update(
                            task_id=task_progress_ids[i],
                            completed=state['progress'],
                            total=state['total'],
                            description=state['status'],
                            metrics=dict2str(state['info'])                            
                        )
 
                    time.sleep(0.02) 

                for p in processes:
                    p.join()                   
 
        except Exception as e:
            print(f"格网多进程训练出错：\n{e}")
        
        print(f"======================================All Grids created successfully, {len(self.grids)} grids created in total======================================\n\n\n\n\n\n\n\n\n\n")
    
    def finetune_grids(self,imgs,need_finetune_grids):
        finetune_grid_num = len(need_finetune_grids)
        try:
            mp.set_start_method("spawn", force=True)
            gpu_num = torch.cuda.device_count()
            world_size = min(gpu_num,finetune_grid_num)
            manager = mp.Manager()
            task_queue = manager.Queue()
            task_states = manager.dict()

            for i in range(finetune_grid_num):
                task_id = i + 1
                task_queue.put((task_id,need_finetune_grids[i].output_path,need_finetune_grids[i].output_path))
                task_states[task_id] = {
                    "status":f"Grid {task_id}:等待分配GPU",
                    "progress":0,
                    "total":1,
                    "info":{

                    }
                }
            for _ in range(world_size):
                task_queue.put(None)

            processes = []
            for rank in track(range(world_size), description="[bold green]正在启动工作进程..."):
                p = mp.Process(target=finetune_grid_worker,args=(rank, task_queue, task_states, self.encoder.state_dict(), imgs, self.options))
                p.start()
                processes.append(p)

            progress = Progress(
                TextColumn("[bold blue]{task.description}"),
                BarColumn(bar_width=None,finished_style='green'),
                "[progress.percentage]{task.percentage:>3.1f}%",
                "•",
                TextColumn("[bold yellow]{task.fields[metrics]}"),
                expand=True
            )
            task_progress_ids = [progress.add_task(f"{i+1}", total=1, metrics = "") for i in range(finetune_grid_num)]
            progress_table = Table.grid(expand=True)
            progress_table.add_row(progress)

            with Live(progress_table, refresh_per_second=50, screen=False, transient=False) as live:
                acitive_workers = world_size
                while acitive_workers > 0:
                    acitive_workers = 0
                    for p in processes:
                        if p.is_alive():
                            acitive_workers += 1
                    
                    for i in range(finetune_grid_num):
                        task_id = i + 1
                        state = task_states[task_id]
                        progress.update(
                            task_id=task_progress_ids[i],
                            completed=state['progress'],
                            total=state['total'],
                            description=state['status'],
                            metrics=dict2str(state['info'])                            
                        )

                    time.sleep(0.02) 

                for p in processes:
                    p.join()                   

        except Exception as e:
            print(f"格网多进程微调出错：\n{e}")
        
        print(f"======================================All Grids finetuned successfully, {len(self.grids)} grids in total======================================\n\n\n\n\n\n\n\n\n\n")


    def __overlap__(self,tl1:np.ndarray,tl2:np.ndarray,br1:np.ndarray,br2:np.ndarray):
        """
        return : [tl,br] [x,y] np.ndarray
        """
        if tl1[0] > br2[0] or tl1[1] < br2[1] or br1[0] < tl2[0] or br1[1] > tl2[1]:
            return None
        tl = np.array([max(tl1[0],tl2[0]),min(tl1[1],tl2[1])])
        br = np.array([min(br1[0],br2[0]),max(br1[1],br2[1])])
        return np.stack([tl,br],axis=0)

    def __calculate_transform__(self,src:torch.Tensor,tgt_mu:torch.Tensor,tgt_sigma:torch.Tensor,valid_scores:torch.Tensor) -> torch.Tensor:

        print(f"valid_scores: {valid_scores.min()} \t {valid_scores.max()} \t {valid_scores.mean()} \t {valid_scores.median()}")
        avg_sigma = torch.norm(tgt_sigma,dim=-1).mean()
        print(f"avg_sigma:{avg_sigma.item()}")

        fitter = AffineFitter()

        valid_mask = valid_scores > .5
        src = src[valid_mask]
        tgt_mu = tgt_mu[valid_mask]
        tgt_sigma = tgt_sigma[valid_mask]

        
        total_num = len(valid_scores)
        _,mask = cv2.estimateAffine2D(src.cpu().numpy(),tgt_mu.cpu().numpy(),method=cv2.RANSAC,ransacReprojThreshold=avg_sigma.item())
        inliers = mask.ravel() == 1

        src = src[inliers]
        tgt_mu = tgt_mu[inliers]
        tgt_sigma = tgt_sigma[inliers]
        # conf_valid_idx = valid_scores > .5
        # src = src[conf_valid_idx]
        # tgt_mu = tgt_mu[conf_valid_idx]
        # tgt_sigma = tgt_sigma[conf_valid_idx]
        # valid_scores = valid_scores[conf_valid_idx]
        print(f"valid filter :{inliers.sum()}/{valid_mask.sum()}/{total_num}")

        fitted_matrix,res = fitter.fit(src,tgt_mu,tgt_sigma,True)
        # dis = np.linalg.norm(locals + (np.mean(targets,axis=0)[None] - np.mean(locals,axis=0)[None]) - targets,axis=-1)
        # print("mean_dis:",dis.mean())
        # dis_valid_idx = dis < dis.mean() + dis.std()
        # locals = locals[dis_valid_idx]
        # targets = targets[dis_valid_idx]
        # offset = np.mean(targets,axis=0) - np.mean(locals,axis=0)
        
        return fitted_matrix,res

    def load_grids(self,path = None,clear = True):
        if path is None:
            path = os.path.join(self.root,'grids')
        grid_num = self.options.grid_num
        grid_paths = [i for i in os.listdir(path) if 'grid_' in i]
        if grid_num <= 0:
            grid_num = len(grid_paths)
        good_grids_num = 0
        bad_grids_num = 0
        if clear:
            self.grids = []
        for grid_path in grid_paths[:grid_num]:
            new_grid = Grid(self.options,self.encoder,os.path.join(path,grid_path),grid_path=os.path.join(path,grid_path))
            if True or new_grid.status == new_grid.STATES.WELL_TRAINED:
                self.grids.append(new_grid)
                good_grids_num += 1
            else:
                bad_grids_num += 1
        print(f"{len(grid_paths)} grids loaded \t including {good_grids_num} good grids and {bad_grids_num} bad grids \t total {len(self.grids)} grids in RSEA now")
    
    


    def adjust(self,adjust_images:List[RSImage]):        
        
        adjust_list = []
        not_adjust_list = []

        for img_idx,image in enumerate(adjust_images):
            all_src = []
            all_tgt_mu = []
            all_tgt_sigma = []
            all_valid_scores = []
            for grid_idx,grid in enumerate(self.grids):
                print(f"processing grid {grid_idx}")
                overlap_diag = self.__overlap__(grid.diag[0],image.corner_xys[0],grid.diag[1],image.corner_xys[3])
                if overlap_diag is None :
                    print(f"no overlap in grid {grid_idx}")
                    continue
                image.overlap_grids.append(grid_idx)
                img_raw,dem,local_hw2 = grid.get_overlap_image(image,mode="interpolate")
                cv2.imwrite(os.path.join(grid.output_path,f'adjust_img_{img_idx}.png'),img_raw)
                
                pred_res = grid.pred_xyh(img_raw,local_hw2)

                mu_linesamp,sigma_linesamp = image.rpc.xy_distribution_to_linesamp(pred_res['mu_xyh_P3'],pred_res['sigma_xyh_P3'])
                local_linesamp = pred_res['locals_P2']
                conf = pred_res['confs_P1']
                valid_score = pred_res['valid_score_P1']

                all_src.append(local_linesamp)
                all_tgt_mu.append(mu_linesamp)
                all_tgt_sigma.append(sigma_linesamp)
                all_valid_scores.append(valid_score)
            if len(all_src) == 0:
                not_adjust_list.append(img_idx)
                print(f"no overlap in image {img_idx}")
                continue
            all_src = torch.concatenate(all_src,dim=0).detach()
            all_tgt_mu = torch.concatenate(all_tgt_mu,dim=0).detach()
            all_tgt_sigma = torch.concatenate(all_tgt_sigma,dim=0).detach()
            all_valid_scores = torch.concatenate(all_valid_scores,dim=0).detach()

            transform,residual = self.__calculate_transform__(all_src,all_tgt_mu,all_tgt_sigma,all_valid_scores)
            
            if residual > self.options.residual_threshold:
                not_adjust_list.append(grid_idx)
                print(f"Image {img_idx} not adjust well, residual = {residual}")
                continue

            image.rpc.Update_Adjust(transform)
            print(f"adjust params of img {img_idx}:",image.rpc.adjust_params.cpu().numpy())
            adjust_list.append(img_idx)
        
        return adjust_list,not_adjust_list

    def adjust_zero(self,image_folders:List[str]):        
        need_adjust_images:List[RSImage] = []
        
        print("Loading Adjust Images")
        for image_id,image_folder in tqdm(enumerate(image_folders)):
            image = RSImage(self.options,image_folder,image_id)
            need_adjust_images.append(image)
        print(f"{len(need_adjust_images)} adjust images loaded")

        self.adjusted_images:List[RSImage] = []

        while True:
            self.load_grids(clear = True)
            if len(self.grids) == 0:
                init_grid_diags = self.find_init_grids(np.array([need_adjust_images[0].corner_xys]),self.options.grid_size)
                self.create_grids(imgs = need_adjust_images[0:1],
                                  grid_diags = init_grid_diags,
                                  max_grid_num = self.options.grid_num)
                self.adjusted_images.append(need_adjust_images[0])
                need_adjust_images = need_adjust_images[1:]
            else:
                adjust_list,not_adjust_list = self.adjust(need_adjust_images)
                if len(adjust_list) == 0:
                    break

                newly_adjust_images:List[RSImage] = [need_adjust_images[i] for i in adjust_list]
                self.adjusted_images.append([need_adjust_images[i] for i in adjust_list])
                need_adjust_images = [need_adjust_images[i] for i in not_adjust_list]
                
                #微调现有网格
                need_finetune_grids = []
                for new_image in newly_adjust_images:
                    for overlap_grid in new_image.overlap_grids:
                        need_finetune_grids.append(overlap_grid)
                need_finetune_grids:List[int] = list(set(need_finetune_grids))
                need_finetune_grids:List[Grid] = [self.grids[i] for i in need_finetune_grids]
                self.finetune_grids(self.adjusted_images,need_finetune_grids)
                
                #创建新网格
                cur_corners = np.stack([image.corner_xys for image in self.adjusted_images],axis=0)
                exist_diags = np.stack([grid.diag for grid in self.grids],axis=0)
                new_grid_diags = self.find_new_grids(cur_corners,exist_diags,self.options.grid_size)
                self.create_grids(imgs = newly_adjust_images,
                                  grid_diags = new_grid_diags,
                                  max_grid_num = self.options.grid_num)

            if len(need_adjust_images) == 0:
                break
        
        self.adjust(self.adjusted_images)
        
        errors = self.check_error(os.path.join('./log',f'adjust_log_{self.options.log_postfix}.csv'),self.adjusted_images)
        info = f"error:\nmax:{errors.max()}\nmin:{errors.min()}\nmean:{errors.mean()}\nmedian:{np.median(errors)}\n<1px:{(errors < 1.).sum() * 1. / len(errors)}\n<3px:{(errors < 3.).sum() * 1. / len(errors)}\n<5px:{(errors < 5.).sum() * 1. / len(errors)}"
        print(info)




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
            df = pd.DataFrame(columns=['mapper_blocks_num','grid_size','median','mean','max','min','@0.5m','@1m','@3m'])
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
            'mapper_blocks_num':self.options.mapper_blocks_num,
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
        
    
    