import os
import logging

from sklearn import metrics

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
from model.encoder_dino_0927 import EncoderDino
from model.solver import AffineFitter
import cv2
from utils import mercator2lonlat, project_mercator
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
from typing import List,Dict
from sklearn.manifold import TSNE

from rs_image import RSImage
from grid_1012 import Grid

from utils import visualize_subset_points

def train_grid_worker(rank:int, task_queue, task_state, encoder_state_dict, imgs, options):
    device = torch.device(f'cuda:{rank}')
    while True:
        try:
            task_config = task_queue.get(timeout = 1)
            if task_config is None:
                break
        except queue.Empty:
            break

        if options.resume_training:
            task_id,grid_path,output_path = task_config
            task_state[task_id]['status'] = f"Grid {task_id} 状态：正在初始化"
            os.makedirs(output_path,exist_ok=True)
            encoder = EncoderDino(os.path.join(options.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'))
            encoder.load_adapter(os.path.join(options.encoder_path,'adapter.pth'))
            grid = Grid(options = options,
                        encoder = encoder,
                        grid_path = grid_path,
                        output_path = output_path,
                        device = device
                        )
        else:
            task_id,diag,output_path = task_config
            task_state[task_id]['status'] = f"Grid {task_id} 状态：正在初始化"
            os.makedirs(output_path,exist_ok=True)
            encoder = EncoderDino(os.path.join(options.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'))
            encoder.load_adapter(os.path.join(options.encoder_path,'adapter.pth'))
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
        grid.train(task_info = {'state':task_state,'id':task_id})

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
        self.encoder = EncoderDino(os.path.join(options.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'))
        self.encoder.load_adapter(os.path.join(options.encoder_path,'adapter.pth'))
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
                np.stack([x0 + self.options.grid_offset_x, y0 + self.options.grid_offset_y], axis=1),
                np.stack([x1 + self.options.grid_offset_x, y1 + self.options.grid_offset_y], axis=1)
            ], axis=1)
            
            return diags

        if self.options.resume_training:
            grid_names = os.listdir(self.grid_root)
            grid_names = sorted(grid_names, key=lambda s: int(s.split('_')[1]))
            grid_paths = [os.path.join(self.grid_root,i) for i in grid_names]
            grid_num = len(grid_paths)
            print(f"{len(grid_paths)} grids is going to resume creating")
        else:
            corners = np.stack([image.corner_xys for image in self.imgs])
            grid_diags = find_grids(corners,grid_size) # M,2,2
            if max_grid_num > 0:
                grid_diags = grid_diags[:max_grid_num]
            grid_num = len(grid_diags)
            self.imgs[0].vis_grid(grid_diags,os.path.join(self.root,'all_grids.png'))
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
                if self.options.resume_training:
                    task_queue.put((task_id,grid_paths[i],os.path.join(self.grid_root,f"grid_{task_id}")))
                else:
                    task_queue.put((task_id,grid_diags[i],os.path.join(self.grid_root,f"grid_{task_id}")))
                task_states[task_id] = {
                    "status":f"Grid {task_id}:等待分配GPU",
                    "progress":0,
                    "total":1,
                    "info":{
                        # 'lr':0,
                        # 'dist':0,
                        # 's':0,
                        # 'obj':0,
                        # 'photo':0,
                        # 'h':0,
                        # 'reg':0,
                        # 'min':0
                    }
                }
            for _ in range(world_size):
                task_queue.put(None)
            

            # pbars = []
            # for i in range(grid_num):
            #     task_id = i + 1
            #     # print(f"============================== Grid {task_id} ==============================")
            #     bar = tqdm(total=1,
            #                desc=f"Grid {task_id} 状态：等待初始化",
            #                position=i * 2 + 1,
            #                leave=True)
            #     pbars.append(bar)

            processes = []
            for rank in track(range(world_size), description="[bold green]正在启动工作进程..."):
                p = mp.Process(target=train_grid_worker,args=(rank, task_queue, task_states, self.encoder.state_dict(), self.imgs, self.options))
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

            

            with Live(progress_table, refresh_per_second=50, screen=False) as live:
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
                        # bar = pbars[i]
                        # bar.set_description(f"{state['status']}")
                        # bar.total = state['total']
                        # bar.n = state['progress']
                        # bar.set_postfix(state['info'])
                        # bar.refresh() 
                    time.sleep(0.02) 

            # for bar in pbars:
            #     bar.close()
                for p in processes:
                    p.join()                   
            # round_num = int(np.ceil(grid_num / world_size))
            # for round_idx in range(round_num):
            #     grids_to_train = self.grids[round_idx * world_size : (round_idx + 1) * world_size]
            #     mp.spawn(train_grid_worker,
            #             args=(world_size,round_idx,grids_to_train),
            #             nprocs=len(grids_to_train),
            #             join=True)
        except Exception as e:
            print(f"格网多进程训练出错：\n{e}")
        
        print(f"======================================All Grids created successfully, {len(self.grids)} grids created in total======================================\n\n\n\n\n\n\n\n\n\n")
    
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

        raw_dis = torch.norm(src - tgt_mu,dim=-1)
        plt.hist(raw_dis.cpu().numpy(),bins=100)
        plt.savefig(os.path.join(self.root,'raw_dis_hist.png'))
        plt.close()
        visualize_subset_points(src.cpu().numpy()[:10000],tgt_mu.cpu().numpy()[:10000],os.path.join(self.root,'raw_points.png'),point_radius=2)
        print(f"raw_dis: {raw_dis.min()} \t {raw_dis.max()} \t {raw_dis.mean()} \t {raw_dis.median()}")
        print(f"valid_scores: {valid_scores.min()} \t {valid_scores.max()} \t {valid_scores.mean()} \t {valid_scores.median()}")       
        avg_sigma = torch.norm(tgt_sigma,dim=-1).mean()
        print(f"avg_sigma:{avg_sigma.item()}")

        fitter = AffineFitter()

        valid_mask = valid_scores > .5
        src = src[valid_mask]
        tgt_mu = tgt_mu[valid_mask]
        tgt_sigma = tgt_sigma[valid_mask]

        
        total_num = len(valid_scores)
        _,mask = cv2.estimateAffine2D(src.cpu().numpy(),tgt_mu.cpu().numpy(),method=cv2.RANSAC,ransacReprojThreshold= 200)
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

        fitted_matrix = fitter.fit(src,tgt_mu,tgt_sigma)
        # dis = np.linalg.norm(locals + (np.mean(targets,axis=0)[None] - np.mean(locals,axis=0)[None]) - targets,axis=-1)
        # print("mean_dis:",dis.mean())
        # dis_valid_idx = dis < dis.mean() + dis.std()
        # locals = locals[dis_valid_idx]
        # targets = targets[dis_valid_idx]
        # offset = np.mean(targets,axis=0) - np.mean(locals,axis=0)
        
        return fitted_matrix

    def load_grids(self,path = None):
        if path is None:
            path = os.path.join(self.root,'grids')
        grid_num = self.options.grid_num
        grid_paths = [i for i in os.listdir(path) if 'grid_' in i]
        if grid_num <= 0:
            grid_num = len(grid_paths)
        good_grids_num = 0
        bad_grids_num = 0
        for grid_path in grid_paths[:grid_num]:
            new_grid = Grid(self.options,self.encoder,os.path.join(path,grid_path),grid_path=os.path.join(path,grid_path))
            # if new_grid.status == new_grid.STATES.WELL_TRAINED:
            #     self.grids.append(new_grid)
            #     good_grids_num += 1
            # else:
            #     bad_grids_num += 1
            self.grids.append(new_grid)
            good_grids_num += 1
        print(f"{len(grid_paths)} grids loaded \t including {good_grids_num} good grids and {bad_grids_num} bad grids \t total {len(self.grids)} grids in RSEA now")
    
    def _visualize_error_vectors(self, pred_xyh: torch.Tensor, local_linesamp: torch.Tensor, image_to_adjust: RSImage, save_path: str):
        """
        可视化预测误差向量。
        
        Args:
            pred_xyh (torch.Tensor): 模型预测的地理坐标 (X,Y,H)。
            local_linesamp (torch.Tensor): 预测点在影像上的原始像素坐标 (line, samp)。
            image_to_adjust (RSImage): 正在被调整的影像对象。
            save_path (str): 图像保存路径。
        """
        print("正在计算并可视化误差向量...")
        pred_xyh_np = pred_xyh.cpu().numpy()
        local_linesamp_np = local_linesamp.cpu().numpy()

        # 使用影像自身的RPC，从像素坐标反算出一个 "基准" 地理坐标
        # 注意：这里的DEM高度是一个近似值，可以使用区域平均高程或直接使用预测高程
        heights = pred_xyh_np[:, 2] 
        lats_true, lons_true = image_to_adjust.rpc.RPC_PHOTO2OBJ(local_linesamp_np[:, 1], local_linesamp_np[:, 0], heights, 'numpy')
        xy_true = project_mercator(np.stack([lats_true, lons_true], axis=-1))[:, [1, 0]]
        
        # 计算误差向量 (在墨卡托投影下)
        error_vectors_xy = pred_xyh_np[:, :2] - xy_true
        
        # 为了绘图清晰，随机采样一部分点
        num_points = len(error_vectors_xy)
        sample_size = min(num_points, 2000)
        indices = np.random.choice(num_points, sample_size, replace=False)

        sampled_points = xy_true[indices]
        sampled_vectors = error_vectors_xy[indices]

        # 绘图
        plt.figure(figsize=(15, 15))
        # 使用 quiver 绘制向量场
        plt.quiver(sampled_points[:, 0], sampled_points[:, 1], 
                   sampled_vectors[:, 0], sampled_vectors[:, 1], 
                   color='r', angles='xy', scale_units='xy', scale=1, width=0.001)
        
        # 也可以绘制点的位置作为参考
        plt.scatter(sampled_points[:, 0], sampled_points[:, 1], s=1, c='b', alpha=0.5, label='Error Vector Origins')

        plt.title('Prediction Error Vector Field')
        plt.xlabel('Mercator X (meters)')
        plt.ylabel('Mercator Y (meters)')
        ax = plt.gca()
        ax.set_aspect('equal', adjustable='box')
        plt.legend()
        plt.grid(True)
        
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"误差向量图已保存至: {save_path}")
        
        # 打印统计信息
        errors_meters = np.linalg.norm(error_vectors_xy, axis=1)
        print(f"误差统计 (米): 平均值={np.mean(errors_meters):.2f}, 中位数={np.median(errors_meters):.2f}, 最大值={np.max(errors_meters):.2f}")

    def adjust(self,image_folders:List[str]):        
        adjust_images:List[RSImage] = []
        
        print("Loading Adjust Images")
        for image_id,image_folder in enumerate(tqdm(image_folders)):
            image = RSImage(self.options,image_folder,image_id)
            adjust_images.append(image)
        print(f"{len(adjust_images)} adjust images loaded")
        
        for img_idx,image in enumerate(adjust_images):
            all_src = []
            all_tgt_mu = []
            all_tgt_sigma = []
            all_valid_scores = []
            
            # 用于误差可视化的临时容器
            all_pred_xyh_for_vis = []
            all_locals_for_vis = []

            for grid_idx,grid in enumerate(self.grids):
                print(f"processing grid {grid_idx}")
                overlap_diag = self.__overlap__(grid.diag[0],image.corner_xys[0],grid.diag[1],image.corner_xys[3])
                if overlap_diag is None :
                    print(f"no overlap in grid {grid_idx}")
                    continue
                img_raw,dem,local_hw2 = grid.get_overlap_image(image,mode="interpolate")
                cv2.imwrite(os.path.join(grid.output_path,f'adjust_img_{img_idx}.png'),img_raw)
                
                pred_res = grid.pred_xyh(img_raw,local_hw2)

                # 新增逻辑：收集用于可视化的数据
                if pred_res and pred_res['mu_xyh_P3'].numel() > 0:
                    all_pred_xyh_for_vis.append(pred_res['mu_xyh_P3'])
                    all_locals_for_vis.append(pred_res['locals_P2'])
                else:
                    # 如果预测结果为空，则跳过此grid
                    continue
                
                mu_linesamp,sigma_linesamp = image.rpc.xy_distribution_to_linesamp(pred_res['mu_xyh_P3'],pred_res['sigma_xyh_P3'])
                local_linesamp = pred_res['locals_P2']
                conf = pred_res['confs_P1']
                valid_score = pred_res['valid_score_P1']

                all_src.append(local_linesamp)
                all_tgt_mu.append(mu_linesamp)
                all_tgt_sigma.append(sigma_linesamp)
                all_valid_scores.append(valid_score)

            # 新增调用
            if all_pred_xyh_for_vis:
                pred_xyh_vis = torch.cat(all_pred_xyh_for_vis, dim=0)
                locals_vis = torch.cat(all_locals_for_vis, dim=0)
                vis_save_path = os.path.join(self.root, f'adjust_img_{img_idx}_error_vectors.png')
                self._visualize_error_vectors(pred_xyh_vis, locals_vis, image, vis_save_path)

            if not all_src: # 如果没有任何预测结果，则跳过后续
                print(f"影像 {img_idx} 未能从任何Grid中获得预测结果，跳过调整。")
                continue

            all_src = torch.concatenate(all_src,dim=0).detach()
            all_tgt_mu = torch.concatenate(all_tgt_mu,dim=0).detach()
            all_tgt_sigma = torch.concatenate(all_tgt_sigma,dim=0).detach()
            all_valid_scores = torch.concatenate(all_valid_scores,dim=0).detach()

            transform = self.__calculate_transform__(all_src,all_tgt_mu,all_tgt_sigma,all_valid_scores)
            image.rpc.Update_Adjust(transform)
            print(image.rpc.adjust_params.cpu().numpy())

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
            # orthorectify_image(image.image[:,:,0],image.dem,image.rpc,os.path.join(image.root,'dom.tif'))
            # image.rpc.save_rpc_to_file(os.path.join(image.root,'rpc_corrected.txt'))
            # timestamp  = time.strftime("%Y%m%d%H%M%S")
            # with open(os.path.join(image.root,f'adjust_info_{timestamp}.txt'),'w') as f:
            #     for k,v in vars(options).items():
            #         info = info + f"{k}:{v}\n"
            #     f.write(info)

            # return adjust_images
        errors = self.check_error(os.path.join('./log',f'adjust_log_{self.options.log_postfix}.csv'),adjust_images)
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
        
    def visualize_feature_distribution(self, source_image_folder: str, target_image_folder: str, grid_idx_to_use: int = 0, sample_size: int = 5000):
        """
        可视化来自两个不同域（例如，不同卫星）的影像特征分布。
        
        Args:
            source_image_folder (str): 源域影像的文件夹路径 (参与训练的卫星)。
            target_image_folder (str): 目标域影像的文件夹路径 (新卫星)。
            grid_idx_to_use (int): 使用哪个已加载的Grid来进行特征提取。
            sample_size (int): 每个域随机采样多少个特征点进行可视化，以避免计算量过大。
        """
        print("开始进行特征分布诊断...")
        if not self.grids:
            print("错误：请先加载Grid (load_grids)。")
            return
        if grid_idx_to_use >= len(self.grids):
            print(f"错误：grid_idx_to_use={grid_idx_to_use} 超出范围，只有 {len(self.grids)} 个grids。")
            return
        
        # 将grid移动到主进程的默认GPU上
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
        grid = self.grids[grid_idx_to_use]
        grid.to_device(device)

        print(f"将使用 Grid {grid_idx_to_use} 在设备 {device} 上进行特征提取。")

        # 1. 加载影像并提取特征
        features_all = []
        labels_all = []

        for domain_idx, folder in enumerate([source_image_folder, target_image_folder]):
            domain_name = "Source (Train)" if domain_idx == 0 else "Target (New)"
            print(f"正在处理 {domain_name} 域影像: {folder}")
            
            if not os.path.exists(folder):
                print(f"错误: 路径不存在 {folder}")
                continue

            image = RSImage(self.options, folder, 999 + domain_idx) # 临时ID
            
            # 获取与Grid重叠的影像部分
            img_raw, _, _ = grid.get_overlap_image(image, mode="interpolate")
            if img_raw is None or img_raw.size == 0:
                print(f"警告: 影像与Grid {grid_idx_to_use} 没有重叠，跳过。")
                continue
            
            # 提取特征
            features = grid._extract_full_features(img_raw).cpu().numpy()
            
            # 2. 随机下采样
            if len(features) > sample_size:
                indices = np.random.choice(len(features), sample_size, replace=False)
                features = features[indices]
            
            features_all.append(features)
            labels_all.extend([domain_idx] * len(features))

        if not features_all:
            print("未能提取到任何特征，诊断中止。")
            return

        features_all = np.concatenate(features_all, axis=0)
        labels_all = np.array(labels_all)

        # 3. 运行 t-SNE
        print("特征提取完成，正在运行 t-SNE... (这可能需要几分钟)")
        tsne = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=300, random_state=42)
        tsne_results = tsne.fit_transform(features_all)

        # 4. 绘图
        print("t-SNE 计算完成，正在绘图...")
        plt.figure(figsize=(12, 10))
        scatter = plt.scatter(tsne_results[:,0], tsne_results[:,1], c=labels_all, cmap=plt.cm.get_cmap("jet", 2), alpha=0.6)
        plt.title('Feature Distribution Visualization (t-SNE)')
        plt.xlabel('t-SNE Component 1')
        plt.ylabel('t-SNE Component 2')
        handles, _ = scatter.legend_elements()
        plt.legend(handles=handles, labels=["Source (Train)", "Target (New)"])
        
        save_path = os.path.join(self.root, 'feature_distribution_diagnosis.png')
        plt.savefig(save_path, dpi=300)
        plt.close()
        print(f"诊断图已保存至: {save_path}")
        print("请检查图片：如果两类点云分离明显，则证明存在显著的域偏移。")

