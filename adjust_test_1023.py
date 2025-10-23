import os
import argparse
import random
from matplotlib.rcsetup import validate_markevery
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from pykeops.torch import LazyTensor
import numpy as np
import cv2
# 使用恢复的、高效的 RSImage 类
from rs_image_1022 import RSImage
from rpc import RPCModelParameterTorch
from model.encoder_dino_0927 import EncoderDino
import scheduler
from utils import find_grids,vis_feat_twin,vis_conf,downsample_average

# DDP相关的库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple

import warnings
warnings.filterwarnings("ignore")

# DDP Step 1: DDP环境初始化函数
def setup_ddp():
    """初始化DDP环境"""
    dist.init_process_group(backend='nccl')
    # torchrun 会自动设置 'LOCAL_RANK' 环境变量
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    print(f"DDP setup on rank {local_rank} with device cuda:{local_rank}")
    return local_rank

# DDP Step 2: 将需要优化的参数封装为 nn.Module
class AffineModel(nn.Module):
    """
    将仿射变换参数R和T封装成一个PyTorch模块，以便DDP管理。
    """
    def __init__(self, init_line=0.0, init_samp=0.0):
        super().__init__()
        # R 和 T 必须是 nn.Parameter 才能被DDP和优化器追踪
        # 为了DDP性能，我们使用 float32
        self.R = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        self.T = nn.Parameter(torch.tensor([init_line, init_samp], dtype=torch.float32))

    def forward(self):
        # "forward" 就返回仿射矩阵
        return torch.concatenate([self.R, self.T.unsqueeze(-1)], dim=-1)

class BundleAffineModel(nn.Module):
    """
    管理所有N-1个可学习仿射变换的模型。
    img_0 被假定为锚点(anchor)，其变换固定为单位矩阵。
    """
    def __init__(self, num_images, init_line=0.0, init_samp=0.0):
        super().__init__()
        self.num_images = num_images
        
        # 我们有 N 张影像, 但只为 img_1 ... img_N-1 创建可学习模型
        self.models = nn.ModuleList()
        for _ in range(num_images - 1):
            self.models.append(AffineModel(init_line, init_samp))
            
    def get_affine(self, index: int) -> torch.Tensor:
        """
        获取第 index 张影像的仿射变换矩阵.
        index 0 (锚点) 返回固定的单位矩阵.
        index > 0   返回其对应的可学习矩阵.
        """
        if index == 0:
            # 返回一个固定的、float32的单位仿射矩阵
            # 它需要和 model 在同一个 device 上 (通过第一个模型获取)
            # (修正) 确保在模型为空时也能工作
            device = "cuda" if torch.cuda.is_available() else "cpu"
            if len(self.models) > 0:
                device = self.models[0].R.device
            return torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], 
                                dtype=torch.float32, device=device)
        else:
            # 返回第 (index - 1) 个子模型的仿射矩阵
            # 调用子模型的 forward()
            return self.models[index - 1]()


class Window():
    def __init__(self,img:np.ndarray,local:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch):
        self.img = img
        self.local = torch.from_numpy(local)
        self.dem = torch.from_numpy(dem)
        self.rpc = rpc
        self.feature = None
        self.conf = None
        
    
    def to_gpu(self):
        # 这里的 .cuda() 会自动使用 torch.cuda.set_device 设置的当前卡
        self.local = self.local.cuda()
        self.dem = self.dem.cuda()
        self.rpc.to_gpu()

    

class Window_Pair():
    def __init__(self,args,diag:np.ndarray,img_0:RSImage,img_1:RSImage,id:int):
        """
        (平差版修改): img_0 和 img_1 现在是任意的 img_i 和 img_j
        """
        self.id = id
        resample_size = 1024
        
        # 计算 img_i (img_0) 的角点
        corners_sampline_0 = img_0.xy_to_sampline(np.array([diag[0],
                                                            [diag[1,0],diag[0,1]],
                                                            diag[1],
                                                            [diag[0,0],diag[1,1]]]))
        # 计算 img_j (img_1) 的角点
        corners_sampline_1 = img_1.xy_to_sampline(np.array([diag[0],
                                                            [diag[1,0],diag[0,1]],
                                                            diag[1],
                                                            [diag[0,0],diag[1,1]]]))
        
        # (修正) 现在从内存中的 self.image 重采样
        img_0_raw,local_0 = img_0.resample_image_by_sampline(corners_sampline_0,(resample_size,resample_size),need_local=True)
        img_1_raw,local_1 = img_1.resample_image_by_sampline(corners_sampline_1,(resample_size,resample_size),need_local=True)
        
        dem_0 = img_0.resample_dem_by_sampline(corners_sampline_0,(resample_size,resample_size))
        dem_1 = img_1.resample_dem_by_sampline(corners_sampline_1,(resample_size,resample_size))

        # Window_0 对应 img_i, Window_1 对应 img_j
        self.window_0 = Window(img_0_raw,local_0,dem_0,img_0.rpc)
        self.window_1 = Window(img_1_raw,local_1,dem_1,img_1.rpc)

        self.debug_output_path = os.path.join(args.debug_output_path,f'window_pair_{self.id}')
        
        # 只在主进程上保存调试图像，避免文件写入冲突
        if dist.get_rank() == 0:
            os.makedirs(self.debug_output_path,exist_ok=True)
            cv2.imwrite(os.path.join(self.debug_output_path,f'img_raw_{img_0.id}.png'),img_0_raw)
            cv2.imwrite(os.path.join(self.debug_output_path,f'img_raw_{img_1.id}.png'),img_1_raw)


    @torch.no_grad()
    def __extract_one_img_feature__(self,encoder:EncoderDino,img_raw:np.ndarray):
        # encoder 会被移动到当前进程对应的GPU
        encoder = encoder.cuda().eval()
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                    ])
        img_tensor = transform(img_raw)
        img_tensor = img_tensor[None].cuda()
        feature,conf = encoder(img_tensor)
        
        return feature,conf

    def extract_features(self,encoder:EncoderDino):
        feature_0,conf_0 = self.__extract_one_img_feature__(encoder,self.window_0.img)
        feature_1,conf_1 = self.__extract_one_img_feature__(encoder,self.window_1.img)
        h,w = feature_0.shape[-2:]
        self.window_0.feature = feature_0[0].permute(1,2,0).flatten(0,1)
        self.window_0.conf = conf_0.squeeze().flatten(0,1)
        self.window_0.local = downsample_average(self.window_0.local,encoder.SAMPLE_FACTOR).flatten(0,1).to(self.window_0.feature.device)
        self.window_0.dem = downsample_average(self.window_0.dem,encoder.SAMPLE_FACTOR).flatten(0,1).to(self.window_0.feature.device)
        self.window_1.feature = feature_1[0].permute(1,2,0).flatten(0,1)
        self.window_1.conf = conf_1.squeeze().flatten(0,1)
        self.window_1.local = downsample_average(self.window_1.local,encoder.SAMPLE_FACTOR).flatten(0,1).to(self.window_1.feature.device)
        self.window_1.dem = downsample_average(self.window_1.dem,encoder.SAMPLE_FACTOR).flatten(0,1).to(self.window_1.feature.device)

        # 只在主进程上保存调试图像
        if dist.get_rank() == 0:
            feat_0_vis = self.window_0.feature.cpu().numpy().reshape(h,w,-1)
            feat_1_vis = self.window_1.feature.cpu().numpy().reshape(h,w,-1)
            conf_0_vis = self.window_0.conf.cpu().numpy().reshape(h,w)
            conf_1_vis = self.window_1.conf.cpu().numpy().reshape(h,w)
            feat_vis_img = vis_feat_twin(feat_0_vis,feat_1_vis)
            conf_cont_0,conf_div_0 = vis_conf(conf_0_vis,self.window_0.img,encoder.SAMPLE_FACTOR,div=args.conf_threshold)
            conf_cont_1,conf_div_1 = vis_conf(conf_1_vis,self.window_1.img,encoder.SAMPLE_FACTOR,div=args.conf_threshold)
            cv2.imwrite(os.path.join(self.debug_output_path,'feat_vis.png'),feat_vis_img)
            cv2.imwrite(os.path.join(self.debug_output_path,'conf_cont_0.png'),conf_cont_0)
            cv2.imwrite(os.path.join(self.debug_output_path,'conf_div_0.png'),conf_div_0)
            cv2.imwrite(os.path.join(self.debug_output_path,'conf_cont_1.png'),conf_cont_1)
            cv2.imwrite(os.path.join(self.debug_output_path,'conf_div_1.png'),conf_div_1)

        self.window_0.to_gpu()
        self.window_1.to_gpu()

        
def load_imgs_bundle(args) -> List[RSImage]:
    """加载所有影像 (包含完整的图像数据)。"""
    base_path = os.path.join(args.root, 'adjust_images')
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    img_folders = [img_folders[i] for i in select_img_idxs]
    
    images = []
    print(f"[Rank {dist.get_rank()}] Found {len(img_folders)} image folders. Loading all...")
    for idx, folder in enumerate(img_folders):
        img_path = os.path.join(base_path, folder)
        try:
            images.append(RSImage(args, img_path, idx))
            print(f"[Rank {dist.get_rank()}] Loaded image {idx} from {folder}.")
        except Exception as e:
            print(f"[Rank {dist.get_rank()}] Failed to load image {idx} from {folder}: {e}")
            
    print(f"[Rank {dist.get_rank()}] Successfully loaded {len(images)} images into memory.")
    return images

def find_overlapping_pairs(images: List[RSImage]) -> List[Tuple[int, int]]:
    """通过检查地理坐标BBox，找出所有重叠的影像对。"""
    bboxes = []
    for img in images:
        min_x = img.corner_xys[:, 0].min()
        max_x = img.corner_xys[:, 0].max()
        min_y = img.corner_xys[:, 1].min()
        max_y = img.corner_xys[:, 1].max()
        bboxes.append((min_x, min_y, max_x, max_y))

    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            b1 = bboxes[i]
            b2 = bboxes[j]
            
            # 检查是否不相交
            is_disjoint = (b1[2] < b2[0] or  # b1.maxX < b2.minX
                           b1[0] > b2[2] or  # b1.minX > b2.maxX
                           b1[3] < b2[1] or  # b1.maxY < b2.minY
                           b1[1] > b2[3])   # b1.minY > b2.maxY
            
            if not is_disjoint:
                pairs.append((i, j))
                
    print(f"Found {len(pairs)} overlapping pairs.")
    return pairs


def warp_local(local:torch.Tensor,dem:torch.Tensor,rpc_src:RPCModelParameterTorch,rpc_dst:RPCModelParameterTorch,affine_matrix:torch.Tensor):
    # RPC内部计算是float64，但affine_matrix是float32，需要转换
    affine_matrix_double = affine_matrix.to(torch.double)
    ones = torch.ones(local.shape[0],1).to(device=local.device,dtype=local.dtype)
    local_homo = torch.cat([local,ones],dim=-1)
    
    # 转换为double进行RPC计算
    trans_local = local_homo.to(torch.double) @ affine_matrix_double.T

    lats,lons = rpc_src.RPC_PHOTO2OBJ(trans_local[:,1],trans_local[:,0],dem)
    samps,lines = rpc_dst.RPC_OBJ2PHOTO(lats,lons,dem)
    warped_local = torch.stack([lines,samps],dim=-1).to(torch.float32) # 输出转回float32
    return warped_local

def feature_sampling(feature:torch.Tensor, conf:torch.Tensor, local:torch.Tensor, query:torch.Tensor,k = 16):
    point_base = LazyTensor(local.contiguous().unsqueeze(0))
    query_lazy = LazyTensor(query.contiguous().unsqueeze(1))
    dist_ij:LazyTensor = ((query_lazy - point_base) ** 2).sum(-1)
    dists,idxs = dist_ij.Kmin_argKmin(K = k, dim=1)

    locals_kmin = local[idxs] # n,k,2
    dists = torch.cdist(query.unsqueeze(1),locals_kmin,p=2).squeeze(1)

    valid_mask = (dists.min(dim=1).values < 8)
    dists = dists[valid_mask]
    idxs = idxs[valid_mask]
    
    if dists.shape[0] == 0: # 如果没有有效的点
        return None, None, valid_mask

    dists_ratio = dists / torch.sum(dists,dim=1,keepdim=True) # n,k
    reverse_dists_ratio = 1. / (dists_ratio + 1e-6)
    weights = reverse_dists_ratio / torch.sum(reverse_dists_ratio,dim=1,keepdim=True)

    feature_sample_p3d = feature[idxs]
    feature_sample_pd = torch.sum(feature_sample_p3d * weights.unsqueeze(-1),dim=1).to(torch.float32)

    conf_sample_p3 = conf[idxs]
    conf_sample_p = torch.sum(conf_sample_p3 * weights,dim=1).to(torch.float32)

    return feature_sample_pd,conf_sample_p,valid_mask

def fit_affine_bundle(args, 
                      local_tasks: list, 
                      images: List[RSImage], 
                      model_ddp: DDP, 
                      optimizer_r: torch.optim.Adam, 
                      optimizer_t: torch.optim.Adam, 
                      scheduler_r, 
                      scheduler_t, 
                      local_rank:int, 
                      world_size:int):
    """
    使用DDP并行计算 *对称损失* 并优化 *所有* 影像的仿射矩阵。
    local_tasks: [(i, j, window_pair_ij), ...]
    images: [RSImage_0, RSImage_1, ...] (包含完整图像)
    model_ddp, optimizers, schedulers: 从 main 传入
    """
    
    num_images = len(images)
    if num_images < 2 and local_rank == 0:
        print("Error: Need at least 2 images for bundle adjustment.")
        return
    
    # 4. 迭代优化
    for iter in range(args.max_iter):
        optimizer_r.zero_grad()
        optimizer_t.zero_grad()
        
        local_total_loss = torch.tensor(0.0, device=local_rank)
        num_valid_pairs = 0
        
        # 5. 只在 *本地* 的任务子集上循环
        if len(local_tasks) == 0:
            pass # loss为0，梯度也为0，是安全的
        else:
            for (i, j, window_pair_ij) in local_tasks:
                
                # 从 DDP 模型中获取 *当前* 的仿射矩阵
                # 直接调用 .module.get_affine 绕过 DDP 包装器
                A_i = model_ddp.module.get_affine(i)
                A_j = model_ddp.module.get_affine(j)
                
                # 获取 Window_Pair 中的数据
                window_i = window_pair_ij.window_0 # 对应 img_i
                window_j = window_pair_ij.window_1 # 对应 img_j
                
                # 获取RPC (RPC已在to_gpu()时移动到对应卡)
                rpc_i = images[i].rpc
                rpc_j = images[j].rpc
                
                # --- 计算对称损失 ---
                
                # 1. Warp j -> i
                warp_j_to_i = warp_local(window_j.local.float(), window_j.dem, rpc_j, rpc_i, A_j)
                feat_j_in_i, conf_j_in_i, valid_j = feature_sampling(window_i.feature.float(), window_i.conf.float(), window_i.local.float(), warp_j_to_i)
                
                # 2. Warp i -> j
                warp_i_to_j = warp_local(window_i.local.float(), window_i.dem, rpc_i, rpc_j, A_i)
                feat_i_in_j, conf_i_in_j, valid_i = feature_sampling(window_j.feature.float(), window_j.conf.float(), window_j.local.float(), warp_i_to_j)
                
                # 3. 计算 loss_a (j -> i)
                loss_a = torch.tensor(0.0, device=local_rank)
                if feat_j_in_i is not None:
                    feat_j_orig = window_j.feature[valid_j].float()
                    conf_cov_a = window_j.conf[valid_j].float() * conf_j_in_i
                    weight_a = conf_cov_a / (conf_cov_a.mean() + 1e-8)
                    loss_a = (torch.norm(feat_j_orig - feat_j_in_i, dim=-1) * weight_a).mean() * 10000.

                # 4. 计算 loss_b (i -> j)
                loss_b = torch.tensor(0.0, device=local_rank)
                if feat_i_in_j is not None:
                    feat_i_orig = window_i.feature[valid_i].float()
                    conf_cov_b = window_i.conf[valid_i].float() * conf_i_in_j
                    weight_b = conf_cov_b / (conf_cov_b.mean() + 1e-8)
                    loss_b = (torch.norm(feat_i_orig - feat_i_in_j, dim=-1) * weight_b).mean() * 10000.
                
                pair_loss = loss_a + loss_b
                
                if not torch.isnan(pair_loss) and not torch.isinf(pair_loss):
                    local_total_loss = local_total_loss + pair_loss
                    num_valid_pairs += 1

            # 6. 计算本地平均 loss
            if num_valid_pairs > 0:
                local_total_loss = local_total_loss / num_valid_pairs
            
        # 7. 反向传播 (DDP 在此处自动计算并同步所有进程的梯度平均值)
        local_total_loss.backward()
        
        optimizer_r.step()
        optimizer_t.step()

        # 8. 日志记录 (只在 rank 0 上打印)
        if (iter + 1) % 10 == 0:
            global_loss_sum = local_total_loss.clone().detach()
            dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
            global_avg_loss = global_loss_sum / world_size

            if local_rank == 0:
                print(f"iter:{iter+1}/{args.max_iter} \t loss:{global_avg_loss.item():.4f} \t lr:{scheduler_t.get_lr()[0]:.2e}")
        
        scheduler_r.step()
        scheduler_t.step()

    # 优化循环结束
    if local_rank == 0:
        print("Bundle adjustment optimization finished.")

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

def check_pair_error(img_i: RSImage, img_j: RSImage) -> np.ndarray:
    """计算单对影像 (i, j) 之间的连接点误差"""
    
    if img_i.tie_points is None or img_j.tie_points is None:
        print(f"Skipping error check for pair ({img_i.id}, {img_j.id}): Missing tie points.")
        return np.array([])
        
    if len(img_i.tie_points) != len(img_j.tie_points):
        print(f"Skipping error check for pair ({img_i.id}, {img_j.id}): Mismatched tie points count.")
        return np.array([])
    
    if len(img_i.tie_points) == 0:
        return np.array([])

    # 投影 img_i 的连接点
    lines_i = img_i.tie_points[:,0]
    samps_i = img_i.tie_points[:,1]
    heights_i = img_i.dem[lines_i,samps_i]
    lats_i, lons_i = img_i.rpc.RPC_PHOTO2OBJ(samps_i, lines_i, heights_i, 'numpy')
    coords_i = np.stack([lats_i, lons_i], axis=-1)
    
    # 投影 img_j 的连接点
    lines_j = img_j.tie_points[:,0]
    samps_j = img_j.tie_points[:,1]
    heights_j = img_j.dem[lines_j,samps_j]
    lats_j, lons_j = img_j.rpc.RPC_PHOTO2OBJ(samps_j, lines_j, heights_j, 'numpy')
    coords_j = np.stack([lats_j, lons_j], axis=-1)
    
    # 计算地理距离
    distances = haversine_distance(coords_i, coords_j)
    return distances

def check_all_pairs_error(images: List[RSImage], overlapping_pairs: List[Tuple[int, int]]) -> np.ndarray:
    """在所有重叠对上计算并汇总误差"""
    all_distances = []
    print("--- Global Error Report ---")
    for (i, j) in overlapping_pairs:
        distances = check_pair_error(images[i], images[j])
        if len(distances) > 0:
            all_distances.append(distances)
            print(f"Pair ({i}, {j}) | Points: {len(distances)} | Mean Error: {distances.mean():.4f} m | Median Error: {np.median(distances):.4f} m")

    if not all_distances:
        print("No valid tie points found for any overlapping pair. Cannot generate report.")
        return np.array([0.0])
        
    all_distances = np.concatenate(all_distances)
    return all_distances


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')

    parser.add_argument('--dino_path', type=str, default='weights',
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--encoder_path', type=str, default='weights/pretrain_swt_cnn_r2_0409_large/backbone.pth',
                        help='file containing pre-trained encoder weights')
    
    parser.add_argument('--max_lr', type=float, default=0.0001,
                        help='highest learning rate')

    parser.add_argument('--max_iter', type=int, default=1000)

    parser.add_argument('--conf_threshold',type=float,default=.5)

    parser.add_argument('--kmin_k',type=int,default=16)

    parser.add_argument('--window_size', type=int, default=2000,help='window size in meter(m)')

    parser.add_argument('--select_imgs',type=str,default='0,1') 

    parser.add_argument('--init_offset_line',type=float,default=0.)

    parser.add_argument('--init_offset_samp',type=float,default=0.)

    parser.add_argument('--grid_offset_x',type=float,default=0)

    parser.add_argument('--grid_offset_y',type=float,default=0)

    parser.add_argument('--grid_num',type=int,default=1)

    args = parser.parse_args()

    # DDP 初始化
    local_rank = setup_ddp()
    world_size = dist.get_world_size() # 总进程数

    args.debug_output_path = os.path.join(args.root,'debug_output')
    if local_rank == 0:
        os.makedirs(args.debug_output_path,exist_ok=True)

    images = load_imgs_bundle(args)
    if len(images) < 2:
        if local_rank == 0:
            print("Error: Found less than 2 images. Bundle adjustment requires at least 2.")
        dist.destroy_process_group()
        exit()

    # DDP Step 3: 任务生成与分片
    all_tasks = [] # [(i, j, diag), ...]
    overlapping_pairs = [] # [(i, j), ...]
    
    # 只在主进程 (rank 0) 上生成任务列表
    if local_rank == 0:
        print("Rank 0: Finding overlapping pairs...")
        overlapping_pairs = find_overlapping_pairs(images)
        
        print("Rank 0: Generating all_tasks list from overlapping pairs...")
        task_id = 0
        for (i, j) in overlapping_pairs:
            # 使用两张影像的 corners 计算重叠区的 grids
            corners = np.stack([images[i].corner_xys, images[j].corner_xys], axis=0)
            diags = find_grids(corners, args.window_size, offset_x=args.grid_offset_x, offset_y=args.grid_offset_y)
            print(f"Select {args.grid_num} grids from total {len(diags)} grids")
            if args.grid_num > 0:
                indices = [int((i + 1) * len(diags) / (args.grid_num + 1.)) for i in range(args.grid_num)]
                diags = [diags[i] for i in indices]
            
            for diag in diags:
                all_tasks.append( (i, j, diag, task_id) ) # (i, j, diag, global_task_id)
                task_id += 1
        
        # 为了负载均衡，打乱任务列表
        random.shuffle(all_tasks)
        print(f"Rank 0: Found {len(all_tasks)} total tasks across {len(overlapping_pairs)} pairs.")

    # 将任务列表广播给所有进程
    tasks_to_broadcast = [all_tasks] if local_rank == 0 else [None]
    dist.broadcast_object_list(tasks_to_broadcast, src=0)
    all_tasks = tasks_to_broadcast[0]
    
    # 将重叠对列表也广播
    pairs_to_broadcast = [overlapping_pairs] if local_rank == 0 else [None]
    dist.broadcast_object_list(pairs_to_broadcast, src=0)
    overlapping_pairs = pairs_to_broadcast[0]

    # 每个进程根据自己的rank获取数据子集
    my_tasks = all_tasks[local_rank::world_size] 

    # 每个进程都加载特征提取器
    encoder = EncoderDino(os.path.join(args.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'))
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    if local_rank == 0:
        print("Encoder Loaded by all processes")

    # 每个进程在自己的任务子集上创建 Window_Pair
    local_tasks_with_data = [] # [(i, j, window_pair_ij), ...]
    print(f"[Rank {local_rank}] Total tasks: {len(all_tasks)}, assigned: {len(my_tasks)}.")
    for (i, j, diag, global_task_id) in my_tasks:
        try:
            window_pair = Window_Pair(args, diag, images[i], images[j], global_task_id)
            window_pair.extract_features(encoder)
            local_tasks_with_data.append( (i, j, window_pair) )
            print(f"[Rank {local_rank}] Task {global_task_id} (pair {i},{j}) created on cuda:{local_rank}")
        except Exception as e:
            print(f"[Rank {local_rank}] !! FAILED to create task {global_task_id} (pair {i},{j}). Error: {e}")

    model = BundleAffineModel(len(images), args.init_offset_line, args.init_offset_samp).to(local_rank)
    model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    all_R_params = [m.R for m in model_ddp.module.models]
    all_T_params = [m.T for m in model_ddp.module.models]
    optimizer_r = torch.optim.Adam(all_R_params, lr=args.max_lr * 0.000001)
    optimizer_t = torch.optim.Adam(all_T_params, lr=args.max_lr)
    
    scheduler_r = torch.optim.lr_scheduler.OneCycleLR(optimizer_r, max_lr=args.max_lr * 0.000001, total_steps=args.max_iter,pct_start=0.1)
    scheduler_t = torch.optim.lr_scheduler.OneCycleLR(optimizer_t, max_lr=args.max_lr, total_steps=args.max_iter,pct_start=0.1)

    fit_affine_bundle(args, 
                      local_tasks_with_data, 
                      images, 
                      model_ddp, 
                      optimizer_r, 
                      optimizer_t, 
                      scheduler_r, 
                      scheduler_t, 
                      local_rank, 
                      world_size)

    # 同步点，确保所有进程都完成了优化
    dist.barrier()
    
    # 只在主进程上进行最终的模型更新和精度验证
    if local_rank == 0:
        print("\n" + "="*30)
        print("All processes finished optimization.")
        print("Applying final affine matrices to RPC models (Rank 0)...")
        
        for i in range(1, len(images)):
            # 从 Rank 0 的模型中获取最终仿射矩阵
            final_A_i = model_ddp.module.get_affine(i).detach()
            print(f"Final affine matrix for image {i}: \n {final_A_i.cpu().numpy()}")
            # 更新 image[i] 的 RPC 对象
            images[i].rpc.Update_Adjust(final_A_i)
            
        print("\nStarting final error check on Rank 0...")
        all_errors = check_all_pairs_error(images, overlapping_pairs)
        
        if len(all_errors) > 0 and all_errors.mean() != 0.0:
            
            print("\n--- Global Error Report (Summary) ---")
            print(f"Total tie points checked: {len(all_errors)}")
            print(f"Mean Error:   {all_errors.mean():.4f} m")
            print(f"Median Error: {np.median(all_errors):.4f} m")
            print(f"Max Error:    {all_errors.max():.4f} m")
            print(f"RMSE:         {np.sqrt(np.mean(all_errors**2)):.4f} m")
            print(f"< 1.0 m: {((all_errors < 1.0).sum() * 1. / len(all_errors)) * 100:.2f} %")
            print(f"< 3.0 m: {((all_errors < 3.0).sum() * 1. / len(all_errors)) * 100:.2f} %")
            print(f"< 5.0 m: {((all_errors < 5.0).sum() * 1. / len(all_errors)) * 100:.2f} %")
        else:
            print("No valid tie points found. Final error check skipped.")
    
    # 清理DDP进程组
    dist.destroy_process_group()

