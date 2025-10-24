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
# (修改) 导入 rs_image_1023.py 以匹配您上传的文件
from rs_image_1023 import RSImage
from rpc import RPCModelParameterTorch
from model.encoder_dino_0927 import EncoderDino
import scheduler
# (修改) 导入 vis_feat_pca 替换 vis_feat_twin
from utils import find_grids, vis_feat_pca, vis_conf, downsample_average

# DDP相关的库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple, Dict

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

# (新增) 辅助函数：用于在可视化图像上绘制网格
def draw_grid(image: np.ndarray, grid_coords: np.ndarray, line_color=(0, 255, 0), thickness=1):
    """
    在图像上绘制网格线。
    :param image: (H, W, 3) 图像
    :param grid_coords: (grid_H, grid_W, 2) 坐标网格，(x, y) 或 (samp, line) 格式
    """
    vis_img = image.copy()
    if vis_img.ndim == 2:
        vis_img = cv2.cvtColor(vis_img, cv2.COLOR_GRAY2BGR)
        
    grid_H, grid_W = grid_coords.shape[:2]
    
    # 绘制水平线 (沿 W 方向)
    for i in range(grid_H):
        for j in range(grid_W - 1):
            pt1 = (int(grid_coords[i, j, 0]), int(grid_coords[i, j, 1]))
            pt2 = (int(grid_coords[i, j + 1, 0]), int(grid_coords[i, j + 1, 1]))
            cv2.line(vis_img, pt1, pt2, line_color, thickness)
            
    # 绘制垂直线 (沿 H 方向)
    for j in range(grid_W):
        for i in range(grid_H - 1):
            pt1 = (int(grid_coords[i, j, 0]), int(grid_coords[i, j, 1]))
            pt2 = (int(grid_coords[i + 1, j, 0]), int(grid_coords[i + 1, j, 1]))
            cv2.line(vis_img, pt1, pt2, line_color, thickness)
            
    return vis_img

# (修改) Window 类：替换 Window_Pair，现在代表单个影像在单个格网上的数据
class Window():
    def __init__(self, args, diag:np.ndarray, img:RSImage, grid_id:int, image_id:int):
        """
        (重构版)
        diag: 全局格网的地理坐标 (2, 2)
        img: 这块窗口所属的 RSImage 对象
        grid_id: 全局格网的索引
        image_id: 影像的索引
        """
        self.grid_id = grid_id
        self.image_id = image_id
        self.rpc = img.rpc
        resample_size = 1024
        
        # 计算 img (image_id) 在格网 (grid_id) 上的角点
        corners_sampline = img.xy_to_sampline(np.array([diag[0],
                                                        [diag[1,0],diag[0,1]],
                                                        diag[1],
                                                        [diag[0,0],diag[1,1]]]))
        
        # 重采样 img, local, dem
        img_raw, local = img.resample_image_by_sampline(corners_sampline,(resample_size,resample_size),need_local=True)
        dem = img.resample_dem_by_sampline(corners_sampline,(resample_size,resample_size))

        self.img = img_raw  # 临时存储，提取特征后释放
        self.local = torch.from_numpy(local) # (H, W, 2)
        self.local_shape = local.shape[:2] # 存储原始形状 (H, W)
        self.dem = torch.from_numpy(dem) # (H, W)
        
        self.feature = None
        self.conf = None

        # (修改) 调试路径现在包含 image_id 和 grid_id
        self.debug_output_path = os.path.join(args.debug_output_path, f'window_i{self.image_id}_k{self.grid_id}')
        
        # 只在主进程上保存调试图像，避免文件写入冲突
        if dist.get_rank() == 0:
            os.makedirs(self.debug_output_path, exist_ok=True)
            # 保存重采样的原始图像，供后续可视化使用
            cv2.imwrite(os.path.join(self.debug_output_path, f'img_raw_{self.image_id}.png'), img_raw)
            np.save(os.path.join(self.debug_output_path, f'local_{self.image_id}.npy'), local)


    @torch.no_grad()
    def __extract_one_img_feature__(self, encoder:EncoderDino, img_raw:np.ndarray):
        # (同旧版)
        encoder = encoder.cuda().eval()
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                    ])
        img_tensor = transform(img_raw)
        img_tensor = img_tensor[None].cuda()
        feature, conf = encoder(img_tensor)
        
        return feature, conf

    def extract_features(self, encoder:EncoderDino):
        # (修改) 只提取自己的特征
        feature, conf = self.__extract_one_img_feature__(encoder, self.img)
        h, w = feature.shape[-2:]
        
        self.feature = feature[0].permute(1,2,0).flatten(0,1)
        self.conf = conf.squeeze().flatten(0,1)
        self.local = downsample_average(self.local, encoder.SAMPLE_FACTOR).flatten(0,1)
        self.dem = downsample_average(self.dem, encoder.SAMPLE_FACTOR).flatten(0,1)

        # 只在主进程上保存调试图像
        if dist.get_rank() == 0:
            feat_vis = self.feature.cpu().numpy().reshape(h, w, -1)
            conf_vis = self.conf.cpu().numpy().reshape(h, w)
            conf_cont, conf_div = vis_conf(conf_vis, self.img, encoder.SAMPLE_FACTOR, div=args.conf_threshold)
            
            # (修改) 保存唯一的特征(PCA)和置信度图
            # 使用 vis_feat_pca 替换 vis_feat_twin
            feat_pca_vis_img = vis_feat_pca(feat_vis)
            cv2.imwrite(os.path.join(self.debug_output_path, f'feat_pca_{self.image_id}.png'), feat_pca_vis_img)
            cv2.imwrite(os.path.join(self.debug_output_path, f'conf_cont_{self.image_id}.png'), conf_cont)
            cv2.imwrite(os.path.join(self.debug_output_path, f'conf_div_{self.image_id}.png'), conf_div)

        # (关键) 释放原始图像占用的CPU/GPU内存
        self.img = None
    
    def to_gpu(self):
        # (修改) 将所有张量数据移动到当前进程的GPU
        # 这里的 .cuda() 会自动使用 torch.cuda.set_device 设置的当前卡
        self.local = self.local.cuda()
        self.dem = self.dem.cuda()
        self.rpc.to_gpu()
        
        # (新增) 确保特征和置信度也在当前GPU
        # 这在 all_gather_object 之后至关重要，因为数据会先被同步到CPU
        if self.feature is not None:
            self.feature = self.feature.cuda()
        if self.conf is not None:
            self.conf = self.conf.cuda()

        
# (删除) Window_Pair 类被移除

def load_imgs_bundle(args) -> List[RSImage]:
    """加载所有影像 (包含完整的图像数据)。(逻辑不变)"""
    base_path = os.path.join(args.root, 'adjust_images')
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    img_folders = [img_folders[i] for i in select_img_idxs]
    
    images = []
    # (修改) 仅 Rank 0 打印加载信息
    if dist.get_rank() == 0:
        print(f"Found {len(img_folders)} image folders. Loading all...")
        
    for idx, folder in enumerate(img_folders):
        img_path = os.path.join(base_path, folder)
        try:
            images.append(RSImage(args, img_path, idx))
            if dist.get_rank() == 0:
                print(f"Loaded image {idx} from {folder}.")
        except Exception as e:
            if dist.get_rank() == 0:
                print(f"Failed to load image {idx} from {folder}: {e}")
            
    if dist.get_rank() == 0:
        print(f"Successfully loaded {len(images)} images into memory by all processes.")
    return images

# (删除) find_overlapping_pairs 函数不再需要，
# find_grids 会处理公共重叠区，loss_calculation_tasks 会自动生成所有对。

def warp_local(local:torch.Tensor,dem:torch.Tensor,rpc_src:RPCModelParameterTorch,rpc_dst:RPCModelParameterTorch,affine_matrix:torch.Tensor):
    # (逻辑不变)
    # RPC内部计算是float64，但affine_matrix是float32，需要转换
    affine_matrix_double = affine_matrix.to(torch.double)
    
    # (修改) 确保 local 也是 float32
    local = local.float()
    ones = torch.ones(local.shape[0],1).to(device=local.device,dtype=local.dtype)
    local_homo = torch.cat([local,ones],dim=-1)
    
    # 转换为double进行RPC计算
    trans_local = local_homo.to(torch.double) @ affine_matrix_double.T

    lats,lons = rpc_src.RPC_PHOTO2OBJ(trans_local[:,1],trans_local[:,0],dem)
    samps,lines = rpc_dst.RPC_OBJ2PHOTO(lats,lons,dem)
    warped_local = torch.stack([lines,samps],dim=-1).to(torch.float32) # 输出转回float32
    return warped_local

def feature_sampling(feature:torch.Tensor, conf:torch.Tensor, local:torch.Tensor, query:torch.Tensor,k = 16):
    # (逻辑不变)
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

# (修改) fit_affine_bundle 函数：
# 现在接收 my_loss_tasks 和 global_windows_db
def fit_affine_bundle(args, 
                      my_loss_tasks: List[Tuple[int, int, int]], 
                      global_windows_db: Dict[Tuple[int, int], Window],
                      images: List[RSImage], 
                      diags: List[np.ndarray],
                      model_ddp: DDP, 
                      optimizer_r: torch.optim.Adam, 
                      optimizer_t: torch.optim.Adam, 
                      scheduler_r, 
                      scheduler_t, 
                      local_rank:int, 
                      world_size:int):
    
    num_images = len(images)
    if num_images < 2 and local_rank == 0:
        print("Error: Need at least 2 images for bundle adjustment.")
        return
    
    # (新增) 可视化输出目录
    vis_output_dir = None
    if local_rank == 0:
        vis_output_dir = os.path.join(args.debug_output_path, 'training_vis')
        os.makedirs(vis_output_dir, exist_ok=True)

    # 4. 迭代优化
    for iter in range(args.max_iter):
        optimizer_r.zero_grad()
        optimizer_t.zero_grad()
        
        local_total_loss = 0
        num_valid_pairs = 0
        
        # 5. 只在 *本地* 的损失计算任务子集上循环
        if len(my_loss_tasks) == 0:
            pass # loss为0，梯度也为0，是安全的
        else:
            # for (i, j, window_pair_ij) in local_tasks:
            # (修改) 遍历 (i, j, k) 任务
            for (i, j, k) in my_loss_tasks:
                
                # 从 DDP 模型中获取 *当前* 的仿射矩阵
                A_i = model_ddp.module.get_affine(i)
                A_j = model_ddp.module.get_affine(j)
                
                # (修改) --- 从全局数据库中检索 Window ---
                try:
                    window_i = global_windows_db[(i, k)]
                    window_j = global_windows_db[(j, k)]
                except KeyError:
                    # 理论上不应该发生
                    print(f"[Rank {local_rank}] Warning: Could not find Window for pair (i={i}, k={k}) or (j={j}, k={k}). Skipping.")
                    continue 

                # (修改) --- 关键：将在 all_gather 后位于CPU的数据移至当前GPU ---
                window_i.to_gpu()
                window_j.to_gpu()
                
                # 获取RPC (RPC已在to_gpu()时移动到对应卡)
                rpc_i = images[i].rpc
                rpc_j = images[j].rpc
                
                # --- 计算对称损失 (此部分逻辑完全不变) ---
                
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
        # 确保即使 loss 为 0 也要反向传播，以同步所有进程
        if torch.is_grad_enabled():
            local_total_loss.backward()
        
        optimizer_r.step()
        optimizer_t.step()

        # 8. 日志记录
        if (iter + 1) % 10 == 0:
            global_loss_sum = local_total_loss.clone().detach()
            dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
            global_avg_loss = global_loss_sum / world_size

            if local_rank == 0:
                print(f"iter:{iter+1}/{args.max_iter} \t loss:{global_avg_loss.item():.4f} \t lr_T:{scheduler_t.get_lr()[0]:.2e}")
        
        # (新增) 可视化逻辑
        if (iter + 1) % 100 == 0 and local_rank == 0 and len(my_loss_tasks) > 0:
            try:
                # 选取本地的第一个任务进行可视化
                vis_task = my_loss_tasks[0]
                i, j, k = vis_task
                
                # 获取数据
                window_i = global_windows_db[(i, k)]
                window_j = global_windows_db[(j, k)]
                A_i = model_ddp.module.get_affine(i).detach()
                A_j = model_ddp.module.get_affine(j).detach()
                
                # 重新加载原始图像 (必须存在)
                vis_path_i = window_i.debug_output_path
                vis_path_j = window_j.debug_output_path
                img_i_raw = cv2.imread(os.path.join(vis_path_i, f'img_raw_{i}.png'))
                img_j_raw = cv2.imread(os.path.join(vis_path_j, f'img_raw_{j}.png'))
                
                # 重新加载原始 local 坐标
                local_i_grid_np = np.load(os.path.join(vis_path_i, f'local_{i}.npy'))
                local_j_grid_np = np.load(os.path.join(vis_path_j, f'local_{j}.npy'))
                
                if img_i_raw is None or img_j_raw is None:
                    print(f"Visualization Error: Could not load raw images for i={i}, j={j}, k={k}")
                else:
                    # 计算 warped grid
                    H, W = window_i.local_shape
                    
                    # 我们需要原始的 (H, W, 2) local 坐标，但 window_i.local 已经被降采样和展平
                    # 因此我们使用 numpy 重新加载的
                    local_i_flat_vis = torch.from_numpy(local_i_grid_np).flatten(0, 1).cuda().float()
                    # dem 也需要重新降采样
                    dem_i_flat_vis = downsample_average(torch.from_numpy(window_i.dem.cpu().numpy().reshape(H, W)), encoder.SAMPLE_FACTOR).flatten(0,1).cuda()
                    
                    warp_i_to_j = warp_local(local_i_flat_vis, dem_i_flat_vis, images[i].rpc, images[j].rpc, A_i)
                    warp_i_to_j_grid = warp_i_to_j.reshape(H, W, 2).cpu().numpy()
                    
                    # 绘制
                    vis_img_j_warped = draw_grid(img_j_raw, warp_i_to_j_grid[:, :, [1, 0]]) # (samp, line) -> (x, y)
                    vis_img_j_orig = draw_grid(img_j_raw, local_j_grid_np[:, :, [1, 0]]) # (samp, line) -> (x, y)
                    comparison_img = cv2.hconcat([vis_img_j_orig, vis_img_j_warped])
                    
                    cv2.imwrite(os.path.join(vis_output_dir, f'iter_{iter+1}_warp_{i}_to_{j}_grid_{k}.png'), comparison_img)
                    print(f"Saved visualization for iter {iter+1}, pair ({i}, {j}), grid {k}.")

            except Exception as e:
                print(f"Visualization Error: {e}")

        scheduler_r.step()
        scheduler_t.step()

    # 优化循环结束
    if local_rank == 0:
        print("Bundle adjustment optimization finished.")

def haversine_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    # (逻辑不变)
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
    """计算单对影像 (i, j) 之间的连接点误差 (逻辑不变)"""
    
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
    """在所有重叠对上计算并汇总误差 (逻辑不变)"""
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
    
    # (修改) 确保所有进程都等待 Rank 0 创建好目录
    dist.barrier()

    # (修改) 所有进程都加载 RSImage 对象
    images = load_imgs_bundle(args)
    if len(images) < 2:
        if local_rank == 0:
            print("Error: Found less than 2 images. Bundle adjustment requires at least 2.")
        dist.destroy_process_group()
        exit()

    # (修改) 所有进程都加载特征提取器
    encoder = EncoderDino(os.path.join(args.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'),upsample_times=0)
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    if local_rank == 0:
        print("Encoder Loaded by all processes")

    # (修改) DDP Step 3: 任务生成(Rank 0)
    diags = []
    window_creation_tasks = [] # List[(i, k)]
    loss_calculation_tasks = [] # List[(i, j, k)]
    overlapping_pairs = [] # List[(i, j)]
    
    if local_rank == 0:
        print("Rank 0: Generating global grids...")
        # (修改) 收集所有影像的 corners
        all_corners = np.stack([img.corner_xys for img in images], axis=0) # (M, 4, 2)
        
        # (修改) find_grids 现在基于所有影像的公共重叠区
        diags = find_grids(all_corners, args.window_size, offset_x=args.grid_offset_x, offset_y=args.grid_offset_y)
        
        print(f"Rank 0: Found {len(diags)} global grids common to all {len(images)} images.")
        if args.grid_num > 0 and len(diags) > 0:
            indices = [int((i + 1) * len(diags) / (args.grid_num + 1.)) for i in range(args.grid_num)]
            diags = [diags[i] for i in indices]
            print(f"Rank 0: Selected {len(diags)} grids based on grid_num={args.grid_num}.")

        N = len(diags)
        M = len(images)
        
        # (修改) 生成 窗口创建任务 M * N
        window_creation_tasks = [(i, k) for i in range(M) for k in range(N)]
        # (修改) 增加注释，解释为何要随机打乱
        # 随机打乱任务列表以实现 DDP 负载均衡
        # 这可以防止某个GPU被分配到连续的、计算/IO密集型任务（例如都属于同一个大影像）
        # 从而避免该GPU成为“掉队者”，导致所有其他GPU在同步点（如all_gather）空等
        random.shuffle(window_creation_tasks)
        
        # (修改) 生成 损失计算任务 (M*(M-1)/2) * N
        loss_calculation_tasks = [(i, j, k) for k in range(N) for i in range(M) for j in range(i + 1, M)]
        # (修改) 增加注释，解释为何要随机打乱
        # 随机打乱损失计算任务以实现 DDP 负载均衡
        # 这确保了每个GPU在优化循环的每一步中都处理混合的格网(k)和影像对(i,j)
        # 避免了内存访问热点（例如一个GPU只访问grid_0的数据）和计算负载不均
        random.shuffle(loss_calculation_tasks)
        
        # (修改) 生成用于最终验证的重叠对列表
        overlapping_pairs = list(set([(i, j) for i, j, k in loss_calculation_tasks]))
        
        print(f"Rank 0: Generated {len(window_creation_tasks)} window creation tasks.")
        print(f"Rank 0: Generated {len(loss_calculation_tasks)} loss calculation tasks.")
        
        data_to_broadcast = [diags, window_creation_tasks, loss_calculation_tasks, overlapping_pairs]
    else:
        data_to_broadcast = [None] * 4

    # (修改) DDP Step 4: 广播任务列表
    dist.broadcast_object_list(data_to_broadcast, src=0)
    diags, window_creation_tasks, loss_calculation_tasks, overlapping_pairs = data_to_broadcast
    
    # (修改) DDP Step 5: 任务分片 (每个进程)
    my_creation_tasks = window_creation_tasks[local_rank::world_size] 
    my_loss_tasks = loss_calculation_tasks[local_rank::world_size]

    # (修改) DDP Step 6: 并行特征提取
    local_windows_storage: Dict[Tuple[int, int], Window] = {}
    print(f"[Rank {local_rank}] Assigned {len(my_creation_tasks)} window creation tasks.")
    
    for (i, k) in my_creation_tasks:
        try:
            diag_k = diags[k]
            window = Window(args, diag_k, images[i], grid_id=k, image_id=i)
            window.extract_features(encoder)
            local_windows_storage[(i, k)] = window # Key: (image_id, grid_id)
        except Exception as e:
            print(f"[Rank {local_rank}] !! FAILED to create window (i={i}, k={k}). Error: {e}")
    
    print(f"[Rank {local_rank}] Finished feature extraction. Gathering all windows...")
    
    # (修改) DDP Step 7: 全局数据同步
    gathered_list = [None] * world_size
    dist.all_gather_object(gathered_list, local_windows_storage)
    
    # (修改) 合并所有数据
    final_global_windows: Dict[Tuple[int, int], Window] = {}
    for d in gathered_list:
        final_global_windows.update(d)
    
    del gathered_list, local_windows_storage # 释放内存
    print(f"[Rank {local_rank}] All {len(final_global_windows)} windows gathered.")

    # (修改) DDP Step 8: 模型与优化器设置 (逻辑不变)
    model = BundleAffineModel(len(images), args.init_offset_line, args.init_offset_samp).to(local_rank)
    model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    
    # (修改) 确保在模型中存在参数时才创建优化器
    all_R_params = [m.R for m in model_ddp.module.models]
    all_T_params = [m.T for m in model_ddp.module.models]
    
    optimizer_r = None
    optimizer_t = None
    scheduler_r = None
    scheduler_t = None
    
    if all_T_params: # 仅当有可优化的参数时 (M > 1)
        optimizer_r = torch.optim.Adam(all_R_params, lr=args.max_lr * 0.000001)
        optimizer_t = torch.optim.Adam(all_T_params, lr=args.max_lr)
        
        scheduler_r = torch.optim.lr_scheduler.OneCycleLR(optimizer_r, max_lr=args.max_lr * 0.000001, total_steps=args.max_iter,pct_start=0.1)
        scheduler_t = torch.optim.lr_scheduler.OneCycleLR(optimizer_t, max_lr=args.max_lr, total_steps=args.max_iter,pct_start=0.1)
    else:
        if local_rank == 0:
            print("Warning: Only one image loaded, no parameters to optimize.")
            
    # (修改) DDP Step 9: 调用优化循环
    if (len(my_loss_tasks) > 0 or dist.get_rank() == 0) and len(images) > 1: # 确保 M > 1
        fit_affine_bundle(args, 
                          my_loss_tasks, 
                          final_global_windows,
                          images, 
                          diags,
                          model_ddp, 
                          optimizer_r, 
                          optimizer_t, 
                          scheduler_r, 
                          scheduler_t, 
                          local_rank, 
                          world_size)
    else:
        if local_rank == 0:
            print("No loss tasks to run. Skipping optimization.")

    # (修改) DDP Step 10: 同步与最终验证
    dist.barrier()
    
    # 只在主进程上进行最终的模型更新和精度验证
    if local_rank == 0:
        print("\n" + "="*30)
        print("All processes finished optimization.")
        
        if len(images) > 1:
            print("Applying final affine matrices to RPC models (Rank 0)...")
            
            for i in range(1, len(images)):
                # 从 Rank 0 的模型中获取最终仿射矩阵
                final_A_i = model_ddp.module.get_affine(i).detach()
                print(f"Final affine matrix for image {i}: \n {final_A_i.cpu().numpy()}")
                # 更新 image[i] 的 RPC 对象
                images[i].rpc.Update_Adjust(final_A_i)
                
            print("\nStarting final error check on Rank 0...")
            # (修改) 使用从 Rank 0 广播来的 overlapping_pairs
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
        else:
            print("Only one image. No adjustments or error checks performed.")
    
    # 清理DDP进程组
    dist.destroy_process_group()

