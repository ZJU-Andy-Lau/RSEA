import os
import argparse
import random
import itertools # 导入 itertools 用于生成像对
from matplotlib.rcsetup import validate_markevery
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from pykeops.torch import LazyTensor
import numpy as np
import cv2
from rs_image_1022 import RSImage
from rpc import RPCModelParameterTorch
from model.encoder_dino_0927 import EncoderDino
import scheduler
from utils import find_grids,vis_feat_twin,vis_conf,downsample_average

# DDP相关的库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple, Dict # 导入 Dict

import warnings
import time # <-- [新] 添加
warnings.filterwarnings("ignore")

def format_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# DDP Step 1: DDP环境初始化函数
def setup_ddp():
    """初始化DDP环境"""
    dist.init_process_group(backend='nccl')
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    print(f"DDP setup on rank {local_rank} with device cuda:{local_rank}")
    return local_rank

# DDP Step 2: 将需要优化的参数封装为 nn.Module
class AffineModel(nn.Module):
    """
    将仿射变换参数R和T封装成一个PyTorch模块，以便DDP管理。
    """
    def __init__(self):
        super().__init__()
        self.R = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        self.T = nn.Parameter(torch.tensor([0., 0.], dtype=torch.float32))

    def forward(self):
        return torch.concatenate([self.R, self.T.unsqueeze(-1)], dim=-1)

class BundleAffineModel(nn.Module):
    """
    管理所有N-1个可学习仿射变换的模型。
    img_0 被假定为锚点(anchor)，其变换固定为单位矩阵。
    """
    def __init__(self, num_images):
        super().__init__()
        self.num_images = num_images
        
        # 我们有 N 张影像, 但只为 img_1 ... img_N-1 创建可学习模型
        self.models = nn.ModuleList()
        for _ in range(num_images - 1):
            self.models.append(AffineModel())
            
    def get_affine(self, index: int) -> torch.Tensor:
        """
        获取第 index 张影像的仿射变换矩阵.
        index 0 (锚点) 返回固定的单位矩阵.
        index > 0   返回其对应的可学习矩阵.
        """
        if index == 0:
            # 返回一个固定的、float32的单位仿射矩阵
            # 它需要和 model 在同一个 device 上 (通过第一个模型获取)
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
        self.img = img # 原始图像块 (将在特征提取后被删除以节省显存)
        self.local = torch.from_numpy(local)
        self.dem = torch.from_numpy(dem)
        self.rpc = rpc
        self.feature = None
        self.conf = None
        
    
    def to_gpu(self):
        # 这里的 .cuda() 会自动使用 torch.cuda.set_device 设置的当前卡
        self.local = self.local.cuda()
        self.dem = self.dem.cuda()
        self.rpc.to_gpu() # RPC 参数也需要到GPU
        # 特征和置信度在提取时已在GPU上
        if self.feature is not None:
            self.feature = self.feature.cuda()
        if self.conf is not None:
            self.conf = self.conf.cuda()


class SharedGrid():
    def __init__(self, args, diag: np.ndarray, all_rs_images: List[RSImage], grid_id: int):
        """
        (新) 代表一个公共地理格网，管理所有在此重叠的影像数据。

        args: 命令行参数
        diag: 格网的地理坐标对角线 (2, 2)
        all_rs_images: 工程中 *所有* RSImage 对象的列表
        grid_id: 该格网的全局唯一ID (用于调试)
        """
        self.args = args
        self.diag = diag
        self.id = grid_id
        self.resample_size = 1024
        
        # self.windows 是核心：它是一个字典，映射 {image_id -> Window}
        # image_id 是 RSImage.id (即影像在列表中的索引 0, 1, 2...)
        self.windows: Dict[int, Window] = {}
        
        # 存储实际重叠在该格网上的影像ID列表
        self.overlapping_image_ids: List[int] = []

        # --- 1. 创建所有影像的 Window ---
        corners_geo = np.array([
            diag[0],
            [diag[1,0], diag[0,1]],
            diag[1],
            [diag[0,0], diag[1,1]]
        ])
        
        for img in all_rs_images:
            try:
                # 步骤 1: 将地理格网(diag)反算回每张影像的像方坐标
                corners_sampline_i = img.xy_to_sampline(corners_geo) # [cite: rs_image_1022.py, line 85]
                
                # (可选) 检查反算的像方坐标是否在影像范围内
                if (corners_sampline_i.min() < 0 or 
                    corners_sampline_i[:, 0].max() > img.W or 
                    corners_sampline_i[:, 1].max() > img.H):
                    # print(f"[Grid {self.id}] Info: Image {img.id} does not cover this grid.")
                    continue # 跳过这张影像

                # 步骤 2: 从影像中重采样出 1024x1024 的图像块
                img_raw, local_i = img.resample_image_by_sampline(corners_sampline_i, (self.resample_size, self.resample_size), need_local=True) # [cite: rs_image_1022.py, line 150]
                dem_i = img.resample_dem_by_sampline(corners_sampline_i, (self.resample_size, self.resample_size)) # [cite: rs_image_1022.py, line 158]

                # 步骤 3: 创建并存储 Window 对象
                self.windows[img.id] = Window(img_raw, local_i, dem_i, img.rpc)
                self.overlapping_image_ids.append(img.id)
                
            except Exception as e:
                # 如果某个影像在该格网反算失败或无法重采样，跳过它
                print(f"[Grid {self.id}] Warning: Failed to create window for image {img.id}. Error: {e}")

        # 确保该格网至少有2张影像重叠，否则无意义
        if len(self.overlapping_image_ids) < 2:
            raise ValueError(f"Grid {self.id} has {len(self.overlapping_image_ids)} overlapping images. Need at least 2.")
        
        if dist.get_rank() == 0:
            self.debug_output_path = os.path.join(args.debug_output_path, f'grid_{self.id}')
            os.makedirs(self.debug_output_path, exist_ok=True)
            for img_id in self.overlapping_image_ids:
                cv2.imwrite(os.path.join(self.debug_output_path, f'img_raw_{img_id}.png'), self.windows[img_id].img)


    @torch.no_grad()
    def extract_features_sequentially(self, encoder: EncoderDino, local_rank: int):
        """
        依次提取此格网中所有影像的特征。
        """
        encoder = encoder.cuda(local_rank).eval()
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                    ])
        
        if dist.get_rank() == 0:
            os.makedirs(self.debug_output_path, exist_ok=True)
        
        for img_id in self.overlapping_image_ids:
            window = self.windows[img_id]
            
            # --- 特征提取 ---
            img_tensor = transform(window.img)[None].cuda(local_rank)
            feature, conf = encoder(img_tensor)
            
            # --- 存储特征 ---
            h, w = feature.shape[-2:]
            window.feature = feature[0].permute(1,2,0).flatten(0,1)
            window.conf = conf.squeeze().flatten(0,1)
            window.local = downsample_average(window.local, encoder.SAMPLE_FACTOR).flatten(0,1)
            window.dem = downsample_average(window.dem, encoder.SAMPLE_FACTOR).flatten(0,1)

            # 立刻删除已不再需要的原始图像块，释放显存
            original_img_for_vis = window.img.copy() # 复制一份用于可视化
            del window.img
            
            # --- 将特征数据移至GPU ---
            window.to_gpu() # to_gpu 会自动使用 local_rank 对应的卡

            if dist.get_rank() == 0:
                feat_vis = window.feature.cpu().numpy().reshape(h,w,-1)
                conf_vis = window.conf.cpu().numpy().reshape(h,w)
                
                # 使用 PCA 可视化特征
                feat_pca = (feat_vis.reshape(-1, feat_vis.shape[-1]) @ (torch.pca_lowrank(window.feature, q=3)[2]).cpu().numpy())
                feat_pca = feat_pca.reshape(h, w, 3)
                feat_pca = (feat_pca - feat_pca.min(axis=(0,1))) / (feat_pca.max(axis=(0,1)) - feat_pca.min(axis=(0,1)) + 1e-6)
                cv2.imwrite(os.path.join(self.debug_output_path, f'feat_vis_{img_id}.png'), (feat_pca * 255).astype(np.uint8))
                
                # 可视化置信度
                conf_cont, conf_div = vis_conf(conf_vis, original_img_for_vis, encoder.SAMPLE_FACTOR, div=self.args.conf_threshold)
                cv2.imwrite(os.path.join(self.debug_output_path, f'conf_cont_{img_id}.png'), conf_cont)
                cv2.imwrite(os.path.join(self.debug_output_path, f'conf_div_{img_id}.png'), conf_div)


    def calculate_all_pairs_loss(self, model_ddp: DDP, images: List[RSImage], local_rank: int) -> torch.Tensor:
        """
         计算此格网内所有影像两两之间的对称损失。
        """
        grid_total_loss = torch.tensor(0.0, device=local_rank)
        num_valid_pairs_in_grid = 0
        
        # 遍历所有唯一的像对 (i, j)
        for (i, j) in itertools.combinations(self.overlapping_image_ids, 2):
            
            # --- 这部分逻辑与原 fit_affine_bundle 中的循环体完全一致 ---
            A_i = model_ddp.module.get_affine(i)
            A_j = model_ddp.module.get_affine(j)
            
            window_i = self.windows[i]
            window_j = self.windows[j]
            
            rpc_i = images[i].rpc
            rpc_j = images[j].rpc
            
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
                grid_total_loss = grid_total_loss + pair_loss
                num_valid_pairs_in_grid += 1

        if num_valid_pairs_in_grid > 0:
            return grid_total_loss / num_valid_pairs_in_grid # 返回该格网的平均损失
        else:
            return torch.tensor(0.0, device=local_rank)

        
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
    """(保留) 通过检查地理坐标BBox，找出所有重叠的影像对。
    (注意) 此函数现在 *只* 用于生成最终的 *验证* 列表。"""
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
                
    print(f"Found {len(pairs)} overlapping pairs for validation.")
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

    valid_mask = (dists.min(dim=1).values < 64)
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
                      local_shared_grids: List[SharedGrid], # 接收 SharedGrid 列表
                      images: List[RSImage], 
                      model_ddp: DDP, 
                      optimizer_r: torch.optim.Adam, 
                      optimizer_t: torch.optim.Adam, 
                      scheduler_r, 
                      scheduler_t, 
                      local_rank:int, 
                      world_size:int,
                      patience: int,           
                      min_loss_threshold: float,
                      overlapping_pairs: List[Tuple[int, int]] 
                      ) -> List[Dict[str, torch.Tensor]]: 
    """
    使用DDP并行计算 *对称损失* 并优化 *所有* 影像的仿射矩阵。
    local_shared_grids: [(SharedGrid_0), (SharedGrid_1), ...]
    images: [RSImage_0, RSImage_1, ...] (包含完整图像)
    model_ddp, optimizers, schedulers: 从 main 传入
    """
    
    num_images = len(images)
    if num_images < 2 and local_rank == 0:
        print("Error: Need at least 2 images for bundle adjustment.")
        return [] 
    
    # --- [新] 初始化早停和最佳模型变量 ---
    best_model_state = [] # 只有 Rank 0 会填充它
    if local_rank == 0:
        min_loss = float('inf')
        patience_counter = 0
        print(f"Starting optimization with patience={patience} and min_loss_threshold={min_loss_threshold}")
        start_time = time.time()
        
    stop_signal = torch.tensor(0.0, device=local_rank)

    # 4. 迭代优化
    for iter in range(args.max_iter):
        optimizer_r.zero_grad()
        optimizer_t.zero_grad()
        
        local_total_loss = torch.tensor(0.0, device=local_rank)
        num_valid_grids = 0 
        
        # 5. 只在 *本地* 的格网子集上循环
        if len(local_shared_grids) == 0:
            pass 
        else:
            # (修改) 循环格网，而不是像对任务
            for grid in local_shared_grids:
                
                # (修改) 在格网内部计算所有像对的损失
                # 这个函数在 SharedGrid 类中定义
                grid_avg_loss = grid.calculate_all_pairs_loss(model_ddp, images, local_rank)
                
                if not torch.isnan(grid_avg_loss) and not torch.isinf(grid_avg_loss) and grid_avg_loss > 0:
                    local_total_loss = local_total_loss + grid_avg_loss
                    num_valid_grids += 1
                else:
                    if grid_avg_loss > 0: # 仅在非零时打印警告
                        print(f"[Rank{local_rank}]: Detect invalid loss:{grid_avg_loss.item()} in Grid {grid.id}")

            # 6. 计算本地平均 loss (按格网平均)
            if num_valid_grids > 0:
                local_total_loss = local_total_loss / num_valid_grids
            
        # 7. 反向传播 (DDP 在此处自动计算并同步所有进程的梯度平均值)
        # 即使 local_total_loss 为 0，backward() 也是安全的
        local_total_loss.backward()
        
        optimizer_r.step()
        optimizer_t.step()
            
        # 1.  在所有进程上获取全局平均损失
        global_loss_sum = local_total_loss.clone().detach()
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
        global_avg_loss = (global_loss_sum / world_size).item() # .item() 转换
        
        # 2. Rank 0 进行决策
        if local_rank == 0:
            # 检查损失是否有显著改善
            if (min_loss - global_avg_loss) > min_loss_threshold:
                # 显著改善
                min_loss = global_avg_loss
                patience_counter = 0
                
                # [新] 保存最佳模型状态
                best_model_state = []
                # 遍历 nn.ModuleList
                for sub_model in model_ddp.module.models: 
                    # .data.clone() 确保复制的是值，而不是引用
                    best_model_state.append({
                        'R': sub_model.R.data.clone(), 
                        'T': sub_model.T.data.clone()
                    })
            else:
                # 没有显著改善
                patience_counter += 1

            # 检查是否需要早停
            if patience_counter >= patience:
                print(f"--- Early stopping triggered at iter {iter+1} ---")
                print(f"Loss ({global_avg_loss:.4f}) did not improve by {min_loss_threshold} for {patience} iterations. Min loss: {min_loss:.4f}")
                stop_signal.fill_(1.0) # 设置停止信号

            # 日志记录
            if (iter + 1) % 10 == 0:
                lr_r = scheduler_r.get_last_lr()[0] if scheduler_r else args.max_lr * 1e-5
                lr_t = scheduler_t.get_last_lr()[0] if scheduler_t else args.max_lr
                
                # ---时间计算 ---
                elapsed_time_sec = time.time() - start_time
                elapsed_time_str = format_time(elapsed_time_sec)
                
                avg_iter_time = elapsed_time_sec / (iter + 1)
                remaining_iter = args.max_iter - (iter + 1)
                remaining_time_sec = avg_iter_time * remaining_iter
                remaining_time_str = format_time(remaining_time_sec)
                # ---时间计算结束 ---

                # ---可选的精度检查 ---
                mean_err, median_err = 0.0, 0.0
                err_log_str = "" # 用于日志的空字符串

                if args.check_error_during_train: # <-- 检查功能开关
                    
                    # 1. 存储所有 RPC 对象的原始(上一轮)仿射参数
                    original_params_list = [img.rpc.adjust_params.clone() for img in images]
                    original_params_inv_list = [img.rpc.adjust_params_inv.clone() for img in images]
                    
                    try:
                        # 2. 临时将 DDP 模型中的 *当前* 仿射参数应用到 RPC 对象
                        with torch.no_grad():
                            for i in range(1, num_images): # img 0 是锚点，不更新
                                current_A_i = model_ddp.module.get_affine(i).detach()
                                images[i].rpc.Update_Adjust(current_A_i) # [cite: rpc.py, line 290]
                        
                        # 3. 使用 *更新后* 的 rpc 对象计算误差
                        mean_err, median_err = get_current_error_stats(images, overlapping_pairs)

                    finally:
                        # 4. (关键) 无论检查是否成功，都 *必须* 恢复 RPC 对象的原始状态
                        with torch.no_grad():
                            for i in range(num_images):
                                images[i].rpc.adjust_params = original_params_list[i]
                                images[i].rpc.adjust_params_inv = original_params_inv_list[i]
                    
                    # 准备日志字符串
                    err_log_str = f"\t mean:{mean_err:.4f}m \t median:{median_err:.4f}m"
                
                # ---精度检查结束 ---

                #更新 print 语句以包含新信息
                print(f"iter:{iter+1}/{args.max_iter} \t loss:{global_avg_loss:.4f} \t min_l:{min_loss:.4f} \t {err_log_str} \t pat:{patience_counter}/{patience} \t lr_t:{lr_t:.2e} \t lr_r:{lr_r:.2e} \t elapsed:{elapsed_time_str} \t eta:{remaining_time_str}")
        
        # 3.Rank 0 将停止信号广播给所有其他进程
        dist.broadcast(stop_signal, src=0)

        # 4.所有进程检查停止信号
        if stop_signal.item() == 1.0:
            print(f"Rank {local_rank}: Received stop signal. Breaking optimization loop.")
            break # 退出循环
        
        # --- 逻辑结束 ---

        if scheduler_r:
            scheduler_r.step()
        if scheduler_t:
            scheduler_t.step()

    # 优化循环结束
    if local_rank == 0:
        print("Bundle adjustment optimization finished.")

    # 返回 Rank 0 上的最佳模型状态
    return best_model_state

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

def get_current_error_stats(images: List[RSImage], overlapping_pairs: List[Tuple[int, int]]) -> Tuple[float, float]:
    """
    (新) 专门用于在优化循环中调用的函数，仅计算并返回误差的均值和中位数。
    """
    all_distances = []
    
    # (此逻辑与 check_all_pairs_error 相同)
    for (i, j) in overlapping_pairs:
        distances = check_pair_error(images[i], images[j])
        if len(distances) > 0:
            all_distances.append(distances)

    if not all_distances:
        return 0.0, 0.0
        
    all_distances = np.concatenate(all_distances)
    
    if len(all_distances) == 0:
        return 0.0, 0.0

    mean_error = np.mean(all_distances)
    median_error = np.median(all_distances)
    
    return mean_error, median_error

def check_pair_error(img_i: RSImage, img_j: RSImage) -> np.ndarray:
    """(保留) 计算单对影像 (i, j) 之间的连接点误差"""
    
    if img_i.tie_points is None or img_j.tie_points is None:
        # print(f"Skipping error check for pair ({img_i.id}, {img_j.id}): Missing tie points.")
        return np.array([])
        
    if len(img_i.tie_points) != len(img_j.tie_points):
        # print(f"Skipping error check for pair ({img_i.id}, {img_j.id}): Mismatched tie points count.")
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
    """(保留) 在所有重叠对上计算并汇总误差"""
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

    parser.add_argument('--grid_offset_x',type=float,default=0)

    parser.add_argument('--grid_offset_y',type=float,default=0)

    parser.add_argument('--grid_num',type=int,default=1)

    parser.add_argument('--patience', type=int, default=100, 
                        help='Patience for early stopping (e.g., 100 iterations)')
    
    parser.add_argument('--min_loss_threshold', type=float, default=1e-4, 
                        help='Minimum improvement threshold for min_loss to reset patience (e.g., 1e-4)')

    parser.add_argument('--check_error_during_train', action='store_true',
                        help='If set, check tie point error every 10 iterations during training (Rank 0 only).')


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

    all_tasks = []
    overlapping_pairs = []
    
    # 只在主进程 (rank 0) 上生成任务列表
    if local_rank == 0:
        print("Rank 0: Finding overlapping pairs (for final validation)...")
        overlapping_pairs = find_overlapping_pairs(images)

        print("\nStarting initial error check on Rank 0...")
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
        
        print("Rank 0: Finding common grids from ALL images...")
        # 堆叠所有影像的 corners
        all_corners = np.stack([img.corner_xys for img in images], axis=0)
        
        #  一次性调用 find_grids 得到 M 个公共格网
        all_common_diags = find_grids(all_corners, args.window_size, offset_x=args.grid_offset_x, offset_y=args.grid_offset_y)
        
        print(f"Select {args.grid_num} grids from total {len(all_common_diags)} common grids")
        if args.grid_num > 0 and len(all_common_diags) > args.grid_num:
            # 采样逻辑
            indices = [int((i + 1) * len(all_common_diags) / (args.grid_num + 1.)) for i in range(args.grid_num)]
            all_tasks = [all_common_diags[i] for i in indices]
        else:
            all_tasks = all_common_diags # all_tasks 是 diags 列表
        
        # 为了负载均衡，打乱任务列表
        random.shuffle(all_tasks)
        print(f"Rank 0: Found {len(all_tasks)} total common grids (tasks).")

    # 将 *格网任务列表* 广播给所有进程
    tasks_to_broadcast = [all_tasks] if local_rank == 0 else [None]
    dist.broadcast_object_list(tasks_to_broadcast, src=0)
    all_tasks = tasks_to_broadcast[0] # all_tasks 是 [diag1, diag2, ...]
    
    #将 *重叠对列表* 广播 (仅用于验证)
    pairs_to_broadcast = [overlapping_pairs] if local_rank == 0 else [None]
    dist.broadcast_object_list(pairs_to_broadcast, src=0)
    overlapping_pairs = pairs_to_broadcast[0] # overlapping_pairs 是 [(i, j), ...]

    # 每个进程根据自己的rank获取 *格网* 子集
    my_tasks = all_tasks[local_rank::world_size] # my_tasks 是 [diag_k, diag_l, ...]

    # 每个进程都加载特征提取器
    encoder = EncoderDino(os.path.join(args.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'),upsample_times=0)
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    if local_rank == 0:
        print("Encoder Loaded by all processes")

    #  每个进程在自己的 *格网* 子集上创建 SharedGrid
    local_shared_grids: List[SharedGrid] = [] # 新的数据列表
    print(f"[Rank {local_rank}] Total grids: {len(all_tasks)}, assigned: {len(my_tasks)}.")
    
    #循环格网 (diags)
    for idx, diag in enumerate(my_tasks):
        global_grid_id = local_rank + idx * world_size # 构造一个全局唯一的ID
        try:
            # 1. 创建 SharedGrid，此时会重采样所有影像块
            #  传入 *所有* 影像
            grid = SharedGrid(args, diag, images, global_grid_id) 
            
            # 2.依次提取特征，并释放img内存
            grid.extract_features_sequentially(encoder, local_rank)
            
            local_shared_grids.append(grid)
            print(f"[Rank {local_rank}] Grid {global_grid_id} (with {len(grid.overlapping_image_ids)} images) created and features extracted on cuda:{local_rank}")
        except Exception as e:
            print(f"[Rank {local_rank}] !! FAILED to create grid {global_grid_id}. Error: {e}")


    # DDP模型和优化器设置
    model = BundleAffineModel(len(images)).to(local_rank)
    model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False) # 如果所有参数都用到了，设为False
    
    all_R_params = [m.R for m in model_ddp.module.models]
    all_T_params = [m.T for m in model_ddp.module.models]
    
    # 确保有参数可优化
    optimizer_r = None
    optimizer_t = None
    scheduler_r = None
    scheduler_t = None

    if all_R_params:
        optimizer_r = torch.optim.Adam(all_R_params, lr=args.max_lr * 1e-5)
        scheduler_r = torch.optim.lr_scheduler.OneCycleLR(optimizer_r, max_lr=args.max_lr * 1e-5, total_steps=args.max_iter,pct_start=50 / args.max_iter)
    
    if all_T_params:
        optimizer_t = torch.optim.Adam(all_T_params, lr=args.max_lr)
        scheduler_t = torch.optim.lr_scheduler.OneCycleLR(optimizer_t, max_lr=args.max_lr, total_steps=args.max_iter,pct_start=50 / args.max_iter)
    
    # 用于保存最佳模型状态的变量
    best_model_state = []

    if optimizer_r is None and optimizer_t is None and local_rank == 0:
        print("Warning: No parameters to optimize (only one image provided?). Skipping optimization.")
        # 如果没有可优化的参数（例如只有一张影像），则跳过fit
    else:
        #  调用已修改的 fit_affine_bundle
        #捕获返回的最佳模型状态
        # --- #调用 fit_affine_bundle ---
        best_model_state = fit_affine_bundle(args, 
                                             local_shared_grids, 
                                             images, 
                                             model_ddp, 
                                             optimizer_r, 
                                             optimizer_t, 
                                             scheduler_r, 
                                             scheduler_t, 
                                             local_rank, 
                                             world_size,
                                             patience=args.patience, 
                                             min_loss_threshold=args.min_loss_threshold,
                                             overlapping_pairs=overlapping_pairs 
                                             )



    #  同步点，确保所有进程都完成了优化
    dist.barrier()
    
    # 只在主进程上进行最终的模型更新和精度验证
    if local_rank == 0:
        print("\n" + "="*30)
        print("All processes finished optimization.")
        
        # --- 加载最佳模型状态 ---
        if best_model_state: # 检查 best_model_state 是否有效（非空）
            print(f"Loading best model state (from min_loss) back into model_ddp.module... (Total {len(best_model_state)} states)")
            # 使用 torch.no_grad() 确保在加载状态时不计算梯度
            with torch.no_grad():
                # best_model_state 存储了 N-1 个模型的状态
                for i, state in enumerate(best_model_state):
                    # model_ddp.module.models[i] 对应 images[i+1]
                    if i < len(model_ddp.module.models):
                        model_ddp.module.models[i].R.data.copy_(state['R'])
                        model_ddp.module.models[i].T.data.copy_(state['T'])
                    else:
                        print(f"Warning: Mismatch in best_model_state (len {len(best_model_state)}) and models (len {len(model_ddp.module.models)})")
                        break
        else:
            print("Warning: No best model state saved (e.g., no improvement found or only 1 image). Using final iteration state.")
        # --- 结束 ---

        print("Applying final (best) affine matrices to RPC models (Rank 0)...")
        
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
    
    dist.destroy_process_group()

