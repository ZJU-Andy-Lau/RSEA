import os
import argparse
import random
import itertools
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
from utils import find_grids,vis_conf,downsample_average

# DDP相关的库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple, Dict # 导入 Dict

import warnings
import time
from tqdm import tqdm

# --- [新导入] ---
import rasterio
from rasterio.transform import from_origin
from pyproj import CRS
from scipy.interpolate import RegularGridInterpolator
# --- [新导入结束] ---


warnings.filterwarnings("ignore")

def format_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

# --- [新函数 1: 正射校正] ---
def orthorectify_patch_mercator(rs_image: RSImage, 
                              grid_diag: np.ndarray, 
                              resolution: float, 
                              output_path: str) -> Tuple[np.ndarray, rasterio.Affine]:
    """
    (新) 使用调整后的RPC和Mercator网格，对单个RSImage进行正射校正。
    
    Args:
        rs_image: 包含 *完整* 影像、DEM和 *已调整* RPC 的 RSImage 对象。
        grid_diag: np.array([[x1, y1], [x2, y2]])，Mercator坐标，顺序不固定。
        resolution: 输出分辨率 (米)。
        output_path: 输出 GeoTIFF 路径。
        
    Returns:
        (ortho_image_array, transform): 返回生成的影像数组和其地理变换。
    """
    
    # 1. 定义输出网格 (Mercator, EPSG:3857)
    # --- [修改开始] ---
    # 显式查找 min/max 坐标，不依赖角点顺序
    all_x = grid_diag[:, 0]
    all_y = grid_diag[:, 1]
    min_x = np.min(all_x)
    max_x = np.max(all_x)
    min_y = np.min(all_y)
    max_y = np.max(all_y)
    # --- [修改结束] ---
    
    out_W = int(np.ceil((max_x - min_x) / resolution))
    out_H = int(np.ceil((max_y - min_y) / resolution))
    
    if out_W <= 0 or out_H <= 0:
        raise ValueError(f"输出尺寸为零或负数 (W:{out_W}, H:{out_H})。请检查 grid_diag 和 resolution。Grid Diag: {grid_diag}")

    # 注意：Y轴在地理坐标中向上，但在影像中向下
    # from_origin 需要左上角 (ul_x, ul_y)，所以 x 是 min_x, y 是 max_y
    transform = from_origin(min_x, max_y, resolution, resolution)
    
    # 计算网格中心点坐标
    out_x_coords = np.linspace(min_x + resolution / 2, max_x - resolution / 2, out_W)
    out_y_coords = np.linspace(max_y - resolution / 2, min_y + resolution / 2, out_H) # Y轴反向
    
    out_xx, out_yy = np.meshgrid(out_x_coords, out_y_coords)
    
    # 2. 创建源影像和DEM的插值器 (基于 'line' 和 'samp')
    H_src, W_src = rs_image.image.shape[:2]
    lines_src = np.arange(H_src)
    samps_src = np.arange(W_src)
    
    # 影像在 RSImage 中被统一处理为 3 通道 [cite: rs_image_1022.py, line 33]
    is_color = True
    image_interpolator_r = RegularGridInterpolator((lines_src, samps_src), rs_image.image[..., 0], method='linear', bounds_error=False, fill_value=0)
    image_interpolator_g = RegularGridInterpolator((lines_src, samps_src), rs_image.image[..., 1], method='linear', bounds_error=False, fill_value=0)
    image_interpolator_b = RegularGridInterpolator((lines_src, samps_src), rs_image.image[..., 2], method='linear', bounds_error=False, fill_value=0)


    # 3. 准备输出数组
    ortho_image = np.zeros((out_H, out_W, 3), dtype=rs_image.image.dtype)
    
    # 4. 分块处理 (Ground-to-Image)
    block_size = 1024 # 可调
    for i in range(0, out_H, block_size):
        i_end = min(i + block_size, out_H)
        for j in range(0, out_W, block_size):
            j_end = min(j + block_size, out_W)
            
            # 提取块内的 Mercator 坐标
            block_xx = out_xx[i:i_end, j:j_end]
            block_yy = out_yy[i:i_end, j:j_end]
            
            xy_points = np.stack([block_xx.ravel(), block_yy.ravel()], axis=-1)
            
            # 5. (关键) 使用 rs_image.xy_to_sampline 进行投影
            # 此函数使用 *已调整* 的 self.rpc，并自动迭代DEM [cite: rs_image_1022.py, line 85]
            # 它返回 (samp, line)
            try:
                sampline_pred = rs_image.xy_to_sampline(xy_points) 
            except Exception as e:
                print(f"警告: xy_to_sampline 在投影时失败 (Grid: {output_path}): {e}")
                continue # 跳过这个块
                
            # 准备插值坐标 (line, samp)
            points_to_sample = np.stack([sampline_pred[:, 1], sampline_pred[:, 0]], axis=-1) # (line, samp)
            
            # 6. 采样像素值
            pixel_vals_r = image_interpolator_r(points_to_sample).reshape(block_xx.shape)
            pixel_vals_g = image_interpolator_g(points_to_sample).reshape(block_xx.shape)
            pixel_vals_b = image_interpolator_b(points_to_sample).reshape(block_xx.shape)
            ortho_image[i:i_end, j:j_end] = np.stack([pixel_vals_r, pixel_vals_g, pixel_vals_b], axis=-1).astype(rs_image.image.dtype)

    # 7. 写入 GeoTIFF
    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=out_H,
        width=out_W,
        count=3, # 始终为 3 通道
        dtype=ortho_image.dtype,
        crs=CRS.from_epsg(3857), # Web Mercator
        transform=transform
    ) as dst:
        dst.write(ortho_image[..., 0], 1)
        dst.write(ortho_image[..., 1], 2)
        dst.write(ortho_image[..., 2], 3)
            
    return ortho_image, transform

# --- [新函数 2: 棋盘格] ---
def create_checkerboard(ortho1: np.ndarray, 
                        ortho2: np.ndarray, 
                        transform: rasterio.Affine,
                        output_path: str, 
                        block_size: int = 50):
    """
    (新) 将两个已对齐的正射影像合并为棋盘格。
    
    Args:
        ortho1: 第一个正射影像 (H, W, 3)
        ortho2: 第二个正射影像 (H, W, 3) (必须同形状)
        transform: 用于保存 GeoTIFF 的地理变换。
        output_path: 输出路径。
        block_size: 棋盘格的大小 (像素)。
    """
    if ortho1.shape != ortho2.shape:
        print(f"警告: 棋盘格影像形状不匹配: {ortho1.shape} vs {ortho2.shape}。跳过 {output_path}")
        return

    H, W = ortho1.shape[:2]
    checkerboard_img = np.zeros_like(ortho1)

    for i in range(0, H, block_size):
        for j in range(0, W, block_size):
            # 确定块索引
            i_block = (i // block_size) % 2
            j_block = (j // block_size) % 2
            
            # (i_block % 2) == (j_block % 2) -> (0,0) or (1,1) -> 使用影像1
            if i_block == j_block:
                checkerboard_img[i:min(i+block_size, H), j:min(j+block_size, W)] = \
                    ortho1[i:min(i+block_size, H), j:min(j+block_size, W)]
            else:
                checkerboard_img[i:min(i+block_size, H), j:min(j+block_size, W)] = \
                    ortho2[i:min(i+block_size, H), j:min(j+block_size, W)]
    
    # 写入 GeoTIFF
    try:
        # 假设 checkerboard_img 是 RGB 顺序，转换为 BGR 以供 cv2 使用
        if checkerboard_img.ndim == 3 and checkerboard_img.shape[2] == 3:
            checkerboard_img_bgr = cv2.cvtColor(checkerboard_img, cv2.COLOR_RGB2BGR)
        else:
            # 如果是灰度图或其他情况，直接使用
            checkerboard_img_bgr = checkerboard_img

        success = cv2.imwrite(output_path, checkerboard_img_bgr)
        if not success:
            print(f"警告: cv2.imwrite 未能成功保存 PNG 文件到 {output_path}")
    except Exception as e:
        print(f"警告: 保存 PNG 文件到 {output_path} 时出错: {e}")


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
    def __init__(self, args, diag: np.ndarray, all_rs_images: List[RSImage], grid_id: str):
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
                # [修改] 增加 grid_id 打印
                print(f"[Grid {self.id}] Warning: Failed to create window for image {img.id}. Error: {e}")

        # 确保该格网至少有2张影像重叠，否则无意义
        if len(self.overlapping_image_ids) < 2:
            raise ValueError(f"Grid {self.id} has {len(self.overlapping_image_ids)} overlapping images. Need at least 2.")
        
        if dist.get_rank() == 0:
            # [修改] 路径中加入 level 信息
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
                cv2.imwrite(os.path.join(self.debug_output_path, f'conf_cont_{img_id}.png'), cv2.cvtColor(conf_cont,cv2.COLOR_RGB2BGR))
                cv2.imwrite(os.path.join(self.debug_output_path, f'conf_div_{img_id}.png'), cv2.cvtColor(conf_div,cv2.COLOR_RGB2BGR))


    def calculate_all_pairs_loss(self, model_ddp: DDP, images: List[RSImage], local_rank: int) -> torch.Tensor:
        """
         (已修改) 计算此格网内所有影像两两之间的 *非对称* 损失 (j -> i, j > i)。
        """
        grid_total_loss = torch.tensor(0.0, device=local_rank)
        num_valid_pairs_in_grid = 0
        
        # 遍历所有唯一的像对 (i, j)
        for (i, j) in itertools.combinations(self.overlapping_image_ids, 2):
            
            # --- 这部分逻辑与原 fit_affine_bundle 中的循环体完全一致 ---
            # i < j, A_i 是目标 (可能固定也可能移动), A_j 是源 (总是移动)
            A_i = model_ddp.module.get_affine(i)
            A_j = model_ddp.module.get_affine(j)
            
            window_i = self.windows[i]
            window_j = self.windows[j]
            
            rpc_i = images[i].rpc
            rpc_j = images[j].rpc
            
            # 1. Warp j -> i  (始终将索引号大的 j 投影到索引号小的 i)
            warp_j_to_i = warp_local(window_j.local.float(), window_j.dem, rpc_j, rpc_i, A_j)
            feat_j_in_i, conf_j_in_i, valid_j = feature_sampling(window_i.feature.float(), window_i.conf.float(), window_i.local.float(), warp_j_to_i, k = self.args.kmin_k)

            # 3. 计算 loss_a (j -> i)
            loss_a = torch.tensor(0.0, device=local_rank)
            if feat_j_in_i is not None:
                feat_j_orig = window_j.feature[valid_j].float()
                conf_cov_a = window_j.conf[valid_j].float() * conf_j_in_i
                weight_a = conf_cov_a / (conf_cov_a.mean() + 1e-8)
                loss_a = (torch.norm(feat_j_orig - feat_j_in_i, dim=-1) * weight_a).mean() * 10000.

            pair_loss = loss_a
            
            if not torch.isnan(pair_loss) and not torch.isinf(pair_loss) and pair_loss > 0:
                grid_total_loss = grid_total_loss + pair_loss
                num_valid_pairs_in_grid += 1
            else:
                if pair_loss > 0: # 仅在非零时打印警告 (虽然isnan和isinf已经覆盖了)
                    print(f"[Rank{local_rank}]: Detect invalid loss:{pair_loss.item()} in Grid {self.id} for pair ({i}, {j})")


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

def feature_sampling(feature:torch.Tensor, conf:torch.Tensor, local:torch.Tensor, query:torch.Tensor,k = 4):
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

# --- [修改开始]: 更新 fit_affine_bundle 函数签名和内部逻辑 ---
def fit_affine_bundle(args, 
                      local_shared_grids: List[SharedGrid], 
                      images: List[RSImage], 
                      model_ddp: DDP, 
                      optimizer_r: torch.optim.Adam, 
                      optimizer_t: torch.optim.Adam, 
                      scheduler_r, 
                      scheduler_t, 
                      local_rank:int, 
                      world_size:int,
                      patience: int,           
                      # min_loss_threshold: float, # 改为从 args 获取
                      overlapping_pairs: List[Tuple[int, int]],
                      current_level: int 
                      ) -> List[Dict[str, torch.Tensor]]: 
    """
    (已修改) 使用DDP并行计算损失并优化仿射矩阵，支持基于loss或error的早停。
    """
    
    num_images = len(images)
    if num_images < 2 and local_rank == 0:
        print("Error: Need at least 2 images for bundle adjustment.")
        return [] 
    
    # ---初始化早停和最佳模型变量 ---
    best_model_state = [] 
    if local_rank == 0:
        # (修改) 使用通用变量名
        min_metric_val = float('inf') 
        patience_counter = 0
        criterion = args.stop_criterion
        loss_threshold = args.min_loss_threshold
        error_threshold = args.min_error_threshold
        print(f"Starting optimization with criterion='{criterion}', patience={patience}.")
        if criterion == 'loss':
            print(f"Using min_loss_threshold={loss_threshold}")
        else: # criterion == 'error'
            print(f"Using min_error_threshold={error_threshold}m")
            if not args.check_error_during_train:
                 print("Warning: stop_criterion='error' requires tie point error checking. " 
                       "Error will only be checked every 10 iterations.")
                       
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
            for grid in local_shared_grids:
                grid_avg_loss = grid.calculate_all_pairs_loss(model_ddp, images, local_rank)
                
                if not torch.isnan(grid_avg_loss) and not torch.isinf(grid_avg_loss) and grid_avg_loss > 0:
                    local_total_loss = local_total_loss + grid_avg_loss
                    num_valid_grids += 1
                else:
                    if grid_avg_loss > 0: 
                        print(f"[Rank{local_rank}]: Detect invalid loss:{grid_avg_loss.item()} in Grid {grid.id}")

            if num_valid_grids > 0:
                local_total_loss = local_total_loss / num_valid_grids
            
        # 7. 反向传播
        local_total_loss.backward()
        
        optimizer_r.step()
        optimizer_t.step()
            
        # 1. 获取全局平均损失 (所有进程都需要)
        global_loss_sum = local_total_loss.clone().detach()
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
        global_avg_loss = (global_loss_sum / world_size).item() 
        
        # 2. Rank 0 进行决策
        if local_rank == 0:
            
            # --- (修改) 早停和最优模型判断逻辑 ---
            mean_err = 0.0 # 初始化
            median_err = 0.0
            
            # 确定是否需要在本轮计算 error
            should_calculate_error = (args.check_error_during_train or args.stop_criterion == 'error') and (iter + 1) % 10 == 0
            
            # 计算 error (如果需要)
            if should_calculate_error:
                # 精度检查逻辑 (与之前相同)
                original_params_list = [img.rpc.adjust_params.clone() for img in images]
                original_params_inv_list = [img.rpc.adjust_params_inv.clone() for img in images]
                try:
                    with torch.no_grad():
                        for i in range(1, num_images): 
                            current_A_i = model_ddp.module.get_affine(i).detach()
                            images[i].rpc.Update_Adjust(current_A_i) 
                    mean_err, median_err = get_current_error_stats(images, overlapping_pairs)
                finally:
                    with torch.no_grad():
                        for i in range(num_images):
                            images[i].rpc.adjust_params = original_params_list[i]
                            images[i].rpc.adjust_params_inv = original_params_inv_list[i]

            # 确定本轮用于判断的指标和阈值
            current_metric_val = 0.0
            current_threshold = 0.0
            perform_check_this_iter = False 
            
            if args.stop_criterion == 'loss':
                current_metric_val = global_avg_loss
                current_threshold = args.min_loss_threshold
                perform_check_this_iter = True # loss 每轮都检查
            elif args.stop_criterion == 'error' and should_calculate_error: # 只有计算了 error 的轮次才检查
                current_metric_val = mean_err 
                current_threshold = args.min_error_threshold
                perform_check_this_iter = True
            
            # 执行判断 (仅在 perform_check_this_iter 为 True 时)
            if perform_check_this_iter and current_metric_val > 0: # 增加 > 0 检查，防止 error 为 0 时误判
                # 检查是否有显著改善 (注意: error 是越小越好)
                if (min_metric_val - current_metric_val) > current_threshold:
                    # 显著改善
                    # print(f"  Improvement detected based on '{args.stop_criterion}': {min_metric_val:.4f} -> {current_metric_val:.4f}")
                    min_metric_val = current_metric_val
                    patience_counter = 0
                    
                    best_model_state = []
                    for sub_model in model_ddp.module.models: 
                        best_model_state.append({
                            'R': sub_model.R.data.clone(), 
                            'T': sub_model.T.data.clone()
                        })
                else:
                    # 没有显著改善
                    patience_counter += 1
            elif args.stop_criterion == 'error' and not should_calculate_error:
                 # 如果是 error 标准，但本轮未计算 error，则不增加 patience 计数器
                 pass
            elif perform_check_this_iter and current_metric_val <= 0 and args.stop_criterion == 'error':
                print(f"  Warning: Mean error is {current_metric_val:.4f}. Skipping best model check for this iteration.")


            # 检查是否需要早停
            if patience_counter >= patience:
                print(f"--- Early stopping triggered at iter {iter+1} based on '{args.stop_criterion}' ---")
                if args.stop_criterion == 'loss':
                    print(f"Loss ({global_avg_loss:.4f}) did not improve by {args.min_loss_threshold} for {patience} iterations. Min loss: {min_metric_val:.4f}")
                else: # error
                     print(f"Mean Error ({current_metric_val:.4f}m) did not improve by {args.min_error_threshold}m for {patience} check intervals. Min error: {min_metric_val:.4f}m")
                stop_signal.fill_(1.0) 

            # --- (修改) 日志记录 ---
            if (iter + 1) % 10 == 0:
                lr_r = scheduler_r.get_last_lr()[0] if scheduler_r else args.max_lr * 1e-5
                lr_t = scheduler_t.get_last_lr()[0] if scheduler_t else args.max_lr
                
                elapsed_time_sec = time.time() - start_time
                elapsed_time_str = format_time(elapsed_time_sec)
                avg_iter_time = elapsed_time_sec / (iter + 1)
                remaining_iter = args.max_iter - (iter + 1)
                remaining_time_sec = avg_iter_time * remaining_iter
                remaining_time_str = format_time(remaining_time_sec)

                # 准备 error 字符串 (如果计算了)
                err_log_str = ""
                if should_calculate_error: # 仅在计算了error的轮次显示
                     err_log_str = f"\t mean:{mean_err:.4f}m \t median:{median_err:.4f}m"

                # 动态显示 min 值
                min_metric_log_str = ""
                if args.stop_criterion == 'loss':
                    min_metric_log_str = f"min_l:{min_metric_val:.4f}"
                else: # error
                    min_metric_log_str = f"min_e:{min_metric_val:.4f}m"


                print(f"Lvl:{current_level + 1}/{args.num_levels} iter:{iter+1}/{args.max_iter} \t loss:{global_avg_loss:.4f} {min_metric_log_str} {err_log_str} \t pat:{patience_counter}/{patience} \t lr_t:{lr_t:.2e}  lr_r:{lr_r:.2e} \t elapsed:{elapsed_time_str}  eta:{remaining_time_str}")
        
        # --- [修改结束] ---
        
        # 3.广播停止信号
        dist.broadcast(stop_signal, src=0)

        # 4.检查停止信号
        if stop_signal.item() == 1.0:
            print(f"Rank {local_rank}: Received stop signal. Breaking optimization loop.")
            break 
        
        if scheduler_r:
            scheduler_r.step()
        if scheduler_t:
            scheduler_t.step()

    # 优化循环结束
    if local_rank == 0:
        print("Bundle adjustment optimization finished for this level.")

    return best_model_state
# --- [修改结束] ---

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
    """ 计算单对影像 (i, j) 之间的连接点误差"""
    
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


# --- [新功能] ---
def visualize_grid_selection(args, all_candidate_info: List[Dict], selected_diags: List[np.ndarray], ref_image: RSImage, level: int):
    """
    绘制格网选择示意图 (仅在 Rank 0 上调用)
    
    Args:
        args: 命令行参数
        all_candidate_info: 包含所有 *有效* 候选格网信息(diag, center, score)的字典列表
        selected_diags: 最终被选中的格网(diag)的列表
        ref_image: 参考影像 (例如 images[0]), 用于确定地理边界
        level: [新] 当前金字塔层级
    """
    print(f"Rank 0: Generating grid selection visualization for Level {level}...")
    try:
        # 1. 获取参考影像的地理边界
        min_x = ref_image.corner_xys[:, 0].min()
        max_x = ref_image.corner_xys[:, 0].max()
        min_y = ref_image.corner_xys[:, 1].min()
        max_y = ref_image.corner_xys[:, 1].max()
        
        geo_w = max_x - min_x
        geo_h = max_y - min_y
        
        if geo_w == 0 or geo_h == 0:
            print("Rank 0: Invalid geographic bounds for visualization.")
            return

        # 2. 创建画布
        vis_h = 1000 # 固定高度
        aspect_ratio = geo_w / geo_h
        vis_w = int(vis_h * aspect_ratio)
        canvas = np.ones((vis_h, vis_w, 3), dtype=np.uint8) * 255 # 白色背景

        # 3. 定义地理坐标到画布像素坐标的映射
        def geo_to_canvas(xy: np.ndarray) -> Tuple[int, int]:
            px = int((xy[0] - min_x) / geo_w * (vis_w - 1))
            py = int((max_y - xy[1]) / geo_h * (vis_h - 1)) # Y轴翻转 (地图坐标 -> 图像坐标)
            return (px, py)

        # 4. 绘制所有候选格网 (浅灰色)
        for info in all_candidate_info:
            diag = info['diag']
            # 构造完整的4个角点
            corners_geo = np.array([
                diag[0], [diag[1,0], diag[0,1]],
                diag[1], [diag[0,0], diag[1,1]]
            ])
            canvas_corners = [geo_to_canvas(pt) for pt in corners_geo]
            cv2.polylines(canvas, [np.array(canvas_corners, dtype=np.int32)], isClosed=True, color=(200, 200, 200), thickness=1)

        # 5. 绘制所有选中的格网 (绿色)
        for diag in selected_diags:
            corners_geo = np.array([
                diag[0], [diag[1,0], diag[0,1]],
                diag[1], [diag[0,0], diag[1,1]]
            ])
            canvas_corners = [geo_to_canvas(pt) for pt in corners_geo]
            cv2.polylines(canvas, [np.array(canvas_corners, dtype=np.int32)], isClosed=True, color=(0, 200, 0), thickness=2) # 亮绿色

        # 6. 保存图像
        output_path = os.path.join(args.debug_output_path, f'grid_selection_visualization_level_{level}.png')
        cv2.imwrite(output_path, canvas)
        print(f"Rank 0: Saved grid selection visualization to {output_path}")

    except Exception as e:
        print(f"Rank 0: FAILED to generate grid visualization. Error: {e}")

def subdivide_grids(parent_diags: List[np.ndarray]) -> List[np.ndarray]:
    """
    Takes a list of geographic grid diagonals (diags) and returns a new list
    containing the 4 sub-grids (quad-tree split) for each parent grid.
    """
    sub_grids = []
    for diag in parent_diags:
        # diag is np.array([[min_x, min_y], [max_x, max_y]])
        # 显式查找min/max，防止顺序问题
        min_x = np.min(diag[:, 0])
        max_x = np.max(diag[:, 0])
        min_y = np.min(diag[:, 1])
        max_y = np.max(diag[:, 1])
        
        mid_x = (min_x + max_x) / 2.0
        mid_y = (min_y + max_y) / 2.0

        # 1. 左下 (Bottom-Left)
        sub_grids.append(np.array([[min_x, min_y], [mid_x, mid_y]]))
        # 2. 右下 (Bottom-Right)
        sub_grids.append(np.array([[mid_x, min_y], [max_x, mid_y]]))
        # 3. 左上 (Top-Left)
        sub_grids.append(np.array([[min_x, mid_y], [mid_x, max_y]]))
        # 4. 右上 (Top-Right)
        sub_grids.append(np.array([[mid_x, mid_y], [max_x, max_y]]))
        
    return sub_grids


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

    parser.add_argument('--window_size', type=int, default=2000,
                        help='INITIAL window size in meter(m) for the coarsest level.')

    parser.add_argument('--select_imgs',type=str,default='0,1') 

    parser.add_argument('--grid_offset_x',type=float,default=0)

    parser.add_argument('--grid_offset_y',type=float,default=0)

    parser.add_argument('--grid_num',type=int,default=1)

    parser.add_argument('--patience', type=int, default=100, 
                        help='Patience for early stopping (e.g., 100 iterations)')
    
    parser.add_argument('--min_loss_threshold', type=float, default=1e-4, 
                        help='Minimum improvement threshold for min_loss to reset patience (e.g., 1e-4)')

    parser.add_argument('--check_error_during_train', action='store_true',
                        help='If set, check tie point error every 10 iterations (and after each level).')

    parser.add_argument('--num_levels', type=int, default=1,
                        help='Total number of pyramid levels for adjustment (default: 1, same as original behavior).')
    
    parser.add_argument('--vis_resolution', type=float, default=1.0, 
                        help='Resolution (in meters) for output orthophotos and checkerboards.')

    # --- [新参数]: 早停标准 ---
    parser.add_argument('--stop_criterion', type=str, choices=['loss', 'error'], default='loss',
                        help="Criterion for early stopping and best model selection ('loss' or 'error').")
    
    parser.add_argument('--min_error_threshold', type=float, default=0.01,
                        help="Minimum improvement threshold (in meters) for mean_error to reset patience when stop_criterion='error'.")
    # --- [新参数结束] ---


    args = parser.parse_args()

    # DDP 初始化
    local_rank = setup_ddp()
    world_size = dist.get_world_size() # 总进程数

    # --- [新逻辑]: 强制检查 error ---
    if args.stop_criterion == 'error' and not args.check_error_during_train:
        if local_rank == 0:
            print("Info: stop_criterion is set to 'error', automatically enabling --check_error_during_train.")
        args.check_error_during_train = True
    # --- [新逻辑结束] ---


    args.debug_output_path = os.path.join(args.root,'debug_output')
    if local_rank == 0:
        os.makedirs(args.debug_output_path,exist_ok=True)

    images = load_imgs_bundle(args)
    if len(images) < 2:
        if local_rank == 0:
            print("Error: Found less than 2 images. Bundle adjustment requires at least 2.")
        dist.destroy_process_group()
        exit()

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
    
    selected_diags_for_level = []
        
    for level in range(args.num_levels):
        
        current_window_size = args.window_size / (2**level)
        all_tasks = [] # 重置当前层级的任务列表
        
        if local_rank == 0:
            print("\n" + "="*50)
            print(f"--- Starting Pyramid Level {level + 1} / {args.num_levels} ---")
            print(f"--- Current Window Size: {current_window_size:.2f} m ---")
            print("="*50 + "\n")
            
        # --- 步骤 1: (Rank 0) 格网生成 ---
        if local_rank == 0:
            if level == 0:
                # [层级 0: 执行初始格网生成、评估和筛选]
                print("Rank 0: Level 0. Finding, assessing, and selecting initial grids...")
                
                # 1. 查找初始格网
                all_corners = np.stack([img.corner_xys for img in images], axis=0)
                all_common_diags = find_grids(all_corners, current_window_size, 
                                            offset_x=args.grid_offset_x, 
                                            offset_y=args.grid_offset_y)
                print(f"Rank 0: Found {len(all_common_diags)} total common grids.")

                # 2. 评估和筛选格网
                print("Rank 0: Loading encoder for grid quality assessment...")
                encoder_assess = EncoderDino(os.path.join(args.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'),upsample_times=0)
                encoder_assess.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
                encoder_assess.cuda(local_rank) # local_rank is 0
                encoder_assess.eval()

                transform_assess = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                ])
                
                all_valid_grids_info = [] # 存储所有有效格网的信息
                print("Rank 0: Assessing quality for all candidate grids (using AVG confidence)...")
                resample_size = 1024 # 与SharedGrid中使用的尺寸保持一致
                
                with torch.no_grad():
                    for diag in tqdm(all_common_diags, desc="Assessing Grids"):
                        total_conf_score = 0.0
                        overlapping_img_count = 0
                        
                        for img in images:
                            try:
                                corners_geo = np.array([
                                    diag[0], [diag[1,0], diag[0,1]],
                                    diag[1], [diag[0,0], diag[1,1]]
                                ])
                                corners_sampline = img.xy_to_sampline(corners_geo)

                                if (corners_sampline.min() < 0 or 
                                    corners_sampline[:, 0].max() > img.W or 
                                    corners_sampline[:, 1].max() > img.H):
                                    continue 

                                img_patch, _ = img.resample_image_by_sampline(corners_sampline, 
                                                                                    (resample_size, resample_size), 
                                                                                    need_local=True) 

                                img_tensor = transform_assess(img_patch)[None].cuda(local_rank)
                                _, conf = encoder_assess(img_tensor)

                                total_conf_score += conf.sum().item()
                                overlapping_img_count += 1
                                
                            except Exception as e:
                                continue
                        
                        if overlapping_img_count >= 2:
                            average_conf = total_conf_score / overlapping_img_count
                            center_xy = diag.mean(axis=0)
                            all_valid_grids_info.append({
                                'score': average_conf,
                                'center': center_xy,
                                'diag': diag
                            })
                
                # --- [修改开始]: 实现 NMS + 置信度补齐 ---
                # 3. 执行空间抑制选择算法
                if args.grid_num > 0 and len(all_valid_grids_info) > args.grid_num:
                    print(f"Rank 0: Found {len(all_valid_grids_info)} valid grids. Selecting {args.grid_num} using confidence-based spatial selection...")
                    
                    # (新) 1. 保存原始的、按置信度排序的列表
                    all_valid_grids_sorted = sorted(all_valid_grids_info, key=lambda x: x['score'], reverse=True)
                    # (新) 2. 使用副本进行 NMS 循环
                    candidate_grids_for_nms = all_valid_grids_sorted.copy()
                    
                    # (新) 3. 存储完整的格网信息 (dict)
                    selected_grids_info_nms = [] 
                    suppression_radius = current_window_size * 1.5 
                    print(f"Rank 0: Using suppression radius {suppression_radius:.2f} m...")

                    # 4. NMS 循环
                    while len(candidate_grids_for_nms) > 0 and len(selected_grids_info_nms) < args.grid_num:
                        best_grid = candidate_grids_for_nms.pop(0)
                        selected_grids_info_nms.append(best_grid) # 存储完整信息
                        
                        remaining_grids = []
                        for grid_info in candidate_grids_for_nms:
                            distance = np.linalg.norm(best_grid['center'] - grid_info['center'])
                            if distance > suppression_radius:
                                remaining_grids.append(grid_info)
                        candidate_grids_for_nms = remaining_grids 
                    
                    # (新) 5. NMS 循环结束，开始执行 "补齐" 逻辑
                    num_selected_by_nms = len(selected_grids_info_nms)
                    
                    # 检查 NMS 选中的数量是否小于目标
                    if num_selected_by_nms < args.grid_num:
                        print(f"Rank 0: NMS 选中了 {num_selected_by_nms} 个格网 (目标: {args.grid_num})。")
                        print(f"Rank 0: 正在从高置信度列表中补齐剩余格网...")
                        
                        num_to_backfill = args.grid_num - num_selected_by_nms
                        
                        # (新) 使用 Set 快速查找已被 NMS 选中的格网
                        # 我们使用 diag 的 .tostring() 作为唯一的 hashable key
                        selected_diags_set = {info['diag'].tostring() for info in selected_grids_info_nms}
                        
                        backfill_grids_info = []
                        
                        # (新) 遍历*原始的、排序好的*列表 (all_valid_grids_sorted)
                        for grid_info in all_valid_grids_sorted:
                            # 如果这个格网 *未被* NMS 选中
                            if grid_info['diag'].tostring() not in selected_diags_set:
                                backfill_grids_info.append(grid_info)
                                # 如果补齐了足够的数量，立刻停止
                                if len(backfill_grids_info) == num_to_backfill:
                                    break
                        
                        print(f"Rank 0: 已补齐 {len(backfill_grids_info)} 个格网。")
                        # (新) 将 NMS 选中的列表与补齐的列表合并
                        final_selected_grids_info = selected_grids_info_nms + backfill_grids_info
                        
                    else:
                        # (新) 如果 NMS 选中的数量足够，则直接使用 NMS 的结果
                        final_selected_grids_info = selected_grids_info_nms

                    # (新) 最终从合并后的列表中提取 diag
                    all_tasks = [info['diag'] for info in final_selected_grids_info]
                    print(f"Rank 0: 最终选中 {len(all_tasks)} 个格网。")
                
                else:
                    # (保持不变) 如果 grid_num 为 0 或候选总数本就 <= grid_num，则使用所有
                    print(f"Rank 0: grid_num ({args.grid_num}) 为 0 或 >= 有效格网总数。使用所有 {len(all_valid_grids_info)} 个有效格网。")
                    all_tasks = [info['diag'] for info in all_valid_grids_info]
                
                # --- [修改结束] ---

                # 4. 调用可视化
                visualize_grid_selection(args, all_valid_grids_info, all_tasks, images[0], level)
                
                # 5. 清理Encoder，释放显存
                del encoder_assess, transform_assess
                torch.cuda.empty_cache()
                
                # 6. 保存结果给下一层级
                selected_diags_for_level = all_tasks
                
            else:
                # [层级 > 0: 执行格网四叉树划分]
                print(f"Rank 0: Level {level+1}. Subdividing {len(selected_diags_for_level)} grids from previous level.")
                
                # 1. 调用新函数进行划分
                all_tasks = subdivide_grids(selected_diags_for_level)
                
                # 2. 保存结果给下一层级
                selected_diags_for_level = all_tasks
                
                print(f"Rank 0: Created {len(all_tasks)} new sub-grids for processing.")
            
            # 7. [通用] 为DDP负载均衡打乱任务列表
            random.shuffle(all_tasks)
            print(f"Rank 0: Final task list for level {level+1} has {len(all_tasks)} grids.")
        
        # --- 步骤 2: (所有 Rank) 广播和执行当前层级的平差 ---
        
        # 1. 广播 *格网任务列表*
        tasks_to_broadcast = [all_tasks] if local_rank == 0 else [None]
        dist.broadcast_object_list(tasks_to_broadcast, src=0)
        all_tasks = tasks_to_broadcast[0] # all_tasks 是 [diag1, diag2, ...]
        
        # 2. 广播 *重叠对列表* (仅用于验证)
        pairs_to_broadcast = [overlapping_pairs] if local_rank == 0 else [None]
        dist.broadcast_object_list(pairs_to_broadcast, src=0)
        overlapping_pairs = pairs_to_broadcast[0] # overlapping_pairs 是 [(i, j), ...]

        # 3. 每个进程根据自己的rank获取 *格网* 子集
        my_tasks = all_tasks[local_rank::world_size] # my_tasks 是 [diag_k, diag_l, ...]

        # 4. 每个进程都加载自己的特征提取器
        encoder = EncoderDino(os.path.join(args.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'),upsample_times=0)
        encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
        if local_rank == 0:
            print(f"Encoder Loaded by all processes for feature extraction on Level {level+1}")

        # 5. 每个进程在自己的 *格网* 子集上创建 SharedGrid
        local_shared_grids: List[SharedGrid] = [] # 新的数据列表
        print(f"[Rank {local_rank}] Level {level+1}: Total grids: {len(all_tasks)}, assigned: {len(my_tasks)}.")
        
        # 循环格网 (diags)
        for idx, diag in enumerate(my_tasks):
            # 构造一个包含层级信息的全局唯一ID
            global_grid_id = f"L{level}_R{local_rank}_{idx}" 
            try:
                # 1. 创建 SharedGrid，此时会重采样所有影像块
                #  传入 *所有* 影像
                grid = SharedGrid(args, diag, images, global_grid_id) 
                
                # 2. 依次提取特征，并释放img内存
                grid.extract_features_sequentially(encoder, local_rank)
                
                local_shared_grids.append(grid)
                print(f"[Rank {local_rank}] Grid {global_grid_id} (with {len(grid.overlapping_image_ids)} images) created and features extracted on cuda:{local_rank}")
            except Exception as e:
                print(f"[Rank {local_rank}] !! FAILED to create grid {global_grid_id}. Error: {e}")


        # 6. DDP模型和优化器设置
        # [重要] 每次循环都重新创建，用于求解本层级的“增量”
        model = BundleAffineModel(len(images)).to(local_rank)
        model_ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False) 
        
        all_R_params = [m.R for m in model_ddp.module.models]
        all_T_params = [m.T for m in model_ddp.module.models]
        
        optimizer_r = None
        optimizer_t = None
        scheduler_r = None
        scheduler_t = None

        if all_R_params:
            optimizer_r = torch.optim.Adam(all_R_params, lr=args.max_lr * 1e-5 / (10 ** level))
            scheduler_r = torch.optim.lr_scheduler.OneCycleLR(optimizer_r, max_lr=args.max_lr * 1e-5, total_steps=args.max_iter,pct_start=50 / args.max_iter)
        
        if all_T_params:
            optimizer_t = torch.optim.Adam(all_T_params, lr=args.max_lr / (10 ** level))
            scheduler_t = torch.optim.lr_scheduler.OneCycleLR(optimizer_t, max_lr=args.max_lr, total_steps=args.max_iter,pct_start=50 / args.max_iter)
        
        best_model_state = []

        if optimizer_r is None and optimizer_t is None and local_rank == 0:
            print(f"Warning: No parameters to optimize for level {level+1} (only one image provided?). Skipping optimization.")
        else:
            # 7. 调用 fit_affine_bundle
            best_model_state = fit_affine_bundle(args, # 传入 args
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
                                                 # min_loss_threshold 从 args 获取
                                                 overlapping_pairs=overlapping_pairs,
                                                 current_level=level 
                                                 )



        # 8. 同步点，确保所有进程都完成了优化
        dist.barrier()
        
        # 9. (Rank 0) [重要] 将本层级的结果“烘焙”到RPC模型中
        # --- [修改开始]: 广播最佳模型，所有 Ranks 都执行烘焙 ---

        # 1. Rank 0 广播 best_model_state
        state_to_broadcast = [best_model_state] if local_rank == 0 else [None]
        dist.broadcast_object_list(state_to_broadcast, src=0)
        best_model_state = state_to_broadcast[0]

        # 2. 所有 Ranks 加载最佳模型状态到 *本地* 的 DDP 模型
        if local_rank == 0:
            print(f"\n[All Ranks] Applying (best) adjustments from Level {level+1} to RPC models...")
            
        if best_model_state: 
            with torch.no_grad():
                for i, state in enumerate(best_model_state):
                    if i < len(model_ddp.module.models):
                        # 直接操作 .data 来更新参数
                        model_ddp.module.models[i].R.data.copy_(state['R'])
                        model_ddp.module.models[i].T.data.copy_(state['T'])
        else:
            if local_rank == 0:
                print(f"Warning: No best model state found for Level {level+1}. Using final iteration state.")
        
        # 3. 所有 Ranks 将*本地* DDP 模型中的仿射变换 "烘焙" 到*本地*的 images RPC 列表中
        for i in range(1, len(images)):
            final_A_i_level = model_ddp.module.get_affine(i).detach()
            if local_rank == 0: # 仅 Rank 0 打印，避免日志混乱
                print(f"Level {level+1} Affine Delta for image {i}: \n {final_A_i_level.cpu().numpy()}")
            # rpc.Update_Adjust 会将新的变换(final_A_i_level)
            # 与已有的变换进行矩阵复合 [cite: rpc.py, line 290]
            images[i].rpc.Update_Adjust(final_A_i_level)
        
        if local_rank == 0:
            print(f"Rank 0: Level {level+1} adjustments applied by all ranks.")
        
        # --- [修改结束] ---
        
        
        # 10. (Rank 0) 打印本层级后的精度
        if local_rank == 0:
            if args.check_error_during_train or level == args.num_levels - 1:
                print(f"\n--- Error Report After Level {level+1} ---")
                all_errors_level = check_all_pairs_error(images, overlapping_pairs)
                if len(all_errors_level) > 0 and all_errors_level.mean() != 0.0:
                    print(f"Total tie points checked: {len(all_errors_level)}")
                    print(f"Mean Error:   {all_errors_level.mean():.4f} m")
                    print(f"Median Error: {np.median(all_errors_level):.4f} m")
                    print(f"RMSE:         {np.sqrt(np.mean(all_errors_level**2)):.4f} m")
                else:
                    print("No valid tie points found for intermediate check.")

        
        # --- [新步骤: 并行可视化] ---
        # 此刻, 所有 Ranks 上的 images[i].rpc 都已更新
        if local_rank == 0:
            print(f"\n[All Ranks] Starting parallel visualization for Level {level+1} (Res: {args.vis_resolution}m)...")
        
        vis_resolution = args.vis_resolution # 使用命令行参数
        
        # 每个 Rank 并行处理自己的格网
        for grid in local_shared_grids:
            grid_ortho_cache = {} # 缓存本格网的正射影像，用于棋盘格
            
            # 使用 grid.id 创建唯一的输出文件夹
            # grid.id 已经是 "L{level}_R{local_rank}_{idx}" 格式
            grid_vis_path = os.path.join(args.debug_output_path, f"vis_{grid.id}") # 加一个 "vis_" 前缀
            os.makedirs(grid_vis_path, exist_ok=True)
            
            # 1. 生成正射影像
            for img_id in grid.overlapping_image_ids:
                rs_image = images[img_id] # 获取包含 *已调整* RPC 的 RSImage
                ortho_output_path = os.path.join(grid_vis_path, f"ortho_img_{img_id}.tif")
                
                try:
                    ortho_array, transform = orthorectify_patch_mercator(
                        rs_image, 
                        grid.diag, # [cite: adjust_test_1023.py, line 161]
                        resolution=vis_resolution,
                        output_path=ortho_output_path
                    )
                    grid_ortho_cache[img_id] = (ortho_array, transform)
                except Exception as e:
                    print(f"[Rank {local_rank}] FAILED orthorectification for {grid.id}/img_{img_id}. Error: {e}")

            # 2. 生成棋盘格
            for (i, j) in itertools.combinations(grid.overlapping_image_ids, 2):
                if i in grid_ortho_cache and j in grid_ortho_cache:
                    ortho_i, transform_i = grid_ortho_cache[i]
                    ortho_j, transform_j = grid_ortho_cache[j]
                    
                    checker_output_path = os.path.join(grid_vis_path, f"checker_{i}_vs_{j}.png")
                    try:
                        create_checkerboard(
                            ortho_i, ortho_j, 
                            transform_i, # 变换应该是相同的
                            checker_output_path, 
                            block_size=50 # 棋盘格大小 (像素)
                        )
                    except Exception as e:
                        print(f"[Rank {local_rank}] FAILED checkerboard for {grid.id}/({i},{j}). Error: {e}")
        
        # [新] 添加一个同步点
        # 确保所有 Rank 都完成了文件写入，然后再进入下一层或清理资源
        dist.barrier()
        if local_rank == 0:
            print(f"[All Ranks] Visualization for Level {level+1} complete.")
        # --- [新步骤结束] ---


        # 11. 清理本层级的资源，为下个层级做准备
        del local_shared_grids, model, model_ddp, optimizer_r, optimizer_t, scheduler_r, scheduler_t, encoder
        torch.cuda.empty_cache()
        dist.barrier() # 确保所有进程都清理完毕
        
    # --- 金字塔循环结束 ---
    
    if local_rank == 0:
        print("\n" + "="*50)
        print(f"--- Multi-level Bundle Adjustment Finished ({args.num_levels} levels) ---")
        print("="*50 + "\n")
    
    # 最终清理
    dist.destroy_process_group()

