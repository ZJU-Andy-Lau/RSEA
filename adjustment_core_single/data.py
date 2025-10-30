import os
import torch
from torchvision import transforms
import numpy as np
import cv2
from typing import List, Dict

# 假设的外部依赖 (确保这些文件/模块路径正确)
from rs_image_1022 import RSImage
from rpc import RPCModelParameterTorch
from model.encoder_dino_0927 import EncoderDino
from utils import vis_conf, downsample_average

# --- [修改] DDP 移除 ---
# import torch.distributed as dist

# 从同一核心模块导入
from adjustment_core.loop import warp_local, feature_sampling

class Window():
    def __init__(self,img:np.ndarray,local:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch):
        self.img = img # 原始图像块 (将在特征提取后被删除以节省显存)
        self.local = torch.from_numpy(local)
        self.dem = torch.from_numpy(dem)
        self.rpc = rpc
        self.feature = None
        self.conf = None
        
    
    # --- [修改] 签名：新增 device 参数 ---
    def to_gpu(self, device: torch.device):
        # --- [修改] .cuda() -> .to(device) ---
        self.local = self.local.to(device)
        self.dem = self.dem.to(device)
        self.rpc.to_gpu(device) # 假设 RPCModelParameterTorch.to_gpu 也被修改为接受 device
        # 特征和置信度在提取时已在GPU上
        if self.feature is not None:
            self.feature = self.feature.to(device)
        if self.conf is not None:
            self.conf = self.conf.to(device)


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
        
        # --- [修改] 移除 dist.get_rank() == 0 判断 ---
        if not args.auto:
            self.debug_output_path = os.path.join(args.debug_output_path, f'grid_{self.id}')
            os.makedirs(self.debug_output_path, exist_ok=True)
            for img_id in self.overlapping_image_ids:
                cv2.imwrite(os.path.join(self.debug_output_path, f'img_raw_{img_id}.png'), self.windows[img_id].img)


    @torch.no_grad()
    # --- [修改] 签名：local_rank -> device ---
    def extract_features_sequentially(self, encoder: EncoderDino, device: torch.device):
        """
        依次提取此格网中所有影像的特征。
        [Refactored] encoder 现在作为参数传入，且假定已在
        """
        # [Refactored] 移除 encoder 的加载和 .cuda()
        encoder_eval = encoder.eval() # 确保是 eval 模式
        
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                    ])
        
        # --- [修改] 移除 dist.get_rank() == 0 判断 ---
        if not self.args.auto:
            os.makedirs(self.debug_output_path, exist_ok=True)
        
        for img_id in self.overlapping_image_ids:
            window = self.windows[img_id]
            
            # --- 特征提取 ---
            # --- [修改] .cuda(local_rank) -> .to(device) ---
            img_tensor = transform(window.img)[None].to(device)
            feature, conf = encoder_eval(img_tensor) # 使用传入的 encoder
            
            # --- 存储特征 ---
            h, w = feature.shape[-2:]
            window.feature = feature[0].permute(1,2,0).flatten(0,1)
            window.conf = conf.squeeze().flatten(0,1)
            window.local = downsample_average(window.local, encoder.SAMPLE_FACTOR).flatten(0,1)
            window.dem = downsample_average(window.dem, encoder.SAMPLE_FACTOR).flatten(0,1)

            # --- [修改] 移除 dist.get_rank() == 0 判断 ---
            if not self.args.auto:
                original_img_for_vis = window.img.copy() # 复制一份用于可视化
            
            # 立刻删除已不再需要的原始图像块，释放显存
            del window.img
            
            # --- 将特征数据移至GPU ---
            # --- [修改] 传入 device ---
            window.to_gpu(device) 

            # --- [修改] 移除 dist.get_rank() == 0 判断 ---
            if not self.args.auto:
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

    # --- [修改] 签名：model_ddp -> model, local_rank -> device ---
    def calculate_all_pairs_loss(self, model: torch.nn.Module, images: List[RSImage], device: torch.device) -> torch.Tensor:
        """
         (已修改) 计算此格网内所有影像两两之间的 *非对称* 损失 (j -> i, j > i)。
         [Refactored] 依赖外部导入的 warp_local 和 feature_sampling
        """
        import itertools # 确保导入
        
        # --- [修改] device=local_rank -> device=device ---
        grid_total_loss = torch.tensor(0.0, device=device)
        num_valid_pairs_in_grid = 0
        
        # 遍历所有唯一的像对 (i, j)
        for (i, j) in itertools.combinations(self.overlapping_image_ids, 2):
            
            # --- 这部分逻辑与原 fit_affine_bundle 中的循环体完全一致 ---
            # i < j, A_i 是目标 (可能固定也可能移动), A_j 是源 (总是移动)
            # --- [修改] model_ddp.module -> model ---
            A_i = model.get_affine(i)
            A_j = model.get_affine(j)
            
            window_i = self.windows[i]
            window_j = self.windows[j]
            
            rpc_i = images[i].rpc
            rpc_j = images[j].rpc
            
            # 1. Warp j -> i  (始终将索引号大的 j 投影到索引号小的 i)
            warp_j_to_i = warp_local(window_j.local.float(), window_j.dem, rpc_j, rpc_i, A_j)
            feat_j_in_i, conf_j_in_i, valid_j = feature_sampling(window_i.feature.float(), window_i.conf.float(), window_i.local.float(), warp_j_to_i, k = self.args.kmin_k)

            # 3. 计算 loss_a (j -> i)
            # --- [修改] device=local_rank -> device=device ---
            loss_a = torch.tensor(0.0, device=device)
            if feat_j_in_i is not None:
                feat_j_orig = window_j.feature[valid_j].float()
                conf_cov_a = window_j.conf[valid_j].float() * conf_j_in_i
                weight_a = conf_cov_a / (conf_cov_a.mean() + 1e-3)
                loss_a = (torch.norm(feat_j_orig - feat_j_in_i, dim=-1) * weight_a).mean() * 10000.

            pair_loss = loss_a
            
            if not torch.isnan(pair_loss) and not torch.isinf(pair_loss) and pair_loss > 0:
                grid_total_loss = grid_total_loss + pair_loss
                num_valid_pairs_in_grid += 1
            else:
                if pair_loss > 0: # 仅在非零时打印警告 (虽然isnan和isinf已经覆盖了)
                    # --- [修改] Rank{local_rank} -> SingleGPU ---
                    print(f"[SingleGPU]: Detect invalid loss:{pair_loss.item()} in Grid {self.id} for pair ({i}, {j})")


        if num_valid_pairs_in_grid > 0:
            return grid_total_loss / num_valid_pairs_in_grid # 返回该格网的平均损失
        else:
            # --- [修改] device=local_rank -> device=device ---
            return torch.tensor(0.0, device=device)
