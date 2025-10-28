import os
import torch
import torch.nn.functional as F # [新增] 导入 F 用于
from torchvision import transforms
import numpy as np
import cv2
from typing import List, Dict

# 假设的外部依赖 (确保这些文件/模块路径正确)
from rs_image_1022 import RSImage
from rpc import RPCModelParameterTorch
from model.encoder_dino_0927 import EncoderDino
from utils import vis_conf, downsample_average

import torch.distributed as dist

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
        
        if dist.get_rank() == 0 and not args.auto:
            self.debug_output_path = os.path.join(args.debug_output_path, f'grid_{self.id}')
            os.makedirs(self.debug_output_path, exist_ok=True)
            for img_id in self.overlapping_image_ids:
                cv2.imwrite(os.path.join(self.debug_output_path, f'img_raw_{img_id}.png'), self.windows[img_id].img)


    @torch.no_grad()
    def extract_features_sequentially(self, encoder: EncoderDino, local_rank: int):
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
        
        if dist.get_rank() == 0 and not self.args.auto:
            os.makedirs(self.debug_output_path, exist_ok=True)
        
        for img_id in self.overlapping_image_ids:
            window = self.windows[img_id]
            
            # --- 特征提取 ---
            img_tensor = transform(window.img)[None].cuda(local_rank)
            feature, conf = encoder_eval(img_tensor) # 使用传入的 encoder
            
            # --- 存储特征 ---
            h, w = feature.shape[-2:]
            window.feature = feature[0].permute(1,2,0).flatten(0,1)
            window.conf = conf.squeeze().flatten(0,1)
            window.local = downsample_average(window.local, encoder.SAMPLE_FACTOR).flatten(0,1)
            window.dem = downsample_average(window.dem, encoder.SAMPLE_FACTOR).flatten(0,1)

            if dist.get_rank() == 0 and not self.args.auto:
                original_img_for_vis = window.img.copy() # 复制一份用于可视化
            
            # 立刻删除已不再需要的原始图像块，释放显存
            del window.img
            
            # --- 将特征数据移至GPU ---
            window.to_gpu() # to_gpu 会自动使用 local_rank 对应的卡

            if dist.get_rank() == 0 and not self.args.auto:
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

    def calculate_all_pairs_loss(self, model_ddp, images: List[RSImage], local_rank: int) -> torch.Tensor:
        """
         (已修改) 计算此格网内所有影像两两之间的 *非对称* 损失 (j -> i, j > i)。
         [Refactored] 依赖外部导入的 warp_local 和 feature_sampling
         [!! 已修改] 损失函数逻辑已根据新需求重写。
        """
        import itertools # 确保导入
        
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

            # 2. [已修改] 调用新的 feature_sampling
            #    它现在只返回 K 近邻的空间距离、索引和有效掩码
            spatial_dists_k, neighbor_idxs_k, valid_j = feature_sampling(
                window_i.local.float(),  # 基础点云 (在...中查找)
                warp_j_to_i,             # 查询点云
                k = self.args.kmin_k
            )

            # 3. [已修改] 计算 loss_a (j -> i)
            loss_a = torch.tensor(0.0, device=local_rank)

            # 检查 feature_sampling 是否返回了有效点
            if spatial_dists_k is None:
                # 没有有效点，损失为 0，跳到下一个像对
                pair_loss = loss_a
            else:
                # --- START: 实施新的损失逻辑 ---
                
                # 4. 获取查询点 j 的原始特征 (feat_j_orig)
                # [N_valid, D] D是特征维度
                feat_j_valid = window_j.feature[valid_j].float() 

                # 5. 获取 K 个邻近点 i 的特征 (feat_i_k)
                # [N_valid, k, D]
                feat_i_k = window_i.feature.float()[neighbor_idxs_k] 

                # 6. 计算 Cosine 相似度
                # (N_valid, 1, D)
                feat_j_expanded = feat_j_valid.unsqueeze(1) 
                
                # dim=2 表示在特征维度 D 上计算相似度
                # [N_valid, k]
                cos_sim_k = F.cosine_similarity(feat_j_expanded, feat_i_k, dim=2)

                # 7. (可选) 引入 temperature 缩放，使 Softmax 更敏感
                # temperature = 0.1 # (可以作为超参数)
                # cos_sim_k = cos_sim_k / temperature

                # 8. 计算 Softmax 权重
                # dim=1 表示在 K 个近邻上计算 Softmax
                # [N_valid, k]
                weights_k = F.softmax(cos_sim_k, dim=1)

                # 9. 获取 K 个空间距离 (已由 feature_sampling 在步骤 2 返回)
                # spatial_dists_k 形状为 [N_valid, k]
                
                # 10. 计算每个查询点的损失：(空间距离 * Softmax权重) 的加权平均
                # (N_valid, k) * (N_valid, k) -> sum(dim=1) -> (N_valid)
                loss_per_query = torch.sum(spatial_dists_k * weights_k, dim=1) # [N_valid]

                # 11. 计算每个查询点的置信度
                # 11.a 获取 K 个邻近点 i 的置信度
                # [N_valid, k]
                conf_i_k = window_i.conf.float()[neighbor_idxs_k]
                
                # 11.b 用 Softmax 权重加权平均，得到匹配到 i 上的置信度
                # [N_valid]
                conf_per_query_i = torch.sum(conf_i_k * weights_k, dim=1)

                # 11.c 获取查询点 j 的原始置信度
                # [N_valid]
                conf_j_valid = window_j.conf[valid_j].float() 
                
                # 11.d 最终置信度 (相乘)
                # [N_valid]
                final_conf_per_query = conf_per_query_i * conf_j_valid

                # 12. 计算该像对的总损失：(loss_per_query * final_conf_per_query) 的加权平均
                
                # 使用归一化的置信度作为权重
                conf_sum = final_conf_per_query.sum()
                if conf_sum > 1e-6:
                    # 保留 * 10000 因子以维持损失的量级
                    loss_a = ((loss_per_query * final_conf_per_query).sum() / conf_sum)  * 1000.
                else:
                    # 如果所有点置信度都为0，则损失为0
                    loss_a = torch.tensor(0.0, device=local_rank)
                
                # --- END: 新的损失逻辑 ---

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
