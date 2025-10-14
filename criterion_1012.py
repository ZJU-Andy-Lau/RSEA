import torch.nn as nn 
import torch 
import torch.nn.functional as F
import numpy as np
import math
from rpc import RPCModelParameterTorch
from utils import project_mercator,mercator2lonlat
import time
from typing import List

def calculate_laplacian_loss(pred_patch):
    """
    计算二阶平滑度损失（拉普拉斯正则化）。
    直接惩罚预测坐标场中的高频抖动。
    """
    # 定义一个固定的2D拉普拉斯卷积核
    laplacian_kernel = torch.tensor([
        [0, 1, 0],
        [1, -4, 1],
        [0, 1, 0]
    ], dtype=torch.float32, device=pred_patch.device).reshape(1, 1, 3, 3)

    # (输入通道=3, 输出通道=3, group=3 实现逐通道卷积)
    laplacian_kernel_xyz = laplacian_kernel.repeat(3, 1, 1, 1)

    # F.conv2d需要 (N, C, H, W) 格式
    laplacian_response = F.conv2d(pred_patch, laplacian_kernel_xyz, padding=1, groups=3)
    
    # 损失是拉普拉斯响应的L1范数，鼓励响应趋近于0
    return torch.mean(torch.abs(laplacian_response))

def calculate_consistency_loss(pred_patch, gt_patch):
    """
    计算patch内的坐标一致性损失 (在绝对地理坐标空间中计算)。
    这个损失函数旨在惩罚模型预测的局部几何变形。
    它要求在一个patch内部，相邻点之间的相对位移向量在预测结果和真实结果中应该保持一致。

    Args:
        pred_patch (torch.Tensor): 预测的地理坐标均值, shape (N, 3, ph, pw)
        gt_patch (torch.Tensor): 真实的地理坐标, shape (N, 3, ph, pw)
    """
    # --- 计算水平方向上相邻点之间的位移向量 ---
    delta_pred_h = pred_patch[..., :, 1:] - pred_patch[..., :, :-1]
    delta_gt_h = gt_patch[..., :, 1:] - gt_patch[..., :, :-1]
    
    # --- 计算垂直方向上相邻点之间的位移向量 ---
    delta_pred_v = pred_patch[..., 1:, :] - pred_patch[..., :-1, :]
    delta_gt_v = gt_patch[..., 1:, :] - gt_patch[..., :-1, :]
    
    # --- [修改] 计算L2损失 (均方误差) ---
    loss_h = torch.nn.functional.mse_loss(delta_pred_h, delta_gt_h)
    loss_v = torch.nn.functional.mse_loss(delta_pred_v, delta_gt_v)
    
    # 返回水平和垂直方向损失之和
    return loss_h + loss_v


class CriterionTrainGrid(nn.Module):
    """
    用于训练Grid中Mapper的核心损失函数。
    """
    def __init__(self, consistency_weight: float = 1.0, laplacian_weight: float = 0.0):
        super().__init__()
        # --- 初始化各项损失的权重和参数 ---
        self.consistency_weight = consistency_weight
        self.laplacian_weight = laplacian_weight
        self.height_weight = 10.0          # 高程损失的权重
        self.photo_weight = 1.0            # 重投影损失的权重
        self.clamp_max = 1000              # 用于tanh_clamp函数，限制重投影损失的最大值
        self.bce = nn.BCEWithLogitsLoss()  # 用于计算valid_score的二元交叉熵损失
        self.valid_score_weight = 10.0     # valid_score损失的权重
        self.warmup_iters = 5           # 训练预热期，在此期间只使用L2损失

    def forward(self, epoch, max_epoch, pred_mu_absolute_patch, pred_log_sigma_patch, gt_absolute_patch, conf_patch, linesamp_patch, elements: list, element_indices: list, valid_score_patch, valid_labels, num_positive_samples):
        """
        损失函数的前向传播计算。
        """
        
        # 确保所有输入张量的数据类型为float32
        pred_mu_absolute_patch, pred_log_sigma_patch, gt_absolute_patch, conf_patch, linesamp_patch, valid_score_patch, valid_labels = \
            [i.to(torch.float32) for i in [pred_mu_absolute_patch, pred_log_sigma_patch, gt_absolute_patch, conf_patch, linesamp_patch, valid_score_patch, valid_labels]]
        
        progress = 1. * epoch / max_epoch

        # --- 1. Valid Score损失 (在所有样本上计算) ---
        loss_valid = self.bce(valid_score_patch, valid_labels)

        # 如果批次中没有正样本，则只返回valid_score损失
        if num_positive_samples == 0:
            total_loss = loss_valid * self.valid_score_weight
            loss_details = {'d': 0.0, 'obj': 0.0, 'h': 0.0, 'p': 0.0, 'c': 0.0, 'lap': 0.0, 'v': loss_valid.item()}
            return total_loss, loss_details

        # --- 从批次中分离出所有正样本的数据 ---
        pred_mu_absolute_pos = pred_mu_absolute_patch[:num_positive_samples]
        pred_log_sigma_pos = pred_log_sigma_patch[:num_positive_samples]
        gt_absolute_pos = gt_absolute_patch[:num_positive_samples]
        conf_pos = conf_patch[:num_positive_samples]
        linesamp_pos = linesamp_patch[:num_positive_samples]
        element_indices_pos = element_indices[:num_positive_samples]

        # --- 将Encoder的特征置信度转换为损失权重 ---
        conf_weights = conf_pos.clone()
        conf_weights[conf_weights > 0.5] = 0.5 + progress * 0.4
        conf_weights[conf_weights < 0.5] = 0.5 - progress * 0.4
        conf_weights = torch.clip(conf_weights - conf_weights.mean() + 1., min=0.)

        # --- [修改] 2. 绝对地理坐标L2损失 (均方误差) ---
        error_squared = (pred_mu_absolute_pos - gt_absolute_pos) ** 2
        # loss_obj 计算XY平面上的均方误差
        loss_obj = (error_squared[:, :2, ...].sum(dim=1, keepdim=True) * conf_weights).mean()
        # loss_height 计算高程的均方误差
        loss_height = (error_squared[:, 2:3, ...] * conf_weights).mean()

        # --- 3. 像素空间重投影损失 (核心修复) ---
        loss_photo_total = torch.tensor(0.0, device=pred_mu_absolute_pos.device)
        unique_element_indices = np.unique(element_indices_pos)
        
        for element_idx in unique_element_indices:
            # 筛选出属于当前 element 的 patch 的掩码
            mask = torch.tensor([i == element_idx for i in element_indices_pos], device=pred_mu_absolute_pos.device)
            
            if not mask.any(): continue

            # 提取这部分 patch 对应的所有数据
            pred_mu_group = pred_mu_absolute_pos[mask]
            linesamp_gt_group = linesamp_pos[mask]
            
            # 使用正确的 RPC 模型进行重投影
            rpc = elements[element_idx].rpc
            
            # 为了高效计算，先将patch展平
            pred_xyh_flat = pred_mu_group.permute(0, 2, 3, 1).reshape(-1, 3)
            linesamp_gt_flat = linesamp_gt_group.permute(0, 2, 3, 1).reshape(-1, 2)

            latlon_pred_flat = mercator2lonlat(pred_xyh_flat[:, [1, 0]])
            linesamp_pred_flat = torch.stack(rpc.RPC_OBJ2PHOTO(latlon_pred_flat[:, 0], latlon_pred_flat[:, 1], pred_xyh_flat[:, 2]), dim=1)[:, [1, 0]]
            
            reprojection_error_pixels = torch.norm(linesamp_pred_flat - linesamp_gt_flat, dim=1)
            
            loss_photo_group = reprojection_error_pixels.mean()
            loss_photo_total += loss_photo_group

        loss_photo = loss_photo_total / len(unique_element_indices) if len(unique_element_indices) > 0 else torch.tensor(0.0, device=pred_mu_absolute_pos.device)

        # --- 4. 几何一致性损失 ---
        loss_consistency = calculate_consistency_loss(pred_mu_absolute_pos, gt_absolute_pos)
        loss_laplacian = calculate_laplacian_loss(pred_mu_absolute_pos)

        # --- 5. 组合总损失 (引入预热逻辑) ---
        if epoch < self.warmup_iters:
            # 在预热期，只使用L2损失强制模型学习均值
            loss_regression = loss_obj + loss_height * self.height_weight
            # 对sigma施加一个小的正则化，防止其在预热期乱跑
            sigma_regularization = (pred_log_sigma_pos ** 2).mean() * 0.01 
            
            total_loss = (loss_regression + sigma_regularization + 
                          loss_consistency * self.consistency_weight + 
                          loss_laplacian * self.laplacian_weight + 
                          loss_photo * self.photo_weight + 
                          loss_valid * self.valid_score_weight)
            
            loss_distribution = torch.tensor(0.0) # 在预热期，分布损失为0
        else:
            # 预热期后，使用完整的负对数似然损失
            sigma_absolute_pos = torch.exp(pred_log_sigma_pos)
            term1 = ((gt_absolute_pos - pred_mu_absolute_pos) ** 2) / (2 * sigma_absolute_pos**2 + 1e-8)
            term2 = pred_log_sigma_pos
            loss_distribution = ((term1 + term2) * conf_weights).mean()
            
            total_loss = (loss_distribution + loss_obj + loss_height * self.height_weight + 
                          loss_photo * self.photo_weight + 
                          loss_consistency * self.consistency_weight + 
                          loss_laplacian * self.laplacian_weight +
                          loss_valid * self.valid_score_weight)
        
        # 构建一个包含各分项损失的字典，用于日志打印
        loss_details = {
            'd': loss_distribution.item(),
            'obj': np.sqrt(loss_obj.item()),
            'h': loss_height.item(),
            'p': loss_photo.item(),
            'c': loss_consistency.item(),
            'lap': loss_laplacian.item(),
            'v': loss_valid.item()
        }

        return total_loss, loss_details

