import torch.nn as nn 
import torch 
import torch.nn.functional as F
import numpy as np
import math
from rpc import RPCModelParameterTorch
from utils import project_mercator,mercator2lonlat
import time
from typing import List

def calculate_affine_loss_differentiable(src_points, dst_points):
    """
    计算可微分的像方仿射一致性损失。
    它求解将src映射到dst的最佳仿射变换T，然后惩罚T对src自身造成的位移。
    理想情况下，T应该是一个单位矩阵，损失应趋近于0。

    Args:
        src_points (torch.Tensor): 源点云 (像方真值), shape (K, 2)
        dst_points (torch.Tensor): 目标点云 (像方预测), shape (K, 2)

    Returns:
        torch.Tensor: 计算出的仿射损失标量。
    """
    # 确保有足够的点来求解仿射变换 (至少3个点)
    if src_points.shape[0] < 3:
        return torch.tensor(0.0, device=src_points.device, dtype=src_points.dtype)

    # 1. 构造增广矩阵 A 用于最小二乘问题 A * x = b
    #    对于每个点 (x, y), 我们有方程组:
    #    a*x + b*y + c = x'
    #    d*x + e*y + f = y'
    #    这里我们求解一个简化的6参数仿射变换
    ones = torch.ones(src_points.shape[0], 1, device=src_points.device, dtype=src_points.dtype)
    A = torch.cat([src_points, ones], dim=1)  # Shape: (K, 3)

    # 2. 使用 torch.linalg.lstsq 解出仿射参数 params
    #    A @ params = dst_points
    #    params 的形状将是 (3, 2)
    try:
        solution = torch.linalg.lstsq(A, dst_points)
        params = solution.solution
    except torch.linalg.LinAlgError:
        # 如果矩阵不可逆或出现其他线性代数错误，则不计算此损失
        return torch.tensor(0.0, device=src_points.device, dtype=src_points.dtype)


    # 3. 将求解出的变换应用回 src_points
    src_points_transformed = A @ params  # Shape: (K, 2)

    # 4. 计算变换后的源点云与原始源点云之间的均方误差
    #    这个损失惩罚了任何偏离单位矩阵的变换
    loss = F.mse_loss(src_points_transformed, src_points)
    
    return loss

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
    
    loss_h = torch.nn.functional.mse_loss(delta_pred_h, delta_gt_h)
    loss_v = torch.nn.functional.mse_loss(delta_pred_v, delta_gt_v)
    
    return loss_h + loss_v


class CriterionTrainGrid(nn.Module):
    """
    用于训练Grid中Mapper的核心损失函数。
    """
    def __init__(self, consistency_weight: float = 50.0, affine_weight: float = 1.0):
        super().__init__()
        # --- 初始化各项损失的权重和参数 ---
        self.consistency_weight = consistency_weight
        self.affine_weight = affine_weight
        self.height_weight = 10.0
        self.photo_weight = 1.0
        self.clamp_max = 1000
        self.bce = nn.BCEWithLogitsLoss()
        self.valid_score_weight = 10.0
        self.warmup_iters = 5

    def forward(self, epoch, max_epoch, pred_mu_absolute_patch, pred_log_sigma_patch, gt_absolute_patch, conf_patch, linesamp_patch, elements: list, element_indices: list, valid_score_patch, valid_labels, num_positive_samples):
        """
        损失函数的前向传播计算。
        """
        
        pred_mu_absolute_patch, pred_log_sigma_patch, gt_absolute_patch, conf_patch, linesamp_patch, valid_score_patch, valid_labels = \
            [i.to(torch.float32) for i in [pred_mu_absolute_patch, pred_log_sigma_patch, gt_absolute_patch, conf_patch, linesamp_patch, valid_score_patch, valid_labels]]
        
        progress = 1. * epoch / max_epoch

        # --- 1. Valid Score损失 (在所有样本上计算) ---
        loss_valid = self.bce(valid_score_patch, valid_labels)

        # 如果批次中没有正样本，则只返回valid_score损失
        if num_positive_samples == 0:
            total_loss = loss_valid * self.valid_score_weight
            loss_details = {'d': 0.0, 'obj': 0.0, 'h': 0.0, 'p': 0.0, 'c': 0.0, 'aff': 0.0, 'v': loss_valid.item()}
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

        # --- 2. 绝对地理坐标L2损失 (均方误差) ---
        error_squared = (pred_mu_absolute_pos - gt_absolute_pos) ** 2
        loss_obj = (error_squared[:, :2, ...].sum(dim=1, keepdim=True) * conf_weights).mean()
        loss_height = (error_squared[:, 2:3, ...] * conf_weights).mean()

        # --- 3. 像素空间重投影损失 & 像方仿射损失 ---
        loss_photo_total = torch.tensor(0.0, device=pred_mu_absolute_pos.device)
        loss_affine_total = torch.tensor(0.0, device=pred_mu_absolute_pos.device)
        unique_element_indices = np.unique(element_indices_pos)
        
        for element_idx in unique_element_indices:
            mask = torch.tensor([i == element_idx for i in element_indices_pos], device=pred_mu_absolute_pos.device)
            if not mask.any(): continue

            pred_mu_group = pred_mu_absolute_pos[mask]
            linesamp_gt_group = linesamp_pos[mask]
            
            rpc = elements[element_idx].rpc
            
            pred_xyh_flat = pred_mu_group.permute(0, 2, 3, 1).reshape(-1, 3)
            linesamp_gt_flat = linesamp_gt_group.permute(0, 2, 3, 1).reshape(-1, 2)

            latlon_pred_flat = mercator2lonlat(pred_xyh_flat[:, [1, 0]])
            linesamp_pred_flat = torch.stack(rpc.RPC_OBJ2PHOTO(latlon_pred_flat[:, 0], latlon_pred_flat[:, 1], pred_xyh_flat[:, 2]), dim=1)[:, [1, 0]]
            
            reprojection_error_pixels = torch.norm(linesamp_pred_flat - linesamp_gt_flat, dim=1)
            loss_photo_group = reprojection_error_pixels.mean()
            loss_photo_total += loss_photo_group

            # --- [核心修改] 新增: 计算并累加 loss_affine ---
            loss_affine_group = calculate_affine_loss_differentiable(linesamp_gt_flat, linesamp_pred_flat)
            loss_affine_total += loss_affine_group

        loss_photo = loss_photo_total / len(unique_element_indices) if len(unique_element_indices) > 0 else torch.tensor(0.0, device=pred_mu_absolute_pos.device)
        loss_affine = loss_affine_total / len(unique_element_indices) if len(unique_element_indices) > 0 else torch.tensor(0.0, device=pred_mu_absolute_pos.device)

        # --- 4. 物方空间几何一致性损失 ---
        loss_consistency = calculate_consistency_loss(pred_mu_absolute_pos, gt_absolute_pos)

        # --- 5. 组合总损失 (引入预热逻辑) ---
        if epoch < self.warmup_iters:
            loss_regression = loss_obj + loss_height * self.height_weight
            sigma_regularization = (pred_log_sigma_pos ** 2).mean() * 0.01 
            
            total_loss = (loss_regression + sigma_regularization + 
                          loss_consistency * self.consistency_weight + 
                          loss_photo * self.photo_weight + 
                          loss_affine * self.affine_weight + 
                          loss_valid * self.valid_score_weight)
            
            loss_distribution = torch.tensor(0.0)
        else:
            sigma_absolute_pos = torch.exp(pred_log_sigma_pos)
            term1 = ((gt_absolute_pos - pred_mu_absolute_pos) ** 2) / (2 * sigma_absolute_pos**2 + 1e-8)
            term2 = pred_log_sigma_pos
            loss_distribution = ((term1 + term2) * conf_weights).mean()
            
            total_loss = (loss_distribution + loss_obj + loss_height * self.height_weight + 
                          loss_photo * self.photo_weight + 
                          loss_consistency * self.consistency_weight + 
                          loss_affine * self.affine_weight +
                          loss_valid * self.valid_score_weight)
        
        loss_details = {
            'd': loss_distribution.item(),
            'obj': np.sqrt(loss_obj.item()),
            'h': loss_height.item(),
            'p': loss_photo.item(),
            'c': loss_consistency.item(),
            'aff': loss_affine.item(),
            'v': loss_valid.item()
        }

        return total_loss, loss_details

