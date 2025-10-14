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
    if src_points.shape[0] < 3:
        return torch.tensor(0.0, device=src_points.device, dtype=src_points.dtype)

    ones = torch.ones(src_points.shape[0], 1, device=src_points.device, dtype=src_points.dtype)
    A = torch.cat([src_points, ones], dim=1)

    try:
        solution = torch.linalg.lstsq(A, dst_points)
        params = solution.solution
    except torch.linalg.LinAlgError:
        return torch.tensor(0.0, device=src_points.device, dtype=src_points.dtype)

    src_points_transformed = A @ params
    loss = F.mse_loss(src_points_transformed, src_points)
    
    return loss

def calculate_consistency_loss(pred_patch, gt_patch):
    """
    计算patch内的坐标一致性损失 (在绝对地理坐标空间中计算)。
    """
    delta_pred_h = pred_patch[..., :, 1:] - pred_patch[..., :, :-1]
    delta_gt_h = gt_patch[..., :, 1:] - gt_patch[..., :, :-1]
    
    delta_pred_v = pred_patch[..., 1:, :] - pred_patch[..., :-1, :]
    delta_gt_v = gt_patch[..., 1:, :] - gt_patch[..., :-1, :]
    
    loss_h = torch.nn.functional.mse_loss(delta_pred_h, delta_gt_h)
    loss_v = torch.nn.functional.mse_loss(delta_pred_v, delta_gt_v)
    
    return loss_h + loss_v


class CriterionTrainGrid(nn.Module):
    """
    [核心修改] 用于训练Grid中Mapper的核心损失函数。
    职责被简化，只计算与坐标回归相关的损失。
    """
    def __init__(self, consistency_weight: float = 50.0, affine_weight: float = 1.0):
        super().__init__()
        self.consistency_weight = consistency_weight
        self.affine_weight = affine_weight
        self.height_weight = 10.0
        self.photo_weight = 1.0
        self.warmup_iters = 5

    def forward(self, epoch, max_epoch, pred_mu_absolute_patch, pred_log_sigma_patch, gt_absolute_patch, conf_patch, linesamp_patch, elements: list, element_indices: list):
        """
        损失函数的前向传播计算 (只处理正样本)。
        """
        
        progress = 1. * epoch / max_epoch

        conf_weights = conf_patch.clone()
        conf_weights[conf_weights > 0.5] = 0.5 + progress * 0.4
        conf_weights[conf_weights < 0.5] = 0.5 - progress * 0.4
        conf_weights = torch.clip(conf_weights - conf_weights.mean() + 1., min=0.)

        error_squared = (pred_mu_absolute_patch - gt_absolute_patch) ** 2
        loss_obj = (error_squared[:, :2, ...].sum(dim=1, keepdim=True) * conf_weights).mean()
        loss_height = (error_squared[:, 2:3, ...] * conf_weights).mean()

        loss_photo_total = torch.tensor(0.0, device=pred_mu_absolute_patch.device)
        loss_affine_total = torch.tensor(0.0, device=pred_mu_absolute_patch.device)
        unique_element_indices = np.unique(element_indices)
        
        for element_idx in unique_element_indices:
            mask = torch.tensor([i == element_idx for i in element_indices], device=pred_mu_absolute_patch.device)
            if not mask.any(): continue

            pred_mu_group = pred_mu_absolute_patch[mask]
            linesamp_gt_group = linesamp_patch[mask]
            
            rpc = elements[element_idx].rpc
            
            pred_xyh_flat = pred_mu_group.permute(0, 2, 3, 1).reshape(-1, 3)
            linesamp_gt_flat = linesamp_gt_group.permute(0, 2, 3, 1).reshape(-1, 2)

            latlon_pred_flat = mercator2lonlat(pred_xyh_flat[:, [1, 0]])
            linesamp_pred_flat = torch.stack(rpc.RPC_OBJ2PHOTO(latlon_pred_flat[:, 0], latlon_pred_flat[:, 1], pred_xyh_flat[:, 2]), dim=1)[:, [1, 0]].to(torch.float32)
            
            reprojection_error_pixels = torch.norm(linesamp_pred_flat - linesamp_gt_flat, dim=1)
            loss_photo_group = reprojection_error_pixels.mean()
            loss_photo_total += loss_photo_group

            loss_affine_group = calculate_affine_loss_differentiable(linesamp_gt_flat, linesamp_pred_flat)
            loss_affine_total += loss_affine_group

        loss_photo = loss_photo_total / len(unique_element_indices) if len(unique_element_indices) > 0 else torch.tensor(0.0, device=pred_mu_absolute_patch.device)
        loss_affine = loss_affine_total / len(unique_element_indices) if len(unique_element_indices) > 0 else torch.tensor(0.0, device=pred_mu_absolute_patch.device)

        loss_consistency = calculate_consistency_loss(pred_mu_absolute_patch, gt_absolute_patch)

        if epoch < self.warmup_iters:
            loss_regression = loss_obj + loss_height * self.height_weight
            sigma_regularization = (pred_log_sigma_patch ** 2).mean() * 0.01 
            
            total_loss = (loss_regression + sigma_regularization + 
                          loss_consistency * self.consistency_weight + 
                          loss_photo * self.photo_weight + 
                          loss_affine * self.affine_weight)
            
            loss_distribution = torch.tensor(0.0)
        else:
            sigma_absolute_pos = torch.exp(pred_log_sigma_patch)
            term1 = ((gt_absolute_patch - pred_mu_absolute_patch) ** 2) / (2 * sigma_absolute_pos**2 + 1e-8)
            term2 = pred_log_sigma_patch
            loss_distribution = ((term1 + term2) * conf_weights).mean()
            
            total_loss = (loss_distribution + loss_obj + loss_height * self.height_weight + 
                          loss_photo * self.photo_weight + 
                          loss_consistency * self.consistency_weight + 
                          loss_affine * self.affine_weight)
        
        loss_details = {
            'd': loss_distribution.item(),
            'obj': np.sqrt(loss_obj.item()),
            'h': loss_height.item(),
            'p': loss_photo.item(),
            'c': loss_consistency.item(),
            'aff': loss_affine.item()
        }

        return total_loss, loss_details

