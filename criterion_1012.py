import torch.nn as nn 
import torch 
import torch.nn.functional as F
import numpy as np
import math
from rpc import RPCModelParameterTorch
from utils import project_mercator,mercator2lonlat
import time
from typing import List

def calculate_consistency_loss(pred_patch, gt_patch):
    """
    计算patch内的坐标一致性损失
    Args:
        pred_patch (torch.Tensor): 预测的地理坐标均值, shape (N, 3, ph, pw)
        gt_patch (torch.Tensor): 真实的地理坐标, shape (N, 3, ph, pw)
    """
    # 水平方向的相对位移向量
    delta_pred_h = pred_patch[..., :, 1:] - pred_patch[..., :, :-1]
    delta_gt_h = gt_patch[..., :, 1:] - gt_patch[..., :, :-1]
    
    # 垂直方向的相对位移向量
    delta_pred_v = pred_patch[..., 1:, :] - pred_patch[..., :-1, :]
    delta_gt_v = gt_patch[..., 1:, :] - gt_patch[..., :-1, :]
    
    # 计算水平和垂直方向上的L1损失
    loss_h = torch.nn.functional.l1_loss(delta_pred_h, delta_gt_h)
    loss_v = torch.nn.functional.l1_loss(delta_pred_v, delta_gt_v)
    
    return loss_h + loss_v


class CriterionTrainGrid(nn.Module):

    def __init__(self):
        super().__init__()
        self.consistency_weight = 0.1
        self.height_weight = 10.0
        self.clamp_max = 1000
        self.bce = nn.BCELoss()
        self.valid_score_weight = 100.0

    def forward(self, epoch, max_epoch, pred_mu_patch, pred_log_sigma_patch, gt_patch, conf_patch, linesamp_patch, rpc: RPCModelParameterTorch, valid_score_patch, valid_labels, num_positive_samples):
        
        # 确保数据类型正确
        pred_mu_patch, pred_log_sigma_patch, gt_patch, conf_patch, linesamp_patch, valid_score_patch, valid_labels = \
            [i.to(torch.float32) for i in [pred_mu_patch, pred_log_sigma_patch, gt_patch, conf_patch, linesamp_patch, valid_score_patch, valid_labels]]
        
        progress = 1. * epoch / max_epoch

        # --- 只对正样本计算回归相关的损失 ---
        pred_mu_pos = pred_mu_patch[:num_positive_samples]
        pred_log_sigma_pos = pred_log_sigma_patch[:num_positive_samples]
        gt_pos = gt_patch[:num_positive_samples]
        conf_pos = conf_patch[:num_positive_samples]
        linesamp_pos = linesamp_patch[:num_positive_samples]

        # --- 根据原始逻辑处理置信度，将其转换为损失权重 ---
        conf_weights = conf_pos.clone()
        conf_weights[conf_weights > 0.5] = 0.5 + progress * 0.4
        conf_weights[conf_weights < 0.5] = 0.5 - progress * 0.4
        conf_weights = torch.clip(conf_weights - conf_weights.mean() + 1., min=0.)

        # --- 1. 核心损失：概率分布损失 (Negative Log-Likelihood) ---
        sigma_pos = torch.exp(pred_log_sigma_pos)
        term1 = ((gt_pos - pred_mu_pos) ** 2) / (2 * sigma_pos**2 + 1e-8)
        term2 = pred_log_sigma_pos
        loss_distribution = ((term1 + term2) * conf_weights).mean()

        # --- 2. 辅助损失：主要目标损失 (Absolute Position Loss) ---
        loss_obj_unreduced = torch.norm(pred_mu_pos[:, :2, ...] - gt_pos[:, :2, ...], dim=1, keepdim=True)
        loss_obj = (loss_obj_unreduced * conf_weights).mean()

        loss_height_unreduced = torch.abs(pred_mu_pos[:, 2, ...] - gt_pos[:, 2, ...]).unsqueeze(1)
        loss_height = (loss_height_unreduced * conf_weights).mean()

        # --- 3. 辅助损失：光度/投影损失 (Photometric/Reprojection Loss) ---
        N_pos, _, ph, pw = pred_mu_pos.shape
        pred_xyh_flat = pred_mu_pos.permute(0, 2, 3, 1).reshape(-1, 3)
        latlon_pred_flat = mercator2lonlat(pred_xyh_flat[:, [1, 0]])
        linesamp_pred_flat = torch.stack(rpc.RPC_OBJ2PHOTO(latlon_pred_flat[:, 0], latlon_pred_flat[:, 1], pred_xyh_flat[:, 2]), dim=1)[:, [1, 0]]
        linesamp_gt_flat = linesamp_pos.reshape(-1, 2)
        conf_flat = conf_weights.permute(0, 2, 3, 1).reshape(-1, 1)
        reprojection_error = torch.norm(linesamp_pred_flat - linesamp_gt_flat, dim=1)
        w = np.sqrt(1 - progress**2)
        t = w * self.clamp_max + 1
        loss_photo = (t * torch.tanh(reprojection_error / t) * conf_flat.squeeze()).mean()

        # --- 4. 辅助损失：坐标一致性损失 (Consistency Loss) ---
        loss_consistency = calculate_consistency_loss(pred_mu_pos, gt_pos)

        # --- 5. 新增损失：有效分数的二元交叉熵损失 (Valid Score Loss) ---
        loss_valid = self.bce(valid_score_patch, valid_labels)

        # --- 6. 组合总损失 ---
        total_loss = loss_distribution + loss_obj + loss_height * self.height_weight + loss_photo + self.consistency_weight * loss_consistency + loss_valid * self.valid_score_weight
        
        loss_details = {
            'dist': loss_distribution.item(),
            'obj': loss_obj.item(),
            'height': loss_height.item(),
            'photo': loss_photo.item(),
            'consistency': loss_consistency.item(),
            'valid': loss_valid.item()
        }

        return total_loss, loss_details

