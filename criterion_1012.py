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
        pred_patch (torch.Tensor): 预测的地理坐标, shape (N, 3, ph, pw)
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
        self.bce = nn.BCELoss()
        self.clamp_max = 1000

    def forward(self, epoch, max_epoch, pred_patch, gt_patch, conf_patch, linesamp_patch, rpc: RPCModelParameterTorch):
        
        # 确保数据类型正确
        pred_patch, gt_patch, conf_patch, linesamp_patch = [i.to(torch.float32) for i in [pred_patch, gt_patch, conf_patch, linesamp_patch]]
        
        progress = 1. * epoch / max_epoch

        # 1. 主要目标损失 (Absolute Position Loss)
        # 直接计算预测坐标和真实坐标之间的L1距离
        # 乘以置信度作为权重
        loss_obj = torch.nn.functional.l1_loss(pred_patch, gt_patch, reduction='none')
        loss_obj = (loss_obj * conf_patch).mean()

        # 2. 坐标一致性损失 (Consistency Loss)
        loss_consistency = calculate_consistency_loss(pred_patch, gt_patch)

        # 3. 光度/投影损失 (Photometric/Reprojection Loss)
        # 将预测的地理坐标批量反投影回影像坐标
        N, _, ph, pw = pred_patch.shape
        # (N, 3, ph, pw) -> (N*ph*pw, 3)
        pred_xyh_flat = pred_patch.permute(0, 2, 3, 1).reshape(-1, 3)
        
        # mercator to latlon
        latlon_pred_flat = mercator2lonlat(pred_xyh_flat[:, [1, 0]])
        
        # RPC back projection
        linesamp_pred_flat = torch.stack(rpc.RPC_OBJ2PHOTO(latlon_pred_flat[:, 0], latlon_pred_flat[:, 1], pred_xyh_flat[:, 2]), dim=1)[:, [1, 0]]
        
        # (N, ph, pw, 2) -> (N*ph*pw, 2)
        linesamp_gt_flat = linesamp_patch.reshape(-1, 2)
        
        # (N, 1, ph, pw) -> (N*ph*pw, 1)
        conf_flat = conf_patch.permute(0, 2, 3, 1).reshape(-1, 1)

        # 计算反投影误差
        reprojection_error = torch.norm(linesamp_pred_flat - linesamp_gt_flat, dim=1)
        
        # 使用tanh_clamp平滑损失
        w = np.sqrt(1 - progress**2)
        t = w * self.clamp_max + 1
        loss_photo = (t * torch.tanh(reprojection_error / t) * conf_flat.squeeze()).mean()


        # 4. 总损失
        total_loss = loss_obj + self.consistency_weight * loss_consistency + loss_photo
        
        loss_details = {
            'obj': loss_obj.item(),
            'consistency': loss_consistency.item(),
            'photo': loss_photo.item()
        }

        return total_loss, loss_details
