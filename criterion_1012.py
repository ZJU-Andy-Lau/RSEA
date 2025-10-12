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
    delta_pred_h = pred_patch[..., :, 1:] - pred_patch[..., :, :-1]
    delta_gt_h = gt_patch[..., :, 1:] - gt_patch[..., :, :-1]
    
    delta_pred_v = pred_patch[..., 1:, :] - pred_patch[..., :-1, :]
    delta_gt_v = gt_patch[..., 1:, :] - gt_patch[..., :-1, :]
    
    loss_h = torch.nn.functional.l1_loss(delta_pred_h, delta_gt_h)
    loss_v = torch.nn.functional.l1_loss(delta_pred_v, delta_gt_v)
    
    return loss_h + loss_v


class CriterionTrainGrid(nn.Module):

    def __init__(self):
        super().__init__()
        self.consistency_weight = 0.1
        self.height_weight = 10.0
        self.clamp_max = 1000
        self.bce = nn.BCEWithLogitsLoss() # Use BCEWithLogitsLoss for numerical stability with raw model outputs
        self.valid_score_weight = 10.0 # Reduced weight

    def denormalize_coords(self, coords_normalized, center, scale):
        return coords_normalized * scale.view(1, 3, 1, 1) + center.view(1, 3, 1, 1)

    def forward(self, epoch, max_epoch, pred_mu_normalized_patch, pred_log_sigma_patch, gt_normalized_patch, conf_patch, linesamp_patch, rpc: RPCModelParameterTorch, valid_score_patch, valid_labels, num_positive_samples, block_center, block_scale):
        
        pred_mu_normalized_patch, pred_log_sigma_patch, gt_normalized_patch, conf_patch, linesamp_patch, valid_score_patch, valid_labels, block_center, block_scale = \
            [i.to(torch.float32) for i in [pred_mu_normalized_patch, pred_log_sigma_patch, gt_normalized_patch, conf_patch, linesamp_patch, valid_score_patch, valid_labels, block_center, block_scale]]
        
        progress = 1. * epoch / max_epoch

        # --- 1. Calculate Valid Score Loss on ALL samples ---
        loss_valid = self.bce(valid_score_patch, valid_labels)

        # --- Only calculate regression losses on POSITIVE samples ---
        if num_positive_samples == 0:
            total_loss = loss_valid * self.valid_score_weight
            loss_details = {'dist': 0.0, 'obj': 0.0, 'height': 0.0, 'photo': 0.0, 'consistency': 0.0, 'valid': loss_valid.item()}
            return total_loss, loss_details

        pred_mu_normalized_pos = pred_mu_normalized_patch[:num_positive_samples]
        pred_log_sigma_pos = pred_log_sigma_patch[:num_positive_samples]
        gt_normalized_pos = gt_normalized_patch[:num_positive_samples]
        conf_pos = conf_patch[:num_positive_samples]
        linesamp_pos = linesamp_patch[:num_positive_samples]

        # --- Convert confidences to loss weights ---
        conf_weights = conf_pos.clone()
        conf_weights[conf_weights > 0.5] = 0.5 + progress * 0.4
        conf_weights[conf_weights < 0.5] = 0.5 - progress * 0.4
        conf_weights = torch.clip(conf_weights - conf_weights.mean() + 1., min=0.)

        # --- 2. Core Loss: Negative Log-Likelihood on NORMALIZED coordinates ---
        sigma_pos = torch.exp(pred_log_sigma_pos)
        term1 = ((gt_normalized_pos - pred_mu_normalized_pos) ** 2) / (2 * sigma_pos**2 + 1e-8)
        term2 = pred_log_sigma_pos
        loss_distribution = ((term1 + term2) * conf_weights).mean()

        # --- 3. Auxiliary Loss: Absolute Position Loss on NORMALIZED coordinates ---
        loss_obj_unreduced = torch.norm(pred_mu_normalized_pos[:, :2, ...] - gt_normalized_pos[:, :2, ...], dim=1, keepdim=True)
        loss_obj = (loss_obj_unreduced * conf_weights).mean()

        loss_height_unreduced = torch.abs(pred_mu_normalized_pos[:, 2, ...] - gt_normalized_pos[:, 2, ...]).unsqueeze(1)
        loss_height = (loss_height_unreduced * conf_weights).mean()

        # --- 4. Auxiliary Loss: Reprojection Loss on ABSOLUTE coordinates ---
        pred_mu_absolute_pos = self.denormalize_coords(pred_mu_normalized_pos, block_center, block_scale)
        
        N_pos, _, ph, pw = pred_mu_absolute_pos.shape
        pred_xyh_flat = pred_mu_absolute_pos.permute(0, 2, 3, 1).reshape(-1, 3)
        latlon_pred_flat = mercator2lonlat(pred_xyh_flat[:, [1, 0]])
        linesamp_pred_flat = torch.stack(rpc.RPC_OBJ2PHOTO(latlon_pred_flat[:, 0], latlon_pred_flat[:, 1], pred_xyh_flat[:, 2]), dim=1)[:, [1, 0]]
        
        linesamp_gt_flat = linesamp_pos.permute(0,2,3,1).reshape(-1, 2)
        conf_flat = conf_weights.permute(0, 2, 3, 1).reshape(-1, 1)
        reprojection_error = torch.norm(linesamp_pred_flat - linesamp_gt_flat, dim=1)
        
        w = np.sqrt(1 - progress**2)
        t = w * self.clamp_max + 1
        loss_photo = (t * torch.tanh(reprojection_error / t) * conf_flat.squeeze()).mean()

        # --- 5. Auxiliary Loss: Consistency Loss on NORMALIZED coordinates ---
        loss_consistency = calculate_consistency_loss(pred_mu_normalized_pos, gt_normalized_pos)

        # --- 6. Combine total loss ---
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

