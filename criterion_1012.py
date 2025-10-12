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
    计算patch内的坐标一致性损失 (在绝对地理坐标空间中计算)。
    这个损失函数旨在惩罚模型预测的局部几何变形。
    它要求在一个patch内部，相邻点之间的相对位移向量在预测结果和真实结果中应该保持一致。

    Args:
        pred_patch (torch.Tensor): 预测的地理坐标均值, shape (N, 3, ph, pw)
        gt_patch (torch.Tensor): 真实的地理坐标, shape (N, 3, ph, pw)
    """
    # --- 计算水平方向上相邻点之间的位移向量 ---
    # pred_patch[..., :, 1:] 表示所有patch在宽度维度上从第二个像素到最后一个像素的部分
    # pred_patch[..., :, :-1] 表示所有patch在宽度维度上从第一个像素到倒数第二个像素的部分
    # 两者相减，得到每个点与其右侧相邻点的位移向量
    delta_pred_h = pred_patch[..., :, 1:] - pred_patch[..., :, :-1]
    delta_gt_h = gt_patch[..., :, 1:] - gt_patch[..., :, :-1]
    
    # --- 计算垂直方向上相邻点之间的位移向量 ---
    # 同理，计算每个点与其下方相邻点的位移向量
    delta_pred_v = pred_patch[..., 1:, :] - pred_patch[..., :-1, :]
    delta_gt_v = gt_patch[..., 1:, :] - gt_patch[..., :-1, :]
    
    # --- 计算L1损失 ---
    # L1损失（绝对值误差）惩罚预测位移向量与真实位移向量之间的差异
    loss_h = torch.nn.functional.l1_loss(delta_pred_h, delta_gt_h)
    loss_v = torch.nn.functional.l1_loss(delta_pred_v, delta_gt_v)
    
    # 返回水平和垂直方向损失之和
    return loss_h + loss_v


class CriterionTrainGrid(nn.Module):
    """
    用于训练Grid中Mapper的核心损失函数。
    这个类严格遵循您原始代码的设计思路，所有损失都在绝对地理坐标（米）或像素空间中进行计算。
    它整合了多种损失项来从不同角度监督模型的学习。
    """
    def __init__(self):
        super().__init__()
        # --- 初始化各项损失的权重和参数 ---
        self.consistency_weight = 0.5      # 坐标一致性损失的权重
        self.height_weight = 10.0          # 高程损失的权重 (与原始代码保持一致)
        self.clamp_max = 1000              # 用于tanh_clamp函数，限制重投影损失的最大值，防止梯度爆炸
        self.bce = nn.BCEWithLogitsLoss()  # 用于计算valid_score的二元交叉熵损失
        self.valid_score_weight = 10.0     # valid_score损失的权重

    def forward(self, epoch, max_epoch, pred_mu_absolute_patch, pred_log_sigma_patch, gt_absolute_patch, conf_patch, linesamp_patch, rpc: RPCModelParameterTorch, valid_score_patch, valid_labels, num_positive_samples):
        """
        损失函数的前向传播计算。
        Args:
            epoch (int): 当前训练轮次。
            max_epoch (int): 总训练轮次。
            pred_mu_absolute_patch (torch.Tensor): 模型预测的绝对地理坐标均值。
            pred_log_sigma_patch (torch.Tensor): 模型预测的绝对地理坐标log(标准差)。
            gt_absolute_patch (torch.Tensor): 真实的绝对地理坐标。
            conf_patch (torch.Tensor): Encoder提取的特征置信度。
            linesamp_patch (torch.Tensor): 真实的影像行列号坐标。
            rpc (RPCModelParameterTorch): RPC相机模型。
            valid_score_patch (torch.Tensor): 模型预测的有效性分数（logits）。
            valid_labels (torch.Tensor): 真实有效性标签 (0或1)。
            num_positive_samples (int): 批次中正样本的数量。
        """
        
        # 确保所有输入张量的数据类型为float32，以保证计算稳定性
        pred_mu_absolute_patch, pred_log_sigma_patch, gt_absolute_patch, conf_patch, linesamp_patch, valid_score_patch, valid_labels = \
            [i.to(torch.float32) for i in [pred_mu_absolute_patch, pred_log_sigma_patch, gt_absolute_patch, conf_patch, linesamp_patch, valid_score_patch, valid_labels]]
        
        # 计算当前训练进度，用于调整置信度权重和tanh_clamp
        progress = 1. * epoch / max_epoch

        # --- 损失计算第一部分: Valid Score损失 (在所有样本上计算) ---
        # valid_score用于判断一个特征是否属于当前Block的地理范围，是一个二分类任务。
        # BCEWithLogitsLoss结合了Sigmoid和BCELoss，数值上更稳定。
        loss_valid = self.bce(valid_score_patch, valid_labels)

        # --- 损失计算第二部分: 回归损失 (只在正样本上计算) ---
        # 如果批次中没有正样本，则只返回valid_score损失
        if num_positive_samples == 0:
            total_loss = loss_valid * self.valid_score_weight
            loss_details = {'dist': 0.0, 'obj': 0.0, 'height': 0.0, 'photo': 0.0, 'consistency': 0.0, 'valid': loss_valid.item()}
            return total_loss, loss_details

        # 从批次中分离出所有正样本的数据
        pred_mu_absolute_pos = pred_mu_absolute_patch[:num_positive_samples]
        pred_log_sigma_pos = pred_log_sigma_patch[:num_positive_samples]
        gt_absolute_pos = gt_absolute_patch[:num_positive_samples]
        conf_pos = conf_patch[:num_positive_samples]
        linesamp_pos = linesamp_patch[:num_positive_samples]

        # --- 将Encoder的特征置信度转换为损失权重 ---
        # 这种权重调整策略使得在训练初期更信任高置信度的点，随着训练深入，权重趋向于均匀。
        conf_weights = conf_pos.clone()
        conf_weights[conf_weights > 0.5] = 0.5 + progress * 0.4
        conf_weights[conf_weights < 0.5] = 0.5 - progress * 0.4
        conf_weights = torch.clip(conf_weights - conf_weights.mean() + 1., min=0.) # 归一化处理

        # --- 2. 核心损失: 负对数似然损失 (Negative Log-Likelihood Loss) ---
        # 这是坐标回归的主要损失。它同时惩罚均值的偏差和方差的估计。
        # 模型被鼓励在预测不准的地方输出更大的方差。
        sigma_absolute_pos = torch.exp(pred_log_sigma_pos)
        # 似然损失的第一项：(真实值-预测值)^2 / (2*方差^2)
        term1 = ((gt_absolute_pos - pred_mu_absolute_pos) ** 2) / (2 * sigma_absolute_pos**2 + 1e-8)
        # 似然损失的第二项：log(方差)
        term2 = pred_log_sigma_pos # 注意：这里使用log(sigma)而不是log(sigma^2)，与原始代码保持一致
        # 应用置信度权重并求平均
        loss_distribution = ((term1 + term2) * conf_weights).mean()

        # --- 3. 辅助损失: 绝对地理坐标L1损失 ---
        # 直接惩罚预测坐标与真实坐标之间的L1距离（米）
        error_absolute = torch.abs(pred_mu_absolute_pos - gt_absolute_pos)
        # 分别计算水平(obj)和高程(height)的损失
        loss_obj = (torch.norm(error_absolute[:, :2, ...], dim=1, keepdim=True) * conf_weights).mean()
        loss_height = (error_absolute[:, 2:3, ...] * conf_weights).mean()

        # --- 4. 辅助损失: 像素空间重投影损失 ---
        # 将预测的地理坐标通过RPC模型反向投影回影像，计算与原始像素位置的误差。
        N_pos, _, ph, pw = pred_mu_absolute_pos.shape
        # 为了高效计算，先将patch展平
        pred_xyh_flat = pred_mu_absolute_pos.permute(0, 2, 3, 1).reshape(-1, 3)
        # 地理坐标 -> 经纬度 -> 影像行列号
        latlon_pred_flat = mercator2lonlat(pred_xyh_flat[:, [1, 0]])
        linesamp_pred_flat = torch.stack(rpc.RPC_OBJ2PHOTO(latlon_pred_flat[:, 0], latlon_pred_flat[:, 1], pred_xyh_flat[:, 2]), dim=1)[:, [1, 0]]
        
        linesamp_gt_flat = linesamp_pos.permute(0,2,3,1).reshape(-1, 2)
        conf_flat = conf_weights.permute(0, 2, 3, 1).reshape(-1, 1)
        
        # 计算重投影误差（像素）
        reprojection_error_pixels = torch.norm(linesamp_pred_flat - linesamp_gt_flat, dim=1)
        
        # 使用tanh_clamp函数来限制损失值，防止因个别离谱的预测点导致梯度爆炸
        w = np.sqrt(1 - progress**2)
        t = w * self.clamp_max + 1
        loss_photo = (t * torch.tanh(reprojection_error_pixels / t) * conf_flat.squeeze()).mean()

        # --- 5. 辅助损失: 坐标一致性损失 ---
        # 计算patch内部的局部几何一致性损失
        loss_consistency = calculate_consistency_loss(pred_mu_absolute_pos, gt_absolute_pos)

        # --- 6. 组合总损失 ---
        # 将所有损失项按照各自的权重相加
        total_loss = loss_distribution + loss_obj + loss_height * self.height_weight + loss_photo + self.consistency_weight * loss_consistency + loss_valid * self.valid_score_weight
        
        # 构建一个包含各分项损失的字典，用于日志打印
        loss_details = {
            'dist': loss_distribution.item(),
            'obj': loss_obj.item(),
            'height': loss_height.item(),
            'photo': loss_photo.item(),
            'consistency': loss_consistency.item(),
            'valid': loss_valid.item()
        }

        return total_loss, loss_details

