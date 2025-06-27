import torch.nn as nn 
import torch 
import torch.nn.functional as F
import numpy as np
import math
from rpc import RPCModelParameterTorch
from utils import project_mercator,mercator2lonlat
import time
from typing import List

class SmoothL1Loss(nn.Module):
    def __init__(self, beta=1.0):
        super(SmoothL1Loss, self).__init__()
        self.beta = beta

    def forward(self, input, target):
        diff = torch.norm(input - target,dim=1)

        loss = torch.where(diff < self.beta,
                           0.5 * diff ** 2 / self.beta,
                           diff - 0.5 * self.beta)

        return loss

def tanh_clamp(loss,r,max):
    w = np.sqrt(1 - r**2)
    t = w * max + 1
    return t * torch.tanh(loss / t)

def get_dis_matrix(points_a:torch.Tensor,points_b:torch.Tensor):
    # return torch.sqrt(torch.sum((points_a.unsqueeze(1) - points_b.unsqueeze(0)) ** 2,dim=-1) + 1e-6)
    return torch.cdist(points_a,points_b,p=2)

@torch.no_grad()
def get_near_points(points1,points2,batch_size,threshold):
    point_num = len(points1)
    min_dis = torch.full((point_num,),1e9,device=points1.device,dtype=points1.dtype)
    min_dis_idx = torch.full((point_num,),-1,device=points1.device,dtype=int)
    batch_num = int(np.ceil(point_num / batch_size))
    for b1 in range(batch_num):
        for b2 in range(batch_num):
            dis = torch.cdist(points1[b1 * batch_size : (b1 + 1) * batch_size],points2[b2 * batch_size : (b2 + 1) * batch_size])
            # print("diag:",dis.diag(),torch.max(dis.diag()),dis[-1,260])
            min_dis_batch,min_dis_idx_batch = torch.min(dis,dim=1)
            # print("min_dis_batch:",min_dis_batch,torch.max(min_dis_batch))
            # print("idx_batch:",min_dis_idx_batch)
            # print("min_idx_dis:",dis[torch.arange(dis.shape[0]),min_dis_idx_batch])
            # print(torch.stack([min_dis_batch,min_dis_idx_batch],dim=-1))

            update_mask = min_dis_batch < min_dis[b1 * batch_size : (b1 + 1) * batch_size]

            min_dis_idx[b1 * batch_size : (b1 + 1) * batch_size][update_mask] = min_dis_idx_batch[update_mask] + b2 * batch_size
            min_dis[b1 * batch_size : (b1 + 1) * batch_size][update_mask] = min_dis_batch[update_mask]

    valid_mask = min_dis < threshold
    # print(len(valid_mask),valid_mask.sum())
    anchor_points = torch.arange(point_num,device=points1.device,dtype=int)[valid_mask]
    positive_points = min_dis_idx[valid_mask]
    return anchor_points,positive_points


@torch.no_grad()
def get_far_points(anchor_points,candidates,threshold=100):
    anchor_points_num = len(anchor_points)
    sample_idx = torch.randperm(len(candidates),device=candidates.device)[:anchor_points_num]
    while len(sample_idx) < anchor_points_num:
        sample_idx = torch.cat([sample_idx,torch.randperm(len(candidates),device=candidates.device)[:anchor_points_num - len(sample_idx)]])
    dis = torch.norm(anchor_points - candidates[sample_idx],dim=1)
    invalid_idx = torch.where(dis<threshold)[0]
    count = 0
    while(len(invalid_idx) > 0 and count < 3):
        append_idx = torch.randperm(len(candidates),device=candidates.device)[:len(invalid_idx)]
        while len(append_idx) < len(invalid_idx):
            append_idx = torch.cat([append_idx,torch.randperm(len(candidates),device=candidates.device)[:len(invalid_idx) - len(append_idx)]])
        sample_idx[invalid_idx] = append_idx
        dis = torch.norm(anchor_points - candidates[sample_idx],dim=1)
        invalid_idx = torch.where(dis<threshold)[0]
        count += 1
    return sample_idx


def affine_loss(local:torch.Tensor,pred:torch.Tensor,conf:torch.Tensor):
    def affine_trans(ori, dst):
        # 添加齐次坐标
        ones = torch.ones(ori.shape[0], 1, device=ori.device)
        ori_homogeneous = torch.cat([ori, ones], dim=1)  # (N, 3)
        
        # 构建并求解线性系统以估计仿射变换
        X = ori_homogeneous  # (N, 3)
        B = dst  # (N, 2)
        
        # 计算伪逆: (X^T·X)^(-1)·X^T
        XTX = torch.matmul(X.transpose(0, 1), X)  # (3, 3)
        XTX_inv = torch.inverse(XTX)  # (3, 3)
        X_pinv = torch.matmul(XTX_inv, X.transpose(0, 1))  # (3, N)
        
        # 计算变换矩阵: (3, 2)，对应于 [A|b]^T
        transform_matrix = torch.matmul(X_pinv, B)  # (3, 2)
        
        # 应用变换到原始点集
        ori_trans = torch.matmul(ori_homogeneous, transform_matrix)  # (N, 2)
        
        return ori_trans
    
    dis = torch.norm(local + (torch.mean(pred,dim=0)[None] - torch.mean(local,dim=0)[None]) - pred,dim=-1)
    valid_idx = (conf > .5) & (dis < dis.mean() + dis.std())
    local = local[valid_idx]
    pred = pred[valid_idx]
    
    reg = affine_trans(local,pred)
    loss_affine = torch.norm(reg - local,dim=-1).mean()
    return loss_affine


    # center_local = torch.sum(conf[valid_idx,None] * local[valid_idx],dim=0) / torch.sum(conf[valid_idx])
    # center_pred = torch.sum(conf[valid_idx,None] * pred[valid_idx],dim=0) / torch.sum(conf[valid_idx])
    # local_centered = local - center_local
    # pred_centerd = pred - center_pred
    # H = torch.matmul(local_centered.T * conf,pred_centerd)
    # U,_,Vt = torch.svd(H)
    # R = torch.matmul(U,Vt.T)
    # if torch.det(R) < 0:
    #     R[:,-1] *= -1
    # t = center_pred - torch.matmul(center_local,R)
    # pred_regularized = torch.matmul(local,R) + t
    # # dis = center_pred - center_local
    # # pred_regularized = local + dis
    # return pred_regularized

def conf_norm(conf):
    conf = conf.clone().detach()
    conf_norm = conf - conf.mean() + 1.
    return torch.clip(conf_norm,min=0.)

def residual2conf(residual,t = 6.):
    conf = torch.full(residual.shape,.5,device=residual.device,dtype=residual.dtype)
    conf[residual > t] = .1
    conf[residual < t] = .9
    conf[torch.isnan(residual)] = .5
    return conf

class CriterionFinetuneNormal(nn.Module):
    def __init__(self):
        super().__init__()
        self.residual_thresholds = 5.
        self.gamma = .05
        self.bce = nn.BCELoss()

    def forward(self,epoch,
                feat1_PD,feat2_PD,
                pred1_P3,pred2_P3,
                conf1_P,conf2_P,
                obj_P3,
                residual1_P,residual2_P,
                H,W
                ):
        
        P = H*W

        res_mid = torch.median(torch.cat([residual1_P,residual2_P])[~torch.isnan(torch.cat([residual1_P,residual2_P]))])
        if not torch.isnan(res_mid):
            self.residual_thresholds = (1. - self.gamma) * self.residual_thresholds + self.gamma * res_mid
        else:
            print("all res nan")
        conf1_gt_P = residual2conf(residual1_P,self.residual_thresholds)
        conf2_gt_P = residual2conf(residual2_P,self.residual_thresholds)
        robust_mask = (residual1_P <= self.residual_thresholds) & (residual2_P <= self.residual_thresholds)

        conf_valid1 = ~torch.isnan(residual1_P)
        conf_valid2 = ~torch.isnan(residual2_P)
        weights1_P = conf_norm(conf1_gt_P)
        weights2_P = conf_norm(conf2_gt_P)

        loss_obj = (.5 * (torch.norm(pred1_P3[:,:2] - obj_P3[:,:2],dim=-1) * weights1_P) + .5 * (torch.norm(pred2_P3[:,:2] - obj_P3[:,:2],dim=-1) * weights2_P)).mean()
        loss_height = ((.5 * (torch.abs(pred1_P3[:,2] - obj_P3[:,2]) * weights1_P) + .5 * (torch.abs(pred2_P3[:,2] - obj_P3[:,2]) * weights2_P))).mean() * 100

        loss_conf = (.5 * torch.abs(conf1_P[conf_valid1] - conf1_gt_P[conf_valid1]) + .5 * torch.abs(conf2_P[conf_valid2] - conf2_gt_P[conf_valid2])).mean()
        loss_conf *= 1000 * min(1.,epoch / 3.)

        shift_amount = torch.randint(low=-P // 2,high = P // 2,size=(1,))[0].item()
        feat1_PD = F.normalize(feat1_PD,dim=1)
        feat2_PD = F.normalize(feat2_PD,dim=1)
        simi_positive = torch.concatenate([torch.sum(feat1_PD[robust_mask] * feat2_PD[robust_mask],dim=1),
                                           torch.sum(feat2_PD[robust_mask] * feat1_PD[robust_mask],dim=1)])
        simi_negative = torch.concatenate([torch.sum(feat1_PD[robust_mask] * torch.roll(feat2_PD,shift_amount)[robust_mask],dim=1),
                                           torch.sum(feat2_PD[robust_mask] * torch.roll(feat1_PD,shift_amount)[robust_mask],dim=1)])
        margin = .7
        loss_feat = torch.clip(simi_negative - simi_positive + margin,min=0.).mean() * 10000

        loss = loss_obj + loss_height + loss_conf + loss_feat

        return loss,loss_obj,loss_height,loss_conf,loss_feat,self.residual_thresholds 
        
class CriterionFinetuneDis(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self,
                pred1_P3,pred2_P3,
                residual1_P,residual2_P,
                residual_thresholds,
                ):
        
        robust_mask = (residual1_P <= residual_thresholds) & (residual2_P <= residual_thresholds)

        dis_obj = torch.norm(pred1_P3[robust_mask,:2] - pred2_P3[robust_mask,:2],dim=-1).mean()
        dis_height = torch.abs(pred1_P3[robust_mask,2] - pred2_P3[robust_mask,2]).mean() * 100

        loss_dis = dis_obj + dis_height
        return loss_dis,dis_obj,dis_height

        
    
class CriterionTrainOneImg(nn.Module):

    def __init__(self):
        super().__init__()
        self.height_tolerance = 5
        self.clamp_max = 1000
        self.loss_height_weight = 10
        self.conf_clamp_rate = 2      

    def forward(self,epoch,max_epoch,pred,conf,photo_gt,dem_gt,rpc:RPCModelParameterTorch):
        
        pred,conf,photo_gt,dem_gt = [i.to(torch.float64) for i in [pred,conf,photo_gt,dem_gt]]


        progress = 1. * epoch / max_epoch

        # if conf.max() == conf.min():
        #     conf = conf - conf + 1.
        # else:
        #     conf = (conf-conf.min()) * self.conf_clamp_rate / (conf.max() - conf.min())
        #     conf = conf / conf.mean()
        valid_idx = conf > .5
        conf[conf > .5] = .5 + progress * .4
        conf[conf < .5] = .5 - progress * .4
        conf = torch.clip(conf - conf.mean() + 1.,min=0.)

        obj_proj_pred,dem_pred = pred[:,:2],pred[:,2]

        photo_gt[:,[0,1]] = photo_gt[:,[1,0]] # line,samp -> samp,line

        obj_gt = torch.stack(rpc.RPC_PHOTO2OBJ(photo_gt[:,0],photo_gt[:,1],dem_gt),dim=-1)
        obj_proj_gt = project_mercator(obj_gt)

        loss_obj = torch.norm(obj_proj_pred - obj_proj_gt,dim=1) * conf
        
        height_dis = torch.abs(dem_pred - dem_gt)
        loss_height = tanh_clamp(height_dis,progress,self.clamp_max)  * conf

        height_valid_mask = height_dis < self.height_tolerance
        dem_pred[~height_valid_mask] = dem_gt[~height_valid_mask]
        
        obj_pred = mercator2lonlat(obj_proj_pred)
        photo_pred = torch.stack(rpc.RPC_OBJ2PHOTO(obj_pred[:,0],obj_pred[:,1],dem_pred),dim=1)
        photo_dis = torch.norm(photo_pred - photo_gt,dim=1)
        loss_photo = tanh_clamp(photo_dis,progress,self.clamp_max) * conf
        real_photo_loss = tanh_clamp(photo_dis,progress,self.clamp_max)[valid_idx].mean()

        bias = tanh_clamp(photo_pred - photo_gt,progress,self.clamp_max) 
        loss_bias =((bias[:,0] * conf).mean() ** 2 + (bias[:,1] * conf).mean() ** 2) ** 0.5

        # photo_pred_regularized = regularization(torch.cat([photo_gt,bwd_photo_gt]),torch.cat([photo_pred,bwd_photo_pred]),conf_pred)
        # loss_photo_regularized = tanh_clamp(torch.norm(photo_pred_regularized - torch.cat([photo_gt,bwd_photo_gt]),dim=1),progress,self.clamp_max) * conf_pred

        # reg,red_idx = regularization(photo_gt,photo_pred,conf)
        # loss_photo_regularized = (tanh_clamp(torch.norm(reg - photo_gt[red_idx],dim=1),progress,self.clamp_max) * conf[red_idx]).mean()
        loss_photo_regularized = affine_loss(photo_gt,photo_pred,conf)
        
        loss = loss_obj.mean() + loss_height.mean() * self.loss_height_weight + loss_photo.mean() + loss_bias + loss_photo_regularized

        return loss, loss_obj.mean() ,loss_height.mean() ,loss_photo.mean(), real_photo_loss,loss_bias.item(),loss_photo_regularized.item()
   
