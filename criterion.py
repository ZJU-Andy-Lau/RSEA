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

class CriterionFinetune(nn.Module):
    def __init__(self,dataset_num):
        super().__init__()
        self.dataset_num = dataset_num
        self.height_tolerance = 10
        self.clamp_max = 1000
        self.loss_height_weight = 1
        self.near_threshold = 8.
        self.residual_thresholds = [4.] * dataset_num
        self.gamma = .05
        self.bce = nn.BCELoss()

    def forward(self,epoch,max_epoch,dataset_idx,
                feat1_PD,feat2_PD,
                pred1_P3,pred2_P3,
                conf1_P,conf2_P,
                obj1_P3,obj2_P3,
                local1_P2,local2_P2,
                residual1_P,residual2_P,
                rpcs:List[RPCModelParameterTorch],v1,v2,
                B,H,W
                ):
        P = H*W
        #--------------------------------------------
        # t0 = time.perf_counter()
        #--------------------------------------------

        pred1_P3,pred2_P3,conf1_P,conf2_P,obj1_P3,obj2_P3,local1_P2,local2_P2,residual1_P,residual2_P= \
            [i.to(torch.float64) for i in [pred1_P3,pred2_P3,conf1_P,conf2_P,obj1_P3,obj2_P3,local1_P2,local2_P2,residual1_P,residual2_P]]
        feat1_PD,feat2_PD = feat1_PD.to(torch.float32),feat2_PD.to(torch.float32)
        
        res_mid = torch.median(torch.cat([residual1_P,residual2_P])[~torch.isnan(torch.cat([residual1_P,residual2_P]))])
        if not torch.isnan(res_mid):
            self.residual_thresholds[dataset_idx] = (1. - self.gamma) * self.residual_thresholds[dataset_idx] + self.gamma * res_mid
        else:
            print("all res nan")   

        feat1_PD = F.normalize(feat1_PD,dim=1)
        feat2_PD = F.normalize(feat2_PD,dim=1)
        feat_BHWD = torch.concatenate([feat1_PD.reshape(B,H,W,-1),feat2_PD.reshape(B,H,W,-1)],dim=0)
        feat_BPD = torch.concatenate([feat1_PD.reshape(B,H*W,-1),feat2_PD.reshape(B,H*W,-1)],dim=0)
        conf_valid1 = ~torch.isnan(residual1_P)
        conf_valid2 = ~torch.isnan(residual2_P)
        conf1_gt_P = residual2conf(residual1_P,self.residual_thresholds[dataset_idx])
        conf2_gt_P = residual2conf(residual2_P,self.residual_thresholds[dataset_idx])

        obj_yx_pred1_P2,dem_pred1_P = pred1_P3[:,:2],pred1_P3[:,2]
        obj_yx_pred2_P2,dem_pred2_P = pred2_P3[:,:2],pred2_P3[:,2]
        obj_yx_gt1_P2,dem_gt1_P = obj1_P3[:,[1,0]],obj1_P3[:,2] # (y,x) (h)
        obj_yx_gt2_P2,dem_gt2_P = obj2_P3[:,[1,0]],obj2_P3[:,2]

        local1_P2[:,[0,1]] = local1_P2[:,[1,0]] # line,samp -> samp,line
        local2_P2[:,[0,1]] = local2_P2[:,[1,0]]


        # 通过投影获取匹配点

        obj_latlon_gt2_P2 = mercator2lonlat(obj_yx_gt2_P2)

        local1_Bp2 = local1_P2.reshape(B,P,2)
        obj_latlon_gt2_Bp2 = obj_latlon_gt2_P2.reshape(B,P,2)
        dem_gt2_Bp = dem_gt2_P.reshape(B,P)
        anchor_points = []
        positive_points = []
        for v in torch.unique(v1):
            rpc = rpcs[v]
            img_idxs = torch.where(v1 == v)[0].to(local1_Bp2.device)
            img_num = len(img_idxs)
            local1_bp2 = local1_Bp2[img_idxs]
            obj_latlon_gt2_p2 = obj_latlon_gt2_Bp2[img_idxs].reshape(-1,2)
            dem_gt2_p = dem_gt2_Bp[img_idxs].reshape(-1) 
            local_proj1_bp2 = torch.stack(rpc.RPC_OBJ2PHOTO(obj_latlon_gt2_p2[:,0],obj_latlon_gt2_p2[:,1],dem_gt2_p),dim=-1).reshape(img_num,P,2)
            local_proj1_bp2 -= local1_bp2[:,0:1,:]
            local_max = (local1_bp2[:,-1,:] - local1_bp2[:,0,:])[:,None]
            mask = (local_proj1_bp2 > -8.).all(dim=2) & (local_proj1_bp2 < local_max).all(dim=2)
            near_idx2 = torch.nonzero(mask).to(device=img_idxs.device)
            near_idx2 = img_idxs[near_idx2[:,0]] * P + near_idx2[:,1]
            local_proj1_bp2 = torch.round(local_proj1_bp2 / 16.).to(int)
            near_idx1 = local_proj1_bp2[...,1] * W + local_proj1_bp2[...,0] + (img_idxs * P)[:,None]
            near_idx1 = near_idx1[mask].reshape(-1)
            # near_idx1 = near_idx1[...,1] * 16 + near_idx1[...,0] + img_idxs * P
            anchor_points.append(near_idx1.reshape(-1))
            positive_points.append(near_idx2.reshape(-1))


        anchor_points = torch.concatenate(anchor_points)
        positive_points = torch.concatenate(positive_points)

        valid_mask = torch.norm(obj_yx_gt1_P2[anchor_points] - obj_yx_gt2_P2[positive_points],dim=1) < self.near_threshold
        anchor_points = anchor_points[valid_mask]
        positive_points = positive_points[valid_mask]
        negative_points = get_far_points(obj_yx_gt1_P2[anchor_points],obj_yx_gt2_P2,threshold=10)

        dis_obj_proj1_P = torch.norm(obj_yx_pred1_P2 - obj_yx_gt1_P2,dim=1)
        dis_obj_proj2_P = torch.norm(obj_yx_pred2_P2 - obj_yx_gt2_P2,dim=1)
        loss_obj = .5 * dis_obj_proj1_P.mean() + .5 * dis_obj_proj2_P.mean()

        dis_pred = torch.norm(obj_yx_pred1_P2[anchor_points] - obj_yx_pred2_P2[positive_points],dim=1)

        
        dis_gt = torch.norm(obj_yx_gt1_P2[anchor_points] - obj_yx_gt2_P2[positive_points],dim=1)
        loss_dis = torch.clip(dis_pred - dis_gt,0.).mean()

        obj_yx_pred_bhw2 = torch.concatenate([obj_yx_pred1_P2.reshape(-1,H,W,2),obj_yx_pred2_P2.reshape(-1,H,W,2)],dim=0)
        obj_yx_gt_bhw2 = torch.concatenate([obj_yx_gt1_P2.reshape(-1,H,W,2),obj_yx_gt2_P2.reshape(-1,H,W,2)],dim=0)
        dis_adj_pred_line = torch.norm(obj_yx_pred_bhw2[:,1:,:] - obj_yx_pred_bhw2[:,:-1,:],dim=-1)
        dis_adj_pred_samp = torch.norm(obj_yx_pred_bhw2[:,:,1:] - obj_yx_pred_bhw2[:,:,:-1],dim=-1)
        dis_adj_gt_line = torch.norm(obj_yx_gt_bhw2[:,1:,:] - obj_yx_gt_bhw2[:,:-1,:],dim=-1)
        dis_adj_gt_samp = torch.norm(obj_yx_gt_bhw2[:,:,1:] - obj_yx_gt_bhw2[:,:,:-1],dim=-1)
        loss_adj = .5 * torch.abs(dis_adj_pred_line - dis_adj_gt_line).mean() + .5 * torch.abs(dis_adj_pred_samp - dis_adj_gt_samp).mean()



        # margin = min(1. * epoch / min(max_epoch,50) + 2.3, 2.7)
        # margin = .3
        temperature = .1
        feat_sim_positive = torch.sum(feat1_PD[anchor_points] * feat2_PD[positive_points],dim=1)
        feat_sim_negative = torch.sum(feat1_PD[anchor_points] * feat1_PD[negative_points],dim=1)
        sp = feat_sim_positive.mean().detach()
        sn = feat_sim_negative.mean().detach()
        # feat_sim_all = torch.bmm(feat_BPD,feat_BPD.transpose(1,2))
        weights = feat_sim_negative.detach()
        weights[weights > .8] = .9
        weights[weights < .8] = .1
        weights = weights - weights.mean() + 1.
        # mask = (feat_sim_all < 1.) & (feat_sim_all > .8)
        # feat_sim_all = feat_sim_all[mask]
        # idxs = torch.randperm(len(feat_sim_all))[:feat_sim_positive.shape[0]]
        # feat_sim_all = feat_sim_all[idxs]
        feat_sim_positive = torch.sum(feat1_PD[anchor_points] * feat2_PD[positive_points],dim=1)
        feat_sim_negative = torch.sum(feat1_PD[anchor_points] * feat2_PD[negative_points],dim=1)
        feat_sim_line = torch.sum(feat_BHWD[:,:-1,:] * feat_BHWD[:,1:,:],dim=-1)
        feat_sim_samp = torch.sum(feat_BHWD[:,:,:-1] * feat_BHWD[:,:,1:],dim=-1)
        feat_sim_diag = torch.sum(feat_BHWD[:,:-1,:-1] * feat_BHWD[:,1:,1:],dim=-1)
        feat_sim_adiag = torch.sum(feat_BHWD[:,1:,:-1] * feat_BHWD[:,:-1,1:],dim=-1)
        feat_sim_adj = torch.clip(feat_sim_line - sp + .05,min=0.).mean() + torch.clip(feat_sim_samp - sp + .05,min=0.).mean() + torch.clip(feat_sim_diag - sp + .05,min=0.).mean() + torch.clip(feat_sim_adiag - sp + .05,min=0.).mean()
        sa = (feat_sim_line.mean() + feat_sim_samp.mean() + feat_sim_diag.mean() + feat_sim_adiag.mean()) * .25
        loss_feat = torch.clip(1. - feat_sim_positive,min=0.).mean() * 10000 + torch.clip((feat_sim_negative - .7) * weights,min=0.).mean() * 1000 + feat_sim_adj * 1000  #-torch.log(torch.sum(torch.exp(feat_sim_positive / temperature)) / torch.sum(torch.exp(feat_sim_all / temperature))) * 100 + 

        
        # sa = feat_sim_all.mean()

        loss_conf = .5 * self.bce(conf1_P[conf_valid1],conf1_gt_P[conf_valid1]) + .5 * self.bce(conf2_P[conf_valid2],conf2_gt_P[conf_valid2])

        loss_conf *= 1000 * min(1.,epoch / 3.)

        dis_height1 = torch.abs(dem_pred1_P - dem_gt1_P)
        dis_height2 = torch.abs(dem_pred2_P - dem_gt2_P)
        loss_height = torch.cat([dis_height1,dis_height2]).mean()

        #--------------------------------------------
        # t5 = time.perf_counter()
        #--------------------------------------------

        # print(t1 - t0,t2 - t1,t3 - t2,t4 - t3,t5 - t4)

        loss = loss_obj + loss_dis + loss_adj + loss_height * self.loss_height_weight + loss_conf + loss_feat# + loss_simi #+ loss_struct * progress # + loss_photo + loss_bias + loss_photo_regularized 
        return loss,loss_obj,loss_dis, loss_adj ,loss_height ,loss_conf, loss_feat, sp , sn, sa


    
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
   
