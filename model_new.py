import math
import re
import os
from tabnanny import verbose
import torch
import torch.nn as nn
import torch.nn.functional as F
from swin_transformer_v2 import SwinTransformerV2,SwinTransformerBlock,PatchEmbed
from timm.models.layers import trunc_normal_
from timm.models.vision_transformer import VisionTransformer
from functools import partial
from torchvision import transforms
from typing import List
import torch.distributed as dist

def bnac(channels):
    return nn.Sequential(
        nn.BatchNorm2d(channels),
        nn.ReLU()
    )

class DistilledVisionTransformer(VisionTransformer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dist_token = nn.Parameter(torch.zeros(1, 1, self.embed_dim))

        num_patches = self.patch_embed.num_patches
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 2, self.embed_dim))
        self.head_dist = nn.Linear(self.embed_dim, self.num_classes) if self.num_classes > 0 else nn.Identity()

        trunc_normal_(self.dist_token, std=.02)
        trunc_normal_(self.pos_embed, std=.02)
        self.head_dist.apply(self._init_weights)
        self.single = True

    def forward_features(self, x):
        # taken from https://github.com/rwightman/pytorch-image-models/blob/master/timm/models/vision_transformer.py
        # with slight modifications to add the dist_token
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        dist_token = self.dist_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, dist_token, x), dim=1)

        x = x + self.pos_embed
        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)
        return x[:, 0], x[:, 1]

    def forward(self, x):
        x, x_dist = self.forward_features(x)
        x = self.head(x)
        x_dist = self.head_dist(x_dist)
        return F.normalize( (x + x_dist) / 2, p=2, dim=1)

class Encoder(nn.Module):

    def __init__(self,cfg = {},verbose = 1,output_global_feature = True):
        super().__init__()
        default_cfg = {
            'input_channels':3,
            'patch_feature_channels':512,
            'global_feature_channels':256,
            'img_size':1024,
            'window_size':8,
            'embed_dim':128,
            'depth':[2,2,18],
            'num_heads':[4,8,16],
            'drop_path_rate':.5,
            'unfreeze_backbone_modules':['head','norm','layers.2.blocks.16','layers.2.blocks.17'],
            'pretrain_window_size':[0,0,0]
        }
        self.cfg = {**default_cfg,**cfg}
        self.verbose = verbose
        self.SAMPLE_FACTOR = 16
        self.input_channels = self.cfg['input_channels']
        self.patch_feature_channels = self.cfg['patch_feature_channels']
        self.global_feature_channels = self.cfg['global_feature_channels']
        self.output_channels = self.cfg['patch_feature_channels'] + self.cfg['global_feature_channels']
        self.output_global_feature = output_global_feature

        self.backbone = SwinTransformerV2(img_size=self.cfg['img_size'],
                                        drop_path_rate=self.cfg['drop_path_rate'],
                                        embed_dim=self.cfg['embed_dim'],
                                        depths=self.cfg['depth'],
                                        num_heads=self.cfg['num_heads'],
                                        window_size=self.cfg['window_size'],
                                        in_chans=self.cfg['input_channels'],
                                        out_chans=self.cfg['patch_feature_channels'],
                                        pretrained_window_sizes=self.cfg['pretrain_window_size']
                                        )
        self.backbone_modules = dict(self.backbone.named_modules())
        self.backbone.requires_grad_(False)

        self.r2former = DistilledVisionTransformer(
            img_size=[480,640], patch_size=16, embed_dim=384, depth=12, num_heads=6, mlp_ratio=4, qkv_bias=True,
            norm_layer=partial(nn.LayerNorm, eps=1e-6), num_classes=self.cfg['global_feature_channels'])
        self.r2former.requires_grad_(False)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
            nn.ReLU(),
            nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
            nn.ReLU(),
            nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels // 16,1,1,0),
            nn.PReLU(),
            nn.Conv2d(self.patch_feature_channels // 16,1,1,1,0),
            nn.Sigmoid()
        )
        self.resize = transforms.Resize([480,640],antialias=True)
        self.unfreeze_backbone(self.cfg['unfreeze_backbone_modules'])


    def unfreeze_backbone(self,module_names:List[str]):
        unfreeze_modules = []
        for name in module_names:
            module = self.backbone_modules.get(name,None)
            if not module is None:
                module.requires_grad_(True)
            unfreeze_modules.append(name)
        if self.verbose > 0:
            print(f"unfreeze modules: {unfreeze_modules}")

    def get_unfreeze_parameters(self):
        params = []
        unfreeze_names = ['conf_head','cnn',*[f'backbone.{i}' for i in self.cfg['unfreeze_backbone_modules']]]
        all_modules = dict(self.named_modules())
        for name in unfreeze_names:
            module = all_modules.get(name,None)
            if not module is None:
                params.extend(list(module.parameters()))
                if self.verbose > 0:
                    print(f"Unfreeze: {name}")
        return params

    def forward(self, x):
        self.r2former.eval()

        feat_backbone = self.backbone(x)
        feat = self.cnn(feat_backbone)
        conf = self.conf_head(F.normalize(feat_backbone,dim=1))

        feat = F.normalize(feat,p=2,dim=1)

        if self.output_global_feature:
            global_feat = self.r2former(self.resize(x)) #F.interpolate(x,size=[480,640],mode='bilinear')
            global_feat = global_feat[:,:,None,None].repeat(1,1,feat.shape[-2],feat.shape[-1])
            return torch.cat([feat,global_feat],dim=1),conf
        else:
            return feat,conf
    
class ProjectHead(nn.Module):
    def __init__(self,input_channels,output_channels = None):
        super().__init__()
        self.input_channels = input_channels
        if output_channels is None:
            output_channels = input_channels // 4
        self.output_channels = output_channels
        self.head = nn.Sequential(
            nn.Conv2d(self.input_channels,self.output_channels,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.output_channels,self.output_channels,1,1,0),
        )
    def forward(self,x):
        return F.normalize(self.head(x),p=2,dim=1)


class DecoderFinetune(nn.Module):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.
    """

    def get_block(self,channels):
        return nn.Sequential(
            nn.Conv2d(channels,channels * 2,1,1,0),
            nn.ReLU(),
            nn.Conv2d(channels * 2,channels * 2,1,1,0),
            nn.ReLU(),
            nn.Conv2d(channels * 2,channels,1,1,0)
        )

    def __init__(self,in_channels=512,block_num=5,use_bn=False):
        super().__init__()
        block_num = max(block_num,1)
        self.use_bn = use_bn
        self.blocks = nn.ModuleList([self.get_block(in_channels) for _ in range(block_num)])
        self.output_xy = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,2,1,1,0),
            nn.Tanh()
        )
        self.output_height = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,1,1,1,0),
            nn.Tanh()
        )
        # self.bn = bnac(in_channels)


    def forward(self, res):
        # res = res / torch.norm(res,dim=1,keepdim=True)
        # if self.use_bn:
        #     res = self.bn(res)
        for block in self.blocks:
            x = block(res)
            res = res + x
        xy_res = self.output_xy(res)
        height_res = self.output_height(res)
        return torch.cat([xy_res,height_res],dim=1)
    
    
class Decoder(nn.Module):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.
    """

    def get_block(self,channels):
        return nn.Sequential(
            nn.Conv2d(channels,channels * 2,1,1,0),
            nn.ReLU(),
            nn.Conv2d(channels * 2,channels * 2,1,1,0),
            nn.ReLU(),
            nn.Conv2d(channels * 2,channels,1,1,0)
        )

    def __init__(self,in_channels=512,block_num=5,use_bn=False):
        super().__init__()
        block_num = max(block_num,1)
        self.use_bn = use_bn
        self.blocks = nn.ModuleList([self.get_block(in_channels) for _ in range(block_num)])
        self.output_xy = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,4,1,1,0)
        )
        self.output_height = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,2,1,1,0)
        )
        self.score_head = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,1,1,1,0),
            nn.Sigmoid()
        )
        # self.bn = bnac(in_channels)


    def forward(self, res):
        # res = res / torch.norm(res,dim=1,keepdim=True)
        # if self.use_bn:
        #     res = self.bn(res)
        for block in self.blocks:
            x = block(res)
            res = res + x
        xy_res = self.output_xy(res)
        height_res = self.output_height(res)
        valid_score = self.score_head(res)
        mu_xy = F.tanh(xy_res[:,:2])
        log_sigma_xy = F.tanh(xy_res[:,2:]) * 10.
        mu_h = F.tanh(height_res[:,:1])
        log_sigma_h = F.tanh(height_res[:,1:]) * 10.

        return torch.cat([mu_xy,mu_h,log_sigma_xy,log_sigma_h],dim=1),valid_score


class AffineFitter:
    """
    使用概率方法拟合一个2D仿射变换。

    该类通过最小化加权的最小二乘损失（等价于负对数似然）来寻找
    一个最佳的仿射变换，将源点映射到目标分布。权重由预测的标准差确定。
    """

    def __init__(self, learning_rate: float = 1e-3, num_iterations: int = 2000, verbose: bool = True):
        """
        初始化拟合器。

        Args:
            learning_rate (float): 优化器的学习率。
            num_iterations (int): 梯度下降的迭代次数。
            verbose (bool): 是否在拟合过程中打印损失信息。
        """
        self.lr = learning_rate
        self.iterations = num_iterations
        self.verbose = verbose
        
        # 最终得到的仿射变换矩阵，(2, 3)
        self.transformation_matrix = None

    def fit(self, 
            source_points: torch.Tensor, 
            pred_means: torch.Tensor, 
            pred_stds: torch.Tensor) -> torch.Tensor:
        """
        执行仿射变换的拟合过程。

        Args:
            source_points (torch.Tensor): 原始点坐标，形状为 (N, 2)。
            pred_means (torch.Tensor): 预测的目标点坐标均值，形状为 (N, 2)。
            pred_stds (torch.Tensor): 预测的目标点坐标标准差，形状为 (N, 2)。

        Returns:
            torch.Tensor: 拟合得到的 (2, 3) 仿射变换矩阵。
        """
        if self.verbose:
            print("开始求解仿射变换")

        # --- 1. 数据校验与准备 ---
        if not (source_points.shape == pred_means.shape == pred_stds.shape and source_points.dim() == 2 and source_points.shape[1] == 2):
            raise ValueError("所有输入张量的形状必须为 (N, 2)。")
        
        device = source_points.device
        dtype = source_points.dtype
        num_points = source_points.shape[0]

        # 仿射变换参数有6个: p = [a, b, c, d, e, f]^T
        # 初始化 H (在文献中常称为 A^T A) 和 b (在文献中常称为 A^T y)
        H = torch.zeros((6, 6), device=device, dtype=dtype)
        b = torch.zeros((6, 1), device=device, dtype=dtype)

        # --- 2. 计算权重 ---
        # 权重 = 1 / sigma^2。增加一个小的 epsilon 防止除以零。
        weights = 1.0 / (pred_stds.pow(2) + 1e-8)
        w_x = weights[:, 0]
        w_y = weights[:, 1]
        
        mu_x = pred_means[:, 0]
        mu_y = pred_means[:, 1]

        # --- 3. 构建线性方程组 Hp = b ---
        # 遍历所有点，累加信息到 H 和 b 中
        for i in range(num_points):
            x_orig, y_orig = source_points[i]

            # 构造与该点相关的向量 u_i 和 v_i
            # x' = u_i^T * p
            # y' = v_i^T * p
            u_i = torch.tensor([x_orig, y_orig, 1, 0, 0, 0], device=device, dtype=dtype).unsqueeze(1) # 6x1
            v_i = torch.tensor([0, 0, 0, x_orig, y_orig, 1], device=device, dtype=dtype).unsqueeze(1) # 6x1

            # 累加到Hessian矩阵 H
            # H = sum(w_xi * u_i*u_i^T + w_yi * v_i*v_i^T)
            H += w_x[i] * (u_i @ u_i.T) + w_y[i] * (v_i @ v_i.T)
            
            # 累加到向量 b
            # b = sum(w_xi * mu_xi * u_i + w_yi * mu_yi * v_i)
            b += w_x[i] * mu_x[i] * u_i + w_y[i] * mu_y[i] * v_i

        # --- 4. 求解线性方程组 ---
        if self.verbose:
            print("线性方程组构建完成，正在求解 Hp = b ...")
        
        try:
            # 使用torch.linalg.solve求解器，它比手动求逆更稳定、更高效
            params_vec = torch.linalg.solve(H, b, out=None) # PyTorch 1.10+
            # 对于旧版本PyTorch，可以使用: params_vec, _ = torch.solve(b, H)
        except torch.linalg.LinAlgError as e:
            print("错误：矩阵H是奇异的或接近奇异的，无法求解。")
            print("这通常发生在输入点共线或点数量过少的情况下。")
            raise e

        # --- 5. 格式化并保存结果 ---
        # 将解向量 p (6x1) 变形为 2x3 的仿射矩阵
        self.transformation_matrix = params_vec.reshape(2, 3)
        
        
        if self.verbose:
            print("求解完成！")
            ori_dis = torch.norm(source_points - pred_means,dim=-1).mean()
            trans_points = self.transform(source_points)
            trans_dis = torch.norm(trans_points - pred_means,dim=-1).mean()
            print(f"初始误差: {ori_dis.item():.2f} \t 变换后误差: {trans_dis.item():.2f}")
           
            
        return self.transformation_matrix

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """
        使用拟合好的变换矩阵来变换新的点。

        Args:
            points (torch.Tensor): 需要变换的点，形状为 (M, 2)。

        Returns:
            torch.Tensor: 变换后的点，形状为 (M, 2)。
        """
        if self.transformation_matrix is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合，然后才能进行变换。")
        
        device = points.device
        num_points = points.shape[0]
        
        points_homogeneous = torch.cat(
            [points, torch.ones(num_points, 1, device=device)],
            dim=1
        )
        
        # 使用存储的矩阵进行变换
        transformed_points = points_homogeneous @ self.transformation_matrix.T
        
        return transformed_points


# self.cnn = nn.Sequential(
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#             bnac(self.patch_feature_channels),
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#             bnac(self.patch_feature_channels),
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#         )