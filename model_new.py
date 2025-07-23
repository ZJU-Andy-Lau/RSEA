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

    def __init__(self,cfg = {},verbose = 1):
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

        global_feat = self.r2former(self.resize(x)) #F.interpolate(x,size=[480,640],mode='bilinear')
        global_feat = global_feat[:,:,None,None].repeat(1,1,feat.shape[-2],feat.shape[-1])
        feat = F.normalize(feat,p=2,dim=1)
        
        return torch.cat([feat,global_feat],dim=1),conf
    
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
        # --- 1. 数据校验与准备 ---
        if not (source_points.shape == pred_means.shape == pred_stds.shape and source_points.dim() == 2 and source_points.shape[1] == 2):
            raise ValueError("所有输入张量的形状必须为 (N, 2)。")
        
        device = source_points.device
        num_points = source_points.shape[0]

        # 将源点转换为齐次坐标 (N, 3)，方便矩阵乘法
        # (x, y) -> (x, y, 1)
        source_homogeneous = torch.cat(
            [source_points, torch.ones(num_points, 1, device=device)], 
            dim=1
        )

        # --- 2. 初始化变换参数和优化器 ---
        # 初始化一个 (2, 3) 的仿射矩阵 [a, b, c; d, e, f]
        # 最佳实践是初始化为单位变换
        affine_matrix = torch.tensor([[1.0, 0.0, 0.0],
                                      [0.0, 1.0, 0.0]], device=device, dtype=source_points.dtype)
        
        
        # 将其封装为可学习参数
        self.params = nn.Parameter(affine_matrix)

        best_params = self.params
        
        # 使用 Adam 优化器
        optimizer = torch.optim.Adam([self.params], lr=self.lr)

        # --- 3. 计算NLL损失的权重 ---
        # 权重 = 1 / sigma^2。增加一个小的 epsilon 防止除以零。
        weights = 1.0 / (pred_stds.pow(2) + 1e-8)

        # --- 4. 优化循环 ---
        if self.verbose:
            print(f"开始拟合... 总共 {self.iterations if self.iterations > 0 else 'no limit'} 次迭代。")
        
        min_loss = 1e9
        iter_count = 0
        no_update_count = 0

        while(True):
            iter_count += 1
            optimizer.zero_grad()

            # 应用仿射变换: (N, 3) @ (3, 2) -> (N, 2)
            # [x, y, 1] @ [[a, d], [b, e], [c, f]] = [ax+by+c, dx+ey+f]
            transformed_points = source_homogeneous @ self.params.T

            # 计算损失 (加权MSE，等价于NLL)
            # L = sum( (x'_i - mu_xi)^2 / sigma_xi^2 + (y'_i - mu_yi)^2 / sigma_yi^2 )
            error = transformed_points - pred_means
            weighted_squared_error = error.pow(2) * weights
            loss = weighted_squared_error.mean()

            if loss < min_loss:
                min_loss = loss.item()
                best_params = self.params
                no_update_count = 0
            else:
                no_update_count += 1

            # 反向传播和优化
            loss.backward()
            optimizer.step()

            if self.verbose and (iter_count % 1000 == 0 or iter_count == self.iterations - 1 or no_update_count > 5000):
                if self.iterations > 0:
                    print(f"iter {iter_count:5d}/{self.iterations}, loss: {loss.item():.4f}, min_loss: {min_loss:.4f}")
                else:
                    print(f"iter {iter_count:5d}, loss: {loss.item():.4f}, min_loss: {min_loss:.4f}")
            
            if self.iterations > 0 and iter_count >= self.iterations:
                break

            if no_update_count > 5000:
                if self.verbose:
                    print("no update for 5000 iterations, early stop")
                break
            

        # --- 5. 保存并返回结果 ---
        # detach() 以防止后续操作被追踪梯度
        self.transformation_matrix = best_params.detach().clone()
        
        if self.verbose:
            print("拟合完成！")
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