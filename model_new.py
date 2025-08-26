import math
import re
import os
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

class Adapter(nn.Module):
    def __init__(self,channel = 512):
        super().__init__()
        self.channels = channel
        self.cnn = nn.Sequential(
            nn.Conv2d(self.channels,self.channels,3,1,1),
            nn.ReLU(),
            nn.Conv2d(self.channels,self.channels,3,1,1),
            nn.ReLU(),
            nn.Conv2d(self.channels,self.channels,3,1,1),
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(self.channels,self.channels // 16,1,1,0),
            nn.PReLU(),
            nn.Conv2d(self.channels // 16,1,1,1,0),
            nn.Sigmoid()
        )
    def forward(self,x):
        feat = self.cnn(x)
        conf = self.conf_head(x)
        return feat,conf

class EncoderDino(nn.Module):

    def __init__(self,dino_weight_path,verbose = 1):
        super().__init__()
        self.verbose = verbose
        self.SAMPLE_FACTOR = 16
        self.input_channels = 3
        self.output_channels = 512

        self.backbone = torch.hub.load('./dinov3','dinov3_vitl16',source='local',weights=dino_weight_path)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.adapter = Adapter()


    def forward(self, x):
        B = x.shape[0]
        H,W = x.shape[-2:]
        feat_backbone = self.backbone(x)
        print("1:",feat_backbone.shape)
        feat_backbone = feat_backbone.reshape(B,H // self.SAMPLE_FACTOR,W // self.SAMPLE_FACTOR,-1).permute(0,3,1,2)
        print("2:",feat_backbone.shape)
        feat,conf = self.adapter(feat_backbone)
        return feat,conf
    
    def load_adapter(self,adapter_path:str):
        self.adapter.load_state_dict({k.replace("module.",""):v for k,v in torch.load(adapter_path,map_location='cpu').items()},strict=True)
    
    def save_adapter(self,output_path:str):
        state_dict = {k:v.detach().cpu() for k,v in self.adapter.state_dict().items()}
        torch.save(state_dict,output_path)
    
    def train(self, mode = True):
        super().train(mode)
        self.backbone.eval()
        return self

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

    def __init__(self, verbose: bool = True):
        """
        初始化拟合器。

        Args:
            learning_rate (float): 优化器的学习率。
            num_iterations (int): 梯度下降的迭代次数。
            verbose (bool): 是否在拟合过程中打印损失信息。
        """
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

        source_homogeneous = torch.cat(
            [source_points, torch.ones(num_points, 1, device=device, dtype=dtype)], 
            dim=1
        )

        # --- 2. 计算权重 ---
        weights = 1.0 / (pred_stds.pow(2) + 1e-8)
        w_x = weights[:, 0].to(dtype=dtype)
        w_y = weights[:, 1].to(dtype=dtype)
        
        mu_x = pred_means[:, 0].to(dtype=dtype)
        mu_y = pred_means[:, 1].to(dtype=dtype)

        # --- 3. 构建线性方程组 Hp = b (向量化版本) ---
        
        # --- 计算 Hessian 矩阵 H ---
        # H 是一个 6x6 的块矩阵:
        # H = [[H_xx,  0   ],
        #      [ 0  ,  H_yy]]
        # 其中 H_xx = sum(w_xi * p_i * p_i^T)
        # H_yy = sum(w_yi * p_i * p_i^T)
        # 这里的 p_i 是齐次坐标 [x_i, y_i, 1]
        
        # 使用矩阵乘法高效计算 H_xx 和 H_yy
        # (P.T @ (w * P)) 等价于 sum(w_i * p_i * p_i^T)
        H_xx = source_homogeneous.T @ (w_x.unsqueeze(1) * source_homogeneous)
        H_yy = source_homogeneous.T @ (w_y.unsqueeze(1) * source_homogeneous)
        
        H = torch.zeros((6, 6), device=device, dtype=dtype)
        H[0:3, 0:3] = H_xx
        H[3:6, 3:6] = H_yy

        # --- 计算向量 b ---
        # b 是一个 6x1 的块向量:
        # b = [[b_x],
        #      [b_y]]
        # 其中 b_x = sum(w_xi * mu_xi * p_i)
        # b_y = sum(w_yi * mu_yi * p_i)
        

        # 使用矩阵-向量乘法高效计算 b_x 和 b_y
        b_x = source_homogeneous.T @ (w_x * mu_x)
        b_y = source_homogeneous.T @ (w_y * mu_y)
        
        b = torch.cat([b_x, b_y]).unsqueeze(1) # 拼接成 6x1 的列向量

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

class HomographyFitter:
    """
    使用非线性优化方法（L-BFGS）来拟合一个2D单应变换。

    该类通过最小化加权的最小二乘损失（等价于负对数似然）来寻找
    最佳的单应变换。由于单应变换对于其参数是非线性的，我们不能
    使用直接解法，而是采用像L-BFGS这样的迭代优化器。
    """

    def __init__(self, max_iterations: int = 2000, lr: float = 1e-3,
                 patience: int = 50, tolerance: float = 1e-7,
                 weight_decay: float = 1e-4, verbose: bool = True):
        """
        初始化拟合器。

        Args:
            max_iterations (int): 优化的最大迭代次数。如果小于等于0，则启用早停策略。
            lr (float): AdamW优化器的学习率。
            patience (int): 早停策略的“耐心值”。
            tolerance (float): 用于判断损失是否“显著下降”的阈值。
            weight_decay (float): AdamW的权重衰减系数。
            verbose (bool): 是否在拟合过程中打印信息。
        """
        self.max_iterations = max_iterations
        self.lr = lr
        self.patience = patience
        self.tolerance = tolerance
        self.weight_decay = weight_decay
        self.verbose = verbose
        # 最终得到的单应变换矩阵，3x3
        self.transformation_matrix = None

    def _get_normalization_matrix(self, points: torch.Tensor) -> torch.Tensor:
        """计算将点归一化的变换矩阵。"""
        mean = points.mean(dim=0)
        cx, cy = mean[0], mean[1]

        # 将点移动到以原点为中心
        centered_points = points - mean
        
        # 计算平均距离，并缩放使其约为sqrt(2)
        avg_dist = (centered_points**2).sum(dim=1).sqrt().mean()
        # 修正: 确保新创建的张量与输入张量有相同的dtype和device
        scale = torch.sqrt(torch.tensor(2.0, dtype=points.dtype, device=points.device)) / (avg_dist + 1e-8)

        # 构建归一化矩阵
        # T = [s, 0, -s*cx]
        #     [0, s, -s*cy]
        #     [0, 0, 1    ]
        T = torch.eye(3, device=points.device, dtype=points.dtype)
        T[0, 0] = T[1, 1] = scale
        T[0, 2] = -scale * cx
        T[1, 2] = -scale * cy
        return T

    def fit(self,
            source_points: torch.Tensor,
            pred_means: torch.Tensor,
            pred_stds: torch.Tensor) -> torch.Tensor:
        """
        执行单应变换的拟合过程。
        """
        if self.verbose:
            print("开始使用AdamW和坐标归一化拟合单应变换...")
            if self.max_iterations <= 0:
                print(f"早停已启用: patience={self.patience}, tolerance={self.tolerance}")

        device = source_points.device
        dtype = source_points.dtype
        
        # --- 1. 坐标归一化 (鲁棒性关键步骤) ---
        T_source = self._get_normalization_matrix(source_points)
        T_target = self._get_normalization_matrix(pred_means)
        T_target_inv = torch.linalg.inv(T_target)

        source_h = torch.cat([source_points, torch.ones(source_points.shape[0], 1, device=device, dtype=dtype)], dim=1)
        pred_means_h = torch.cat([pred_means, torch.ones(pred_means.shape[0], 1, device=device, dtype=dtype)], dim=1)

        # 应用归一化
        norm_source_h = (T_source @ source_h.T).T
        norm_pred_means_h = (T_target @ pred_means_h.T).T
        
        # 清理: 移除未使用的变量 norm_source_points
        norm_pred_means = norm_pred_means_h[:, :2] / norm_pred_means_h[:, 2].unsqueeze(1)
        
        # --- 2. 初始化变换参数和优化器 ---
        initial_params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        self.params = nn.Parameter(initial_params)
        optimizer = torch.optim.AdamW([self.params], lr=self.lr, weight_decay=self.weight_decay)

        # --- 3. 计算损失权重 ---
        # 权重不需要归一化
        weights = 1.0 / (pred_stds.pow(2) + 1e-8)

        # --- 4. 在归一化坐标上进行优化循环 ---
        iteration = 0
        best_loss = float('inf')
        patience_counter = 0

        while True:
            iteration += 1
            optimizer.zero_grad()
            
            h_matrix_norm = torch.cat([self.params, torch.tensor([1.0], device=device, dtype=dtype)]).reshape(3, 3)
            
            # 在归一化空间中进行变换
            transformed_h_norm = norm_source_h @ h_matrix_norm.T
            w_norm = transformed_h_norm[:, 2].unsqueeze(1)
            transformed_points_norm = transformed_h_norm[:, :2] / (w_norm + 1e-8)
            
            # 在归一化空间中计算损失
            error = transformed_points_norm - norm_pred_means
            weighted_squared_error = error.pow(2) * weights
            loss = weighted_squared_error.sum()

            if torch.isnan(loss) or torch.isinf(loss):
                if self.verbose:
                    print(f"迭代 {iteration:4d}, 损失变为无效值(NaN/Inf)，优化失败。")
                self.transformation_matrix = torch.eye(3, device=device, dtype=dtype)
                return self.transformation_matrix

            loss.backward()
            optimizer.step()

            if self.verbose and (iteration % 200 == 0 or iteration == 1):
                print(f"迭代 {iteration:4d}, 损失: {loss.item():.6f}，最小：{best_loss:.6f}")

            if best_loss - loss.item() > self.tolerance:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if self.max_iterations > 0 and iteration >= self.max_iterations:
                if self.verbose:
                    print(f"达到最大迭代次数上限: {self.max_iterations}。")
                break
            
            if self.max_iterations <= 0 and patience_counter >= self.patience:
                if self.verbose:
                    print(f"\n损失在 {self.patience} 次迭代内没有显著下降，提前停止于第 {iteration} 次迭代。")
                break

        # --- 5. 反归一化并保存结果 ---
        final_h_norm = torch.cat([self.params.detach(), torch.tensor([1.0], device=device, dtype=dtype)]).reshape(3, 3)
        self.transformation_matrix = T_target_inv.to(torch.float) @ final_h_norm.to(torch.float) @ T_source.to(torch.float)
        
        # --- 6. 计算并输出最终结果 ---
        if self.verbose:
            # 新增: 计算最终的平均残差 (Reprojection Error)
            final_transformed_points = self.transform(source_points)
            residuals = torch.sqrt(((final_transformed_points - pred_means)**2).sum(dim=1))
            avg_residual = residuals.mean()
            print(f"拟合完成于第 {iteration} 次迭代。最终损失: {best_loss:.6f}, 平均残差: {avg_residual.item():.6f} 像素")
            
        return self.transformation_matrix

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """
        使用拟合好的变换矩阵来变换新的点。
        """
        if self.transformation_matrix is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合，然后才能进行变换。")
        
        device = points.device
        dtype = points.dtype
        num_points = points.shape[0]
        
        points_homogeneous = torch.cat(
            [points, torch.ones(num_points, 1, device=device, dtype=dtype)],
            dim=1
        )
        
        transformed_homogeneous = points_homogeneous @ self.transformation_matrix.T
        
        w = transformed_homogeneous[:, 2].unsqueeze(1)
        transformed_points = transformed_homogeneous[:, :2] / (w + 1e-8)
        
        return transformed_points

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """
        使用拟合好的变换矩阵来变换新的点。
        """
        if self.transformation_matrix is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合，然后才能进行变换。")
        
        device = points.device
        dtype = points.dtype
        num_points = points.shape[0]
        
        points_homogeneous = torch.cat(
            [points, torch.ones(num_points, 1, device=device, dtype=dtype)],
            dim=1
        )
        
        transformed_homogeneous = points_homogeneous @ self.transformation_matrix.T
        
        w = transformed_homogeneous[:, 2].unsqueeze(1)
        transformed_points = transformed_homogeneous[:, :2] / (w + 1e-8)
        
        return transformed_points

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """
        使用拟合好的变换矩阵来变换新的点。
        """
        if self.transformation_matrix is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合，然后才能进行变换。")
        
        device = points.device
        dtype = points.dtype
        num_points = points.shape[0]
        
        points_homogeneous = torch.cat(
            [points, torch.ones(num_points, 1, device=device, dtype=dtype)],
            dim=1
        )
        
        transformed_homogeneous = points_homogeneous @ self.transformation_matrix.T
        
        w = transformed_homogeneous[:, 2].unsqueeze(1)
        transformed_points = transformed_homogeneous[:, :2] / (w + 1e-8)
        
        return transformed_points

# self.cnn = nn.Sequential(
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#             bnac(self.patch_feature_channels),
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#             bnac(self.patch_feature_channels),
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#         )