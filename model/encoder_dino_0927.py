import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class PositionalEncoding(nn.Module):
    """二维正弦位置编码，为特征图注入空间位置信息"""
    def __init__(self, dim: int, max_shape: tuple = (256, 256)):
        super().__init__()
        self.dim = dim
        pe = torch.zeros(dim, *max_shape)
        device = pe.device
        y_position = torch.arange(0, max_shape[0], dtype=torch.float32, device=device).unsqueeze(1)
        x_position = torch.arange(0, max_shape[1], dtype=torch.float32, device=device).unsqueeze(0)

        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * -(torch.log(torch.tensor(10000.0)) / dim))
        
        pe[0::2, :, :] = torch.sin(y_position * div_term.unsqueeze(1).unsqueeze(2))
        pe[1::2, :, :] = torch.cos(y_position * div_term.unsqueeze(1).unsqueeze(2))
        
        pe[0::2, :, :] += torch.sin(x_position * div_term.unsqueeze(1).unsqueeze(2))
        pe[1::2, :, :] += torch.cos(x_position * div_term.unsqueeze(1).unsqueeze(2))

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): (B, D, H, W) 的输入特征图
        Returns:
            torch.Tensor: 添加了位置编码的特征图
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]


# --- 核心模块 1: 增强型Encoder ---
class AttentionBlock(nn.Module):
    """一个通用的注意力模块，可用于自注意力或交叉注意力"""
    def __init__(self, dim: int, num_heads: int = 8):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)

    def forward(self, query: torch.Tensor, key: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query, key, value (torch.Tensor): (B, N, D) shape, N = H * W
        """
        B, N, D = query.shape
        
        # 线性投影并切分为多头
        q = self.q_proj(query).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled Dot-Product Attention
        attn_scores = (q @ k.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn_scores, dim=-1)

        # 加权求和
        x = (attn @ v).transpose(1, 2).reshape(B, N, D)
        
        # 输出线性投影
        return self.out_proj(x)

class FeatureInteractionModule(nn.Module):
    def __init__(self, feature_dim: int, num_heads: int = 8, num_layers: int = 2):
        """
        对两路特征图进行交互。

        Args:
            feature_dim (int): 输入特征图的维度D。
            num_heads (int): 注意力头的数量。
            num_layers (int): 交互层数（自注意+交叉注意算一层）。
        """
        super().__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': AttentionBlock(dim=feature_dim, num_heads=num_heads),
                'cross_attn': AttentionBlock(dim=feature_dim, num_heads=num_heads),
                'norm1': nn.LayerNorm(feature_dim),
                'norm2': nn.LayerNorm(feature_dim),
                'norm3': nn.LayerNorm(feature_dim)
            }) for _ in range(num_layers)
        ])
        
    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_a (torch.Tensor): 来自影像A的特征图 (B, D, H, W)
            feat_b (torch.Tensor): 来自影像B的特征图 (B, D, H, W)
        Returns:
            tuple[torch.Tensor, torch.Tensor]: 交互后的两路特征图
        """
        B, D, H, W = feat_a.shape
        
        # 展平为序列
        seq_a = feat_a.flatten(2).transpose(1, 2)
        seq_b = feat_b.flatten(2).transpose(1, 2)
        
        for layer in self.layers:
            # 1. 自注意力 (内部上下文增强)
            res_a, res_b = seq_a, seq_b
            seq_a = layer['norm1'](res_a + layer['self_attn'](seq_a, seq_a, seq_a))
            seq_b = layer['norm1'](res_b + layer['self_attn'](seq_b, seq_b, seq_b))
            
            # 2. 交叉注意力 (跨视图信息融合)
            res_a, res_b = seq_a, seq_b
            # a 从 b 获取信息
            seq_a = layer['norm2'](res_a + layer['cross_attn'](seq_a, seq_b, seq_b))
            # b 从 a 获取信息
            seq_b = layer['norm3'](res_b + layer['cross_attn'](seq_b, seq_a, seq_a))

        # 还原为图像形状
        out_a = seq_a.transpose(1, 2).view(B, D, H, W)
        out_b = seq_b.transpose(1, 2).view(B, D, H, W)

        return F.normalize(out_a,dim=1), F.normalize(out_b,dim=1)
    
class Adapter(nn.Module):
    def __init__(self,input_channels = 512,output_channels = 512):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels,self.input_channels // 4,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.input_channels // 4,self.output_channels,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.output_channels,self.output_channels,1,1,0),
        )

        self.pos_encoder = PositionalEncoding(dim=output_channels)
        
        # 自注意力模块，借鉴CasP中的TransformerBlock设计
        self.self_attention_block = AttentionBlock(dim=output_channels, num_heads=8)
        self.norm = nn.LayerNorm(output_channels)

        self.conf_head = nn.Sequential(
            nn.Conv2d(self.input_channels,self.input_channels // 4,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.input_channels // 4, self.input_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(self.input_channels // 16, 1 ,1,1,0),
            nn.Sigmoid()
        )
    def forward(self,x):
        raw_feat = self.cnn(x)
        B,D,H,W = raw_feat.shape
        feat_with_pos = self.pos_encoder(raw_feat)
        feat_seq = feat_with_pos.flatten(2).transpose(1,2)
        # feat_seq = raw_feat.flatten(2).transpose(1,2)
        attn_output = self.self_attention_block(feat_seq, feat_seq, feat_seq)
        attended_sequence = self.norm(feat_seq + attn_output)
        feat = F.normalize(attended_sequence.transpose(1, 2).view(B, D, H, W),dim=1)
        conf = self.conf_head(x)
        # feat = F.normalize(feat,dim=1)
        return feat,conf
    
    

class EncoderDino(nn.Module):

    def __init__(self,dino_weight_path,output_channels=512,verbose = 1,layers = [5,11,17,23]):
        super().__init__()
        self.verbose = verbose
        self.layers = layers
        self.SAMPLE_FACTOR = 16
        self.input_channels = 3
        self.output_channels = output_channels

        self.backbone = torch.hub.load('./dinov3','dinov3_vitl16',source='local',weights=dino_weight_path)
        self.backbone.eval()
        self.backbone.requires_grad_(False)

        self.adapter = Adapter(input_channels=1024 * len(layers),output_channels=output_channels)


    def forward(self, x):
        B = x.shape[0]
        H,W = x.shape[-2:]
        feat_backbone = self.backbone.get_intermediate_layers(x = x, n = self.layers)
        feat_backbone = torch.cat(feat_backbone,dim=-1)
        feat_backbone = feat_backbone.reshape(B,H // self.SAMPLE_FACTOR,W // self.SAMPLE_FACTOR,-1).permute(0,3,1,2)
        feat,conf = self.adapter(feat_backbone)
        return feat,conf
    
    def unfreeze_backbone(self,layers:List[int] = []):
        parameters = []
        for layer in layers:
            block = self.backbone.blocks[layer]
            block.requires_grad_(True)
            parameters.extend(list(block.parameters()))
            if self.verbose > 0:
                print(f"Unfreeze backbone layer {layer}")
        return parameters
    
    def load_adapter(self,adapter_path:str):
        self.adapter.load_state_dict({k.replace("module.",""):v for k,v in torch.load(adapter_path,map_location='cpu').items()},strict=True)
    
    def save_adapter(self,output_path:str):
        state_dict = {k:v.detach().cpu() for k,v in self.adapter.state_dict().items()}
        torch.save(state_dict,output_path)

    def save_backbone(self,output_path:str):
        state_dict = {k:v.detach().cpu() for k,v in self.backbone.state_dict().items()}
        torch.save(state_dict,output_path)