import torch
import torch.nn as nn
import torch.nn.functional as F

def window_partition(x: torch.Tensor, window_size: int):
    """
    输入形状: (B, H, W, C)
    输出形状: (B*num_windows, window_size, window_size, C)
    """
    B, H, W, C = x.shape
    H_padded = (H + window_size - 1) // window_size * window_size
    W_padded = (W + window_size - 1) // window_size * window_size
    x = F.pad(x, (0, 0, 0, W_padded - W, 0, H_padded - H))  # 填充右侧和底部
    x = x.view(B, H_padded // window_size, window_size, W_padded // window_size, window_size, C)
    windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
    return windows

def window_reverse(windows: torch.Tensor, window_size: int, B:int, H: int, W: int):
    """
    输入形状: (B*num_windows, window_size, window_size, C)
    输出形状: (B, H, W, C)
    """
    # B = windows.shape[0] // ((H * W) // (window_size**2))
    H_pad = (H + window_size - 1) // window_size * window_size
    W_pad = (W + window_size - 1) // window_size * window_size

    x = windows.view(B, H_pad // window_size, W_pad // window_size, window_size, window_size, -1)
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous()  # (B, H_pad//win, win, W_pad//win, win, C)
    x = x.view(B, H_pad, W_pad, -1)                # (B, H_pad, W_pad, C)
    print(x.shape)
    x = x[:, :H, :W, :]         # 裁剪填充部分
    print(x.shape)
    return x
    # B = windows.shape[0] // (H * W // window_size**2)
    # x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    # x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
    # return x.view(B, H, W, -1)

class ShiftedWindowAttention(nn.Module):
    def __init__(self, embed_dim: int, window_size: int, num_heads: int, shift_size: int = 0):
        super().__init__()
        self.window_size = window_size
        self.shift_size = shift_size
        self.num_heads = num_heads
        
        # 核心模块：基于 MultiheadAttention 的窗口注意力
        self.attn = nn.MultiheadAttention(embed_dim, num_heads)
        
        # 相对位置编码（参考 Swin Transformer 实现）
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) ** 2, num_heads)
        )
        self._init_relative_position_index()

    def _init_relative_position_index(self):
        # 生成窗口内相对位置索引 
        coords = torch.stack(torch.meshgrid(
            [torch.arange(self.window_size), 
             torch.arange(self.window_size)],
            indexing='ij'
        )).flatten(1)
        relative_coords = coords[:, :, None] - coords[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords += self.window_size - 1
        relative_coords[:, :, 0] *= 2 * self.window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))

    def _apply_shift(self, x: torch.Tensor):
        """应用滑动窗口移位 """
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        return x

    def _reverse_shift(self, x: torch.Tensor):
        """恢复滑动窗口位置 """
        if self.shift_size > 0:
            x = torch.roll(x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        return x

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape
        
        # 滑动窗口移位 
        shifted_x = self._apply_shift(x)
        
        # 窗口划分 
        windows = window_partition(shifted_x, self.window_size)  # (B*num_windows, M, M, C)
        windows = windows.view(-1, self.window_size**2, C)      # (B*num_windows, M*M, C)
        
        # 相对位置编码 
        relative_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size**2, self.window_size**2, -1)
        relative_bias = relative_bias.permute(2, 0, 1)  # (num_heads, M*M, M*M)
        
        # 注意力计算（MultiheadAttention 需要形状调整）
        windows = windows.permute(1, 0, 2)  # (seq_len, batch, C) 符合 nn.MultiheadAttention 输入
        attn_output, _ = self.attn(
            query=windows, 
            key=windows, 
            value=windows,
            attn_mask=relative_bias.repeat(windows.size(1), 1, 1)  # 扩展至所有批次
        )
        attn_output = attn_output.permute(1, 0, 2)  # 恢复 (batch*num_windows, M*M, C)
        
        # 窗口复原 
        attn_output = attn_output.view(-1, self.window_size, self.window_size, C)
        shifted_output = window_reverse(attn_output, self.window_size, B, H, W)
        
        # 恢复滑动窗口位置 
        x = self._reverse_shift(shifted_output)
        return x
    
class LocalWindowAttention(nn.Module):
    def __init__(self, dim, window_size=7, num_heads=8, qkv_bias=True):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        assert self.head_dim * num_heads == dim, "dim必须能被num_heads整除"
        
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        
        # 相对位置编码表（每个头独立）
        self.rel_pos_table = nn.Parameter(
            torch.randn(2 * window_size - 1, 2 * window_size - 1, num_heads) * 0.02)
        
        # 预计算相对位置索引
        coords = torch.stack(torch.meshgrid(
            torch.arange(window_size), 
            torch.arange(window_size)), dim=0)
        coords_flatten = coords.view(2, -1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        self.register_buffer("relative_index", relative_coords)

    def forward(self, x):
        B, H, W, C = x.shape
        pad_r = (self.window_size - W % self.window_size) % self.window_size
        pad_b = (self.window_size - H % self.window_size) % self.window_size
        x = F.pad(x, (0, 0, 0, pad_r, 0, pad_b))  # 仅填充右和下
        
        # 生成填充掩码 [B, H_pad, W_pad]
        H_pad, W_pad = H + pad_b, W + pad_r
        mask = torch.ones((B, H_pad, W_pad), dtype=torch.bool, device=x.device)
        mask[:, :H, :W] = False  # 有效区域为False，填充区域为True
        
        # 分块处理输入和掩码
        x = x.view(B, H_pad//self.window_size, self.window_size,
                   W_pad//self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, self.window_size, self.window_size, C)
        
        mask = mask.view(B, H_pad//self.window_size, self.window_size,
                         W_pad//self.window_size, self.window_size)
        mask = mask.permute(0, 1, 3, 2, 4).contiguous()
        mask = mask.view(-1, self.window_size * self.window_size)
        
        # 生成QKV并分头
        qkv = self.qkv(x)
        q, k, v = qkv.chunk(3, dim=-1)
        
        # 转换为多头 [B*num_blocks, num_heads, ws^2, head_dim]
        q = q.view(-1, self.window_size**2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        k = k.view(-1, self.window_size**2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        v = v.view(-1, self.window_size**2, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 注意力计算（加入掩码）
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        
        # 应用填充掩码（关键修改）
        mask = mask.view(-1, 1, 1, self.window_size**2)  # [B*num_blocks, 1, 1, ws^2]
        attn = attn.masked_fill(mask, -1e4)  # 填充区域分数设为负无穷
        
        # 相对位置编码
        h_idx, w_idx = self.relative_index[...,0], self.relative_index[...,1]
        rel_pos_bias = self.rel_pos_table[h_idx, w_idx].permute(2,0,1).contiguous()
        attn += rel_pos_bias.unsqueeze(0)
        
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(-1, self.window_size, self.window_size, C)
        
        # 还原空间维度
        x = x.view(B, H_pad//self.window_size, W_pad//self.window_size,
                   self.window_size, self.window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(B, H_pad, W_pad, C)[:, :H, :W, :]  # 去除填充部分
        return x