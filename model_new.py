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

def bnac(channels):
    return nn.Sequential(
        nn.BatchNorm2d(channels),
        nn.ReLU()
    )

class CNNBlock(nn.Module):
    def __init__(self, input_channels = 512, hidden_channels = 256):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(input_channels,hidden_channels,1,1,0),
            bnac(hidden_channels),
            nn.Conv2d(hidden_channels,hidden_channels,3,1,1),
            bnac(hidden_channels),
            nn.Conv2d(hidden_channels,hidden_channels,3,1,1),
            bnac(hidden_channels),
            nn.Conv2d(hidden_channels,hidden_channels,3,1,1),
            bnac(hidden_channels),
            nn.Conv2d(hidden_channels,input_channels,1,1,0)
        )
    def forward(self,x):
        res = self.block(x)
        x = x + res
        return x

class CNN(nn.Module):
    def __init__(self,depth,channels):
        super().__init__()
        self.blocks = nn.ModuleList([CNNBlock(channels,channels//2) for _ in range(depth)])
        self.output = nn.Sequential(
            nn.Conv2d(channels * (depth + 1),channels * (depth + 1) // 2,1,1,0),
            bnac(channels * (depth + 1) // 2),
            nn.Conv2d(channels * (depth + 1) // 2,channels,1,1,0)
        )
    def forward(self,x):
        sum = x
        for blk in self.blocks:
            x = blk(x)
            sum = torch.concatenate([sum,x],dim=1)
        return self.output(sum)

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

class EncoderCNN(nn.Module):
    def __init__(self,input_channels=3,output_channels=512,depth=10):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = output_channels
        self.cnn = CNN(depth,self.hidden_channels)
        self.head = nn.Sequential(
            nn.Conv2d(input_channels,self.hidden_channels // 8,3,2,1),
            bnac(self.hidden_channels // 8),
            nn.Conv2d(self.hidden_channels // 8,self.hidden_channels // 4,3,2,1),
            bnac(self.hidden_channels // 4),
            nn.Conv2d(self.hidden_channels // 4,self.hidden_channels // 2,3,2,1),
            bnac(self.hidden_channels // 2),
            nn.Conv2d(self.hidden_channels // 2,self.hidden_channels,3,2,1),
            bnac(self.hidden_channels),
        )
        self.conf_output = nn.Sequential(
            nn.Conv2d(self.hidden_channels,self.hidden_channels // 16,1,1,0),
            nn.PReLU(),
            nn.Conv2d(self.hidden_channels // 16,1,1,1,0),
            nn.Sigmoid()
        )
    def forward(self,x):
        x = self.head(x)
        feat = self.cnn(x)
        feat = F.normalize(feat,dim=1)
        conf = self.conf_output(feat)
        return feat,conf 

class Encoder0324(nn.Module):
    """
    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self,cfg = {}):
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
        print(f"unfreeze modules: {unfreeze_modules}")

    def get_unfreeze_parameters(self):
        params = []
        unfreeze_names = ['conf_head',*[f'backbone.{i}' for i in self.cfg['unfreeze_backbone_modules']]]
        all_modules = dict(self.named_modules())
        for name in unfreeze_names:
            module = all_modules.get(name,None)
            if not module is None:
                params.extend(list(module.parameters()))
        return params

    def forward(self, x):
        self.r2former.eval()
        feat = self.backbone(x)
        global_feat = self.r2former(self.resize(x)) #F.interpolate(x,size=[480,640],mode='bilinear')
        global_feat = global_feat[:,:,None,None].repeat(1,1,feat.shape[-2],feat.shape[-1])
        feat = F.normalize(feat,p=2,dim=1)
        conf = self.conf_head(feat)
        return torch.cat([feat,global_feat],dim=1),conf
        # return feat,conf

class Encoder0326(nn.Module):
    """
    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self,input_channels=1,output_channels=512,img_size=1024,window_size=8,swt_depth=16,cnn_depth=16,drop_path_rate=.1):
        super().__init__()

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.hidden_channels = output_channels
        self.img_size = img_size
        self.window_size = window_size
        self.swt_depth = swt_depth
        self.cnn_depth = cnn_depth
        self.norm1 = nn.LayerNorm(self.hidden_channels)
        self.norm2 = nn.LayerNorm(self.hidden_channels)

        self.shift_list = torch.tensor([0,1,3,5,7,11,13,17,19,29,31,37])
        self.shift_list = self.shift_list[self.shift_list < window_size]

        self.head = nn.Sequential(
            nn.Conv2d(input_channels,self.hidden_channels // 8,3,2,1),
            bnac(self.hidden_channels // 8),
            nn.Conv2d(self.hidden_channels // 8,self.hidden_channels // 4,3,2,1),
            bnac(self.hidden_channels // 4),
            nn.Conv2d(self.hidden_channels // 4,self.hidden_channels // 2,3,2,1),
            bnac(self.hidden_channels // 2),
            nn.Conv2d(self.hidden_channels // 2,self.hidden_channels,3,2,1),
            bnac(self.hidden_channels),
        )
    
        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, swt_depth)]
        # self.swt_blocks = nn.ModuleList([SwinTransformerBlock(dim=self.hidden_channels,
        #                                                   input_resolution=(int(img_size // 16),int(img_size // 16)),
        #                                                   num_heads=16,
        #                                                   window_size=int(window_size),
        #                                                   shift_size=int(self.shift_list[i % len(self.shift_list)]), #shift_size=,int((i%2)*(window_size // 2))
        #                                                   drop_path=dpr[i])
                                                          
        #                             for i in range(swt_depth)])

        self.cnn_blocks = nn.ModuleList([CNNBlock(self.hidden_channels,self.hidden_channels // 4) for i in range(self.cnn_depth)])
        
        self.conf_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels,self.output_channels // 16,1,1,0),
            nn.PReLU(),
            nn.Conv2d(self.output_channels // 16,1,1,1,0),
            nn.Sigmoid()
        )

        self.feat_head = nn.Sequential(
            nn.Conv2d(self.hidden_channels,self.output_channels,1,1,0),
            nn.PReLU(),
            nn.Conv2d(self.output_channels,self.output_channels,1,1,0),
        )


    def forward(self, x):
        x = self.head(x) # B,D,h,w]
        # B,D,h,w = x.shape
        # x = x.flatten(2).permute(0,2,1)
        # x = self.norm1(x)
        # for blk in self.swt_blocks:
        #     x = blk(x)
        # x = self.norm2(x)
        # x = x.reshape(B,h,w,D).permute(0,3,1,2)
        for blk in self.cnn_blocks:
            x = blk(x)
        feat = self.feat_head(x)
        feat = F.normalize(feat,p=2,dim=1)
        conf = self.conf_head(feat)
        return feat,conf

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
        self.output_lonlat = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,2,1,1,0),
            nn.Tanh()
        )
        self.output_height = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,1,1,1,0)
        )
        self.bn = bnac(in_channels)


    def forward(self, res):
        # res = res / torch.norm(res,dim=1,keepdim=True)
        if self.use_bn:
            res = self.bn(res)
        for block in self.blocks:
            x = block(res)
            res = res + x
        lonlat_res = self.output_lonlat(res)
        height_res = self.output_height(res)
        return torch.cat([lonlat_res,height_res],dim=1)

class Encoder0409(nn.Module):
    """
    FCN encoder, used to extract features from the input images.

    The number of output channels is configurable, the default used in the paper is 512.
    """

    def __init__(self,cfg = {}):
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
        print(f"unfreeze modules: {unfreeze_modules}")

    def get_unfreeze_parameters(self):
        params = []
        unfreeze_names = ['conf_head','cnn',*[f'backbone.{i}' for i in self.cfg['unfreeze_backbone_modules']]]
        all_modules = dict(self.named_modules())
        for name in unfreeze_names:
            module = all_modules.get(name,None)
            if not module is None:
                params.extend(list(module.parameters()))
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


# self.cnn = nn.Sequential(
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#             bnac(self.patch_feature_channels),
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#             bnac(self.patch_feature_channels),
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#         )