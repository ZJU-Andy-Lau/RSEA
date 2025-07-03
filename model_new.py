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




# self.cnn = nn.Sequential(
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#             bnac(self.patch_feature_channels),
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#             bnac(self.patch_feature_channels),
#             nn.Conv2d(self.patch_feature_channels,self.patch_feature_channels,3,1,1),
#         )