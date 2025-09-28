import torch.nn as nn
import torch.nn.functional as F
from swin_transformer_v2 import SwinTransformerV2
from torchvision import transforms
from typing import List



class Encoder(nn.Module):

    def __init__(self,cfg = {},verbose = 1):
        super().__init__()
        default_cfg = {
            'input_channels':3,
            'output_channels':512,
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
        self.output_channels = self.cfg['output_channels']

        self.backbone = SwinTransformerV2(img_size=self.cfg['img_size'],
                                        drop_path_rate=self.cfg['drop_path_rate'],
                                        embed_dim=self.cfg['embed_dim'],
                                        depths=self.cfg['depth'],
                                        num_heads=self.cfg['num_heads'],
                                        window_size=self.cfg['window_size'],
                                        in_chans=self.cfg['input_channels'],
                                        out_chans=self.cfg['output_channels'],
                                        pretrained_window_sizes=self.cfg['pretrain_window_size']
                                        )
        self.backbone_modules = dict(self.backbone.named_modules())
        self.backbone.requires_grad_(False)

        self.cnn = nn.Sequential(
            nn.Conv2d(self.output_channels,self.output_channels,3,1,1),
            nn.ReLU(),
            nn.Conv2d(self.output_channels,self.output_channels,3,1,1),
            nn.ReLU(),
            nn.Conv2d(self.output_channels,self.output_channels,3,1,1),
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(self.output_channels,self.output_channels // 16,1,1,0),
            nn.PReLU(),
            nn.Conv2d(self.output_channels // 16,1,1,1,0),
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
        feat_backbone = self.backbone(x)
        feat = self.cnn(feat_backbone)
        conf = self.conf_head(F.normalize(feat_backbone,dim=1))

        feat = F.normalize(feat,p=2,dim=1)
        
        return feat,conf