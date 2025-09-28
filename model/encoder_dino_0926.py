import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List

class Adapter(nn.Module):
    def __init__(self,input_channels = 512,output_channels = 512):
        super().__init__()
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.cnn = nn.Sequential(
            nn.Conv2d(self.input_channels,self.input_channels // 4,1,1,0),
            nn.BatchNorm2d(self.input_channels // 4),
            nn.ReLU(),
            nn.Conv2d(self.input_channels // 4,self.output_channels,1,1,0),
            nn.BatchNorm2d(self.output_channels),
            nn.ReLU(),
            nn.Conv2d(self.output_channels,self.output_channels,3,1,1),
        )

        self.conf_head = nn.Sequential(
            nn.Conv2d(self.input_channels,self.input_channels // 4,1,1,0),
            nn.BatchNorm2d(self.input_channels // 4),
            nn.ReLU(),
            nn.Conv2d(self.input_channels // 4, self.input_channels // 16,1,1,0),
            nn.BatchNorm2d(self.input_channels // 16),
            nn.ReLU(),
            nn.Conv2d(self.input_channels // 16, 1 ,3,1,1,padding_mode='replicate'),
            nn.Sigmoid()
        )
    def forward(self,x):
        feat = self.cnn(x)
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