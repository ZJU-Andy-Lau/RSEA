import os
import argparse
from torchvision import transforms
import numpy as np
from model_new import EncoderDino
from utils import vis_conf,vis_feat_pca,get_current_time
import warnings
warnings.filterwarnings("ignore")
import cv2
import torch


@torch.no_grad()
def vis(encoder:EncoderDino,vis_img:np.ndarray,output_folder):
    os.makedirs(output_folder,exist_ok=True)
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.430, 0.411, 0.296), (0.213, 0.156, 0.143)) # (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
                ])
    input = transform(vis_img).unsqueeze(0).to(encoder.device)
    feat,conf = encoder(input)
    h,w,c = feat.shape[-2],feat.shape[-1],feat.shape[1]
    feat = feat.permute(0,2,3,1).reshape(h,w,c).cpu().numpy()
    conf = conf.reshape(h,w).cpu().numpy()
    vis_feat_pca(feat,os.path.join(output_folder,f'feat_pca_dino_{get_current_time()}.png'))
    vis_conf(conf,vis_img,16,os.path.join(output_folder,f'conf_dino_{get_current_time()}.png'))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--encoder_path',type=str,default=None)
    parser.add_argument('--dino_weight_path',type=str,default=None)
    parser.add_argument('--vis_img_path',type=str,default=None)
    parser.add_argument('--output_folder',type=str,default=None)
    args = parser.parse_args()

    encoder = EncoderDino(dino_weight_path=args.dino_weight_path)
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    encoder = encoder.cuda().eval()

    vis_img = cv2.imread(args.vis_img_path)
    vis(encoder,vis_img,args.output_folder)

