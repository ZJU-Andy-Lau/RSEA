import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
import numpy as np
import cv2
from rs_image import RSImage
from rpc import RPCModelParameterTorch
from model.encoder_dino_0927 import EncoderDino
import scheduler
from utils import find_grids,vis_feat_twin,vis_conf,downsample_average

class Window():
    def __init__(self,img:np.ndarray,local:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch):
        self.img = img
        self.local = torch.from_numpy(local)
        self.dem = torch.from_numpy(dem)
        self.rpc = rpc
        self.feature = None
        self.conf = None
        self.affine_matrix = torch.tensor([[1.0,0.0,0.0],
                                            [0.0,1.0,0.0]])
    
    def to_gpu(self):
        self.local = self.local.cuda()
        self.dem = self.dem.cuda()
        self.rpc.to_gpu()
        self.affine_matrix = self.affine_matrix.cuda()
        

def load_imgs(args):
    img_folders = os.listdir(os.path.join(args.root,'adjust_images'))
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    img_0 = RSImage(args,os.path.join(args.root,img_folders[select_img_idxs[0]]),0)
    img_1 = RSImage(args,os.path.join(args.root,img_folders[select_img_idxs[1]]),1)
    return img_0,img_1

@torch.no_grad()
def extract_feature(encoder:EncoderDino,img_raw:np.ndarray):
    """
    img_raw:(H,W,3)
    """
    encoder = encoder.cuda().eval()
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                ])
    img_tensor = transform(img_raw)
    img_tensor = img_tensor[None].cuda()
    feature,conf = encoder(img_tensor)
    
    return feature,conf

def warp_local(local:torch.Tensor,dem:torch.Tensor,rpc_src:RPCModelParameterTorch,rpc_dst:RPCModelParameterTorch,affine_matrix:torch.Tensor):
    local = local.reshape(-1,2)
    dem = dem.reshape(-1)
    ones = torch.ones(local.shape[0],1).to(device=local.device,dtype=local.dtype)
    local_homo = torch.cat([local,ones],dim=-1)
    trans_local = local_homo @ affine_matrix.T
    lats,lons = rpc_src.RPC_PHOTO2OBJ(trans_local[:,1],trans_local[:,0],dem)
    samps,lines = rpc_dst.RPC_OBJ2PHOTO(lats,lons,dem)
    warped_local = torch.stack([lines,samps],dim=-1)
    return warped_local

def feature_sampling(feature:torch.Tensor, local:torch.Tensor, query:torch.Tensor,sharpness = 10.0):
    h, w, d = feature.shape

    feature_flat = feature.reshape(h * w, d)
    local_flat = local.reshape(h * w, 2)

    dist_sq = torch.sum(
        (local_flat.unsqueeze(1) - query.unsqueeze(0))**2,
        dim=-1
    )

    attention_weights = torch.nn.functional.softmax(-dist_sq * sharpness, dim=0)

    sampled_feature = attention_weights.T @ feature_flat

    return sampled_feature

def fit_affine(args,window_0:Window,window_1:Window):
    """
    把window_1 warp到 window_0
    """
    window_0.to_gpu()
    window_1.to_gpu()
    params = nn.Parameter(window_1.affine_matrix).cuda()
    optimizer = torch.optim.Adam([params],lr = args.max_lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer,
                                                    max_lr=args.max_lr,
                                                    total_steps=args.max_iter
                                                    )
    for iter in range(args.max_iter):
        optimizer.zero_grad()
        query_local = warp_local(window_1.local,window_1.dem,window_1.rpc,window_0.rpc,params)
        query_feature = window_1.feature.flatten(0,1) # N,D
        sample_feature = feature_sampling(window_0.feature,window_0.local,query_local,args.sharpness) # N,D
        loss = torch.norm(query_feature - sample_feature,dim=-1).mean() * 100.
        
        loss.backward()
        optimizer.step()

        if (iter + 1) % 10 == 0:
            af = params.reshape(-1).detach().cpu().numpy()
            print(f"iter:{iter}/{args.max_iter} \t loss:{loss.item():.4f} \t lr:{scheduler.get_lr()[0]} \t af:{af}")
        
        
        scheduler.step()
    
    final_affine_matrix = params.detach().cpu().numpy()

    print(f"final affine matrix: \n {final_affine_matrix}")
        


if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')

    parser.add_argument('--dino_path', type=str, default='weights',
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--encoder_path', type=str, default='weights/pretrain_swt_cnn_r2_0409_large/backbone.pth',
                        help='file containing pre-trained encoder weights')
    
    parser.add_argument('--max_lr', type=float, default=0.0001,
                        help='highest learning rate')

    parser.add_argument('--max_iter', type=int, default=1000)

    parser.add_argument('--sharpness',type=float,default=10.0)

    parser.add_argument('--window_size', type=int, default=2000,help='window size in meter(m)')

    parser.add_argument('--select_imgs',type=str,default='0,1') #前期只测试两张图像配准

    args = parser.parse_args()

    debug_output_path = os.path.join(args.root,'debug_output')
    os.makedirs(debug_output_path,exist_ok=True)

    img_0,img_1 = load_imgs(args)

    print("images loaded")

    corners = np.stack([img_0.corner_xys,img_1.corner_xys],axis=0)
    grid_diag = find_grids(corners,args.window_size,grid_num=1)[0]

    print(f"grid:{grid_diag}")

    resample_size = 1024
    corners_sampline_0 = img_0.xy_to_sampline(np.array([grid_diag[0],
                                                        [grid_diag[1,0],grid_diag[0,1]],
                                                        grid_diag[1],
                                                        [grid_diag[0,0],grid_diag[1,1]]]))
    corners_sampline_1 = img_1.xy_to_sampline(np.array([grid_diag[0],
                                                        [grid_diag[1,0],grid_diag[0,1]],
                                                        grid_diag[1],
                                                        [grid_diag[0,0],grid_diag[1,1]]]))
    
    img_0_raw,local_0 = img_0.resample_image_by_sampline(corners_sampline_0,(resample_size,resample_size),need_local=True)
    img_1_raw,local_1 = img_1.resample_image_by_sampline(corners_sampline_1,(resample_size,resample_size),need_local=True)
    dem_0 = img_0.resample_dem_by_sampline(corners_sampline_0,(resample_size,resample_size))
    dem_1 = img_1.resample_dem_by_sampline(corners_sampline_1,(resample_size,resample_size))

    window_0 = Window(img_0_raw,local_0,dem_0,img_0.rpc)
    window_1 = Window(img_1_raw,local_1,dem_1,img_1.rpc)

    cv2.imwrite(os.path.join(debug_output_path,'img_raw_0.png'),img_0_raw)
    cv2.imwrite(os.path.join(debug_output_path,'img_raw_1.png'),img_1_raw)

    encoder = EncoderDino(os.path.join(args.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'))
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    feature_0,conf_0 = extract_feature(encoder,img_0_raw)
    feature_1,conf_1 = extract_feature(encoder,img_1_raw)
    window_0.feature = feature_0[0].permute(1,2,0)
    window_0.conf = conf_0.squeeze()
    window_0.local = downsample_average(window_0.local,encoder.SAMPLE_FACTOR)
    window_0.dem = downsample_average(window_0.dem,encoder.SAMPLE_FACTOR)
    window_1.feature = feature_1[0].permute(1,2,0)
    window_1.conf = conf_1.squeeze()
    window_1.local = downsample_average(window_1.local,encoder.SAMPLE_FACTOR)
    window_1.dem = downsample_average(window_1.dem,encoder.SAMPLE_FACTOR)

    print("=======================window info=======================")
    print(f"sample factor:{encoder.SAMPLE_FACTOR}")
    print(f"feature:{window_0.feature.shape}")
    print(f"conf:{window_0.conf.shape}")
    print(f"local:{window_0.local.shape}")
    print(f"dem:{window_0.dem.shape}")
    print("\n")
    

    feat_0_vis = window_0.feature.cpu().numpy()
    feat_1_vis = window_1.feature.cpu().numpy()
    conf_0_vis = window_0.conf.cpu().numpy()
    conf_1_vis = window_1.conf.cpu().numpy()
    feat_vis_img = vis_feat_twin(feat_0_vis,feat_1_vis)
    conf_cont_0,conf_div_0 = vis_conf(conf_0_vis,img_0_raw,encoder.SAMPLE_FACTOR)
    conf_cont_1,conf_div_1 = vis_conf(conf_1_vis,img_1_raw,encoder.SAMPLE_FACTOR)
    cv2.imwrite(os.path.join(debug_output_path,'feat_vis.png'),feat_vis_img)
    cv2.imwrite(os.path.join(debug_output_path,'conf_cont_0.png'),conf_cont_0)
    cv2.imwrite(os.path.join(debug_output_path,'conf_div_0.png'),conf_div_0)
    cv2.imwrite(os.path.join(debug_output_path,'conf_cont_1.png'),conf_cont_1)
    cv2.imwrite(os.path.join(debug_output_path,'conf_div_1.png'),conf_div_1)

    fit_affine(args,window_0,window_1)




    





