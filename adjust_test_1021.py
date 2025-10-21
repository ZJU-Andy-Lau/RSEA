import os
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from pykeops.torch import LazyTensor
import numpy as np
import cv2
from rs_image import RSImage
from rpc import RPCModelParameterTorch
from model.encoder_dino_0927 import EncoderDino
import scheduler
from utils import find_grids,vis_feat_twin,vis_conf,downsample_average

import warnings
warnings.filterwarnings("ignore")

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
    ones = torch.ones(local.shape[0],1).to(device=local.device,dtype=local.dtype)
    local_homo = torch.cat([local,ones],dim=-1)
    trans_local = local_homo @ affine_matrix.T
    lats,lons = rpc_src.RPC_PHOTO2OBJ(trans_local[:,1],trans_local[:,0],dem)
    samps,lines = rpc_dst.RPC_OBJ2PHOTO(lats,lons,dem)
    warped_local = torch.stack([lines,samps],dim=-1).to(torch.float32)
    return warped_local

def feature_sampling(feature:torch.Tensor, local:torch.Tensor, query:torch.Tensor,k = 16):
    point_base = LazyTensor(local.contiguous().unsqueeze(0))
    query_lazy = LazyTensor(query.contiguous().unsqueeze(1))
    dist_ij:LazyTensor = ((query_lazy - point_base) ** 2).sum(-1)
    dists,idxs = dist_ij.Kmin_argKmin(K = k, dim=1)

    locals_kmin = local[idxs] # n,k,2
    dists = torch.cdist(query.unsqueeze(1),locals_kmin,p=2).squeeze(1)

    valid_mask = (dists.min(dim=1).values < 8)
    dists = dists[valid_mask]
    idxs = idxs[valid_mask]

    dists_ratio = dists / torch.sum(dists,dim=1,keepdim=True) # n,k
    reverse_dists_ratio = 1. / dists_ratio
    weights = reverse_dists_ratio / torch.sum(reverse_dists_ratio,dim=1,keepdim=True)

    feature_sample_p3d = feature[idxs]
    feature_sample_pd = torch.sum(feature_sample_p3d * weights.unsqueeze(-1),dim=1).to(torch.float32)

    return feature_sample_pd,valid_mask


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
        sample_feature,valid_mask = feature_sampling(window_0.feature,window_0.local,query_local,args.kmin_k) # N,D
        query_feature = window_1.feature[valid_mask] # N,D
        loss = torch.norm(query_feature - sample_feature,dim=-1).mean() * 100.
        
        loss.backward()
        optimizer.step()

        if (iter + 1) % 10 == 0:
            af = params.reshape(-1).detach().cpu().numpy()
            with np.printoptions(precision=3, suppress=True):
                print(f"iter:{iter+1}/{args.max_iter} \t loss:{loss.item():.4f} \t lr:{scheduler.get_lr()[0]:.2e} \t af:{af}")
        
        
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

    parser.add_argument('--kmin_k',type=int,default=16)

    parser.add_argument('--window_size', type=int, default=2000,help='window size in meter(m)')

    parser.add_argument('--select_imgs',type=str,default='0,1') #前期只测试两张图像配准

    parser.add_argument('--init_offset_line',type=float,default=0.)

    parser.add_argument('--init_offset_samp',type=float,default=0.)

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

    window_1.affine_matrix[0,2] += args.init_offset_line
    window_1.affine_matrix[1,2] += args.init_offset_samp

    encoder = EncoderDino(os.path.join(args.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'))
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    feature_0,conf_0 = extract_feature(encoder,img_0_raw)
    feature_1,conf_1 = extract_feature(encoder,img_1_raw)
    h,w = feature_0.shape[-2:]
    window_0.feature = feature_0[0].permute(1,2,0).flatten(0,1)
    window_0.conf = conf_0.squeeze().flatten(0,1)
    window_0.local = downsample_average(window_0.local,encoder.SAMPLE_FACTOR).flatten(0,1)
    window_0.dem = downsample_average(window_0.dem,encoder.SAMPLE_FACTOR).flatten(0,1)
    window_1.feature = feature_1[0].permute(1,2,0).flatten(0,1)
    window_1.conf = conf_1.squeeze().flatten(0,1)
    window_1.local = downsample_average(window_1.local,encoder.SAMPLE_FACTOR).flatten(0,1)
    window_1.dem = downsample_average(window_1.dem,encoder.SAMPLE_FACTOR).flatten(0,1)

    window_0.to_gpu()
    window_1.to_gpu()

    print("=======================window info=======================")
    print(f"sample factor:{encoder.SAMPLE_FACTOR}")
    print(f"feature:{window_0.feature.shape}")
    print(f"conf:{window_0.conf.shape}")
    print(f"local:{window_0.local.shape}")
    print(f"dem:{window_0.dem.shape}")
    print("\n")
    

    feat_0_vis = window_0.feature.cpu().numpy().reshape(h,w,-1)
    feat_1_vis = window_1.feature.cpu().numpy().reshape(h,w,-1)
    conf_0_vis = window_0.conf.cpu().numpy().reshape(h,w)
    conf_1_vis = window_1.conf.cpu().numpy().reshape(h,w)
    feat_vis_img = vis_feat_twin(feat_0_vis,feat_1_vis)
    conf_cont_0,conf_div_0 = vis_conf(conf_0_vis,img_0_raw,encoder.SAMPLE_FACTOR)
    conf_cont_1,conf_div_1 = vis_conf(conf_1_vis,img_1_raw,encoder.SAMPLE_FACTOR)
    cv2.imwrite(os.path.join(debug_output_path,'feat_vis.png'),feat_vis_img)
    cv2.imwrite(os.path.join(debug_output_path,'conf_cont_0.png'),conf_cont_0)
    cv2.imwrite(os.path.join(debug_output_path,'conf_div_0.png'),conf_div_0)
    cv2.imwrite(os.path.join(debug_output_path,'conf_cont_1.png'),conf_cont_1)
    cv2.imwrite(os.path.join(debug_output_path,'conf_div_1.png'),conf_div_1)

    fit_affine(args,window_0,window_1)




    





