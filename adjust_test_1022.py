import os
import argparse
from matplotlib.rcsetup import validate_markevery
import torch
import torch.nn.functional as F
import torch.nn as nn
from torchvision import transforms
from pykeops.torch import LazyTensor
import numpy as np
import cv2
from rs_image_1022 import RSImage
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
        
    
    def to_gpu(self):
        self.local = self.local.cuda()
        self.dem = self.dem.cuda()
        self.rpc.to_gpu()

    

class Window_Pair():
    def __init__(self,args,diag:np.ndarray,img_0:RSImage,img_1:RSImage,id:int):
        self.id = id
        resample_size = 1024
        corners_sampline_0 = img_0.xy_to_sampline(np.array([diag[0],
                                                            [diag[1,0],diag[0,1]],
                                                            diag[1],
                                                            [diag[0,0],diag[1,1]]]))
        corners_sampline_1 = img_1.xy_to_sampline(np.array([diag[0],
                                                            [diag[1,0],diag[0,1]],
                                                            diag[1],
                                                            [diag[0,0],diag[1,1]]]))
        
        img_0_raw,local_0 = img_0.resample_image_by_sampline(corners_sampline_0,(resample_size,resample_size),need_local=True)
        img_1_raw,local_1 = img_1.resample_image_by_sampline(corners_sampline_1,(resample_size,resample_size),need_local=True)
        dem_0 = img_0.resample_dem_by_sampline(corners_sampline_0,(resample_size,resample_size))
        dem_1 = img_1.resample_dem_by_sampline(corners_sampline_1,(resample_size,resample_size))

        self.window_0 = Window(img_0_raw,local_0,dem_0,img_0.rpc)
        self.window_1 = Window(img_1_raw,local_1,dem_1,img_1.rpc)

        self.debug_output_path = os.path.join(args.debug_output_path,f'window_pair_{self.id}')
        os.makedirs(self.debug_output_path,exist_ok=True)

        cv2.imwrite(os.path.join(self.debug_output_path,'img_raw_0.png'),img_0_raw)
        cv2.imwrite(os.path.join(self.debug_output_path,'img_raw_1.png'),img_1_raw)

    @torch.no_grad()
    def __extract_one_img_feature__(self,encoder:EncoderDino,img_raw:np.ndarray):
        encoder = encoder.cuda().eval()
        transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)) 
                    ])
        img_tensor = transform(img_raw)
        img_tensor = img_tensor[None].cuda()
        feature,conf = encoder(img_tensor)
        
        return feature,conf

    def extract_features(self,encoder:EncoderDino):
        feature_0,conf_0 = self.__extract_one_img_feature__(encoder,self.window_0.img)
        feature_1,conf_1 = self.__extract_one_img_feature__(encoder,self.window_1.img)
        h,w = feature_0.shape[-2:]
        self.window_0.feature = feature_0[0].permute(1,2,0).flatten(0,1)
        self.window_0.conf = conf_0.squeeze().flatten(0,1)
        self.window_0.local = downsample_average(self.window_0.local,encoder.SAMPLE_FACTOR).flatten(0,1).to(self.window_0.feature.device)
        self.window_0.dem = downsample_average(self.window_0.dem,encoder.SAMPLE_FACTOR).flatten(0,1).to(self.window_0.feature.device)
        self.window_1.feature = feature_1[0].permute(1,2,0).flatten(0,1)
        self.window_1.conf = conf_1.squeeze().flatten(0,1)
        self.window_1.local = downsample_average(self.window_1.local,encoder.SAMPLE_FACTOR).flatten(0,1).to(self.window_1.feature.device)
        self.window_1.dem = downsample_average(self.window_1.dem,encoder.SAMPLE_FACTOR).flatten(0,1).to(self.window_1.feature.device)

        feat_0_vis = self.window_0.feature.cpu().numpy().reshape(h,w,-1)
        feat_1_vis = self.window_1.feature.cpu().numpy().reshape(h,w,-1)
        conf_0_vis = self.window_0.conf.cpu().numpy().reshape(h,w)
        conf_1_vis = self.window_1.conf.cpu().numpy().reshape(h,w)
        feat_vis_img = vis_feat_twin(feat_0_vis,feat_1_vis)
        conf_cont_0,conf_div_0 = vis_conf(conf_0_vis,self.window_0.img,encoder.SAMPLE_FACTOR,div=args.conf_threshold)
        conf_cont_1,conf_div_1 = vis_conf(conf_1_vis,self.window_1.img,encoder.SAMPLE_FACTOR,div=args.conf_threshold)
        cv2.imwrite(os.path.join(self.debug_output_path,'feat_vis.png'),feat_vis_img)
        cv2.imwrite(os.path.join(self.debug_output_path,'conf_cont_0.png'),conf_cont_0)
        cv2.imwrite(os.path.join(self.debug_output_path,'conf_div_0.png'),conf_div_0)
        cv2.imwrite(os.path.join(self.debug_output_path,'conf_cont_1.png'),conf_cont_1)
        cv2.imwrite(os.path.join(self.debug_output_path,'conf_div_1.png'),conf_div_1)

        self.window_0.to_gpu()
        self.window_1.to_gpu()

        

def load_imgs(args):
    img_folders = os.listdir(os.path.join(args.root,'adjust_images'))
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    img_0 = RSImage(args,os.path.join(args.root,img_folders[select_img_idxs[0]]),0)
    img_1 = RSImage(args,os.path.join(args.root,img_folders[select_img_idxs[1]]),1)
    return img_0,img_1


def warp_local(local:torch.Tensor,dem:torch.Tensor,rpc_src:RPCModelParameterTorch,rpc_dst:RPCModelParameterTorch,affine_matrix:torch.Tensor):
    ones = torch.ones(local.shape[0],1).to(device=local.device,dtype=local.dtype)
    local_homo = torch.cat([local,ones],dim=-1)
    trans_local = local_homo @ affine_matrix.T
    lats,lons = rpc_src.RPC_PHOTO2OBJ(trans_local[:,1],trans_local[:,0],dem)
    samps,lines = rpc_dst.RPC_OBJ2PHOTO(lats,lons,dem)
    warped_local = torch.stack([lines,samps],dim=-1).to(torch.float32)
    return warped_local

def feature_sampling(feature:torch.Tensor, conf:torch.Tensor, local:torch.Tensor, query:torch.Tensor,k = 16):
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
    reverse_dists_ratio = 1. / (dists_ratio + 1e-6)
    weights = reverse_dists_ratio / torch.sum(reverse_dists_ratio,dim=1,keepdim=True)

    feature_sample_p3d = feature[idxs]
    feature_sample_pd = torch.sum(feature_sample_p3d * weights.unsqueeze(-1),dim=1).to(torch.float32)

    conf_sample_p3 = conf[idxs]
    conf_sample_p = torch.sum(conf_sample_p3 * weights,dim=1).to(torch.float32)

    return feature_sample_pd,conf_sample_p,valid_mask


def fit_affine(args,window_pairs:list[Window_Pair]):
    """
    把window_1 warp到 window_0
    """
    
    R = nn.Parameter(torch.tensor([[1.0,0.0],
                                [0.0,1.0]]).cuda())
    T = nn.Parameter(torch.tensor([args.init_offset_line,args.init_offset_samp]).cuda())
    optimizer_r = torch.optim.Adam([R],lr = args.max_lr * 0.000001)
    optimizer_t = torch.optim.Adam([T],lr = args.max_lr)
    scheduler_r = torch.optim.lr_scheduler.OneCycleLR(optimizer_r,
                                                        max_lr=args.max_lr * 0.000001,
                                                        total_steps=args.max_iter
                                                        )
    scheduler_t = torch.optim.lr_scheduler.OneCycleLR(optimizer_t,
                                                        max_lr=args.max_lr,
                                                        total_steps=args.max_iter
                                                        )
    for iter in range(args.max_iter):
        optimizer_r.zero_grad()
        optimizer_t.zero_grad()
        af_mat = torch.concatenate([R,T.unsqueeze(-1)],dim=-1)
        total_loss = 0
        for window_pair in window_pairs:
            window_0,window_1 = window_pair.window_0,window_pair.window_1
            query_local = warp_local(window_1.local,window_1.dem,window_1.rpc,window_0.rpc,af_mat)
            sample_feature,sample_conf,valid_mask = feature_sampling(window_0.feature,window_0.conf,window_0.local,query_local,args.kmin_k) # N,D
            query_feature = window_1.feature[valid_mask] # N,D
            query_conf = window_1.conf[valid_mask]
            conf_cov = query_conf * sample_conf
            
            weight = conf_cov / conf_cov.mean()
            loss = (torch.norm(query_feature - sample_feature,dim=-1) * weight).mean() * 10000.
            total_loss = total_loss + loss
        total_loss = total_loss / len(window_pairs)
        total_loss.backward()
        optimizer_r.step()
        optimizer_t.step()

        if (iter + 1) % 10 == 0:
            af = af_mat.detach().cpu().numpy()
            with np.printoptions(precision=5, suppress=False):
                print(f"iter:{iter+1}/{args.max_iter} \t loss:{total_loss.item():.4f} \t lr:{scheduler_t.get_lr()[0]:.2e} \n af:{af}")
        
        
        scheduler_r.step()
        scheduler_t.step()

    window_1.rpc.Update_Adjust(af_mat.detach())
    final_affine_matrix = af_mat.detach().cpu().numpy()

    print(f"final affine matrix: \n {final_affine_matrix}")
        
def check_error(images:list[RSImage]):        
        def haversine_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
            R = 6371000 
            lat1 = coords1[:, 0]
            lon1 = coords1[:, 1]
            lat2 = coords2[:, 0]
            lon2 = coords2[:, 1]

            lat1_rad = np.radians(lat1)
            lon1_rad = np.radians(lon1)
            lat2_rad = np.radians(lat2)
            lon2_rad = np.radians(lon2)

            dlat = lat2_rad - lat1_rad
            dlon = lon2_rad - lon1_rad

            a = np.sin(dlat / 2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2)**2
            c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1 - a))
            distance = R * c
            
            return distance
        
        error_flag = False
        for image in images:
            if image.tie_points is None:
                print(f"image {image.id} has no tie points")
                error_flag = True
        if error_flag:
            print("error check aborted")
            return
        
        coords = []
        distances = []
        for image in images:
            lines = image.tie_points[:,0]
            samps = image.tie_points[:,1]
            heights = image.dem[lines,samps]
            lats,lons = image.rpc.RPC_PHOTO2OBJ(samps,lines,heights,'numpy')
            coords.append(np.stack([lats,lons],axis=-1))
        n = len(coords)
        for i in range(n-1):
            for j in range(i+1,n):
                distances.append(haversine_distance(coords[i],coords[j]))
        
        distances = np.stack(distances,axis=-1).reshape(-1)    

        return distances

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

    parser.add_argument('--conf_threshold',type=float,default=.5)

    parser.add_argument('--kmin_k',type=int,default=16)

    parser.add_argument('--window_size', type=int, default=2000,help='window size in meter(m)')

    parser.add_argument('--select_imgs',type=str,default='0,1') #前期只测试两张图像配准

    parser.add_argument('--init_offset_line',type=float,default=0.)

    parser.add_argument('--init_offset_samp',type=float,default=0.)

    parser.add_argument('--grid_offset_x',type=float,default=0)

    parser.add_argument('--grid_offset_y',type=float,default=0)

    parser.add_argument('--grid_num',type=int,default=1)

    args = parser.parse_args()

    args.debug_output_path = os.path.join(args.root,'debug_output')
    os.makedirs(args.debug_output_path,exist_ok=True)

    img_0,img_1 = load_imgs(args)

    print("images loaded")

    encoder = EncoderDino(os.path.join(args.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'))
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))

    print("Encoder Loaded")

    corners = np.stack([img_0.corner_xys,img_1.corner_xys],axis=0)
    diags = find_grids(corners,args.window_size,offset_x=args.grid_offset_x,offset_y=args.grid_offset_y,grid_num=args.grid_num)

    window_pairs = []
    for id,diag in enumerate(diags):
        window_pair = Window_Pair(args,diag,img_0,img_1,id)
        window_pair.extract_features(encoder)
        window_pairs.append(window_pair)
        print(f"Window pair {id} created")

    
    fit_affine(args,window_pairs)

    errors = check_error([img_0,img_1])
    info = f"error:\nmax:{errors.max()}\nmin:{errors.min()}\nmean:{errors.mean()}\nmedian:{np.median(errors)}\n<1px:{(errors < 1.).sum() * 1. / len(errors)}\n<3px:{(errors < 3.).sum() * 1. / len(errors)}\n<5px:{(errors < 5.).sum() * 1. / len(errors)}"
    print(info)


    





