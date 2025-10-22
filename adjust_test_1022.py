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

# DDP相关的库
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import warnings
warnings.filterwarnings("ignore")

# DDP Step 1: DDP环境初始化函数
def setup_ddp():
    """初始化DDP环境"""
    dist.init_process_group(backend='nccl')
    # torchrun 会自动设置 'LOCAL_RANK' 环境变量
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    print(f"DDP setup on rank {local_rank} with device cuda:{local_rank}")
    return local_rank

# DDP Step 2: 将需要优化的参数封装为 nn.Module
class AffineModel(nn.Module):
    """
    将仿射变换参数R和T封装成一个PyTorch模块，以便DDP管理。
    """
    def __init__(self, init_line=0.0, init_samp=0.0):
        super().__init__()
        # R 和 T 必须是 nn.Parameter 才能被DDP和优化器追踪
        # 为了DDP性能，我们使用 float32
        self.R = nn.Parameter(torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32))
        self.T = nn.Parameter(torch.tensor([init_line, init_samp], dtype=torch.float32))

    def forward(self):
        # "forward" 就返回仿射矩阵
        return torch.concatenate([self.R, self.T.unsqueeze(-1)], dim=-1)


class Window():
    def __init__(self,img:np.ndarray,local:np.ndarray,dem:np.ndarray,rpc:RPCModelParameterTorch):
        self.img = img
        self.local = torch.from_numpy(local)
        self.dem = torch.from_numpy(dem)
        self.rpc = rpc
        self.feature = None
        self.conf = None
        
    
    def to_gpu(self):
        # 这里的 .cuda() 会自动使用 torch.cuda.set_device 设置的当前卡
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
        
        # 只在主进程上保存调试图像，避免文件写入冲突
        if dist.get_rank() == 0:
            os.makedirs(self.debug_output_path,exist_ok=True)
            cv2.imwrite(os.path.join(self.debug_output_path,'img_raw_0.png'),img_0_raw)
            cv2.imwrite(os.path.join(self.debug_output_path,'img_raw_1.png'),img_1_raw)


    @torch.no_grad()
    def __extract_one_img_feature__(self,encoder:EncoderDino,img_raw:np.ndarray):
        # encoder 会被移动到当前进程对应的GPU
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

        # 只在主进程上保存调试图像
        if dist.get_rank() == 0:
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
    # RPC内部计算是float64，但affine_matrix是float32，需要转换
    affine_matrix_double = affine_matrix.to(torch.double)
    ones = torch.ones(local.shape[0],1).to(device=local.device,dtype=local.dtype)
    local_homo = torch.cat([local,ones],dim=-1)
    
    # 转换为double进行RPC计算
    trans_local = local_homo.to(torch.double) @ affine_matrix_double.T

    lats,lons = rpc_src.RPC_PHOTO2OBJ(trans_local[:,1],trans_local[:,0],dem)
    samps,lines = rpc_dst.RPC_OBJ2PHOTO(lats,lons,dem)
    warped_local = torch.stack([lines,samps],dim=-1).to(torch.float32) # 输出转回float32
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
    
    if dists.shape[0] == 0: # 如果没有有效的点
        return None, None, valid_mask

    dists_ratio = dists / torch.sum(dists,dim=1,keepdim=True) # n,k
    reverse_dists_ratio = 1. / (dists_ratio + 1e-6)
    weights = reverse_dists_ratio / torch.sum(reverse_dists_ratio,dim=1,keepdim=True)

    feature_sample_p3d = feature[idxs]
    feature_sample_pd = torch.sum(feature_sample_p3d * weights.unsqueeze(-1),dim=1).to(torch.float32)

    conf_sample_p3 = conf[idxs]
    conf_sample_p = torch.sum(conf_sample_p3 * weights,dim=1).to(torch.float32)

    return feature_sample_pd,conf_sample_p,valid_mask


# DDP Step 4: 修改 fit_affine 函数以适应DDP
def fit_affine(args, local_window_pairs:list[Window_Pair], local_rank:int, world_size:int):
    """
    使用DDP并行计算loss并优化仿射矩阵。
    把window_1 warp到 window_0
    """
    
    # 初始化模型并移动到当前进程的GPU
    model = AffineModel(args.init_offset_line, args.init_offset_samp).to(local_rank)
    
    # 用DDP包装模型
    model_ddp = DDP(model, device_ids=[local_rank])
    
    # 优化器现在优化 DDP 模型的参数
    # 通过 model_ddp.module 访问原始模型
    optimizer_r = torch.optim.Adam([model_ddp.module.R],lr = args.max_lr * 0.000001)
    optimizer_t = torch.optim.Adam([model_ddp.module.T],lr = args.max_lr)
    
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
        
        # 解决方案二: 直接调用 .module 的 forward 方法，绕过DDP的外部包装器
        af_mat = model_ddp.module()
        
        local_total_loss = torch.tensor(0.0, device=local_rank)
        
        # 只在 *本地* 的 window_pairs 子集上循环
        if len(local_window_pairs) == 0:
            # 如果这个卡没分配到数据，创建一个假的loss以同步梯度
            # 这种情况下，我们需要确保backward()被调用
            pass # loss为0，梯度也为0，是安全的
        else:
            num_valid_pairs = 0
            for window_pair in local_window_pairs:
                window_0,window_1 = window_pair.window_0,window_pair.window_1

                # 将输入数据转换为float32以进行匹配
                query_local = warp_local(window_1.local.float(),window_1.dem,window_1.rpc,window_0.rpc,af_mat)
                sample_feature,sample_conf,valid_mask = feature_sampling(window_0.feature.float(),window_0.conf.float(),window_0.local.float(),query_local,args.kmin_k) # N,D

                if sample_feature is None: # 如果没有有效的采样点，跳过这个pair
                    continue

                query_feature = window_1.feature[valid_mask].float() # N,D
                query_conf = window_1.conf[valid_mask].float()
                
                conf_cov = query_conf * sample_conf
                
                # 增加一个小的epsilon防止除以零
                weight = conf_cov / (conf_cov.mean() + 1e-8)
                loss = (torch.norm(query_feature - sample_feature,dim=-1) * weight).mean() * 10000.
                
                if not torch.isnan(loss) and not torch.isinf(loss):
                    local_total_loss = local_total_loss + loss
                    num_valid_pairs += 1

            # 计算本地平均 loss
            if num_valid_pairs > 0:
                local_total_loss = local_total_loss / num_valid_pairs
            
        # 反向传播 (DDP 在此处自动计算并同步所有进程的梯度平均值)
        # 即使loss为0，也需要调用backward来触发同步
        local_total_loss.backward()
        
        optimizer_r.step()
        optimizer_t.step()

        # 日志记录 (只在 rank 0 上打印)
        if (iter + 1) % 10 == 0:
            # 收集所有卡的loss，用于计算全局平均loss以供显示
            global_loss_sum = local_total_loss.clone().detach()
            dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
            global_avg_loss = global_loss_sum / world_size

            if local_rank == 0:
                af = af_mat.detach().cpu().numpy()
                with np.printoptions(precision=5, suppress=False):
                    print(f"iter:{iter+1}/{args.max_iter} \t loss:{global_avg_loss.item():.4f} \t lr:{scheduler_t.get_lr()[0]:.2e} \n af:{af}")
        
        
        scheduler_r.step()
        scheduler_t.step()

    # 在所有进程都完成优化后，获取最终的仿射矩阵
    final_affine_matrix_tensor = model_ddp.module.forward().detach()
    
    # rank 0 返回最终结果
    if local_rank == 0:
        final_affine_matrix_numpy = final_affine_matrix_tensor.cpu().numpy()
        return final_affine_matrix_numpy
    else:
        return None
        
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

    # DDP 初始化
    local_rank = setup_ddp()
    world_size = dist.get_world_size() # 总进程数

    args.debug_output_path = os.path.join(args.root,'debug_output')
    if local_rank == 0:
        os.makedirs(args.debug_output_path,exist_ok=True)

    # 每个进程都加载图像和RPC模型，这是必要的
    img_0,img_1 = load_imgs(args)

    if local_rank == 0:
        print("images loaded by all processes")

    # 每个进程都加载特征提取器
    encoder = EncoderDino(os.path.join(args.dino_path,'dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth'))
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    # encoder 会在 extract_features 中被移动到对应的GPU

    if local_rank == 0:
        print("Encoder Loaded by all processes")

    # 计算总的 grids，但只在主进程上计算一次即可
    # DDP Step 3: 数据分片 (Sharding)
    diags = None
    if local_rank == 0:
        corners = np.stack([img_0.corner_xys,img_1.corner_xys],axis=0)
        diags = find_grids(corners,args.window_size,offset_x=args.grid_offset_x,offset_y=args.grid_offset_y,grid_num=args.grid_num)
    
    # 使用 dist.broadcast_object_list 将主进程的 diags 广播给所有其他进程
    # 这样可以确保所有进程的 diags 列表是完全一致的
    diags_to_broadcast = [diags] if local_rank == 0 else [None]
    dist.broadcast_object_list(diags_to_broadcast, src=0)
    diags = diags_to_broadcast[0]

    # 每个进程根据自己的rank获取数据子集
    my_diags = diags[local_rank::world_size] 

    window_pairs = []
    print(f"[Rank {local_rank}] Total grids: {len(diags)}, assigned: {len(my_diags)}.")
    # 每个进程只在自己的 diags 子集上创建 Window_Pair
    for idx,diag in enumerate(my_diags):
        # 计算一个全局唯一的ID
        global_id = idx * world_size + local_rank
        window_pair = Window_Pair(args,diag,img_0,img_1,global_id)
        window_pair.extract_features(encoder)
        window_pairs.append(window_pair)
        print(f"[Rank {local_rank}] Window pair with global id {global_id} created on cuda:{local_rank}")

    # 调用修改后的 fit_affine
    final_affine_matrix = fit_affine(args,window_pairs, local_rank, world_size)

    # 同步点，确保所有进程都完成了优化
    dist.barrier()

    # 只在主进程上进行最终的模型更新和精度验证
    if local_rank == 0:
        print("All processes finished optimization.")
        if final_affine_matrix is not None:
            print(f"Final affine matrix from Rank 0: \n {final_affine_matrix}")
            # 使用返回的最终矩阵更新主进程中的RPC模型
            img_1.rpc.Update_Adjust(torch.from_numpy(final_affine_matrix).to(img_1.rpc.device))
            
            print("\nStarting final error check on Rank 0...")
            errors = check_error([img_0,img_1])
            if errors is not None:
                info = f"Error Report:\nmax: {errors.max():.4f} m\nmin: {errors.min():.4f} m\nmean: {errors.mean():.4f} m\nmedian: {np.median(errors):.4f} m\n<1m: {((errors < 1.).sum() * 1. / len(errors)) * 100:.2f} %\n<3m: {((errors < 3.).sum() * 1. / len(errors)) * 100:.2f} %\n<5m: {((errors < 5.).sum() * 1. / len(errors)) * 100:.2f} %"
                print(info)
        else:
            print("Optimization finished, but no final matrix was returned from rank 0.")


    # 清理DDP进程组
    dist.destroy_process_group()

