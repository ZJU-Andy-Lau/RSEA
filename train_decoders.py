import os
import math
import h5py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model.encoder_dino_0927 import EncoderDino
from model.decoders import DecoderFinetune
from utils import apply_polynomial,get_map_coef,bilinear_interpolate
from tqdm import tqdm
from scheduler import MultiStageOneCycleLR


# --- 2. 分布式环境设置与清理 ---

def setup_distributed(rank, world_size):
    """初始化分布式进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    # 使用NCCL后端，它为NVIDIA GPU提供了最优的性能
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()


# --- 3. 核心功能函数 ---

def crop_to_windows(image_tensor, label_tensor, window_size=1024):
    """
    将大尺寸图像和标签高效地切分成最小数量的窗口。
    
    参数:
        image_tensor (torch.Tensor): 形状为 (C, H, W) 的图像张量。
        label_tensor (torch.Tensor): 形状为 (C_label, H, W) 的标签张量。
        window_size (int): 窗口的目标边长。
        
    返回:
        (torch.Tensor, torch.Tensor): 包含所有窗口的图像和标签张量元组。
                                      形状为 (N_windows, C, window_size, window_size)。
    """
    _, H, W = image_tensor.shape
    
    # 计算需要填充多少才能被window_size整除
    pad_h = (window_size - H % window_size) % window_size
    pad_w = (window_size - W % window_size) % window_size

    # 使用F.pad进行填充，'constant'模式默认用0填充
    padding = (0, pad_w, 0, pad_h)
    image_padded = torch.nn.functional.pad(image_tensor, padding, "constant", 0)
    label_padded = torch.nn.functional.pad(label_tensor, padding, "constant", 0)
    
    # 使用unfold进行高效的无重叠切分
    # unfold(dimension, size, step)
    # 1. 先对高度维度切分
    image_unfolded_h = image_padded.unfold(1, window_size, window_size)
    label_unfolded_h = label_padded.unfold(1, window_size, window_size)
    
    # 2. 再对宽度维度切分
    # 现在的形状是 (C, num_h_windows, W_padded, window_size)
    # 我们需要在第2个维度(W_padded)上再次unfold
    image_unfolded_hw = image_unfolded_h.unfold(2, window_size, window_size)
    label_unfolded_hw = label_unfolded_h.unfold(2, window_size, window_size)
    
    # 3. 调整形状以得到 (N_windows, C, H, W)
    # 当前形状: (C, num_h, num_w, window_size, window_size)
    image_windows = image_unfolded_hw.permute(1, 2, 0, 3, 4).contiguous()
    label_windows = label_unfolded_hw.permute(1, 2, 0, 3, 4).contiguous()
    
    num_h_windows = image_windows.shape[0]
    num_w_windows = image_windows.shape[1]
    
    image_windows = image_windows.view(num_h_windows * num_w_windows, -1, window_size, window_size)
    label_windows = label_windows.view(num_h_windows * num_w_windows, -1, window_size, window_size)
    
    return image_windows, label_windows

def downsample(arr,ds):
    if ds <= 0:
        return arr
    H,W = arr.shape[:2]
    lines = np.arange(0,H - ds + 1,ds) + (ds - 1.) * 0.5
    samps = np.arange(0,W - ds + 1,ds) + (ds - 1.) * 0.5
    sample_idxs = np.stack(np.meshgrid(samps,lines,indexing='xy'),axis=-1).reshape(-1,2) # x,y
    arr_ds = bilinear_interpolate(arr,sample_idxs)
    arr_ds = arr_ds.reshape(len(lines),len(samps),-1).squeeze()
    return arr_ds

def centerize_obj(obj:np.ndarray):
    x = obj[...,0]
    y = obj[...,1]
    h = obj[...,2]
    x = x - (x.max() + x.min()) * .5
    y = y - (y.max() + y.min()) * .5
    return np.stack([x,y,h],axis=-1)

def warp_by_poly(raw,coefs):
    # raw[:,0] = .5 * (raw[:,0] + 1.) * (bbox['x_max'] - bbox['x_min']) + bbox['x_min']
    # raw[:,1] = .5 * (raw[:,1] + 1.) * (bbox['y_max'] - bbox['y_min']) + bbox['y_min']
    # raw[:,2] = .5 * (raw[:,2] + 1.) * (bbox['h_max'] - bbox['h_min']) + bbox['h_min']
    # x = apply_polynomial(raw[:,0],coefs['x'])
    # y = apply_polynomial(raw[:,1],coefs['y'])
    x = (raw[:,0] + 1.) * .5 * (coefs['x'][1] - coefs['x'][0]) + coefs['x'][0]
    y = (raw[:,1] + 1.) * .5 * (coefs['y'][1] - coefs['y'][0]) + coefs['y'][0]
    h = apply_polynomial(raw[:,2],coefs['h'])
    warped = torch.stack([x,y,h],dim=-1)
    return warped

def train_single_decoder(rank, decoder, buffer, map_coeffs, epochs, lr, save_path):
    """
    在指定的GPU上训练一个Decoder模型。
    
    参数:
        rank (int): 当前GPU的ID。
        decoder (nn.Module): 待训练的Decoder模型实例。
        dataloader (DataLoader): 包含（特征, 标签）的数据加载器。
        epochs (int): 训练轮数。
        lr (float): 学习率。
        save_path (str): 模型权重保存路径。
    """
    decoder.to(rank)
    decoder.train()
    
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    scheduler = MultiStageOneCycleLR(optimizer,
                                     total_steps=epochs,
                                     warmup_ratio=.1,
                                     cooldown_ratio=.5)
    # 假设是回归任务或类似任务，使用MSE损失。
    # 如果是分割任务，应使用 nn.BCEWithLogitsLoss 或 nn.CrossEntropyLoss
    criterion = nn.MSELoss()

    print(f"[GPU {rank}] 开始训练 {os.path.basename(save_path)}. Buffer大小: {len(buffer['features'])} 个样本.")
    features = buffer['features'].permute(1,0)[None,:,:,None] #1,D,P,1
    gt_objs = buffer['objs'] # P,3
    for epoch in range(epochs):
        output = decoder(features)
        output = output.permute(0,2,3,1).flatten(0,2)
        pred_obj = warp_by_poly(output,map_coeffs)
        loss = criterion(pred_obj,gt_objs)
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % 10 == 0: # 每10轮打印一次日志
            print(f"[GPU {rank}] | 任务: {os.path.basename(save_path)} | Epoch [{epoch+1}/{epochs}] | Loss: {loss:.4f}")

    # 保存训练好的Decoder权重
    torch.save(decoder.state_dict(), save_path)
    print(f"[GPU {rank}] 训练完成. Decoder已保存至 {save_path}")


# --- 4. 主工作进程 ---

def main_worker(rank, world_size, args, all_images, all_labels, all_map_coeffs):
    """
    每个GPU上运行的主函数。
    """
    print(f"启动 GPU {rank}/{world_size} 的工作进程...")
    setup_distributed(rank, world_size)

    # 1. 确定当前GPU需要处理的数据集索引
    #    这是一种简单有效的任务均分策略
    num_total_datasets = all_images.shape[0]
    indices_for_this_gpu = list(range(num_total_datasets))[rank::world_size]
    print(f"[GPU {rank}] 分配到 {len(indices_for_this_gpu)} 个训练任务 (Indices: {indices_for_this_gpu[:5]}...).")
    
    label_channels = all_labels.shape[-1]

    # 2. 加载预训练的Encoder
    encoder = EncoderDino(dino_weight_path=args.dino_weight_path)
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    # 模拟加载预训练权重
    # encoder.load_state_dict(torch.load('path/to/your/encoder_weights.pth'))
    encoder.to(rank)
    encoder.eval() # 设置为评估模式
    for param in encoder.parameters():
        param.requires_grad = False # 冻结Encoder的所有参数

    # 3. 遍历分配到的任务，逐一训练Decoder
    for data_idx in indices_for_this_gpu:
        print(f"[GPU {rank}] 开始处理数据集索引: {data_idx}")
        
        # --- a. 从numpy数组中获取数据并转换为Tensor ---
        # numpy数组格式为 H, W, C -> torch需要 C, H, W
        image_np = all_images[data_idx]
        label_np = all_labels[data_idx]
        map_coef = all_map_coeffs[data_idx]

        label_np = downsample(label_np,16)
        
        image = torch.from_numpy(image_np).permute(2, 0, 1).float()
        label = torch.from_numpy(label_np).permute(2, 0, 1).float()

        print(f"[GPU {rank}] 加载数据: Image {image.shape}, Label {label.shape}")
        
        # --- b. 切分窗口 ---
        image_windows, label_windows = crop_to_windows(image, label, args.window_size)
        print(f"[GPU {rank}] 图像和标签被切分为 {image_windows.shape[0]} 个窗口.")
        
        # --- c. 创建特征Buffer (在GPU上) ---
        feature_buffer = []
        
        # 使用一个小的 DataLoader 来批量处理特征提取，防止显存溢出
        temp_dataloader = DataLoader(image_windows, batch_size=args.batch_size, shuffle=False)

        with torch.no_grad(): # 确保不计算梯度
            for image_batch in temp_dataloader:
                image_batch = image_batch.to(rank)
                feature_batch,_ = encoder(image_batch)
                # 将提取的特征直接保留在GPU上
                feature_buffer.append(feature_batch)
        
        # 拼接所有批次的特征，all_features 张量现在在GPU上
        all_features = torch.cat(feature_buffer, dim=0).permute(0,2,3,1).flatten(0,2)
        
        # 将对应的标签窗口也移动到当前GPU
        all_labels_for_features = label_windows.to(rank).permute(0,2,3,1).flatten(0,2)

        buffer = {
            'features':all_features,
            'objs':all_labels_for_features
        }

        # Encoder(MaxPool)会让特征图尺寸减半, Decoder会恢复
        # 这里我们的示例Decoder输入是128x512x512，输出是Dx1024x1024
        # 所以label窗口也应该是1024x1024
        
        # training_dataset = TensorDataset(all_features, all_labels_for_features)
        # training_dataloader = DataLoader(training_dataset, batch_size=args.batch_size, shuffle=True)

        # --- d. 训练新的Decoder ---
        decoder = DecoderFinetune(in_channels=encoder.output_channels,block_num=args.decoder_block_num)
        
        # 定义保存路径
        decoder_name = f"decoder_{data_idx}.pth"
        save_path = os.path.join(args.output_dir, decoder_name)
        
        train_single_decoder(
            rank=rank,
            decoder=decoder,
            buffer=buffer,
            map_coeffs=map_coef,
            epochs=args.epochs,
            lr=args.lr,
            save_path=save_path
        )

        # except Exception as e:
        #     print(f"[GPU {rank}] 处理索引 {data_idx} 时发生错误: {e}")
        #     continue # 跳过这个数据，继续下一个

    cleanup()
    print(f"GPU {rank} 的所有任务已完成。")


# --- 5. 主程序入口 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分布式训练多个独立的Decoder")
    parser.add_argument('--encoder_path',type=str,default=None)
    parser.add_argument('--dino_weight_path',type=str,default=None)
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--dataset_num',type=int,default=None)
    parser.add_argument('--output_dir', type=str, default='./trained_decoders', help='保存训练好的Decoder权重的目录')
    parser.add_argument('--window_size', type=int, default=1024, help='Encoder的输入窗口大小')
    parser.add_argument('--epochs', type=int, default=50, help='每个Decoder的训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=4, help='训练时的批量大小')
    parser.add_argument('--decoder_block_num',type=int,default=1)
    
    args = parser.parse_args()

    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)

    # 1. 从.npy文件加载数据
    try:
        print("加载数据")
        database = h5py.File(os.path.join(args.dataset_path,'train_data.h5'),'r')
        all_keys = list(database.keys())
        dataset_num = args.dataset_num
        if dataset_num is None:
            dataset_num = len(all_keys)
        dataset_indices = torch.randperm(len(all_keys))[:dataset_num].numpy()
        keys = [all_keys[i] for i in dataset_indices]
        all_images = []
        all_labels = []
        all_map_coeffs = []
        for key in tqdm(keys):
            img = database[key]['images']['image_0'][:]
            img = np.stack([img] * 3,axis = -1)
            obj = database[key]['obj'][:]
            obj = centerize_obj(obj)
            map_coef = {
                    'x':np.array([obj[:,:,0].min(),obj[:,:,0].max()]),
                    'y':np.array([obj[:,:,1].min(),obj[:,:,1].max()]),
                    'h':get_map_coef(obj[:,:,2].reshape(-1))
                }
            all_images.append(img)
            all_labels.append(obj)
            all_map_coeffs.append(map_coef)
        
        all_images = np.stack(all_images,axis=0)
        all_labels = np.stack(all_labels,axis=0)

    except FileNotFoundError as e:
        print(f"错误: 无法找到数据文件: {e}")
        exit(1)
    except Exception as e:
        print(f"加载数据时发生错误: {e}")
        exit(1)
        
    num_datasets = all_images.shape[0]
    print(f"共找到 {num_datasets} 个数据集待处理。")
    np.save(os.path.join(args.output_dir,'dataset_indices.npy'),dataset_indices)

    # 2. 启动分布式训练
    world_size = torch.cuda.device_count()
    if world_size < 8:
        print(f"警告：检测到 {world_size} 张GPU，但代码为8张GPU优化。将使用所有可用的GPU。")
    
    print(f"将在 {world_size} 张GPU上启动训练...")
    
    mp.spawn(
        main_worker,
        args=(world_size, args, all_images, all_labels, all_map_coeffs),
        nprocs=world_size,
        join=True
    )
    
    print("所有训练任务已完成！")

