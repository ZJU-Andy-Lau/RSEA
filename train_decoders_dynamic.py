import os
import warnings
warnings.filterwarnings("ignore")
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
import time
import datetime
import socket
from contextlib import closing

from model.encoder_dino_0927 import EncoderDino
from model.decoders import DecoderFinetune
from utils import apply_polynomial,get_map_coef,downsample
from tqdm import tqdm
from scheduler import MultiStageOneCycleLR
import kornia.augmentation as K
import kornia.geometry.transform as KT
import cv2
import matplotlib
# 设置与多进程兼容的matplotlib后端
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --- 分布式环境设置与清理 ---

def setup_distributed(rank, world_size, port):
    """初始化分布式进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = port
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=60))
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()


# --- 核心功能函数 ---

def find_free_port():
    """动态查找一个空闲的端口"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def format_time(seconds):
    """将秒数转换为HH:MM:SS格式的字符串"""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def generate_dynamic_windows(
    image_tensor, label_tensor, num_windows, window_size=1024, 
    min_crop_size=500, max_crop_size=2000, rank=0
):
    """
    【重大修改】为动态训练范式设计的数据生成器。
    在每个Epoch中实时、高效地生成一批随机增强的训练窗口。
    包含了多尺度、随机旋转的几何增强和色彩、模糊的光度增强。

    参数:
        image_tensor (torch.Tensor): 形状为 (C, H, W) 的完整图像张量 (应已在GPU上)。
        label_tensor (torch.Tensor): 形状为 (C_label, H, W) 的完整标签张量 (应已在GPU上)。
        num_windows (int): 需要生成的窗口数量。
        window_size (int): 最终输入模型的窗口目标边长。
        min_crop_size (int): 随机裁切的最小尺寸。
        max_crop_size (int): 随机裁切的最大尺寸。
        rank (int): 当前进程的排名。

    返回:
        (torch.Tensor, torch.Tensor): 经过完整增强和预处理的图像和标签张量元组。
    """
    # --- 代码修复 ---
    # 错误原因: Kornia的旋转(rotate)等几何变换函数在GPU上执行时需要浮点型输入(如float32),
    # 因为双线性插值等操作无法在整数类型(torch.uint8, 即Byte)上实现。
    # 解决方案: 在函数开始时, 就将输入的图像和标签张量从uint8转换为float32。
    image_tensor = image_tensor.float()
    label_tensor = label_tensor.float()
    # --- 修复结束 ---

    C, H, W = image_tensor.shape
    device = image_tensor.device

    image_windows_list = []
    label_windows_list = []

    # 定义光度增强，直接在GPU上操作
    photometric_aug = nn.Sequential(
        K.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.8),
        K.RandomGaussianBlur((3, 3), (0.1, 2.0), p=0.5)
    ).to(device)

    # 确定合法的裁切尺寸范围
    max_allowed_size = min(max_crop_size, H, W)
    min_allowed_size = min(min_crop_size, max_allowed_size)
    
    if min_allowed_size >= max_allowed_size:
        print(f"[GPU {rank}] Warning: Skipping random crops, image dimensions ({H}x{W}) too small for range [{min_crop_size}, {max_crop_size}].")
        return None, None

    num_found = 0
    max_attempts = num_windows * 100 # 设置最大尝试次数以避免死循环
    attempts = 0

    while num_found < num_windows and attempts < max_attempts:
        attempts += 1
        crop_size = torch.randint(min_allowed_size, max_allowed_size + 1, (1,), device=device).item()
        ws_half = crop_size / 2.0
        corners = torch.tensor([[-ws_half, -ws_half], [ws_half, -ws_half], [ws_half, ws_half], [-ws_half, ws_half]], device=device)
        
        angle_deg = torch.rand(1, device=device) * 360
        angle_rad = torch.deg2rad(angle_deg)
        center_x = torch.rand(1, device=device) * W
        center_y = torch.rand(1, device=device) * H

        c, s = torch.cos(angle_rad), torch.sin(angle_rad)
        rot_mat = torch.tensor([[c, -s], [s, c]], device=device)
        rotated_corners = corners @ rot_mat.T + torch.tensor([center_x, center_y], device=device)

        if torch.all(rotated_corners >= 0) and torch.all(rotated_corners[:, 0] <= W) and torch.all(rotated_corners[:, 1] <= H):
            num_found += 1
            
            # 使用Kornia进行高效的旋转和裁切 (现在输入已经是float类型, 不会报错)
            rotated_img = KT.rotate(image_tensor.unsqueeze(0), angle_deg, center=torch.tensor([[center_x, center_y]], device=device), mode='bilinear', align_corners=True)
            rotated_lbl = KT.rotate(label_tensor.unsqueeze(0), angle_deg, center=torch.tensor([[center_x, center_y]], device=device), mode='bilinear', align_corners=True)
            
            # 从旋转后的图像中心裁切出crop_size的窗口，然后缩放到标准window_size
            final_img_win = KT.resize(K.CenterCrop(crop_size)(rotated_img), (window_size, window_size)).squeeze(0)
            final_lbl_win = KT.resize(K.CenterCrop(crop_size)(rotated_lbl), (window_size, window_size)).squeeze(0)

            image_windows_list.append(final_img_win)
            label_windows_list.append(final_lbl_win)

    if not image_windows_list:
        return None, None
        
    image_windows = torch.stack(image_windows_list, dim=0)
    label_windows = torch.stack(label_windows_list, dim=0)

    # 1. 旋转增强 (4种固定角度)
    img_v0 = image_windows
    img_v1 = torch.rot90(image_windows, 1, [2, 3])
    img_v2 = torch.rot90(image_windows, 2, [2, 3])
    img_v3 = torch.rot90(image_windows, 3, [2, 3])
    image_windows_augmented = torch.cat([img_v0, img_v1, img_v2, img_v3], dim=0)

    lbl_v0 = label_windows
    lbl_v1 = torch.rot90(label_windows, 1, [2, 3])
    lbl_v2 = torch.rot90(label_windows, 2, [2, 3])
    lbl_v3 = torch.rot90(label_windows, 3, [2, 3])
    label_windows_rotated = torch.cat([lbl_v0, lbl_v1, lbl_v2, lbl_v3], dim=0)
    
    # 2. 光度增强
    # 因为输入已是[0, 255]范围的float, 所以先除以255.0归一化到[0, 1]
    image_windows_normalized_0_1 = image_windows_augmented / 255.0
    image_windows_photometric = photometric_aug(image_windows_normalized_0_1)

    # 3. 标准化和下采样
    norm_transform = K.Normalize(
        mean=torch.tensor([0.485, 0.456, 0.406], device=device), 
        std=torch.tensor([0.229, 0.224, 0.225], device=device)
    )
    image_windows_normalized = norm_transform(image_windows_photometric)

    label_windows_downsampled = downsample(label_windows_rotated.permute(0, 2, 3, 1), 16)
    label_windows_final = label_windows_downsampled.permute(0, 3, 1, 2)

    return image_windows_normalized, label_windows_final


def centerize_obj(obj:np.ndarray):
    x = obj[...,0]
    y = obj[...,1]
    h = obj[...,2]
    x = x - (x.max() + x.min()) * .5
    y = y - (y.max() + y.min()) * .5
    return np.stack([x,y,h],axis=-1)

def warp_by_poly(raw,coefs):
    # 确保系数在正确的设备上
    device = raw.device
    x_coef = torch.from_numpy(coefs['x']).to(device)
    y_coef = torch.from_numpy(coefs['y']).to(device)
    
    x = (raw[:,0] + 1.) * .5 * (x_coef[1] - x_coef[0]) + x_coef[0]
    y = (raw[:,1] + 1.) * .5 * (y_coef[1] - y_coef[0]) + y_coef[0]
    h = apply_polynomial(raw[:,2],coefs['h'], device) # 确保此函数支持device参数
    warped = torch.stack([x,y,h],dim=-1)
    return warped

def add_noise_to_features(features, min_cos_sim=0.99):
    """
    为单位特征向量添加噪声，确保其仍在单位超球面上，并满足最小余弦相似度约束。
    """
    max_angle_rad = torch.acos(torch.tensor(min_cos_sim, device=features.device))
    theta = torch.rand(features.shape[0], 1, device=features.device) * max_angle_rad
    noise = torch.randn_like(features)
    dot_product = torch.sum(features * noise, dim=1, keepdim=True)
    noise_orthogonal = noise - dot_product * features
    noise_orthogonal_unit = noise_orthogonal / (torch.norm(noise_orthogonal, dim=1, keepdim=True) + 1e-8)
    noisy_features = torch.cos(theta) * features + torch.sin(theta) * noise_orthogonal_unit
    return noisy_features

def train_single_decoder(rank, decoder, encoder, image_full, label_full, map_coeffs, val_img_center, val_lbl_center, val_img_last, val_lbl_last, save_path, vis_save_path, args):
    """
    【重大修改】在动态训练范式下训练一个Decoder模型。
    每个Epoch都会实时生成新的训练数据。
    """
    decoder.to(rank)
    
    optimizer = optim.AdamW(decoder.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = MultiStageOneCycleLR(optimizer,
                                     total_steps=args.epochs,
                                     warmup_ratio=.1,
                                     cooldown_ratio=.7)
    
    start_epoch = 0
    min_loss = 1e9
    checkpoint_base_name = os.path.basename(save_path).replace('.pth', '.pth.tar')
    checkpoint_path = os.path.join(args.output_dir, 'checkpoints', checkpoint_base_name)
    
    if args.resume_training and os.path.exists(checkpoint_path):
        print(f"[GPU {rank}] 发现断点文件，正在恢复训练: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=f'cuda:{rank}')
        decoder.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        min_loss = checkpoint['min_loss']
        print(f"[GPU {rank}] 已从 Epoch {start_epoch} 恢复. 当前最小损失: {min_loss:.4f}")

    last_reported_min_loss = min_loss
    best_state_dict = decoder.state_dict()

    print(f"[GPU {rank}] 开始动态训练 {os.path.basename(save_path)}. 每个Epoch生成 {args.windows_per_epoch * 4} 个窗口.")
    start_time = time.time() 
    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        
        # --- 1. 动态生成本Epoch的训练数据 ---
        image_windows, label_windows = generate_dynamic_windows(
            image_full, label_full,
            num_windows=args.windows_per_epoch,
            window_size=args.window_size,
            min_crop_size=args.min_crop_size,
            max_crop_size=args.max_crop_size,
            rank=rank
        )
        if image_windows is None:
            print(f"[GPU {rank}] Epoch {epoch+1} 数据生成失败，跳过此轮训练。")
            continue

        # --- 2. 动态提取特征 ---
        with torch.no_grad():
            features_for_epoch, _ = encoder(image_windows)
        
        # --- 3. 准备特征和标签用于训练 ---
        all_features = features_for_epoch.permute(0,2,3,1).flatten(0,2)
        all_gt_objs = label_windows.permute(0,2,3,1).flatten(0,2)
        num_total_features = all_features.shape[0]

        # --- 4. Mini-batch 训练循环 ---
        epoch_indices = torch.randperm(num_total_features, device=rank)
        epoch_loss = 0.0
        num_batches = 0
        
        for i in range(0, num_total_features, args.decoder_batch_size):
            batch_indices = epoch_indices[i : i + args.decoder_batch_size]
            
            feature_batch = all_features[batch_indices]
            gt_objs_batch = all_gt_objs[batch_indices]

            if args.add_feature_noise:
                feature_batch = add_noise_to_features(feature_batch, min_cos_sim=args.noise_level)

            feature_batch_reshaped = feature_batch.T.unsqueeze(0).unsqueeze(-1)
            
            optimizer.zero_grad()
            output = decoder(feature_batch_reshaped)
            output = output.permute(0,2,3,1).flatten(0,2)
            pred_obj = warp_by_poly(output, map_coeffs)
            
            loss = torch.norm(pred_obj - gt_objs_batch, dim=1).mean()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            num_batches += 1

        scheduler.step()
        avg_epoch_loss = epoch_loss / num_batches if num_batches > 0 else 0.0

        # --- 5. 验证与日志记录 ---
        if (epoch + 1) % 100 == 0 and (epoch + 1) > 0:
            decoder.eval()
            with torch.no_grad():
                val_img_center_gpu = val_img_center.to(rank)
                val_feat_center, _ = encoder(val_img_center_gpu)
                val_output_center = decoder(val_feat_center).permute(0,2,3,1).flatten(0,2)
                val_pred_obj_center = warp_by_poly(val_output_center, map_coeffs)
                val_gt_obj_center = val_lbl_center.to(rank).permute(0,2,3,1).flatten(0,2)
                val_loss_center = torch.norm(val_pred_obj_center - val_gt_obj_center, dim=1).mean()

                val_img_last_gpu = val_img_last.to(rank)
                val_feat_last, _ = encoder(val_img_last_gpu)
                val_output_last = decoder(val_feat_last).permute(0,2,3,1).flatten(0,2)
                val_pred_obj_last = warp_by_poly(val_output_last, map_coeffs)
                val_gt_obj_last = val_lbl_last.to(rank).permute(0,2,3,1).flatten(0,2)
                val_loss_last = torch.norm(val_pred_obj_last - val_gt_obj_last, dim=1).mean()
            
            elapsed_seconds = time.time() - start_time
            epochs_done = epoch - start_epoch + 1
            total_epochs_in_run = args.epochs - start_epoch
            time_per_epoch = elapsed_seconds / epochs_done if epochs_done > 0 else 0
            remaining_seconds = time_per_epoch * (total_epochs_in_run - epochs_done)
            min_loss_delta = last_reported_min_loss - min_loss
            min_loss_str = f"min Loss: {min_loss:.4f}"
            if min_loss_delta > 1e-6:
                min_loss_str += f" (↓{min_loss_delta:.4f})"
            last_reported_min_loss = min_loss
            
            print(f"[GPU {rank}] 任务: {os.path.basename(save_path)} | Epoch [{epoch+1}/{args.epochs}] | Train Loss: {avg_epoch_loss:.4f} | Val Loss(Center): {val_loss_center.item():.4f} | Val Loss(Last): {val_loss_last.item():.4f} | {min_loss_str} | Elapsed: {format_time(elapsed_seconds)} | ETA: {format_time(remaining_seconds)}")

            if (epoch + 1) % 1000 == 0 and (epoch + 1) > 0:
                pred_coords_center = val_pred_obj_center.cpu().numpy()
                true_coords_center = val_gt_obj_center.cpu().numpy()
                plt.figure(figsize=(10, 10))
                plt.scatter(true_coords_center[:, 0], true_coords_center[:, 1], c='red', label='Ground Truth', s=10, alpha=0.7)
                plt.scatter(pred_coords_center[:, 0], pred_coords_center[:, 1], c='green', label='Prediction', s=10, alpha=0.7)
                plt.legend()
                plt.title(f'Validation (Center): Pred vs GT (Epoch {epoch+1}) - Rank {rank}')
                plt.xlabel('X coordinate'); plt.ylabel('Y coordinate'); plt.grid(True); plt.axis('equal')
                path_parts_center = os.path.splitext(vis_save_path)
                epoch_save_path_center = f"{path_parts_center[0]}_center_epoch_{epoch+1}_rank{rank}{path_parts_center[1]}"
                plt.savefig(epoch_save_path_center); plt.close()

                checkpoint_state = {'epoch': epoch, 'state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'min_loss': min_loss}
                torch.save(checkpoint_state, checkpoint_path)

        if avg_epoch_loss < min_loss:
            min_loss = avg_epoch_loss
            best_state_dict = decoder.state_dict()

    torch.save(best_state_dict, save_path)
    print(f"[GPU {rank}] 最佳模型已保存至 {save_path}")


# --- 主工作进程 ---

def main_worker(rank, world_size, args, all_images, all_labels, all_map_coeffs, padded_task_indices, port):
    """
    每个GPU上运行的主函数。
    """
    setup_distributed(rank, world_size, port)
    
    print(f"[GPU {rank}] 启动工作进程...")
    
    vis_output_dir = os.path.join(args.output_dir, 'vis_output')
    os.makedirs(vis_output_dir, exist_ok=True)
    
    indices_for_this_gpu = padded_task_indices[rank::world_size]
    print(f"[GPU {rank}] 将执行 {len(indices_for_this_gpu)} 轮任务（包含虚拟任务）。")
    
    encoder = EncoderDino(dino_weight_path=args.dino_weight_path)
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    encoder.to(rank)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    for data_idx in indices_for_this_gpu:
        
        is_dummy_task = (data_idx == -1)
        if is_dummy_task:
            continue
        
        img_vis_dir = os.path.join(vis_output_dir, f'data_{data_idx}')
        os.makedirs(img_vis_dir, exist_ok=True)
        
        dist.barrier()
        
        print(f"\n[GPU {rank}] 开始处理数据集索引: {data_idx}")
        
        image_np = all_images[data_idx]
        label_np = all_labels[data_idx]
        map_coef = all_map_coeffs[data_idx]
        
        # 将完整图像和标签转为Tensor，并直接移动到目标GPU
        # 这是为了避免在训练循环中反复传输
        image_full_tensor = torch.from_numpy(image_np).permute(2, 0, 1).to(rank)
        label_full_tensor = torch.from_numpy(label_np).permute(2, 0, 1).to(rank)

        print(f"[GPU {rank}] 加载数据: Image {image_full_tensor.shape}, Label {label_full_tensor.shape}")
        
        # --- 【重大修改】移除静态数据生成和特征提取过程 ---
        # 不再预先生成 image_windows 和 buffer
        
        # --- 准备固定的验证数据 ---
        # 注意：这里的验证数据生成仍然基于CPU Tensor，然后在使用时再移动到GPU
        temp_img_cpu = torch.from_numpy(image_np).permute(2, 0, 1)
        temp_lbl_cpu = torch.from_numpy(label_np).permute(2, 0, 1)
        
        # 验证 1: 中心旋转样本
        val_angle = torch.tensor([45.0])
        center_crop = K.CenterCrop(args.window_size)
        full_img_rotated = KT.rotate(temp_img_cpu.float().unsqueeze(0), val_angle, mode='bilinear', align_corners=True)
        full_lbl_rotated = KT.rotate(temp_lbl_cpu.float().unsqueeze(0), val_angle, mode='bilinear', align_corners=True)
        val_img_unnormalized = center_crop(full_img_rotated)
        val_lbl_rotated = center_crop(full_lbl_rotated)
        norm_transform_cpu = K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        val_img_center = norm_transform_cpu(val_img_unnormalized / 255.0) 
        val_lbl_downsampled_center = downsample(val_lbl_rotated.permute(0,2,3,1), 16)
        val_lbl_center = val_lbl_downsampled_center.permute(0,3,1,2)
        
        # 验证 2: 来自静态裁切的一个固定样本，确保验证的一致性
        # 我们需要生成一个固定的窗口用于验证
        top, left = 0, 0 # 例如，取左上角的窗口
        
        # 确保裁切不会越界
        h, w = temp_img_cpu.shape[1], temp_img_cpu.shape[2]
        crop_h = min(args.window_size, h)
        crop_w = min(args.window_size, w)

        img_win_val = temp_img_cpu[:, top:top+crop_h, left:left+crop_w].unsqueeze(0)
        lbl_win_val = temp_lbl_cpu[:, top:top+crop_h, left:left+crop_w].unsqueeze(0)

        # 如果裁切尺寸小于目标尺寸，进行填充
        if crop_h < args.window_size or crop_w < args.window_size:
            padding_h = args.window_size - crop_h
            padding_w = args.window_size - crop_w
            # (左, 右, 上, 下)
            img_win_val = torch.nn.functional.pad(img_win_val, (0, padding_w, 0, padding_h))
            lbl_win_val = torch.nn.functional.pad(lbl_win_val, (0, padding_w, 0, padding_h))

        val_img_last = norm_transform_cpu(img_win_val.float() / 255.0)
        val_lbl_last_down = downsample(lbl_win_val.permute(0,2,3,1), 16)
        val_lbl_last = val_lbl_last_down.permute(0,3,1,2)

        print(f"[GPU {rank}] 固定的验证数据已创建. 中心样本Shape: {val_img_center.shape}, 固定样本Shape: {val_img_last.shape}")

        decoder = DecoderFinetune(in_channels=encoder.output_channels, block_num=args.decoder_block_num)
        
        decoder_name = f"decoder_{data_idx}.pth"
        save_path = os.path.join(args.output_dir, decoder_name)
        scatter_plot_path = os.path.join(img_vis_dir, 'validation_scatter.png')
        
        train_single_decoder(
            rank=rank,
            decoder=decoder,
            encoder=encoder,
            image_full=image_full_tensor, # 传入完整的GPU Tensor
            label_full=label_full_tensor, # 传入完整的GPU Tensor
            map_coeffs=map_coef,
            val_img_center=val_img_center,
            val_lbl_center=val_lbl_center,
            val_img_last=val_img_last,
            val_lbl_last=val_lbl_last,
            save_path=save_path,
            vis_save_path=scatter_plot_path,
            args=args
        )

    cleanup()
    print(f"[GPU {rank}] 的所有任务已完成。")


# --- 主程序入口 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分布式动态训练多个独立的Decoder")
    # --- 路径与数据参数 ---
    parser.add_argument('--encoder_path',type=str,default=None, help="预训练Encoder适配器路径")
    parser.add_argument('--dino_weight_path',type=str,default=None, help="DINO模型权重路径")
    parser.add_argument('--dataset_path',type=str,default='./datasets', help="数据集H5文件所在目录")
    parser.add_argument('--dataset_num',type=int,default=None, help="要使用的数据集数量（随机选择）")
    parser.add_argument('--dataset_select',type=str,default=None, help="指定使用的数据集索引，以逗号分隔，如'0,5,12'")
    parser.add_argument('--output_dir', type=str, default='./trained_decoders_dynamic', help='保存训练好的Decoder权重的目录')
    
    # --- 训练超参数 ---
    parser.add_argument('--epochs', type=int, default=20000, help='每个Decoder的训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='AdamW优化器的权重衰减')
    parser.add_argument('--decoder_batch_size', type=int, default=4096, help='训练Decoder时的Mini-batch大小')
    parser.add_argument('--decoder_block_num',type=int, default=1, help='Decoder中的block数量')

    # --- 动态数据生成参数 ---
    parser.add_argument('--windows_per_epoch', type=int, default=64, help='【新】每个Epoch动态生成的基准窗口数量 (最终数量会乘以4，因为有旋转增强)')
    parser.add_argument('--window_size', type=int, default=1024, help='Encoder的输入窗口大小')
    parser.add_argument('--min_crop_size', type=int, default=500, help='随机裁切的最小尺寸')
    parser.add_argument('--max_crop_size', type=int, default=2000, help='随机裁切的最大尺寸')

    # --- 增强与正则化参数 ---
    parser.add_argument('--add_feature_noise', action='store_true', help='为特征添加噪声以进行数据增强')
    parser.add_argument('--noise_level', type=float, default=0.99, help='噪声级别，与原特征的最小余弦相似度')

    # --- 其他 ---
    parser.add_argument('--resume_training', action='store_true', help='从最新的断点恢复训练')
    # 移除 batch_size 参数，因为它不再用于静态特征提取
    # 移除 win_num 参数，因为它已被动态生成逻辑取代

    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'vis_output'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    try:
        print("加载数据")
        database = h5py.File(os.path.join(args.dataset_path,'train_data.h5'),'r')
        all_keys = list(database.keys())
        if args.dataset_select is None:
            dataset_num = args.dataset_num if args.dataset_num is not None else len(all_keys)
            dataset_indices = torch.randperm(len(all_keys))[:dataset_num].numpy()
        else:
            dataset_indices = [int(i) for i in args.dataset_select.split(',')]
        
        keys = [all_keys[i] for i in dataset_indices]
        all_images = []
        all_labels = []
        all_map_coeffs = []
        for key in tqdm(keys, desc="Loading data from H5 file"):
            img = database[key]['images']['image_0'][:]
            if img.ndim == 2: img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 1: img = np.concatenate([img] * 3, axis=-1)
            
            obj = database[key]['obj'][:]
            obj = centerize_obj(obj)
            map_coef = {
                'x':np.array([obj[...,0].min(),obj[...,0].max()]),
                'y':np.array([obj[...,1].min(),obj[...,1].max()]),
                'h':get_map_coef(obj[...,2].reshape(-1))
            }
            all_images.append(img)
            all_labels.append(obj)
            all_map_coeffs.append(map_coef)
        
    except FileNotFoundError as e:
        print(f"错误: 无法找到数据文件: {e}"); exit(1)
    except Exception as e:
        print(f"加载数据时发生错误: {e}"); exit(1)
        
    num_datasets = len(all_images)
    print(f"共找到 {num_datasets} 个数据集待处理。")
    np.save(os.path.join(args.output_dir,'dataset_indices.npy'),dataset_indices)

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("错误：没有检测到可用的GPU。"); exit(1)
        
    effective_world_size = min(world_size, num_datasets)
    port = find_free_port()
    print(f"将在 {effective_world_size} 张GPU上启动训练，使用端口 {port}。")

    num_real_tasks = len(all_images)
    padded_task_indices = list(range(num_real_tasks))
    num_to_pad = (effective_world_size - num_real_tasks % effective_world_size) % effective_world_size
    if num_to_pad > 0:
        padded_task_indices.extend([-1] * num_to_pad)
    
    mp.spawn(
        main_worker,
        args=(effective_world_size, args, all_images, all_labels, all_map_coeffs, padded_task_indices, port),
        nprocs=effective_world_size,
        join=True
    )
    
    print("所有训练任务已完成！")

