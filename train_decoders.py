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
import time # 引入time模块
import datetime # 引入datetime模块
import socket # 引入socket模块
from contextlib import closing # 引入closing模块

from model.encoder_dino_0927 import EncoderDino
from model.decoders import DecoderFinetune
from utils import apply_polynomial,get_map_coef,downsample
from tqdm import tqdm
from scheduler import MultiStageOneCycleLR
import kornia.augmentation as K
import kornia.geometry.transform as KT
import cv2
import matplotlib
# --- 修复 1: 设置与多进程兼容的matplotlib后端 ---
# 必须在import pyplot之前设置
matplotlib.use('Agg')
import matplotlib.pyplot as plt


# --- 2. 分布式环境设置与清理 ---

def setup_distributed(rank, world_size, port):
    """初始化分布式进程组"""
    os.environ['MASTER_ADDR'] = 'localhost'
    # --- 修复 2: 使用动态传入的端口号 ---
    os.environ['MASTER_PORT'] = port
    # 使用NCCL后端，它为NVIDIA GPU提供了最优的性能
    dist.init_process_group("nccl", rank=rank, world_size=world_size, timeout=datetime.timedelta(minutes=60))
    torch.cuda.set_device(rank)

def cleanup():
    """清理分布式进程组"""
    dist.destroy_process_group()


# --- 3. 核心功能函数 ---

def find_free_port():
    """动态查找一个空闲的端口"""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return str(s.getsockname()[1])

def format_time(seconds):
    """将秒数转换为HH:MM:SS格式的字符串。"""
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

def crop_to_windows(image_tensor, label_tensor, image_np_for_vis, window_size=1024, win_num=3, output_path=None, rank=0, min_crop_size=500, max_crop_size=2000, sample_factor = 4):
    """
    将大尺寸图像和标签高效地切分成多个窗口，包括均匀窗口和动态多尺度的随机旋转窗口。
    采用拒绝采样方法确保旋转窗口完全在原图内，不含任何padding。
    修改后的版本：对均匀裁切的窗口进行随机两次90度旋转，对随机裁切的窗口不进行额外的90度旋转。
    """
    C, H, W = image_tensor.shape
    
    if output_path:
        os.makedirs(output_path, exist_ok=True)
    
    vis_image = image_np_for_vis.copy()
    image_windows_list = []
    label_windows_list = []

    # 1. 均匀裁切窗口 (Uniform Cropping)
    if win_num > 1:
        stride_h = (H - window_size) / (win_num - 1)
        stride_w = (W - window_size) / (win_num - 1)
    else:
        stride_h = 0
        stride_w = 0

    for i in range(win_num):
        for j in range(win_num):
            top = int(i * stride_h)
            left = int(j * stride_w)
            
            top = min(top, H - window_size)
            left = min(left, W - window_size)

            img_win = image_tensor[:, top:top+window_size, left:left+window_size]
            lbl_win = label_tensor[:, top:top+window_size, left:left+window_size]
            image_windows_list.append(img_win)
            label_windows_list.append(lbl_win)
            
            cv2.rectangle(vis_image, (left, top), (left + window_size, top + window_size), (0, 255, 0), 5)

    # 2. 动态多尺度随机旋转裁切 (Dynamic Multi-Scale Random Rotated Cropping)
    num_random_windows = win_num * win_num
    num_found = 0

    max_allowed_size = min(max_crop_size, H, W)
    min_allowed_size = min(min_crop_size, max_allowed_size)
        
    if min_allowed_size >= max_allowed_size:
        if rank == 0:
            print(f"Skipping random crops: image dimensions ({H}x{W}) are too small for the crop size range [{min_crop_size}, {max_crop_size}].")
    else:
        pbar = tqdm(total=num_random_windows, desc=f"Finding {num_random_windows} valid random crops", leave=False, position=rank, disable=(rank!=0))
        while num_found < num_random_windows:
            crop_size = torch.randint(min_allowed_size, max_allowed_size + 1, (1,)).item()
            ws_half = crop_size / 2.0
            corners = torch.tensor([[-ws_half, -ws_half], [ws_half, -ws_half], [ws_half, ws_half], [-ws_half, ws_half]])
            
            angle_deg = torch.rand(1) * 360
            angle_rad = torch.deg2rad(angle_deg)
            center_x = torch.rand(1) * W
            center_y = torch.rand(1) * H

            c, s = torch.cos(angle_rad), torch.sin(angle_rad)
            rot_mat = torch.tensor([[c, -s], [s, c]])
            rotated_corners = corners @ rot_mat.T + torch.tensor([center_x, center_y])

            if torch.all(rotated_corners[:, 0] >= 0) and torch.all(rotated_corners[:, 0] <= W) and \
               torch.all(rotated_corners[:, 1] >= 0) and torch.all(rotated_corners[:, 1] <= H):
                
                num_found += 1
                if rank == 0: pbar.update(1)

                xmin, ymin = rotated_corners.min(dim=0).values.floor().int()
                xmax, ymax = rotated_corners.max(dim=0).values.ceil().int()
                
                xmin, ymin = max(0, xmin), max(0, ymin)
                xmax, ymax = min(W, xmax), min(H, ymax)

                temp_img = image_tensor[:, ymin:ymax, xmin:xmax]
                temp_lbl = label_tensor[:, ymin:ymax, xmin:xmax]
                
                center_x_new, center_y_new = center_x - xmin, center_y - ymin

                rotated_temp_img = KT.rotate(temp_img.float().unsqueeze(0), angle_deg, center=torch.tensor([[center_x_new, center_y_new]]), mode='bilinear', align_corners=True)
                rotated_temp_lbl = KT.rotate(temp_lbl.float().unsqueeze(0), angle_deg, center=torch.tensor([[center_x_new, center_y_new]]), mode='bilinear', align_corners=True)

                final_img_win = KT.resize(K.CenterCrop(crop_size)(rotated_temp_img), (window_size, window_size)).squeeze(0)
                final_lbl_win = KT.resize(K.CenterCrop(crop_size)(rotated_temp_lbl), (window_size, window_size)).squeeze(0)

                image_windows_list.append(final_img_win)
                label_windows_list.append(final_lbl_win)
                
                cv2.drawContours(vis_image, [rotated_corners.int().numpy()], 0, (255, 0, 0), 5) 
        
        if pbar: pbar.close()

    if not image_windows_list:
        return None, None

    # --- 3. 数据增强 (Data Augmentation) ---
    num_uniform_windows = win_num * win_num
    uniform_image_windows = torch.stack(image_windows_list[:num_uniform_windows], dim=0)
    uniform_label_windows = torch.stack(label_windows_list[:num_uniform_windows], dim=0)

    # 对均匀裁切的窗口，从[0, 90, 180, 270]度中随机选择两种旋转
    rotation_indices = torch.randperm(4)[:2]
    augmented_uniform_imgs = []
    augmented_uniform_lbls = []
    for k in rotation_indices:
        augmented_uniform_imgs.append(torch.rot90(uniform_image_windows, int(k), [2, 3]))
        augmented_uniform_lbls.append(torch.rot90(uniform_label_windows, int(k), [2, 3]))
    
    final_image_windows = torch.cat(augmented_uniform_imgs, dim=0)
    final_label_windows = torch.cat(augmented_uniform_lbls, dim=0)

    # 随机裁切的窗口已经自带旋转，不再进行额外的90度旋转
    if len(image_windows_list) > num_uniform_windows:
        random_image_windows = torch.stack(image_windows_list[num_uniform_windows:], dim=0)
        random_label_windows = torch.stack(label_windows_list[num_uniform_windows:], dim=0)
        final_image_windows = torch.cat([final_image_windows, random_image_windows], dim=0)
        final_label_windows = torch.cat([final_label_windows, random_label_windows], dim=0)

    # --- 4. 标准化和下采样 (Normalization and Downsampling) ---
    transform = K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            )
    image_windows_augmented = transform(final_image_windows.float() / 255.0)

    label_windows_rotated = final_label_windows.permute(0, 2, 3, 1) # (N, H, W, C)
    label_windows_downsampled = downsample(label_windows_rotated, sample_factor)
    label_windows_augmented = label_windows_downsampled.permute(0, 3, 1, 2) # (N, C, H_new, W_new)

    return image_windows_augmented, label_windows_augmented


def centerize_obj(obj:np.ndarray):
    x = obj[...,0]
    y = obj[...,1]
    h = obj[...,2]
    x = x - (x.max() + x.min()) * .5
    y = y - (y.max() + y.min()) * .5
    return np.stack([x,y,h],axis=-1)

def warp_by_poly(raw,coefs):
    x = (raw[:,0] + 1.) * .5 * (coefs['x'][1] - coefs['x'][0]) + coefs['x'][0]
    y = (raw[:,1] + 1.) * .5 * (coefs['y'][1] - coefs['y'][0]) + coefs['y'][0]
    h = apply_polynomial(raw[:,2],coefs['h'])
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

def train_single_decoder(rank, decoder, encoder, buffer, map_coeffs, val_img_center, val_lbl_center, val_img_last, val_lbl_last, save_path, vis_save_path, args):
    """
    在指定的GPU上训练一个Decoder模型。
    修改后的版本：
    1. Buffer为(B, D, H, W)格式，按图像窗口进行批处理。
    2. 引入空间一致性损失。
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

    print(f"[GPU {rank}] 开始训练 {os.path.basename(save_path)}. Buffer大小: {len(buffer['features'])} 个图像窗口.")
    all_features = buffer['features'].to(rank)
    all_gt_objs = buffer['objs'].to(rank)
    num_total_windows = all_features.shape[0]

    start_time = time.time() 
    for epoch in range(start_epoch, args.epochs):
        decoder.train()
        
        epoch_indices = torch.randperm(num_total_windows, device=rank)
        epoch_loss = 0.0
        epoch_recon_loss = 0.0
        epoch_spatial_loss = 0.0
        num_batches = 0
        
        # decoder_batch_size现在代表每个批次中的图像窗口数量
        for i in range(0, num_total_windows, args.decoder_batch_size):
            batch_indices = epoch_indices[i : i + args.decoder_batch_size]
            
            feature_batch = all_features[batch_indices]
            gt_objs_batch = all_gt_objs[batch_indices]

            if args.add_feature_noise:
                B, D, H, W = feature_batch.shape
                feature_batch_flat = feature_batch.permute(0, 2, 3, 1).reshape(-1, D)
                noisy_features_flat = add_noise_to_features(feature_batch_flat, min_cos_sim=args.noise_level)
                feature_batch = noisy_features_flat.reshape(B, H, W, D).permute(0, 3, 1, 2)

            optimizer.zero_grad()
            
            # Decoder直接处理(B,D,H,W)的特征图
            output = decoder(feature_batch)
            
            # --- 损失计算 ---
            output_permuted = output.permute(0, 2, 3, 1)
            B, H, W, C = output_permuted.shape
            output_flat = output_permuted.reshape(-1, C)

            pred_obj_flat = warp_by_poly(output_flat, map_coeffs)
            pred_obj_map = pred_obj_flat.reshape(B, H, W, C)

            gt_obj_map = gt_objs_batch.permute(0, 2, 3, 1)
            gt_obj_flat = gt_obj_map.reshape(-1, C)

            # 1. 重建损失
            loss_recon = torch.norm(pred_obj_flat - gt_obj_flat, dim=1).mean()
            
            # 2. 空间一致性损失
            pred_diff_h = torch.norm(pred_obj_map[:, :, 1:, :] - pred_obj_map[:, :, :-1, :], p=2, dim=-1)
            gt_diff_h = torch.norm(gt_obj_map[:, :, 1:, :] - gt_obj_map[:, :, :-1, :], p=2, dim=-1)
            pred_diff_v = torch.norm(pred_obj_map[:, 1:, :, :] - pred_obj_map[:, :-1, :, :], p=2, dim=-1)
            gt_diff_v = torch.norm(gt_obj_map[:, 1:, :, :] - gt_obj_map[:, :-1, :, :], p=2, dim=-1)
            
            loss_spatial = torch.abs(pred_diff_h - gt_diff_h.detach()).mean() + torch.abs(pred_diff_v - gt_diff_v.detach()).mean()
            
            # 3. 总损失
            loss = loss_recon + args.spatial_loss_weight * loss_spatial

            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            epoch_recon_loss += loss_recon.item()
            epoch_spatial_loss += loss_spatial.item()
            num_batches += 1

        scheduler.step()
        avg_epoch_loss = epoch_loss / num_batches
        avg_recon_loss = epoch_recon_loss / num_batches
        avg_spatial_loss = epoch_spatial_loss / num_batches

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
            if min_loss_delta > 1e-6: min_loss_str += f" (↓{min_loss_delta:.4f})"
            last_reported_min_loss = min_loss
            
            print(f"[GPU {rank}] 任务: {os.path.basename(save_path)} | Epoch [{epoch+1}/{args.epochs}] | "
                  f"Total Loss: {avg_epoch_loss:.4f} | Recon Loss: {avg_recon_loss:.4f} | Spatial Loss: {avg_spatial_loss:.4f} | "
                  f"Val(C): {val_loss_center.item():.4f} | Val(L): {val_loss_last.item():.4f} | {min_loss_str} | "
                  f"Elapsed: {format_time(elapsed_seconds)} | ETA: {format_time(remaining_seconds)}")

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

                pred_coords_last = val_pred_obj_last.cpu().numpy()
                true_coords_last = val_gt_obj_last.cpu().numpy()
                plt.figure(figsize=(10, 10))
                plt.scatter(true_coords_last[:, 0], true_coords_last[:, 1], c='red', label='Ground Truth', s=10, alpha=0.7)
                plt.scatter(pred_coords_last[:, 0], pred_coords_last[:, 1], c='green', label='Prediction', s=10, alpha=0.7)
                plt.legend()
                plt.title(f'Validation (Last Sample): Pred vs GT (Epoch {epoch+1}) - Rank {rank}')
                plt.xlabel('X coordinate'); plt.ylabel('Y coordinate'); plt.grid(True); plt.axis('equal')
                path_parts_last = os.path.splitext(vis_save_path)
                epoch_save_path_last = f"{path_parts_last[0]}_last_sample_epoch_{epoch+1}_rank{rank}{path_parts_last[1]}"
                plt.savefig(epoch_save_path_last); plt.close()

                checkpoint_state = {'epoch': epoch, 'state_dict': decoder.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(), 'min_loss': min_loss}
                torch.save(checkpoint_state, checkpoint_path)

        if avg_epoch_loss < min_loss:
            min_loss = avg_epoch_loss
            best_state_dict = decoder.state_dict()

    torch.save(best_state_dict, save_path)
    print(f"[GPU {rank}] 最佳模型已保存至 {save_path}")


# --- 4. 主工作进程 ---

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
        if not is_dummy_task:
            img_vis_dir = os.path.join(vis_output_dir, f'data_{data_idx}')
            os.makedirs(img_vis_dir, exist_ok=True)
        dist.barrier()
        if is_dummy_task: continue

        print(f"\n[GPU {rank}] 开始处理数据集索引: {data_idx}")
        
        image_np = all_images[data_idx]
        label_np = all_labels[data_idx]
        map_coef = all_map_coeffs[data_idx]
        
        image = torch.from_numpy(image_np).permute(2, 0, 1)
        label = torch.from_numpy(label_np).permute(2, 0, 1)

        print(f"[GPU {rank}] 加载数据: Image {image.shape}, Label {label.shape}")
        
        image_windows, label_windows = crop_to_windows(image, label, image_np, args.window_size, args.win_num, img_vis_dir, rank, args.min_crop_size, args.max_crop_size, encoder.SAMPLE_FACTOR)
        if image_windows is None:
            print(f"[GPU {rank}] 索引 {data_idx} 的图像尺寸过小，无法裁切，已跳过。")
            continue
        
        print(f"[GPU {rank}] 图像和标签被切分为 {image_windows.shape[0]} 个窗口.")
        
        # --- 准备验证数据 ---
        H, W = image.shape[1], image.shape[2]
        val_angle = torch.tensor([45.0])
        center_crop = K.CenterCrop(args.window_size)
        full_img_rotated = KT.rotate(image.float().unsqueeze(0), val_angle, mode='bilinear', align_corners=True)
        full_lbl_rotated = KT.rotate(label.float().unsqueeze(0), val_angle, mode='bilinear', align_corners=True)
        val_img_unnormalized = center_crop(full_img_rotated)
        val_lbl_rotated = center_crop(full_lbl_rotated)
        norm_transform = K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        val_img_center = norm_transform(val_img_unnormalized / 255.0) 
        val_lbl_downsampled_center = downsample(val_lbl_rotated.permute(0,2,3,1), encoder.SAMPLE_FACTOR)
        val_lbl_center = val_lbl_downsampled_center.permute(0,3,1,2)
        
        val_img_last = image_windows[-1].unsqueeze(0) # Use the very last augmented sample for validation
        val_lbl_last = label_windows[-1].unsqueeze(0)
        
        print(f"[GPU {rank}] 验证数据已创建. 中心样本Shape: {val_img_center.shape}, 末尾样本Shape: {val_img_last.shape}")

        # --- 提取特征并构建Buffer ---
        feature_buffer = []
        training_image_windows = image_windows  #[:-1] # Exclude the last sample used for validation
        temp_dataloader = DataLoader(training_image_windows, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for image_batch in temp_dataloader:
                feature_batch,_ = encoder(image_batch.to(rank))
                feature_buffer.append(feature_batch.cpu())
        
        # Buffer现在保持(B, D, H, W)格式
        all_features = torch.cat(feature_buffer, dim=0)
        all_labels_for_features = label_windows #[:-1]

        if args.max_buffer_size and all_features.shape[0] > args.max_buffer_size:
            select_idxs = torch.randperm(all_features.shape[0])[:args.max_buffer_size]
            all_features = all_features[select_idxs]
            all_labels_for_features = all_labels_for_features[select_idxs]

        buffer = {'features': all_features, 'objs': all_labels_for_features}

        decoder = DecoderFinetune(in_channels=encoder.output_channels, block_num=args.decoder_block_num)
        
        decoder_name = f"decoder_{data_idx}.pth"
        save_path = os.path.join(args.output_dir, decoder_name)
        scatter_plot_path = os.path.join(img_vis_dir, 'validation_scatter.png')
        
        train_single_decoder(
            rank=rank, decoder=decoder, encoder=encoder, buffer=buffer,
            map_coeffs=map_coef, val_img_center=val_img_center, val_lbl_center=val_lbl_center,
            val_img_last=val_img_last, val_lbl_last=val_lbl_last, save_path=save_path,
            vis_save_path=scatter_plot_path, args=args
        )

    cleanup()
    print(f"[GPU {rank}] 的所有任务已完成。")


# --- 5. 主程序入口 ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="分布式训练多个独立的Decoder")
    parser.add_argument('--encoder_path',type=str,default=None)
    parser.add_argument('--dino_weight_path',type=str,default=None)
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--dataset_num',type=int,default=None)
    parser.add_argument('--dataset_select',type=str,default=None)
    parser.add_argument('--output_dir', type=str, default='./trained_decoders', help='保存训练好的Decoder权重的目录')
    parser.add_argument('--window_size', type=int, default=1024, help='Encoder的输入窗口大小')
    parser.add_argument('--win_num', type=int, default=3, help='每条边上裁切的窗口数')
    parser.add_argument('--max_buffer_size',type=int,default=None, help='每个任务用于训练的最大图像窗口数量')
    parser.add_argument('--epochs', type=int, default=10000, help='每个Decoder的训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=4, help='特征提取时的批量大小')
    parser.add_argument('--resume_training', action='store_true', help='从最新的断点恢复训练')
    parser.add_argument('--decoder_batch_size', type=int, default=16, help='训练Decoder时每个批次的图像窗口数量')
    parser.add_argument('--add_feature_noise', action='store_true', help='为特征添加噪声以进行数据增强')
    parser.add_argument('--noise_level', type=float, default=0.99, help='噪声级别，与原特征的最小余弦相似度')
    parser.add_argument('--decoder_block_num',type=int, default=1, help='Decoder中的block数量')
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='AdamW优化器的权重衰减')
    parser.add_argument('--min_crop_size', type=int, default=500, help='随机裁切的最小尺寸')
    parser.add_argument('--max_crop_size', type=int, default=2000, help='随机裁切的最大尺寸')
    parser.add_argument('--spatial_loss_weight', type=float, default=0.5, help='空间一致性损失的权重')
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'vis_output'), exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'checkpoints'), exist_ok=True)

    try:
        print("加载数据")
        database = h5py.File(os.path.join(args.dataset_path,'train_data.h5'),'r')
        all_keys = list(database.keys())
        if args.dataset_select is None:
            dataset_num = args.dataset_num
            if dataset_num is None: dataset_num = len(all_keys)
            dataset_indices = torch.randperm(len(all_keys))[:dataset_num].numpy()
        else:
            dataset_indices = [int(i) for i in args.dataset_select.split(',')]
            dataset_num = len(dataset_indices)

        keys = [all_keys[i] for i in dataset_indices]
        all_images, all_labels, all_map_coeffs = [], [], []
        for key in tqdm(keys, desc="Loading data"):
            img = database[key]['images']['image_0'][:]
            if img.ndim == 2: img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 1: img = np.concatenate([img] * 3, axis=-1)
            obj = centerize_obj(database[key]['obj'][:])
            map_coef = {
                'x':np.array([obj[...,0].min(),obj[...,0].max()]),
                'y':np.array([obj[...,1].min(),obj[...,1].max()]),
                'h':get_map_coef(obj[...,2].reshape(-1))
            }
            all_images.append(img)
            all_labels.append(obj)
            all_map_coeffs.append(map_coef)
    except Exception as e:
        print(f"加载数据时发生严重错误: {e}")
        exit(1)
        
    num_datasets = len(all_images)
    print(f"共找到 {num_datasets} 个数据集待处理。")
    np.save(os.path.join(args.output_dir,'dataset_indices.npy'),dataset_indices)

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("错误：没有检测到可用的GPU。")
        exit(1)
    world_size = min(world_size, num_datasets)
    port = find_free_port()
    print(f"使用空闲端口 {port} 进行分布式训练。")

    num_real_tasks = len(all_images)
    padded_task_indices = list(range(num_real_tasks))
    num_to_pad = (world_size - num_real_tasks % world_size) % world_size
    if num_to_pad > 0: padded_task_indices.extend([-1] * num_to_pad)
    
    print(f"将在 {world_size} 张GPU上启动训练... 总任务数（含虚拟任务）: {len(padded_task_indices)}")
    
    mp.spawn(
        main_worker,
        args=(world_size, args, all_images, all_labels, all_map_coeffs, padded_task_indices, port),
        nprocs=world_size,
        join=True
    )
    
    print("所有训练任务已完成！")
