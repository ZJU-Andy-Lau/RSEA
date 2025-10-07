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
from model.encoder_dino_0927 import EncoderDino
from model.decoders import DecoderFinetune
from utils import apply_polynomial,get_map_coef,downsample
from tqdm import tqdm
from scheduler import MultiStageOneCycleLR
import kornia.augmentation as K
import kornia.geometry.transform as KT
import cv2
import matplotlib.pyplot as plt


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

def crop_to_windows(image_tensor, label_tensor, image_np_for_vis, window_size=1024, win_num=3, output_path=None):
    """
    将大尺寸图像和标签高效地切分成多个窗口，包括均匀分布和随机旋转的窗口。
    采用拒绝采样方法确保旋转窗口完全在原图内，不含任何padding。

    参数:
        image_tensor (torch.Tensor): 形状为 (C, H, W) 的图像张量。
        label_tensor (torch.Tensor): 形状为 (C_label, H, W) 的标签张量。
        image_np_for_vis (np.ndarray): 用于可视化的原始Numpy图像 (H, W, C)。
        window_size (int): 窗口的目标边长。
        win_num (int): 每条边上裁切的窗口数量。
        output_path (str): 保存裁切窗口和可视化图像的文件夹路径。

    返回:
        (torch.Tensor, torch.Tensor): 包含所有窗口的图像和标签张量元组。
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

    # 2. 随机裁切和旋转窗口 (Random Cropping and Rotation) - 采用拒绝采样策略
    num_random_windows = win_num * win_num
    num_found = 0
    
    ws_half = window_size / 2.0
    # 预计算窗口四个角的相对坐标
    corners = torch.tensor([[-ws_half, -ws_half], [ws_half, -ws_half], [ws_half, ws_half], [-ws_half, ws_half]])

    # 只有当图像足够大时才进行随机裁切
    if H > window_size and W > window_size:
        with tqdm(total=num_random_windows, desc=f"Finding {num_random_windows} valid random crops", leave=False) as pbar:
            while num_found < num_random_windows:
                angle_deg = torch.rand(1) * 360
                angle_rad = torch.deg2rad(angle_deg)
                
                center_x = torch.rand(1) * W
                center_y = torch.rand(1) * H

                # 构建旋转矩阵
                c, s = torch.cos(angle_rad), torch.sin(angle_rad)
                rot_mat = torch.tensor([[c, -s], [s, c]])
                
                # 计算旋转后四个角的绝对坐标
                rotated_corners = corners @ rot_mat.T + torch.tensor([center_x, center_y])

                # 验证所有角点是否在图像边界内
                if torch.all(rotated_corners[:, 0] >= 0) and torch.all(rotated_corners[:, 0] <= W) and \
                   torch.all(rotated_corners[:, 1] >= 0) and torch.all(rotated_corners[:, 1] <= H):
                    
                    # --- 有效样本，执行裁切 ---
                    num_found += 1
                    pbar.update(1)

                    # 计算旋转窗口的最小轴对齐边界框
                    xmin, ymin = rotated_corners.min(dim=0).values.floor().int()
                    xmax, ymax = rotated_corners.max(dim=0).values.ceil().int()
                    
                    # 确保边界框不越界
                    xmin, ymin = max(0, xmin), max(0, ymin)
                    xmax, ymax = min(W, xmax), min(H, ymax)

                    # 裁切出包含旋转窗口的最小区域
                    temp_img = image_tensor[:, ymin:ymax, xmin:xmax]
                    temp_lbl = label_tensor[:, ymin:ymax, xmin:xmax]
                    
                    # 计算新的旋转中心（相对于裁切出的区域）
                    center_x_new = center_x - xmin
                    center_y_new = center_y - ymin

                    # 旋转这个小区域
                    rotated_temp_img = KT.rotate(temp_img.float().unsqueeze(0), angle_deg, center=torch.tensor([[center_x_new, center_y_new]]), mode='bilinear', align_corners=True)
                    rotated_temp_lbl = KT.rotate(temp_lbl.float().unsqueeze(0), angle_deg, center=torch.tensor([[center_x_new, center_y_new]]), mode='bilinear', align_corners=True)

                    # 从旋转后的小区域中心裁切出最终窗口
                    final_img_win = K.CenterCrop(window_size)(rotated_temp_img).squeeze(0)
                    final_lbl_win = K.CenterCrop(window_size)(rotated_temp_lbl).squeeze(0)

                    image_windows_list.append(final_img_win)
                    label_windows_list.append(final_lbl_win)

                    # 在可视化图上绘制旋转后的矩形框
                    cv2.drawContours(vis_image, [rotated_corners.int().numpy()], 0, (255, 0, 0), 5)
    
    if output_path:
        cv2.imwrite(os.path.join(output_path, 'window_visualization.png'), vis_image)

    if not image_windows_list:
        return None, None

    image_windows = torch.stack(image_windows_list, dim=0)
    label_windows = torch.stack(label_windows_list, dim=0)

    # --- 3. 旋转增强 (Rotation Augmentation) ---
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

    # 保存所有裁切出的小图
    if output_path:
        for i in range(image_windows_augmented.shape[0]):
            # The tensor is float32 with values in [0, 255] range at this point
            tensor_slice = image_windows_augmented[i]
            # Permute, move to CPU, clamp, convert to uint8, then convert to numpy
            img_to_save = tensor_slice.permute(1, 2, 0).cpu().clamp(0, 255).to(torch.uint8).numpy()
            cv2.imwrite(os.path.join(output_path, f'window_{i:04d}.png'), img_to_save)

    # --- 4. 标准化和下采样 (Normalization and Downsampling) ---
    transform = K.Normalize(
                mean=torch.tensor([0.485, 0.456, 0.406]), 
                std=torch.tensor([0.229, 0.224, 0.225])
            )
    image_windows_augmented = transform(image_windows_augmented.float() / 255.0)

    label_windows_rotated = label_windows_rotated.permute(0, 2, 3, 1) # (N, H, W, C)
    label_windows_downsampled = downsample(label_windows_rotated, 16)
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

def train_single_decoder(rank, decoder, encoder, buffer, map_coeffs, val_img, val_lbl, epochs, lr, save_path, vis_save_path):
    """
    在指定的GPU上训练一个Decoder模型, 并在训练中进行验证。
    """
    decoder.to(rank)
    best_state_dict = None
    min_loss = 1e9
    
    optimizer = optim.Adam(decoder.parameters(), lr=lr)
    scheduler = MultiStageOneCycleLR(optimizer,
                                     total_steps=epochs,
                                     warmup_ratio=.1,
                                     cooldown_ratio=.7)
    criterion = nn.MSELoss()

    print(f"[GPU {rank}] 开始训练 {os.path.basename(save_path)}. Buffer大小: {len(buffer['features'])} 个样本.")
    features = buffer['features'].permute(1,0)[None,:,:,None].to(rank)
    gt_objs = buffer['objs'].to(rank)
    
    for epoch in range(epochs):
        decoder.train()
        optimizer.zero_grad()
        output = decoder(features)
        output = output.permute(0,2,3,1).flatten(0,2)
        pred_obj = warp_by_poly(output,map_coeffs)
        loss = torch.norm(pred_obj - gt_objs,dim=1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()

        if (epoch + 1) % 100 == 0 and (epoch + 1) > 0:
            decoder.eval()
            with torch.no_grad():
                val_img_gpu = val_img.to(rank)
                val_lbl_gpu = val_lbl.to(rank)
                val_feat, _ = encoder(val_img_gpu)
                val_output = decoder(val_feat)
                val_output = val_output.permute(0,2,3,1).flatten(0,2)
                val_pred_obj = warp_by_poly(val_output, map_coeffs)
                val_gt_obj = val_lbl_gpu.permute(0,2,3,1).flatten(0,2)
                val_loss = torch.norm(val_pred_obj - val_gt_obj, dim=1).mean()
            
            print(f"[GPU {rank}] | 任务: {os.path.basename(save_path)} | Epoch [{epoch+1}/{epochs}] | Train Loss: {loss:.4f} | Val Loss: {val_loss:.4f} | min Loss: {min_loss:.4f}")

            if (epoch + 1) % 1000 == 0 and (epoch + 1) > 0:
                pred_coords = val_pred_obj.cpu().numpy()
                true_coords = val_gt_obj.cpu().numpy()
                
                plt.figure(figsize=(10, 10))
                plt.scatter(true_coords[:, 0], true_coords[:, 1], c='red', label='Ground Truth', s=10, alpha=0.7)
                plt.scatter(pred_coords[:, 0], pred_coords[:, 1], c='green', label='Prediction', s=10, alpha=0.7)
                plt.legend()
                plt.title(f'Validation: Prediction vs. Ground Truth (Epoch {epoch+1})')
                plt.xlabel('X coordinate')
                plt.ylabel('Y coordinate')
                plt.grid(True)
                plt.axis('equal')

                path_parts = os.path.splitext(vis_save_path)
                epoch_save_path = f"{path_parts[0]}_epoch_{epoch+1}{path_parts[1]}"
                plt.savefig(epoch_save_path)
                plt.close()
                print(f"[GPU {rank}] Validation scatter plot saved to {epoch_save_path}")

        if loss < min_loss:
            best_state_dict = decoder.state_dict()
            min_loss = loss

    torch.save(best_state_dict, save_path)
    print(f"[GPU {rank}] 训练完成. Decoder已保存至 {save_path}")

    print(f"[GPU {rank}] 正在使用最佳模型生成最终验证散点图...")
    decoder.load_state_dict(best_state_dict)
    decoder.eval()
    with torch.no_grad():
        val_img_gpu = val_img.to(rank)
        val_feat, _ = encoder(val_img_gpu)
        val_output = decoder(val_feat).permute(0,2,3,1).flatten(0,2)
        pred_coords = warp_by_poly(val_output, map_coeffs).cpu().numpy()
        true_coords = val_lbl.permute(0,2,3,1).flatten(0,2).cpu().numpy()

    plt.figure(figsize=(10, 10))
    plt.scatter(true_coords[:, 0], true_coords[:, 1], c='red', label='Ground Truth', s=10, alpha=0.7)
    plt.scatter(pred_coords[:, 0], pred_coords[:, 1], c='green', label='Prediction', s=10, alpha=0.7)
    plt.legend()
    plt.title('Final Validation with Best Model: Prediction vs. Ground Truth')
    plt.xlabel('X coordinate')
    plt.ylabel('Y coordinate')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(vis_save_path)
    plt.close()
    print(f"[GPU {rank}] 最终散点图已保存至 {vis_save_path}")


# --- 4. 主工作进程 ---

def main_worker(rank, world_size, args, all_images, all_labels, all_map_coeffs):
    """
    每个GPU上运行的主函数。
    """
    print(f"启动 GPU {rank}/{world_size} 的工作进程...")
    setup_distributed(rank, world_size)
    
    vis_output_dir = os.path.join(args.output_dir, 'vis_output')
    if rank == 0:
        os.makedirs(vis_output_dir, exist_ok=True)
    dist.barrier()

    num_total_datasets = all_images.shape[0]
    indices_for_this_gpu = list(range(num_total_datasets))[rank::world_size]
    print(f"[GPU {rank}] 分配到 {len(indices_for_this_gpu)} 个训练任务 (Indices: {indices_for_this_gpu[:5]}...).")
    
    encoder = EncoderDino(dino_weight_path=args.dino_weight_path)
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    encoder.to(rank)
    encoder.eval()
    for param in encoder.parameters():
        param.requires_grad = False

    for data_idx in indices_for_this_gpu:
        print(f"[GPU {rank}] 开始处理数据集索引: {data_idx}")
        
        image_np = all_images[data_idx]
        label_np = all_labels[data_idx]
        map_coef = all_map_coeffs[data_idx]
        
        image = torch.from_numpy(image_np).permute(2, 0, 1)
        label = torch.from_numpy(label_np).permute(2, 0, 1)

        print(f"[GPU {rank}] 加载数据: Image {image.shape}, Label {label.shape}")
        
        img_vis_dir = os.path.join(vis_output_dir, f'data_{data_idx}')
        if rank == 0:
            os.makedirs(img_vis_dir, exist_ok=True)
        dist.barrier()
        
        image_windows, label_windows = crop_to_windows(image, label, image_np, args.window_size, args.win_num, img_vis_dir)
        if image_windows is None:
            print(f"[GPU {rank}] 索引 {data_idx} 的图像尺寸过小，无法裁切，已跳过。")
            continue
        print(f"[GPU {rank}] 图像和标签被切分为 {image_windows.shape[0]} 个窗口.")
        
        # --- c. 创建验证数据 (新逻辑) ---
        H, W = image.shape[1], image.shape[2]
        val_angle = torch.tensor([45.0])
        center_crop = K.CenterCrop(args.window_size)

        # 1. 先旋转整个图像
        # 添加batch维度以进行旋转
        full_img_rotated = KT.rotate(image.float().unsqueeze(0), val_angle, mode='bilinear', align_corners=True)
        full_lbl_rotated = KT.rotate(label.float().unsqueeze(0), val_angle, mode='bilinear', align_corners=True)

        # 2. 然后从旋转后的图像中心裁切
        val_img_unnormalized = center_crop(full_img_rotated)
        val_lbl_rotated = center_crop(full_lbl_rotated)

        # 3. 保存裁切出的验证样本图像
        val_img_to_save = val_img_unnormalized.squeeze(0).permute(1, 2, 0).cpu().to(torch.uint8).numpy()
        cv2.imwrite(os.path.join(img_vis_dir, 'validation_sample.png'), val_img_to_save)
        print(f"[GPU {rank}] 验证样本图像已保存至 {os.path.join(img_vis_dir, 'validation_sample.png')}")

        # 4. 创建并保存在原图上框出验证区域的可视化图像
        # 将训练样本的可视化结果作为底图
        vis_val_image = cv2.imread(os.path.join(img_vis_dir, 'window_visualization.png'))
        center_x, center_y = W / 2, H / 2
        # OpenCV的旋转角度为逆时针，所以用-45度
        rect = ((center_x, center_y), (args.window_size, args.window_size), -45.0) 
        box_pts = cv2.boxPoints(rect)
        box_pts = np.int0(box_pts)
        # 使用黄色 (0, 255, 255) 框出验证区域
        cv2.drawContours(vis_val_image, [box_pts], 0, (0, 255, 255), 5) 
        cv2.imwrite(os.path.join(img_vis_dir, 'window_visualization_with_validation.png'), vis_val_image)
        print(f"[GPU {rank}] 验证区域可视化图像已保存至 {os.path.join(img_vis_dir, 'window_visualization_with_validation.png')}")

        # 5. 标准化和下采样，准备输入模型
        norm_transform = K.Normalize(mean=torch.tensor([0.485, 0.456, 0.406]), std=torch.tensor([0.229, 0.224, 0.225]))
        val_img = norm_transform(val_img_unnormalized / 255.0) 
        val_lbl_downsampled = downsample(val_lbl_rotated.permute(0,2,3,1), 16)
        val_lbl = val_lbl_downsampled.permute(0,3,1,2)
        print(f"[GPU {rank}] 验证数据已创建. Shape: {val_img.shape}")


        # --- d. 创建特征Buffer (在GPU上) ---
        feature_buffer = []
        temp_dataloader = DataLoader(image_windows, batch_size=args.batch_size, shuffle=False)
        with torch.no_grad():
            for image_batch in temp_dataloader:
                feature_batch,_ = encoder(image_batch.to(rank))
                feature_buffer.append(feature_batch.cpu())
        
        all_features = torch.cat(feature_buffer, dim=0).permute(0,2,3,1).flatten(0,2)
        all_labels_for_features = label_windows.permute(0,2,3,1).flatten(0,2)

        buffer = {
            'features':all_features,
            'objs':all_labels_for_features
        }

        # --- e. 训练新的Decoder ---
        decoder = DecoderFinetune(in_channels=encoder.output_channels,block_num=args.decoder_block_num)
        
        decoder_name = f"decoder_{data_idx}.pth"
        save_path = os.path.join(args.output_dir, decoder_name)
        scatter_plot_path = os.path.join(img_vis_dir, 'validation_scatter.png')
        
        train_single_decoder(
            rank=rank,
            decoder=decoder,
            encoder=encoder,
            buffer=buffer,
            map_coeffs=map_coef,
            val_img=val_img,
            val_lbl=val_lbl,
            epochs=args.epochs,
            lr=args.lr,
            save_path=save_path,
            vis_save_path=scatter_plot_path
        )

    cleanup()
    print(f"GPU {rank} 的所有任务已完成。")


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
    parser.add_argument('--win_num', type=int, default=3, help='每条边上裁切的窗口数，总共裁切 win_num*win_num 个均匀窗口和同样数量的随机窗口')
    parser.add_argument('--epochs', type=int, default=200, help='每个Decoder的训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    parser.add_argument('--batch_size', type=int, default=4, help='特征提取时的批量大小')
    parser.add_argument('--decoder_block_num',type=int,default=1)
    
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(os.path.join(args.output_dir, 'vis_output'), exist_ok=True)

    try:
        print("加载数据")
        database = h5py.File(os.path.join(args.dataset_path,'train_data.h5'),'r')
        all_keys = list(database.keys())
        if args.dataset_select is None:
            dataset_num = args.dataset_num
            if dataset_num is None:
                dataset_num = len(all_keys)
            dataset_indices = torch.randperm(len(all_keys))[:dataset_num].numpy()
        else:
            dataset_indices = [int(i) for i in args.dataset_select.split(',')]
            dataset_num = len(dataset_indices)

        keys = [all_keys[i] for i in dataset_indices]
        all_images = []
        all_labels = []
        all_map_coeffs = []
        for key in tqdm(keys, desc="Loading data"):
            img = database[key]['images']['image_0'][:]
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 1:
                img = np.concatenate([img] * 3, axis=-1)

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

    world_size = torch.cuda.device_count()
    if world_size == 0:
        print("错误：没有检测到可用的GPU。")
        exit(1)
        
    world_size = min(world_size, num_datasets)
    
    print(f"将在 {world_size} 张GPU上启动训练...")
    
    mp.spawn(
        main_worker,
        args=(world_size, args, all_images, all_labels, all_map_coeffs),
        nprocs=world_size,
        join=True
    )
    
    print("所有训练任务已完成！")

