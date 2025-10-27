import os
import argparse
import numpy as np
import cv2
import torch
import rasterio
from rasterio.transform import from_origin
from rasterio.windows import Window
from pyproj import CRS
from scipy.interpolate import RegularGridInterpolator
from tqdm import tqdm
from types import SimpleNamespace # 用于创建模拟的 options 对象
import warnings
from typing import List

# 确保可以导入同目录的 rs_image_1022 和 rpc
try:
    from rs_image_1022 import RSImage
    from rpc import RPCModelParameterTorch
except ImportError:
    print("错误：无法导入 'rs_image_1022.py' 或 'rpc.py'。")
    print("请确保这些文件与 'create_orthomosaic.py' 位于同一目录中。")
    exit(1)

warnings.filterwarnings("ignore")

def load_imgs_bundle_simplified(args) -> List[RSImage]:
    """
    (新) 单进程版本的影像加载器。
    它加载 RSImage 对象，但不执行DDP相关的打印。
    """
    base_path = os.path.join(args.root, 'adjust_images')
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    # 根据索引过滤文件夹
    try:
        img_folders = [img_folders[i] for i in select_img_idxs]
    except IndexError:
        print(f"错误：影像索引超出范围。可用索引为 0 到 {len(img_folders) - 1}。")
        return []

    images = []
    print(f"找到 {len(img_folders)} 个影像文件夹。正在加载...")
        
    # 创建一个模拟的 'options' 对象，因为 RSImage.__init__ 需要它
    mock_options = SimpleNamespace(auto=True) 
    
    for idx, folder in enumerate(img_folders):
        img_path = os.path.join(base_path, folder)
        try:
            # 使用 select_img_idxs 中的原始索引作为 RSImage 的 id
            original_index = select_img_idxs[idx]
            images.append(RSImage(mock_options, img_path, original_index))
            print(f"已加载影像 {original_index} (来自 {folder})。")
        except Exception as e:
            print(f"加载影像 {original_index} (来自 {folder}) 失败: {e}")
    
    print(f"成功加载 {len(images)} 张影像到内存。")
    return images

def create_individual_orthos(args):
    """
    (修改) 使用烘焙后的RPC，为每张影像生成单独的正射影像。
    """
    
    print("--- 步骤 1: 加载影像数据 (影像、DEM、原始RPC) ---")
    images = load_imgs_bundle_simplified(args)
    if not images:
        print("未加载任何影像。退出。")
        return

    print(f"\n--- 步骤 2: 加载并覆盖为调整后的 RPC (来自 {args.rpc_dir}) ---")
    for img in images:
        # 基于 img.id (即它在原始文件夹中的索引) 查找RPC文件
        rpc_filename = f"image_{img.id}_L{args.level}_baked.txt"
        rpc_path = os.path.join(args.rpc_dir, rpc_filename)
        
        if os.path.exists(rpc_path):
            try:
                img.rpc.load_from_file(rpc_path)
                print(f"成功为影像 {img.id} 加载调整后的RPC: {rpc_filename}")
            except Exception as e:
                print(f"警告：为影像 {img.id} 加载RPC {rpc_filename} 失败: {e}。将使用原始RPC。")
        else:
            print(f"警告：找不到影像 {img.id} 调整后的RPC: {rpc_filename}。将使用原始RPC。")
            
        # 将RPC移动到GPU以便快速投影
        if torch.cuda.is_available():
            img.rpc.to_gpu()

    print("\n--- [修改] 步骤 3-5: 循环处理每张影像 ---")
    
    for img in images:
        print(f"\n--- 正在处理影像 {img.id} ---")
        
        # --- 步骤 3 (单张影像): 计算地理边界 ---
        print(f"  步骤 3: 计算影像 {img.id} 的地理边界...")
        corners_xy = img.__get_corner_xys__() 
        
        min_x = corners_xy[:, 0].min()
        max_x = corners_xy[:, 0].max()
        min_y = corners_xy[:, 1].min()
        max_y = corners_xy[:, 1].max()
        
        print(f"  影像 {img.id} 边界 (EPSG:3857):")
        print(f"    X: ({min_x:.2f}, {max_x:.2f})")
        print(f"    Y: ({min_y:.2f}, {max_y:.2f})")

        # --- 步骤 4 (单张影像): 定义输出网格和创建插值器 ---
        print(f"  步骤 4: 定义输出网格并创建插值器...")
        out_W = int(np.ceil((max_x - min_x) / args.resolution))
        out_H = int(np.ceil((max_y - min_y) / args.resolution))
        
        if out_W <= 0 or out_H <= 0:
            print(f"  跳过影像 {img.id}: 无效的输出尺寸 ({out_W}x{out_H})。")
            del img.image # 释放内存
            continue
            
        # GeoTIFF 变换 (左上角)
        transform = from_origin(min_x, max_y, args.resolution, args.resolution)
        print(f"  输出尺寸: {out_W} x {out_H} 像素")
        
        H_src, W_src = img.image.shape[:2]
        lines_src = np.arange(H_src)
        samps_src = np.arange(W_src)
        
        # 为 R, G, B 通道创建插值器
        try:
            inter_r = RegularGridInterpolator((lines_src, samps_src), img.image[..., 0], method='linear', bounds_error=False, fill_value=0)
            inter_g = RegularGridInterpolator((lines_src, samps_src), img.image[..., 1], method='linear', bounds_error=False, fill_value=0)
            inter_b = RegularGridInterpolator((lines_src, samps_src), img.image[..., 2], method='linear', bounds_error=False, fill_value=0)
        except ValueError as e:
            print(f"  跳过影像 {img.id}: 创建插值器失败。可能影像/DEM尺寸不匹配？错误: {e}")
            del img.image
            continue

        # 释放原始影像内存（插值器已持有数据副本）
        del img.image
        
        print(f"  已为影像 {img.id} 创建插值器。")

        # --- 步骤 5 (单张影像): 分块生成正射影像 ---
        output_filename = f"ortho_image_{img.id}_L{args.level}.tif"
        output_filepath = os.path.join(args.output_dir, output_filename)
        print(f"  步骤 5: 开始分块生成 {output_filepath}...")
        
        # 准备GeoTIFF写入
        with rasterio.open(
            output_filepath, 'w',
            driver='GTiff',
            height=out_H,
            width=out_W,
            count=3, # RGB
            dtype=np.uint8, # 直接输出uint8
            crs=CRS.from_epsg(3857),
            transform=transform,
            compress='lzw'
        ) as dst:
            
            block_size = args.block_size
            
            # 使用tqdm创建进度条
            tqdm_iter = range(0, out_H, block_size)
            for i in tqdm(tqdm_iter, desc=f"处理影像 {img.id}", leave=False):
                for j in range(0, out_W, block_size):
                    
                    # 定义当前块的边界
                    i_end = min(i + block_size, out_H)
                    j_end = min(j + block_size, out_W)
                    
                    block_h = i_end - i
                    block_w = j_end - j
                    
                    # 计算块的地理坐标网格
                    out_x_coords_block = np.linspace(
                        min_x + (j + 0.5) * args.resolution,
                        min_x + (j_end - 0.5) * args.resolution,
                        block_w
                    )
                    out_y_coords_block = np.linspace(
                        max_y - (i + 0.5) * args.resolution,
                        max_y - (i_end - 0.5) * args.resolution,
                        block_h
                    )
                    block_xx, block_yy = np.meshgrid(out_x_coords_block, out_y_coords_block)
                    
                    # (N, 2) 形状的物方点
                    xy_points = np.stack([block_xx.ravel(), block_yy.ravel()], axis=-1)
                    
                    try:
                        # 步骤 1: Ground-to-Image 投影
                        sampline_pred = img.xy_to_sampline(xy_points) # (N, 2) -> (samp, line)
                        
                        # 步骤 2: 准备插值坐标 (line, samp)
                        points_to_sample = np.stack([sampline_pred[:, 1], sampline_pred[:, 0]], axis=-1)
                        
                        # 步骤 3: 采样
                        pixel_r = inter_r(points_to_sample).reshape(block_h, block_w)
                        pixel_g = inter_g(points_to_sample).reshape(block_h, block_w)
                        pixel_b = inter_b(points_to_sample).reshape(block_h, block_w)
                        
                        # 步骤 4: 堆叠并设置 Nodata
                        final_block_rgb = np.stack([pixel_r, pixel_g, pixel_b], axis=-1)
                        final_block_rgb[pixel_r == 0] = 0 # 假设 0 是 nodata
                        
                        # 转换回 uint8
                        final_block_rgb = final_block_rgb.astype(np.uint8)
                        
                        # 步骤 5: 写入GeoTIFF
                        window = Window(j, i, block_w, block_h)
                        dst.write(final_block_rgb[..., 0], 1, window=window)
                        dst.write(final_block_rgb[..., 1], 2, window=window)
                        dst.write(final_block_rgb[..., 2], 3, window=window)
                        
                    except Exception as e:
                        # 某张影像在此块投影失败，跳过
                        # print(f"警告：影像 {img.id} 在块 (i={i}, j={j}) 投影失败: {e}")
                        continue
        
        print(f"  影像 {img.id} 处理完成。")
        del inter_r, inter_g, inter_b # 清理插值器内存

    print(f"\n--- 步骤 6: 完成 ---")
    print(f"所有单独的正射影像已成功保存到: {args.output_dir}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="使用调整后的RPC为每张影像生成单独的正射影像")

    parser.add_argument('--root', type=str, required=True,
                        help='包含 "adjust_images" 文件夹的项目根目录。')

    parser.add_argument('--rpc_dir', type=str, required=True,
                        help='包含 "baked" RPC 文件的目录 (例如: .../debug_output/rpc_level_2)')

    parser.add_argument('--level', type=int, required=True,
                        help='要使用的RPC的层级 (例如: 2，对应 rpc_level_2)')

    parser.add_argument('--select_imgs', type=str, required=True,
                        help='要处理的影像索引，以逗号分隔 (例如: "0,1")')

    parser.add_argument('--resolution', type=float, default=1.0,
                        help='输出正射影像的分辨率 (米/像素)')

    parser.add_argument('--output_dir', type=str, required=True,
                        help='用于保存所有输出的 GeoTIFF 文件的目录 (例如: ./ortho_outputs)')

    parser.add_argument('--block_size', type=int, default=2048,
                        help='处理块大小（像素），用于管理内存。')

    args = parser.parse_args()
    
    # [修改] 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    # [修改] 调用新函数
    create_individual_orthos(args)

