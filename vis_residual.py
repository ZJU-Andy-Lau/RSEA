import numpy as np
import h5py
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from utils import get_current_time
from dataloader import residual_average
import argparse

timestamp = get_current_time()

def vis_raw(img:np.ndarray,residual:np.ndarray,output_folder):
    H,W = img.shape[:2]
    if img.shape[:2] != residual.shape:
        raise ValueError("图片和结果数组的H和W维度必须相同。")

    # 1. 创建一个布尔掩码，标记 residual 中非 nan 的位置
    valid_mask = ~np.isnan(residual)

    # 如果 residual 中全是 nan，则不进行标注，直接保存原图
    if not np.any(valid_mask):
        print("警告: residual 数组中所有值均为 nan，不进行任何标注。")
        return

    # 2. 对非 nan 的值进行归一化处理（映射到 0-1 范围）
    res_values = residual[valid_mask]
    min_val = np.min(res_values)
    max_val = np.max(res_values)
    
    # 防止所有值都相同时除以零
    if max_val == min_val:
        normalized_values = np.full_like(res_values, 0.5, dtype=float)
    else:
        normalized_values = (res_values - min_val) / (max_val - min_val)

    # 3. 获取从绿色到红色的色带，并映射颜色
    # Matplotlib 的 'RdYlGn_r' 色带正好是从绿 -> 黄 -> 红
    cmap = plt.cm.get_cmap('RdYlGn_r')
    
    # cmap 返回的是 RGBA 格式，且值范围是 [0, 1]
    # 我们需要转换为 OpenCV 使用的 BGR 格式，且值范围是 [0, 255]
    colors_rgba = cmap(normalized_values)
    colors_bgr = (colors_rgba[:, :3][:, ::-1] * 255).astype(np.uint8)

    # 4. 在图片副本上进行标注
    output_img = img.copy()
    output_img[valid_mask] = colors_bgr

    # 5. 保存结果图片
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_path = os.path.join(output_folder, f"residual_raw_{timestamp}.png")
    
    cv2.imwrite(output_path, output_img)
    print(f"标注后的图片已保存至: {output_path}")

    # --- 创建并保存颜色图例 ---
    fig, ax = plt.subplots(figsize=(2, 8))
    fig.subplots_adjust(right=0.5)
    norm = mpl.colors.Normalize(vmin=min_val, vmax=max_val)
    cb = mpl.colorbar.ColorbarBase(ax, cmap=cmap, norm=norm, orientation='vertical')
    cb.set_label('数值范围', size=12, weight='bold')
    
    output_path_legend = os.path.join(output_folder, f"colorbar_legend_{timestamp}.png")
    plt.savefig(output_path_legend, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"颜色图例已保存至: {output_path_legend}")

def vis_mask(img, res, output_folder, patch_size = 16, alpha=0.6):
    """
    将 numpy 数组的值映射到色带，并以半透明掩膜的形式标注到图片上。

    Args:
        img (np.ndarray): 输入的原始图片，形状为 (H, W, 3)，BGR格式。
        res (np.ndarray): 包含数值和nan的numpy数组，形状为 (H/patch_size, W/patch_size)。
        patch_size (int): res数组中一个元素对应的原始图片中的边长。例如，16。
        output_folder (str): 保存标注后图片的文件夹路径。
        alpha (float): 标注掩膜的透明度，值在 0.0 到 1.0 之间。0.0表示完全透明，1.0表示完全不透明。
    """
    H_img, W_img, _ = img.shape
    H_res, W_res = res.shape

    # 1. 检查尺寸匹配
    if H_img != H_res * patch_size or W_img != W_res * patch_size:
        raise ValueError(
            f"图片尺寸 ({H_img}, {W_img}) 与 res 数组尺寸 ({H_res}, {W_res}) "
            f"和 patch_size ({patch_size}) 不匹配。"
            f"期望图片尺寸为 ({H_res * patch_size}, {W_res * patch_size})。"
        )

    # 2. 创建一个布尔掩码，标记 res 中非 nan 的位置
    valid_mask_res = ~np.isnan(res)

    # 如果 res 中全是 nan，则不进行标注，直接保存原图
    if not np.any(valid_mask_res):
        print("警告: res 数组中所有值均为 nan，不进行任何标注。")
        # 确保目录存在，并保存原始图片，以符合输出要求
        if not os.path.exists(output_folder):
            os.makedirs(output_folder)
        output_path = os.path.join(output_folder, f"residual_mask_{timestamp}.png")
        cv2.imwrite(output_path, img)
        print(f"原图已保存至: {output_path}")
        return

    # 3. 对非 nan 的值进行归一化处理（映射到 0-1 范围）
    res_values = res[valid_mask_res]
    min_val = np.min(res_values)
    max_val = np.max(res_values)
    
    # 防止所有值都相同时除以零
    if max_val == min_val:
        normalized_values = np.full_like(res_values, 0.5, dtype=float)
    else:
        normalized_values = (res_values - min_val) / (max_val - min_val)

    # 4. 获取从绿色到红色的色带，并映射颜色
    cmap = plt.cm.get_cmap('RdYlGn_r') # 'RdYlGn_r' 是从绿到黄再到红的色带
    colors_rgba = cmap(normalized_values)
    # 将 RGBA (0-1) 转换为 BGR (0-255)
    colors_bgr = (colors_rgba[:, :3][:, ::-1] * 255).astype(np.uint8)

    # 5. 构建一个与原始图像同大小的，用于叠加的颜色掩膜层
    # 初始时全黑 (0,0,0)
    color_overlay = np.zeros_like(img, dtype=np.uint8) 

    # 将计算出的颜色填充到 color_overlay 的对应 patch 区域
    color_idx = 0
    for r in range(H_res):
        for c in range(W_res):
            if valid_mask_res[r, c]: # 如果 res[r,c] 是有效值
                # 计算当前 patch 的像素范围
                start_y = r * patch_size
                end_y = (r + 1) * patch_size
                start_x = c * patch_size
                end_x = (c + 1) * patch_size
                
                # 将对应的颜色填充到 overlay 层
                color_overlay[start_y:end_y, start_x:end_x, :] = colors_bgr[color_idx]
                color_idx += 1
    
    # 6. 将颜色掩膜层半透明地叠加到原始图像上
    # output_img = cv2.addWeighted(img, 1 - alpha, color_overlay, alpha, 0)
    # img (BGR), overlay (BGR)
    # 对于叠加，我们只在 color_overlay 中有颜色的地方进行叠加
    # 创建一个与 img 相同大小的 alpha_channel
    alpha_channel = np.zeros((H_img, W_img), dtype=np.float32)

    # 填充 valid_mask_res 对应的区域为 alpha 值
    for r in range(H_res):
        for c in range(W_res):
            if valid_mask_res[r, c]:
                start_y = r * patch_size
                end_y = (r + 1) * patch_size
                start_x = c * patch_size
                end_x = (c + 1) * patch_size
                alpha_channel[start_y:end_y, start_x:end_x] = alpha

    output_img = img.copy().astype(np.float32) # 将 img 转换为浮点数类型以便进行加权混合

    # 遍历图像的每个像素
    for y in range(H_img):
        for x in range(W_img):
            if alpha_channel[y, x] > 0: # 只有在有掩膜的区域才进行混合
                output_img[y, x, :] = (1 - alpha_channel[y, x]) * img[y, x, :] + \
                                       alpha_channel[y, x] * color_overlay[y, x, :]
    
    output_img = output_img.astype(np.uint8)

    # 7. 保存结果图片
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        
    output_path = os.path.join(output_folder, f"residual_mask_{timestamp}.png")
    
    cv2.imwrite(output_path, output_img)
    print(f"标注后的半透明图片已保存至: {output_path}")

def clamp_res(residual:np.ndarray):
    valid_mask = ~np.isnan(residual)
    min_val = residual[valid_mask].min()
    median_val = np.median(residual[valid_mask])
    max_val = 2 * median_val - min_val
    residual[residual > max_val] = max_val
    return residual

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default=None)
    parser.add_argument('--img_idx',type=int,default=None)
    parser.add_argument('--view_idx',type=int,default=0)
    parser.add_argument('--output_folder',type=str,default=None)
    args = parser.parse_args()

    data = h5py.File(args.dataset_path,'r')
    keys = list(data.keys())

    if args.img_idx is None:
        img_idx = np.random.randint(0,len(keys)-1)
    else:
        img_idx = args.img_idx
    
    img = data[keys[img_idx]]['images'][f'image_{args.view_idx}'][:]
    residual_raw = data[keys[img_idx]]['residuals'][f'residual_{args.view_idx}'][:]
    img = np.stack([img] * 3,axis=-1)
    residual_raw = clamp_res(residual_raw)

    os.makedirs(args.output_folder,exist_ok=True)
    vis_raw(img,residual_raw,args.output_folder)
    vis_mask(img,residual_average(residual_raw,16),args.output_folder)



    
