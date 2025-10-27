import os
import numpy as np
import torch
from typing import List, Tuple
from rs_image_1022 import RSImage  # 依赖 rs_image_1022.py
from rpc import RPCModelParameterTorch # 依赖 rpc.py

# --- 数据加载函数 (来自 adjust_test_1023.py) ---

def load_imgs_bundle(root_path: str, select_imgs: str) -> List[RSImage]:
    """加载所有影像 (包含完整的图像数据)。"""
    base_path = os.path.join(root_path, 'adjust_images')
    try:
        select_img_idxs = [int(i) for i in select_imgs.split(',')]
    except ValueError:
        print(f"Error: 'select_imgs' 格式错误. 应为 '0,1,2'。 得到: '{select_imgs}'")
        return []
        
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
    
    selected_folders = []
    for i in select_img_idxs:
        if i < len(img_folders):
            selected_folders.append(img_folders[i])
        else:
            print(f"Warning: 索引 {i} 超出影像文件夹数量 ({len(img_folders)})，已跳过。")
            
    if not selected_folders:
        print("Error: 未找到或未选择任何影像文件夹。")
        return []

    images = []
    print(f"Found {len(selected_folders)} selected image folders. Loading...")
    for idx, folder in enumerate(selected_folders):
        img_path = os.path.join(base_path, folder)
        try:
            # 注意：RSImage 的 __init__ 需要一个 'options' 对象，
            # 在这里我们传递一个 None，因为这个特定实现不使用它。
            images.append(RSImage(options=None, root=img_path, id=idx))
            print(f"Loaded image {idx} from {folder}.")
        except Exception as e:
            print(f"Failed to load image {idx} from {folder}: {e}")
            
    print(f"Successfully loaded {len(images)} images into memory.")
    return images

def find_overlapping_pairs(images: List[RSImage]) -> List[Tuple[int, int]]:
    """通过检查地理坐标BBox，找出所有重叠的影像对。"""
    bboxes = []
    for img in images:
        min_x = img.corner_xys[:, 0].min()
        max_x = img.corner_xys[:, 0].max()
        min_y = img.corner_xys[:, 1].min()
        max_y = img.corner_xys[:, 1].max()
        bboxes.append((min_x, min_y, max_x, max_y))

    pairs = []
    for i in range(len(images)):
        for j in range(i + 1, len(images)):
            b1 = bboxes[i]
            b2 = bboxes[j]
            
            is_disjoint = (b1[2] < b2[0] or  # b1.maxX < b2.minX
                           b1[0] > b2[2] or  # b1.minX > b2.maxX
                           b1[3] < b2[1] or  # b1.maxY < b2.minY
                           b1[1] > b2[3])   # b1.minY > b2.maxY
            
            if not is_disjoint:
                pairs.append((i, j))
                
    print(f"Found {len(pairs)} overlapping pairs for validation.")
    return pairs

# --- 精度检查函数 (来自 adjust_test_1023.py) ---

def haversine_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """计算两组 (Lat, Lon) 坐标之间的球面距离 (米)。"""
    R = 6371000  # 地球半径 (米)
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

def check_pair_error(img_i: RSImage, img_j: RSImage) -> np.ndarray:
    """ 计算单对影像 (i, j) 之间 *检查点* 的物方误差"""
    
    if img_i.tie_points is None or img_j.tie_points is None:
        return np.array([])
        
    if len(img_i.tie_points) != len(img_j.tie_points):
        print(f"Warning: Pair ({img_i.id}, {img_j.id}) 检查点数量不匹配。")
        return np.array([])
    
    if len(img_i.tie_points) == 0:
        return np.array([])

    # 确保 RPC 在 GPU 上 (因为 RSImage 初始化时会移动)
    img_i.rpc.to_gpu()
    img_j.rpc.to_gpu()

    with torch.no_grad():
        # 投影 img_i 的检查点
        lines_i = img_i.tie_points[:,0]
        samps_i = img_i.tie_points[:,1]
        heights_i = img_i.dem_interp(img_i.tie_points) # 使用 dem_interp
        lats_i, lons_i = img_i.rpc.RPC_PHOTO2OBJ(samps_i, lines_i, heights_i, 'numpy')
        coords_i = np.stack([lats_i, lons_i], axis=-1)
        
        # 投影 img_j 的检查点
        lines_j = img_j.tie_points[:,0]
        samps_j = img_j.tie_points[:,1]
        heights_j = img_j.dem_interp(img_j.tie_points) # 使用 dem_interp
        lats_j, lons_j = img_j.rpc.RPC_PHOTO2OBJ(samps_j, lines_j, heights_j, 'numpy')
        coords_j = np.stack([lats_j, lons_j], axis=-1)
    
    # 计算地理距离
    distances = haversine_distance(coords_i, coords_j)
    return distances

def check_all_pairs_error(images: List[RSImage], overlapping_pairs: List[Tuple[int, int]]) -> np.ndarray:
    """在所有重叠对上计算并汇总 *检查点* 误差"""
    all_distances = []
    print("--- Accuracy Report (using check points) ---")
    
    if not overlapping_pairs:
        print("No overlapping pairs found to check.")
        return np.array([0.0])

    for (i, j) in overlapping_pairs:
        distances = check_pair_error(images[i], images[j])
        if len(distances) > 0:
            all_distances.append(distances)
            print(f"Pair ({i}, {j}) | Points: {len(distances)} | Mean Error: {distances.mean():.4f} m | Median Error: {np.median(distances):.4f} m")
        else:
            print(f"Pair ({i}, {j}) | No valid check points found.")

    if not all_distances:
        print("No valid check points found for any overlapping pair. Cannot generate report.")
        return np.array([0.0])
        
    all_distances = np.concatenate(all_distances)
    
    if len(all_distances) > 0:
        print("\n--- Global Error Report (Summary) ---")
        print(f"Total check points: {len(all_distances)}")
        print(f"Mean Error:         {all_distances.mean():.4f} m")
        print(f"Median Error:       {np.median(all_distances):.4f} m")
        print(f"Max Error:          {all_distances.max():.4f} m")
        print(f"RMSE:               {np.sqrt(np.mean(all_distances**2)):.4f} m")
        print(f"< 1.0 m:            {((all_distances < 1.0).sum() / len(all_distances)) * 100:.2f} %")
        print(f"< 3.0 m:            {((all_distances < 3.0).sum() / len(all_distances)) * 100:.2f} %")
    
    return all_distances
