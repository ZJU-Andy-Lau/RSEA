import os
import argparse
import numpy as np
import torch
import warnings
from typing import List, Tuple

# 假设 rpc.py 在同一目录或 Python 路径中
from rpc import RPCModelParameterTorch

warnings.filterwarnings("ignore")

class ErrorImage:
    """
    一个轻量级的影像类，专用于误差评估。
    它只加载RPC和连接点，DEM数据按需从磁盘读取。
    """
    def __init__(self, root: str, id: int):
        self.root = root
        self.id = id
        self.dem_path = os.path.join(root, 'dem.npy')
        self.tie_points_path = os.path.join(root, 'tie_points.txt')
        self.rpc_path = os.path.join(root, 'rpc.txt')

        # 1. 加载RPC (必需)
        if not os.path.exists(self.rpc_path):
            raise FileNotFoundError(f"错误 (Image {self.id}): 未找到 rpc.txt at {self.rpc_path}")
        self.rpc = RPCModelParameterTorch()
        self.rpc.load_from_file(self.rpc_path)
        self.rpc.to_gpu() # 假设评估在GPU上运行以加快RPC计算

        # 2. 加载连接点 (必需)
        self.tie_points = self._load_tie_points()
        
        # 3. 计算角点地理坐标 (用于查找重叠对)
        # 这需要 DEM 的尺寸和 4 个角点的高程
        self.corner_xys = self._get_corner_xys()

    def _load_tie_points(self) -> np.ndarray:
        """加载 tie_points.txt 文件"""
        path = self.tie_points_path
        if not os.path.exists(path):
            print(f"信息 (Image {self.id}): 未找到 tie_points.txt。")
            return None
        
        try:
            tie_points = np.loadtxt(path, dtype=int)
            if tie_points.ndim == 0: # 空文件
                return None
            if tie_points.ndim == 1:
                tie_points = tie_points.reshape(1, -1)
            if tie_points.shape[1] != 2:
                print(f"警告 (Image {self.id}): tie_points 格式错误。")
                return None
            return tie_points
        except Exception as e:
            print(f"警告 (Image {self.id}): 加载 tie_points 失败: {e}")
            return None

    def _get_dem_values_at_coords(self, lines: np.ndarray, samps: np.ndarray) -> np.ndarray:
        """
        [核心功能] 使用内存映射 (mmap) 按需从磁盘加载DEM值。
        """
        if not os.path.exists(self.dem_path):
            print(f"警告 (Image {self.id}): 未找到 dem.npy at {self.dem_path}。将使用RPC平均高程。")
            return np.full(lines.shape, self.rpc.HEIGHT_OFF.item())
            
        try:
            # 'r' 模式 = 只读。这不会将文件加载到RAM。
            dem_mmap = np.load(self.dem_path, mmap_mode='r')
            
            # 检查坐标是否越界
            H, W = dem_mmap.shape
            if np.any(lines < 0) or np.any(lines >= H) or np.any(samps < 0) or np.any(samps >= W):
                 print(f"警告 (Image {self.id}): 连接点坐标越界。DEM 尺寸: ({H}, {W})")
                 # 裁剪坐标以防止错误
                 lines = np.clip(lines, 0, H - 1)
                 samps = np.clip(samps, 0, W - 1)

            # 仅从磁盘读取这几个点的值
            heights = dem_mmap[lines, samps]
            # 必须将结果复制为新数组，否则 mmap 引用会保持打开
            return np.array(heights)
        except Exception as e:
            print(f"警告 (Image {self.id}): 无法从 {self.dem_path} 读取DEM值: {e}")
            # 回退：使用RPC的平均高程
            return np.full(lines.shape, self.rpc.HEIGHT_OFF.item())

    def _get_corner_xys(self) -> np.ndarray:
        """计算4个角的地理坐标 (用于 find_overlapping_pairs)"""
        try:
            # 快速读取DEM的形状，而不加载全部内容
            dem_shape = np.load(self.dem_path, mmap_mode='r').shape
            H, W = dem_shape
        except Exception as e:
            print(f"致命错误 (Image {self.id}): 无法读取 {self.dem_path} 的尺寸: {e}")
            raise e # 允许 load_imgs_bundle 捕获此异常

        # 定义4个角点的 (line, samp) 坐标
        corner_lines = np.array([0, 0, H - 1, H - 1], dtype=int)
        corner_samps = np.array([0, W - 1, 0, W - 1], dtype=int)
        
        # [关键] 按需加载这4个角点的DEM值
        corner_heights = self._get_dem_values_at_coords(corner_lines, corner_samps)

        # 使用RPC计算地理坐标 (转为 tensor 以使用 RPC 类)
        latlons = torch.stack(self.rpc.RPC_PHOTO2OBJ(
            torch.from_numpy(corner_samps),
            torch.from_numpy(corner_lines),
            torch.from_numpy(corner_heights)
        ), dim=-1)
        
        # 使用RPC内置的方法 (lat,lon) -> (y,x)，然后翻转为 (x,y)
        # (这取代了对 utils.py 中 project_mercator 的需求)
        yx = self.rpc.latlon2yx(latlons) # (N, 2) tensor [y, x]
        xy = yx[:, [1, 0]] # [x, y]
        return xy.cpu().numpy()

    def get_heights_for_tie_points(self) -> np.ndarray:
        """
        公开接口：获取所有 tie_points 对应的高程值。
        """
        if self.tie_points is None:
            return np.array([])
        
        lines = self.tie_points[:, 0]
        samps = self.tie_points[:, 1]
        return self._get_dem_values_at_coords(lines, samps)


def load_imgs_bundle(args) -> List[ErrorImage]:
    """
    (已修改) 加载所有影像 (使用轻量级的 ErrorImage 类)。
    这是一个单进程版本，用于评估脚本。
    """
    base_path = args.root # [修改] 直接使用 root 路径
    
    img_folders = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])

    # [修改] 如果提供了 select_imgs，则按索引过滤
    if args.select_imgs:
        try:
            select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
            img_folders = [img_folders[i] for i in select_img_idxs]
        except Exception as e:
            print(f"警告: 解析 --select_imgs 失败 ({e})。将加载所有影像。")
    
    images = []
    print(f"发现 {len(img_folders)} 个影像文件夹。正在加载...")
    for idx, folder in enumerate(img_folders):
        img_path = os.path.join(base_path, folder)
        try:
            # [修改] 使用新的 ErrorImage 类
            images.append(ErrorImage(img_path, idx))
            print(f"已加载影像 {idx} (来自 {folder}).")
        except Exception as e:
            print(f"加载影像 {idx} (来自 {folder}) 失败: {e}")
            
    print(f"成功加载 {len(images)} 张影像到内存。")
    return images

def find_overlapping_pairs(images: List[ErrorImage]) -> List[Tuple[int, int]]:
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
            
            # 检查是否不相交
            is_disjoint = (b1[2] < b2[0] or  # b1.maxX < b2.minX
                           b1[0] > b2[2] or  # b1.minX > b2.maxX
                           b1[3] < b2[1] or  # b1.maxY < b2.minY
                           b1[1] > b2[3])   # b1.minY > b2.maxY
            
            if not is_disjoint:
                pairs.append((i, j))
                
    print(f"找到 {len(pairs)} 个重叠像对。")
    return pairs

def haversine_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """计算两组 (lat, lon) 坐标之间的 Haversine 距离 (米)"""
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

def check_pair_error(img_i: ErrorImage, img_j: ErrorImage) -> np.ndarray:
    """ 计算单对影像 (i, j) 之间的连接点误差"""
    
    if img_i.tie_points is None or img_j.tie_points is None:
        # print(f"Skipping error check for pair ({img_i.id}, {img_j.id}): Missing tie points.")
        return np.array([])
        
    if len(img_i.tie_points) != len(img_j.tie_points):
        print(f"警告: 像对 ({img_i.id}, {img_j.id}) 连接点数量不匹配 ({len(img_i.tie_points)} vs {len(img_j.tie_points)})。跳过...")
        return np.array([])
    
    if len(img_i.tie_points) == 0:
        return np.array([])

    # 投影 img_i 的连接点
    lines_i = img_i.tie_points[:,0]
    samps_i = img_i.tie_points[:,1]
    # [修改] 按需加载高程
    heights_i = img_i.get_heights_for_tie_points()
    if heights_i.size == 0: return np.array([]) # 检查是否加载失败

    # 使用 RPC (可能已被调整) 将 (samp, line, h) -> (lat, lon)
    lats_i, lons_i = img_i.rpc.RPC_PHOTO2OBJ(samps_i, lines_i, heights_i, 'numpy')
    coords_i = np.stack([lats_i, lons_i], axis=-1)
    
    # 投影 img_j 的连接点
    lines_j = img_j.tie_points[:,0]
    samps_j = img_j.tie_points[:,1]
    # [修改] 按需加载高程
    heights_j = img_j.get_heights_for_tie_points()
    if heights_j.size == 0: return np.array([]) # 检查是否加载失败
    
    # 使用 RPC (可能已被调整) 将 (samp, line, h) -> (lat, lon)
    lats_j, lons_j = img_j.rpc.RPC_PHOTO2OBJ(samps_j, lines_j, heights_j, 'numpy')
    coords_j = np.stack([lats_j, lons_j], axis=-1)
    
    # 计算地理距离
    distances = haversine_distance(coords_i, coords_j)
    return distances

def check_all_pairs_error(images: List[ErrorImage], overlapping_pairs: List[Tuple[int, int]]) -> np.ndarray:
    """在所有重叠对上计算并汇总误差"""
    all_distances = []
    print("\n" + "="*50)
    print("--- 逐对误差详细报告 ---")
    print("="*50)
    for (i, j) in overlapping_pairs:
        distances = check_pair_error(images[i], images[j])
        if len(distances) > 0:
            all_distances.append(distances)
            print(f"像对 ({i}, {j}) | 连接点数: {len(distances):>5} | 平均误差: {distances.mean():.4f} m | 中位误差: {np.median(distances):.4f} m")
        else:
            print(f"像对 ({i}, {j}) | 无有效连接点可供计算。")

    if not all_distances:
        print("\n未找到任何有效的连接点。无法生成报告。")
        return np.array([0.0])
        
    all_distances = np.concatenate(all_distances)
    return all_distances


if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description="遥感影像连接点误差评估报告")

    parser.add_argument('--root', type=str, required=True,
                        help="包含所有影像子文件夹的根目录 (例如 '.../adjust_images')")

    parser.add_argument('--select_imgs', type=str, default=None,
                        help="(可选) 指定要评估的影像索引，以逗号分隔 (例如 '0,1,3')。默认为评估所有影像。")

    args = parser.parse_args()

    # 1. 加载影像
    print(f"正在从 {args.root} 加载影像...")
    images = load_imgs_bundle(args)
    
    if len(images) < 2:
        print("错误: 至少需要两张影像才能进行像对评估。")
        exit()

    # 2. 查找重叠对
    overlapping_pairs = find_overlapping_pairs(images)
    
    if not overlapping_pairs:
        print("未找到任何重叠的影像对。")
        exit()

    # 3. 计算所有误差 (这将打印详细报告)
    all_errors = check_all_pairs_error(images, overlapping_pairs)
    
    # 4. 打印全局总结报告
    if len(all_errors) > 0 and all_errors.mean() != 0.0:
        print("\n" + "="*50)
        print("--- 全局误差总结报告 ---")
        print("="*50)
        print(f"总连接点数:   {len(all_errors)}")
        print(f"平均误差 (Mean): {all_errors.mean():.4f} m")
        print(f"中位误差 (Median): {np.median(all_errors):.4f} m")
        print(f"最大误差 (Max):   {all_errors.max():.4f} m")
        print(f"均方根误差 (RMSE): {np.sqrt(np.mean(all_errors**2)):.4f} m")
        print("-" * 50)
        print(f"误差 < 1.0 m 占比: {((all_errors < 1.0).sum() / len(all_errors)) * 100:.2f} %")
        print(f"误差 < 3.0 m 占比: {((all_errors < 3.0).sum() / len(all_errors)) * 100:.2f} %")
        print(f"误差 < 5.0 m 占比: {((all_errors < 5.0).sum() / len(all_errors)) * 100:.2f} %")
        print("="*50)
    else:
        print("\n计算完成，但未找到有效的误差数据点用于全局总结。")

