import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
import torch
from rpc import RPCModelParameterTorch
import os
import argparse
from utils import project_mercator,mercator2lonlat

from shapely.geometry import Polygon
from shapely.errors import GEOSException

def stretch_array_to_uint8(image_array: np.ndarray, 
                           lower_percent: int = 2, 
                           upper_percent: int = 98, 
                           nodata_value: int | float | None = None) -> np.ndarray:
    """
    将高位深（如uint16, float32）的NumPy数组通过百分比截断线性拉伸转换为uint8数组。

    这是将遥感影像数据从高动态范围压缩到8位以便于可视化的最常用和科学的方法之一。

    Args:
        image_array (np.ndarray): 输入的二维(H, W)或三维(H, W, C)NumPy数组。
                                  函数会自动处理多波段情况。
        lower_percent (int, optional): 用于截断的最低百分位数。默认为 2。
        upper_percent (int, optional): 用于截断的最高百分位数。默认为 98。
        nodata_value (int | float | None, optional): 数组中的无效值。
                                                     如果提供，这些值在计算时将被忽略，
                                                     并在输出数组中通常设为0。默认为 None。

    Returns:
        np.ndarray: 经过拉伸和类型转换后的 uint8 数组，形状与输入相同。
    """
    # 检查输入是否为NumPy数组
    if not isinstance(image_array, np.ndarray):
        raise TypeError("输入必须是一个NumPy数组。")
    
    # 创建一个浮点类型的副本用于计算，避免修改原始数组
    stretched_array = image_array.astype(np.float32)

    # 处理多波段影像的情况
    if image_array.ndim == 3:
        # 对每个波段独立进行拉伸
        for i in range(image_array.shape[2]):
            band = image_array[..., i]
            stretched_array[..., i] = stretch_array_to_uint8(band, lower_percent, upper_percent, nodata_value)
        return stretched_array.astype(np.uint8)

    # --- 以下为单波段处理逻辑 ---

    # 确定用于计算直方图的有效像素
    if nodata_value is not None:
        valid_pixels = image_array[image_array != nodata_value]
    else:
        valid_pixels = image_array.flatten()

    # 如果没有有效像素（例如，一个完全是NoData值的数组），返回一个全零数组
    if valid_pixels.size == 0:
        return np.zeros_like(image_array, dtype=np.uint8)

    # 计算用于拉伸的最小和最大值
    min_val, max_val = np.percentile(valid_pixels, (lower_percent, upper_percent))
    
    # 边缘情况处理：如果min和max相等（例如，图像是纯色），避免除以零
    if min_val == max_val:
        # 将所有有效像素设置为中等灰色127，或直接返回0
        stretched_array.fill(0) 
        if nodata_value is not None:
             stretched_array[image_array != nodata_value] = 127
        else:
             stretched_array.fill(127)
        return stretched_array.astype(np.uint8)

    # 将所有小于min_val的像素值设置为min_val，大于max_val的设置为max_val
    stretched_array = np.clip(stretched_array, min_val, max_val)

    # 应用线性拉伸公式
    stretched_array = (stretched_array - min_val) / (max_val - min_val) * 255.0

    # 将结果转换为uint8
    output_array = stretched_array.astype(np.uint8)

    # 如果原始数据有NoData值，将这些位置在最终的uint8图像中设置为0
    if nodata_value is not None:
        output_array[image_array == nodata_value] = 0

    return output_array

def read_tif(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read()
        pan_data = np.mean(data,axis=0).astype(src.profile('dtype'))
        return pan_data

def find_intersection(rect1: np.ndarray, rect2: np.ndarray) -> np.ndarray | None:
    """
    使用 Shapely 库计算两个非轴对齐矩形（或任意凸四边形）的重叠区域。

    Args:
        rect1 (np.ndarray): 第一个矩形的(4, 2) numpy数组。
        rect2 (np.ndarray): 第二个矩形的(4, 2) numpy数组。

    Returns:
        np.ndarray | None: 
            - 如果存在面重叠，则返回一个 NumPy 数组，包含了重叠多边形的顶点坐标。
            - 如果没有重叠或重叠区域为线/点，则返回 None。
    """
    try:
        # 1. 将Numpy数组转换为Shapely的Polygon对象
        poly1 = Polygon(rect1)
        poly2 = Polygon(rect2)
        
        # 2. 检查多边形是否有效（例如，没有自相交）并且是否存在交集
        if not poly1.is_valid or not poly2.is_valid:
            # 对于简单的矩形输入，通常是有效的
            print("警告: 输入的几何图形之一无效。")
            return None

        if not poly1.intersects(poly2):
            return None

        # 3. 计算交集
        intersection_geom = poly1.intersection(poly2)

        # 4. 处理结果
        # 我们只关心有面积的重叠部分（即结果为Polygon）
        if intersection_geom.is_empty or not isinstance(intersection_geom, Polygon):
            return None
        
        # 5. 提取坐标并返回
        # intersection_geom.exterior.coords 返回一个CoordinateSequence对象
        # 我们将其转换为Numpy数组
        return np.array(intersection_geom.exterior.coords)

    except GEOSException as e:
        print(f"处理几何图形时发生错误: {e}")
        return None

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str)
    parser.add_argument('--output_path',type=str)
    args = parser.parse_args()

    root = args.root
    output_path = args.output_path
    os.makedirs(output_path,exist_ok=True)
    names = [i.split('.')[0] for i in os.listdir(root) if 'rpc' in i and 'PAN' in i]
    img1 = read_tif(os.path.join(root,f'{names[0]}.tiff'))
    img2 = read_tif(os.path.join(root,f'{names[1]}.tiff'))
    dem1 = read_tif(os.path.join(root,f'{names[0]}_dem.tif'))
    dem2 = read_tif(os.path.join(root,f'{names[1]}_dem.tif'))
    dem_full = rasterio.open(os.path.join(root,'dem_egm.tif'),'r')
    rpc1 = RPCModelParameterTorch()
    rpc2 = RPCModelParameterTorch()
    rpc1.load_from_file(os.path.join(root,f'{names[0]}.rpc'))
    rpc2.load_from_file(os.path.join(root,f'{names[1]}.rpc'))
    H1,W1 = img1.shape[:2]
    H2,W2 = img2.shape[:2]

    corners_lats1,corners_lons1 = rpc1.RPC_PHOTO2OBJ([0,0,W1-1,W1-1],[0,H1-1,H1-1,0],[dem1[0,0],dem1[-1,0],dem1[-1,-1],dem1[0,-1]])
    corners_lats2,corners_lons2 = rpc2.RPC_PHOTO2OBJ([0,0,W2-1,W2-1],[0,H2-1,H2-1,0],[dem2[0,0],dem2[-1,0],dem2[-1,-1],dem2[0,-1]])

    corners_yx1 = project_mercator(torch.stack([corners_lats1,corners_lons1],dim=-1)).numpy()
    corners_yx2 = project_mercator(torch.stack([corners_lats2,corners_lons2],dim=-1)).numpy()

    intersection_yxs = find_intersection(corners_yx1,corners_yx2)
    intersection_latlons = mercator2lonlat(torch.from_numpy(intersection_yxs)).numpy()
    intersection_heights = [val[0] for val in dem_full.sample(intersection_latlons[:,[1,0]])]
    intersection_samps1,intersection_lines1 = rpc1.RPC_OBJ2PHOTO(intersection_latlons[:,0],intersection_latlons[:,1],intersection_heights,'numpy')
    intersection_samps2,intersection_lines2 = rpc2.RPC_OBJ2PHOTO(intersection_latlons[:,0],intersection_latlons[:,1],intersection_heights,'numpy')

    line_min1,line_max1,samp_min1,samp_max1 = max(0,intersection_lines1.min()), \
                                              min(H1-1,intersection_lines1.max()),\
                                              max(0,intersection_samps1.min()),\
                                              min(W1-1,intersection_samps1.max())
    
    line_min2,line_max2,samp_min2,samp_max2 = max(0,intersection_lines2.min()), \
                                              min(H2-1,intersection_lines2.max()),\
                                              max(0,intersection_samps2.min()),\
                                              min(W2-1,intersection_samps2.max())
    
    img1_output = stretch_array_to_uint8(img1[line_min1:line_max1,samp_min1:samp_max1])
    img2_output = stretch_array_to_uint8(img2[line_min2:line_max2,samp_min2:samp_max2])

    cv2.imwrite(os.path.join(output_path,'img1.png'),img1_output)
    cv2.imwrite(os.path.join(output_path,'img2.png'),img2_output)

    rpc1.LINE_OFF -= line_min1
    rpc1.SAMP_OFF -= samp_min1
    rpc2.LINE_OFF -= line_min2
    rpc2.SAMP_OFF -= samp_min2
    rpc1.save_rpc_to_file(os.path.join(output_path,'img1.rpc'))
    rpc2.save_rpc_to_file(os.path.join(output_path,'img2.rpc'))

    dem1_output = dem1[line_min1:line_max1,samp_min1:samp_max1]
    dem2_output = dem2[line_min2:line_max2,samp_min2:samp_max2]
    np.save(os.path.join(output_path,'img1_dem.npy'),dem1_output)
    np.save(os.path.join(output_path,'img2_dem.npy'),dem2_output)



    



    

        