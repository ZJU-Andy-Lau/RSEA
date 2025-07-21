import warnings
warnings.filterwarnings('ignore')
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

from tqdm import tqdm,trange

def stretch_array_to_uint8(image_array: np.ndarray, 
                           lower_percent: int = 2, 
                           upper_percent: int = 98, 
                           nodata_value = None) -> np.ndarray:
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
    min_val = min_val.astype(np.float32)
    max_val = max_val.astype(np.float32)
    
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
    stretched_array = np.clip(stretched_array, min_val, max_val,dtype=np.float32)
    print(stretched_array.dtype,min_val.dtype,max_val.dtype)

    # 应用线性拉伸公式
    stretched_array = (stretched_array - min_val) / (max_val - min_val) * 255

    # 将结果转换为uint8
    output_array = stretched_array.astype(np.uint8)

    # 如果原始数据有NoData值，将这些位置在最终的uint8图像中设置为0
    if nodata_value is not None:
        output_array[image_array == nodata_value] = 0

    return output_array

def read_tif(tif_path):
    with rasterio.open(tif_path) as src:
        data = src.read()
        pan_data = np.mean(data,axis=0)
        return pan_data

def get_data(src,line,samp):
    window = Window(samp,line,1,1)
    value = src.read(window = window)
    return value[0][0][0]

def find_intersection(quads_array: np.ndarray) -> np.ndarray:
    """
    计算N个凸四边形的公共重叠区域。

    Args:
        quads_array (np.ndarray): 一个形状为 (N, 4, 2) 的Numpy数组，
                                  记录了N个四边形的4个顶点坐标(x, y)。

    Returns:
        np.ndarray: 一个形状为 (M, 2) 的Numpy数组，记录了最终重叠区域
                    多边形的M个顶点坐标。如果没有重叠区域，则返回一个
                    形状为 (0, 2) 的空数组。
    """
    # --- 输入验证 ---
    if not isinstance(quads_array, np.ndarray) or quads_array.ndim != 3 or quads_array.shape[1:] != (4, 2):
        raise ValueError("输入必须是一个形状为 (N, 4, 2) 的Numpy数组。")

    num_quads = quads_array.shape[0]

    # 如果没有四边形，则返回空
    if num_quads == 0:
        return np.empty((0, 2), dtype=float)

    # --- 初始化 ---
    # 将第一个四边形作为初始的重叠区域
    try:
        # Shapely的Polygon会自动闭合，所以我们提供的顶点列表即可
        intersection_poly = Polygon(quads_array[0])
    except Exception as e:
        raise ValueError(f"无法从坐标创建第一个多边形: {e}")

    # 如果只有一个四边形，直接返回其顶点
    if num_quads == 1:
        # Shapely Polygon的exterior.coords返回的坐标会包含重复的闭合点，需要去掉
        return np.array(intersection_poly.exterior.coords)[:-1]

    # --- 迭代计算交集 ---
    for i in range(1, num_quads):
        # 如果当前交集已为空，则后续无需计算，最终交集也为空
        if intersection_poly.is_empty:
            return np.empty((0, 2), dtype=float)

        try:
            current_poly = Polygon(quads_array[i])
            # 计算与当前多边形的交集
            intersection_poly = intersection_poly.intersection(current_poly)
        except Exception as e:
            # 如果某个四边形坐标无效，可以跳过或抛出异常
            print(f"警告: 第 {i+1} 个四边形无效，已跳过。错误: {e}")
            continue

    # --- 结果处理 ---
    # 检查最终的交集类型并提取顶点
    if intersection_poly.is_empty:
        # 没有交集
        return np.empty((0, 2), dtype=float)
    
    # 交集可能是一个点、一条线或一个多边形
    geom_type = intersection_poly.geom_type
    
    if geom_type == 'Polygon':
        # 标准情况：交集是一个多边形
        # exterior.coords的最后一个点与第一个点相同，用于闭合路径，我们需要去掉它
        return np.array(intersection_poly.exterior.coords)[:-1]
    else:
        raise ValueError("没有重叠区域")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str)
    parser.add_argument('--output_path',type=str)
    args = parser.parse_args()

    root = args.root
    output_path = args.output_path
    os.makedirs(output_path,exist_ok=True)
    names = [i.split('.rpc')[0] for i in os.listdir(root) if 'rpc' in i and 'PAN' in i]


    imgs = []
    dems = []
    rpcs = []
    hws = []
    corner_yxs = []
    dem_full = rasterio.open(os.path.join(root,'dtm_egm.tif'),'r')

    for name in tqdm(names):
        img = rasterio.open(os.path.join(root,f'{name}.tiff'),'r')
        dem = rasterio.open(os.path.join(root,f'{name}_height.tif'),'r')
        H,W = img.height,img.width
        rpc = RPCModelParameterTorch()
        rpc.load_from_file(os.path.join(root,f'{name}.rpc'))
        corner_lats,corner_lons = rpc.RPC_PHOTO2OBJ([0,0,W-1,W-1],[0,H-1,H-1,0],[get_data(dem,0,0),get_data(dem,H-1,0),get_data(dem,H-1,W-1),get_data(dem,W-1,0)])
        corner_yxs.append(project_mercator(torch.stack([corner_lats,corner_lons],dim=-1)).numpy())
        imgs.append(img)
        dems.append(dem)
        rpcs.append(rpc)
        hws.append([H,W])
    corner_yxs = np.stack(corner_yxs,axis=0)

    print("load imgs done")

    intersection_yxs = find_intersection(corner_yxs)
    print(intersection_yxs)
    intersection_latlons = mercator2lonlat(torch.from_numpy(intersection_yxs)).numpy()
    intersection_heights = [val[0] for val in dem_full.sample(intersection_latlons[:,[1,0]])]
    
    for i in trange(len(names)):
        rpc = rpcs[i]
        intersection_samps,intersection_lines = rpc.RPC_OBJ2PHOTO(intersection_latlons[:,0],intersection_latlons[:,1],intersection_heights,'numpy')
        line_min,line_max,samp_min,samp_max = int(max(0,intersection_lines.min())), \
                                              int(min(hws[i][0]-1,intersection_lines.max())),\
                                              int(max(0,intersection_samps.min())),\
                                              int(min(hws[i][1]-1,intersection_samps.max()))
        window = Window(samp_min,line_min,samp_max - samp_min,line_max - line_min)
        img_output = imgs[i].read(window = window)[0]

        with rasterio.open(
            os.path.join(output_path,f'{names[i]}.tif'),
            'w',
            driver='GTiff',
            height=img_output.shape[0],
            width=img_output.shape[1],
            count=1,
            dtype=img_output.dtype,
        ) as dst:
            dst.write(img_output,1)

        # img_output = stretch_array_to_uint8(img_output)

        # cv2.imwrite(os.path.join(output_path,f'{names[i]}.png'),img_output)
        
        # dem_output = dems[i].read(window = window).astype(np.float32)
        # np.save(os.path.join(output_path,f'{names[i]}_height.npy'),dem_output)

        # rpc.LINE_OFF -= line_min
        # rpc.SAMP_OFF -= samp_min
        # rpc.save_rpc_to_file(os.path.join(output_path,f'{names[i]}.rpc'))
        



    



    

        