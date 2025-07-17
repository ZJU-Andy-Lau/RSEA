import enum
import numpy as np
import torch
import cv2
import rasterio
from rpc import RPCModelParameterTorch,load_rpc
import os
import argparse
from tqdm import tqdm
from shapely.geometry import Polygon, box
import matplotlib.pyplot as plt
from utils import mercator2lonlat,resample_from_quad

def find_squares_in_intersection(quads: np.ndarray, size: float) -> np.ndarray:
    """
    找出N个四边形的重合区域，并在其中填充不重叠的轴平行正方形。

    参数:
    quads (np.ndarray): 一个形状为 (N, 4, 2) 的Numpy数组，
                        记录了N个四边形的xy坐标。
    size (float):      希望在重叠区域内寻找的正方形的边长。

    返回:
    np.ndarray: 一个形状为 (M, 4, 2) 的Numpy数组，记录了找到的M个
                正方形的顶点坐标。
                顶点顺序: (x_min,y_max)->(x_max,y_max)->(x_max,y_min)->(x_min,y_min)
                如果找不到或没有重叠区域，则返回一个形状为 (0, 4, 2) 的空数组。
    """
    if quads.shape[0] == 0:
        return np.empty((0, 4, 2))

    # 1. 找出所有四边形的重合区域
    # 将第一个四边形转换为Shapely的Polygon对象作为初始交集
    try:
        intersection_area = Polygon(quads[0])
        # 与其余所有四边形逐个求交集
        for i in range(1, quads.shape[0]):
            intersection_area = intersection_area.intersection(Polygon(quads[i]))
    except Exception as e:
        print(f"创建多边形或计算交集时出错: {e}")
        return np.empty((0, 4, 2))


    # 如果没有重叠区域，则返回空数组
    if intersection_area.is_empty:
        return np.empty((0, 4, 2))

    # 2. 在重叠区域内找出尽可能多的不重叠正方形
    found_squares = []
    # 获取重叠区域的边界框 (minx, miny, maxx, maxy)
    min_x, min_y, max_x, max_y = intersection_area.bounds

    # 从边界框的左下角开始，以size为步长进行网格搜索
    y = min_y
    while y + size <= max_y:
        x = min_x
        while x + size <= max_x:
            # 创建一个候选正方形 (使用shapely.geometry.box更方便)
            # box(minx, miny, maxx, maxy)
            candidate_square = box(x, y, x + size, y + size)
            
            # 检查候选正方形是否完全包含在重叠区域内
            # 使用 contains 或 covers 都可以，covers 更严格一些，要求边界接触
            if intersection_area.contains(candidate_square):
                # 如果正方形有效，则记录其顶点
                sq_x_min, sq_y_min = x, y
                sq_x_max, sq_y_max = x + size, y + size
                
                # 按照要求的顺序排列顶点：
                # (x_min,y_max)->(x_max,y_max)->(x_max,y_min)->(x_min,y_min)
                square_vertices = [
                    [sq_x_min, sq_y_max],  # 左上
                    [sq_x_max, sq_y_max],  # 右上
                    [sq_x_max, sq_y_min],  # 右下
                    [sq_x_min, sq_y_min]   # 左下
                ]
                found_squares.append(square_vertices)
            
            x += size
        y += size

    # 3. 返回(M, 4, 2)的Numpy数组
    if not found_squares:
        return np.empty((0, 4, 2))
    
    return np.array(found_squares, dtype=np.float64)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str)
    parser.add_argument('--output_path',type=str)
    parser.add_argument('--dem_path',type=str,default='./datasets/beijing_raw/dtm_egm.tif')
    args = parser.parse_args()

    root = args.root
    output_path = args.output_path
    os.makedirs(output_path,exist_ok=True)
    names = [i.split('.rpc')[0] for i in os.listdir(root) if 'rpc' in i and 'PAN' in i]
    total_num = len(names)

    print("Loading Images")
    imgs = [cv2.imread(os.path.join(root,f'{name}.png'),cv2.IMREAD_GRAYSCALE) for name in tqdm(names)]

    print("Loading Heights")
    heights = [np.load(os.path.join(root,f'{name}_height.npy'),mmap_mode='r')[0] for name in tqdm(names)]

    print("Loading RPCs")
    rpcs = [load_rpc(os.path.join(root,f'{name}.rpc')) for name in tqdm(names)]

    dem_full = rasterio.open(args.dem_path)

    corners = []
    for img,height,rpc in zip(imgs,heights,rpcs):
        H,W = img.shape[:2]
        xs,ys = rpc.RPC_LINESAMP2XY([0,0,H-1,H-1],[0,W-1,W-1,0],[height[0,0],height[0,-1],height[-1,-1],height[-1,0]],'numpy')
        corners.append(np.stack([xs,ys],axis=-1))
    
    corners = np.stack(corners,axis=0)
    rects = find_squares_in_intersection(corners,2000)

    # print("corners:",corners)
    # print("rects:",rects)

    for rect_idx,rect in enumerate(rects):
        rect_latlons = mercator2lonlat(torch.from_numpy(rect[:,[1,0]])).numpy()
        rect_heights = [val[0] for val in dem_full.sample(rect_latlons[:,[1,0]])]
        rect_output_path = os.path.join(output_path,f'rect_{rect_idx}')
        os.makedirs(rect_output_path,exist_ok=True)
        for name,img,height,rpc in zip(names,imgs,heights,rpcs):
            rect_linesamps = np.stack(rpc.RPC_XY2LINESAMP(rect[:,0],rect[:,1],rect_heights,'numpy'),axis=-1)
            img_resample,local_resample = resample_from_quad(img,rect_linesamps,(3000,3000))
            height_resample,_ = resample_from_quad(height,rect_linesamps,(3000,3000))
            local_p2 = local_resample.reshape(-1,2)
            height_p = height_resample.reshape(-1)
            xy_p2 = np.stack(rpc.RPC_LINESAMP2XY(local_p2[:,0],local_p2[:,1],height_p),axis=-1)
            obj_p3 = np.concatenate([xy_p2,height_p[:,None]],axis=-1)
            obj_hw3 = obj_p3.reshape(3000,3000,3)
            cv2.imwrite(os.path.join(rect_output_path,f'{name}.png'),img_resample.astype(np.uint8))
            np.save(os.path.join(rect_output_path,f'{name}_obj.npy'),obj_hw3)



    # x_max,x_min,y_max,y_min = corners[:,:,0].max(),corners[:,:,0].min(),corners[:,:,1].max(),corners[:,:,1].min()
    # print("corners:",corners)
    # print("rects:",rects)

    # corners[:,:,0] = (corners[:,:,0] - x_min) / (x_max - x_min)
    # corners[:,:,1] = (corners[:,:,1] - y_min) / (y_max - y_min)
    # rects[:,:,0] = (rects[:,:,0] - x_min) / (x_max - x_min)
    # rects[:,:,1] = (rects[:,:,1] - y_min) / (y_max - y_min)

    # fig, ax = plt.subplots()
    # # 绘制原始四边形
    # for quad in corners:
    #     p = plt.Polygon(quad, fill=True, alpha=0.3, ec='black', lw=2)
    #     ax.add_patch(p)
    # # 绘制找到的正方形
    # if rects.shape[0] > 0:
    #     for square in rects:
    #         p = plt.Polygon(square, fill=True, color='red', alpha=0.7, ec='black', lw=1)
    #         ax.add_patch(p)
    # plt.show()
    