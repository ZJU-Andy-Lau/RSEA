import os
import cv2
from datetime import datetime
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from skimage.transform import AffineTransform
from skimage.measure import ransac
from PIL import Image
import numpy as np
from copy import deepcopy
import torch
import torch.nn as nn
import torch.nn.init as init
import random
import argparse
from rpc import RPCModelParameterTorch
def crop_rect_from_image(image, rect_points, size):
    """
    从图像中截取矩形区域。

    参数:
    - image: 使用cv2.imread()读取的图像。
    - rect_points: 矩形的四个顶点坐标，按顺时针或逆时针顺序排列。
                  例如: [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

    返回:
    - cropped_image: 截取出的矩形图像。
    """
    # 将四个顶点转换为numpy数组
    rect = np.array(rect_points, dtype="float32")

    # 计算矩形的边界框的宽度和高度
    width_a = np.linalg.norm(rect[0] - rect[1])
    width_b = np.linalg.norm(rect[2] - rect[3])
    max_width = int(max(width_a, width_b))

    height_a = np.linalg.norm(rect[0] - rect[3])
    height_b = np.linalg.norm(rect[1] - rect[2])
    max_height = int(max(height_a, height_b))

    # 目标矩形的四个角的坐标（仿射变换后的坐标）
    dst = np.array([[0, 0], [0, max_width-1], [max_height-1, max_width-1], [max_height-1, 0]], dtype="float32")

    rect_xy = np.array([[p[1],p[0]] for p in rect], dtype="float32")
    dst_xy = np.array([[p[1],p[0]]for p in dst], dtype="float32")

    # 计算仿射变换矩阵
    M = cv2.getPerspectiveTransform(rect_xy, dst_xy)
    M_inv = cv2.getPerspectiveTransform(dst, rect)

    # 使用仿射变换将图像中的矩形区域转换为目标矩形区域
    warped = cv2.warpPerspective(image.astype(np.float32), M, (max_width, max_height))

    if warped.shape[0] < size or warped.shape[1] < size:
        warped = cv2.resize(warped,(size,size))

    return warped.astype(np.float32),M_inv

def random_square_cut_and_affine(images, square_size, angle = None, margin = None):
    H, W = images[0].shape[:2]
    
    if angle is None:
        angle = np.random.uniform(-5,5)  # 随机旋转角度
    theta = np.deg2rad(angle)
    if margin is None:
        margin = int((np.abs(np.sin(theta)) + np.abs(np.cos(theta)))*square_size / 2) + 1
    center_x = np.random.uniform(margin + 1, W - margin - 1)
    center_y = np.random.uniform(margin + 1, H - margin - 1)
    # crop_upperleft = np.array([center_x - square_size / 2,center_y - square_size / 2],dtype=int)
    new_points = np.array([[-square_size / 2, -square_size / 2],
                        [ -square_size / 2, square_size / 2],
                        [ square_size / 2,  square_size / 2],
                        [square_size / 2,  -square_size / 2]])

    
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta),  np.cos(theta)]
    ])
    
    rotated_points = np.matmul(rotation_matrix,new_points.T).T + np.array([center_x, center_y])
    res = [crop_rect_from_image(image,rotated_points,square_size) for image in images]
    af_mat = res[0][1]

    return [i[0] for i in res],af_mat[:2,:],rotated_points


def estimate_affine_ransac(A, B, iterations=1000, threshold=0.1, whole=False, hp_num=3):
    max_inliers_num = 0
    best_affine_matrix = None

    def estimate_affine_transformation(A, B):
        # 中心化点集
        A_centered = A - np.mean(A, axis=0)
        B_centered = B - np.mean(B, axis=0)

        # 计算协方差矩阵
        H = A_centered.T @ B_centered

        # 进行奇异值分解
        U, S, Vt = np.linalg.svd(H)

        # 计算旋转矩阵
        R = Vt.T @ U.T

        # 确保旋转矩阵的行列式为1
        if np.linalg.det(R) < 0:
            Vt[1, :] *= -1
            R = Vt.T @ U.T

        # 计算平移
        t = np.mean(B, axis=0) - R @ np.mean(A, axis=0)

        # 组合成仿射变换矩阵
        affine_matrix = np.eye(3)
        affine_matrix[:2, :2] = R
        affine_matrix[:2, 2] = t

        return affine_matrix

    for _ in range(iterations):
        # 随机选择三个点
        sample_indices = random.sample(range(len(A)), hp_num)
        A_sample = A[sample_indices]
        B_sample = B[sample_indices]

        # 计算仿射变换矩阵
        affine_matrix = estimate_affine_transformation(A_sample, B_sample)

        # 计算在当前变换下的预测值
        B_pred = (affine_matrix[:2, :2] @ A.T).T + affine_matrix[:2, 2]

        # 计算内点
        distances = np.linalg.norm(B - B_pred, axis=1)
        inliers = distances < threshold

        # 更新最佳内点集
        if np.sum(inliers) > max_inliers_num:
            max_inliers_num = np.sum(inliers)
            if whole:
                whole_matrix = estimate_affine_transformation(A[inliers],B[inliers])
                B_pred = (whole_matrix[:2, :2] @ A.T).T + whole_matrix[:2, 2]
                # 计算内点
                distances = np.linalg.norm(B - B_pred, axis=1)
                inliers = distances < threshold
                if np.sum(inliers) > max_inliers_num:
                    max_inliers_num = np.sum(inliers)
                    best_affine_matrix = whole_matrix
                else:
                    best_affine_matrix = affine_matrix
            else:
                best_affine_matrix = affine_matrix

    return best_affine_matrix[:2,:], max_inliers_num

# def estimate_affine_ransac(points1, points2, threshold = 20):
#     model_robust, inliers = ransac((points1, points2), 
#                                 AffineTransform, 
#                                 min_samples=3, 
#                                 residual_threshold=threshold, 
#                                 max_trials=1000)
    
#     return model_robust.params[:2,:], len(inliers[inliers]) 

def calculate_errors(T_true, T_pred):
    # 提取旋转部分 (前2x2矩阵) 和 平移部分 (最后一列)
    R_true = T_true[:, :2]
    R_pred = T_pred[:, :2]
    
    t_true = T_true[:, 2]
    t_pred = T_pred[:, 2]
    
    # 计算平移误差（欧氏距离）
    translation_error = np.linalg.norm(t_pred - t_true)
    
    # 计算旋转角度
    theta_true = np.arctan2(R_true[1, 0], R_true[0, 0])
    theta_pred = np.arctan2(R_pred[1, 0], R_pred[0, 0])
    
    # 计算旋转误差（角度差异）
    rotation_error = np.degrees(np.abs(theta_pred - theta_true))
    
    return rotation_error, translation_error

class TableLogger():
    def __init__(self,folder_path:str,columns:list,prefix:str = 'log'):
        self.df = pd.DataFrame(columns=columns)
        self.folder = Path(folder_path)
        self.file_name = f"{prefix}_{datetime.now().strftime('%Y%m%d%H%M%S')}.csv"
        self.df.to_csv(self.folder / self.file_name,index=False)
    def update(self,row):
        try:
            self.df = self.df._append(row,ignore_index=True)
            self.df.to_csv(self.folder / self.file_name,index=False)
        except:
            self.df = self.df.append(row,ignore_index=True)
            self.df.to_csv(self.folder / self.file_name,index=False)

def average_downsample_matrix(matrix:np.ndarray, n):
    """
    对给定的numpy矩阵进行n倍平均下采样
    :param matrix: 输入的numpy矩阵
    :param n: 下采样倍数
    :return: 下采样后的矩阵
    """
    rows, cols = matrix.shape[:2]
    new_rows = rows // n
    new_cols = cols // n
    downsampled_matrix = np.zeros((new_rows, new_cols,*matrix.shape[2:]))
    for r in range(new_rows):
        for c in range(new_cols):
            downsampled_matrix[r, c] = matrix[r * n:(r + 1) * n, c * n:(c + 1) * n].mean(axis=(0,1))

    return downsampled_matrix

def get_coord_mat(H,W,downsample:int = 0):
    """
    return: [row,col] (H,W,2)
    """
    row_coords, col_coords = np.meshgrid(np.arange(0,H), np.arange(0,W), indexing='ij')
    coord_array = np.stack([row_coords, col_coords], axis=-1).astype(np.float32)  # (H,W, 2)
    if downsample > 0:
        coord_array = average_downsample_matrix(coord_array,downsample)
    return coord_array

def kaiming_init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        if m.bias is not None:
            init.zeros_(m.bias)

def norm_init_weights(m):
    if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
        init.normal_(m.weight,0,0.001)
        if m.bias is not None:
            init.normal_(m.bias,0,0.001)

def str2bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif value.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected")

def warp_by_extend(points:torch.Tensor,extend:torch.Tensor):
    """
    points [x,y,h]
    return [y,x,h]
    """
    points = points.to(torch.double)
    extend = extend.to(torch.double).to(points.device)
    points[:,0] *= extend[6]
    points[:,1] *= extend[7] 
    points[:,2] = (points[:,2] + 1.) * (extend[9] - extend[8]) * 0.5 + extend[8]
    af_mat = extend[:6].reshape(2,3)
    R = af_mat[:2,:2]
    t = af_mat[:2,2]
    points[:,:2] = points[:,:2] @ R.T
    points[:,:2] = points[:,:2] + t
    points[:,[0,1]] = points[:,[1,0]]
    return points

def project_mercator(latlon:torch.Tensor):
    """
    (lat,lon) -> (y,x) N,2
    """
    r = 6378137.
    lon_rad = latlon[:,1] * torch.pi / 180.
    lat_rad = latlon[:,0] * torch.pi / 180.
    x = r * lon_rad
    y = r * torch.log(torch.tan(torch.pi / 4. + lat_rad / 2.))
    return torch.stack([y,x],dim=-1)

def mercator2lonlat(coord:torch.Tensor):
    """
    (y,x) -> (lat,lon) N,2
    """
    coord = torch.tensor(coord).to(torch.float64)
    r = 6378137.
    lon = (180. * coord[:,1]) / (torch.pi * r)
    lat = (2 * torch.atan(torch.exp(coord[:,0] / r)) - torch.pi * 0.5) * 180. / torch.pi
    return torch.stack([lat,lon],dim=-1)

def proj2photo(proj_coord:torch.Tensor,dem:torch.Tensor,rpc:RPCModelParameterTorch):
    lonlat_coord = mercator2lonlat(proj_coord)
    photo_coord = torch.stack(rpc.RPC_OBJ2PHOTO(lonlat_coord[:,0],lonlat_coord[:,1],dem),dim=1)
    return photo_coord

def bilinear_interpolate(array, points):
    """
    输入：
    - array: 二维 (H, W) 或三维 (H, W, C) 的数组
    - points: (N, 2) 的浮点坐标数组，每行表示一个坐标 [x, y]
    输出：
    - 插值结果，形状为 (N,) 或 (N, C)
    """
    array = np.asarray(array)
    points = np.asarray(points)
    
    # 将二维数组扩展为 (H, W, 1) 以统一处理
    if array.ndim == 2:
        array = array[..., np.newaxis]
    
    H, W, C = array.shape
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    
    # 计算整数坐标并约束边界
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, W-1)
    x0 = np.clip(x0, 0, W-1)
    
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, H-1)
    y0 = np.clip(y0, 0, H-1)
    
    # 提取四个角点的值，形状 (N, C)
    Ia = array[y0, x0, :]
    Ib = array[y1, x0, :]
    Ic = array[y0, x1, :]
    Id = array[y1, x1, :]
    
    # 计算权重
    dx = x - x0
    dy = y - y0
    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy
    
    # 加权求和（广播到所有通道）
    result = (
        wa[:, None] * Ia +
        wb[:, None] * Ib +
        wc[:, None] * Ic +
        wd[:, None] * Id
    )
    
    # 压缩多余的维度（若原始输入是二维）
    if array.shape[-1] == 1 and array.ndim == 3:
        result = result.squeeze(axis=1)
    
    return result

def downsample(arr:torch.Tensor,ds):
    if ds <= 0:
        return arr
    if len(arr.shape) < 4:
        arr = arr.unsqueeze(-1)
    arr_ds = []
    for a in arr:
        H,W = a.shape[:2]
        lines = np.arange(0,H - ds + 1,ds) + (ds - 1.) * 0.5
        samps = np.arange(0,W - ds + 1,ds) + (ds - 1.) * 0.5
        sample_idxs = np.stack(np.meshgrid(samps,lines,indexing='xy'),axis=-1).reshape(-1,2) # x,y
        a = torch.tensor(bilinear_interpolate(a,sample_idxs))
        arr_ds.append(a.reshape(len(lines),len(samps),-1).squeeze())
    arr_ds = torch.stack(arr_ds,dim=0)
    return arr_ds