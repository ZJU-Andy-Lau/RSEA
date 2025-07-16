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
from typing import Tuple
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

def bilinear_interpolate(array, points, use_cuda=False):
    """
    在矩阵上进行双线性插值采样，可选择在 CPU (NumPy) 或 GPU (PyTorch) 上运行。

    输入:
    - array: 二维 (H, W) 或三维 (H, W, C) 的 numpy 数组或 torch 张量。
    - points: (N, 2) 的浮点坐标数组或张量，每行表示一个坐标 [x, y]。
    - use_cuda:布尔值。如果为 True，则尝试使用 GPU (CUDA) 加速。

    输出:
    - 插值结果，形状为 (N,) 或 (N, C) 的 numpy 数组。
    """
    
    # ----------- GPU (CUDA) 加速路径 -----------
    if use_cuda:
        # 检查 CUDA 是否可用，如果不可用则警告并回退到 CPU
        if not torch.cuda.is_available():
            print("警告：CUDA 不可用。将回退到 CPU (NumPy) 执行。")
            use_cuda = False
        else:
            device = torch.device('cuda')
            
            # 确保输入是 PyTorch 张量并移至 GPU
            # 使用 torch.as_tensor 避免不必要的数据拷贝
            arr_tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
            pts_tensor = torch.as_tensor(points, dtype=torch.float32, device=device)
            
            # 将二维数组扩展为 (H, W, 1) 以统一处理
            if arr_tensor.dim() == 2:
                arr_tensor = arr_tensor.unsqueeze(-1)
            
            H, W, C = arr_tensor.shape
            x = pts_tensor[:, 0]
            y = pts_tensor[:, 1]
            
            # 计算整数坐标并约束边界
            # torch.floor 的结果是浮点数，需要转为长整型用于索引
            x0 = torch.floor(x).long()
            y0 = torch.floor(y).long()
            
            # 使用 torch.clamp 约束边界，等同于 np.clip
            x1 = torch.clamp(x0 + 1, 0, W - 1)
            x0 = torch.clamp(x0, 0, W - 1)
            y1 = torch.clamp(y0 + 1, 0, H - 1)
            y0 = torch.clamp(y0, 0, H - 1)
            
            # 提取四个角点的值，形状 (N, C)
            # PyTorch 的高级索引方式与 NumPy 相同
            Ia = arr_tensor[y0, x0, :]
            Ib = arr_tensor[y1, x0, :]
            Ic = arr_tensor[y0, x1, :]
            Id = arr_tensor[y1, x1, :]
            
            # 计算权重 (dx, dy 仍然是浮点数)
            dx = x - x0.float()
            dy = y - y0.float()
            
            wa = (1 - dx) * (1 - dy)
            wb = (1 - dx) * dy
            wc = dx * (1 - dy)
            wd = dx * dy
            
            # 加权求和 (使用 unsqueeze(1) 广播到所有通道)
            # wa[:, None] 在 PyTorch 中是 wa.unsqueeze(1)
            result_tensor = (
                wa.unsqueeze(1) * Ia +
                wb.unsqueeze(1) * Ib +
                wc.unsqueeze(1) * Ic +
                wd.unsqueeze(1) * Id
            )
            
            # 压缩多余的维度
            if arr_tensor.shape[-1] == 1 and arr_tensor.dim() == 3:
                result_tensor = result_tensor.squeeze(axis=1)
                
            # 将结果从 GPU 移回 CPU 并转换为 NumPy 数组
            return result_tensor.cpu().numpy()

    # ----------- CPU (NumPy) 原始路径 -----------
    # 如果 use_cuda 为 False，则执行原始逻辑
    array = np.asarray(array)
    points = np.asarray(points)
    
    # 记录原始维度以决定最终输出形状
    original_ndim = array.ndim
    
    if array.ndim == 2:
        array = array[..., np.newaxis]
    
    H, W, C = array.shape
    x = points[:, 0].astype(float)
    y = points[:, 1].astype(float)
    
    x0 = np.floor(x).astype(int)
    x1 = np.clip(x0 + 1, 0, W - 1)
    x0 = np.clip(x0, 0, W - 1)
    
    y0 = np.floor(y).astype(int)
    y1 = np.clip(y0 + 1, 0, H - 1)
    y0 = np.clip(y0, 0, H - 1)
    
    Ia = array[y0, x0, :]
    Ib = array[y1, x0, :]
    Ic = array[y0, x1, :]
    Id = array[y1, x1, :]
    
    dx = x - x0
    dy = y - y0
    wa = (1 - dx) * (1 - dy)
    wb = (1 - dx) * dy
    wc = dx * (1 - dy)
    wd = dx * dy
    
    result = (
        wa[:, None] * Ia +
        wb[:, None] * Ib +
        wc[:, None] * Ic +
        wd[:, None] * Id
    )
    
    if original_ndim == 2:
        result = result.squeeze(axis=1)
    
    return result

def downsample_average(
    input_tensor: torch.Tensor,
    downsample_factor: int,
    use_cuda: bool = False
) -> torch.Tensor:
    """
    使用滑动窗口平均值对图像张量进行下采样。

    该函数接受一个形状为 (H, W) 或 (H, W, C) 的 PyTorch 图像张量，
    并使用一个 (downsample_factor x downsample_factor) 的窗口
    以 downsample_factor 为步长进行不重叠的滑动窗口下采样，
    并取窗口内的像素平均值。

    Args:
        input_tensor (torch.Tensor): 输入的图像张量，形状可以是 (H, W) [灰度图]
                                     或 (H, W, C) [彩色图]。
        downsample_factor (int): 下采样因子，将作为窗口大小和步长。例如，8 表示
                                 使用 8x8 的窗口下采样8倍。
        use_cuda (bool, optional): 如果为 True，则尝试使用 CUDA GPU 进行加速。
                                   如果 CUDA 不可用，将打印警告并回退到 CPU。
                                   默认为 False。

    Returns:
        torch.Tensor: 经过下采样后的图像张量。
                      如果输入是 (H, W)，输出是 (H/factor, W/factor)。
                      如果输入是 (H, W, C)，输出是 (H/factor, W/factor, C)。
                      输出张量将位于计算所用的设备上（CPU 或 CUDA）。

    Raises:
        ValueError: 如果输入张量的维度不是 2 或 3。
        TypeError: 如果输入不是一个 torch.Tensor。
    """
    if not isinstance(input_tensor, torch.Tensor):
        raise TypeError(f"输入必须是 torch.Tensor，但得到的是 {type(input_tensor)}")

    # --- 1. 检查和设置计算设备 (CPU or CUDA) ---
    device = torch.device("cpu")
    if use_cuda:
        if torch.cuda.is_available():
            device = torch.device("cuda")
            # print("CUDA is available. Using GPU for acceleration.")
        else:
            print("警告: 请求使用 CUDA，但 CUDA 不可用。将回退到 CPU。")

    # 将输入张量移动到目标设备
    input_tensor = input_tensor.to(device)

    # --- 2. 预处理：将输入张量调整为 PyTorch 卷积层期望的格式 (N, C, H, W) ---
    # PyTorch 的 2D 卷积/池化层需要一个4D张量作为输入：(批量大小, 通道数, 高, 宽)
    input_dim = input_tensor.dim()
    if input_dim == 2:  # 灰度图 (H, W)
        is_grayscale = True
        # 扩展为 (1, 1, H, W)
        tensor_in = input_tensor.unsqueeze(0).unsqueeze(0)
    elif input_dim == 3:  # 彩色图 (H, W, C)
        is_grayscale = False
        # PyTorch 使用 "channels-first" (C, H, W) 格式，所以需要转换维度
        # (H, W, C) -> (C, H, W)，然后扩展为 (1, C, H, W)
        tensor_in = input_tensor.permute(2, 0, 1).unsqueeze(0)
    else:
        raise ValueError(f"输入张量的维度必须是 2 (H,W) 或 3 (H,W,C)，但得到的是 {input_dim}")

    # 确保输入张量是浮点数类型，以便计算平均值
    tensor_in = tensor_in.float()

    # --- 3. 定义并执行平均池化操作 ---
    # 使用 AvgPool2d 可以高效地完成滑动窗口平均操作
    # kernel_size 是窗口大小
    # stride 是滑动步长
    # 当 kernel_size 和 stride 相同时，窗口不会重叠
    pool = nn.AvgPool2d(kernel_size=downsample_factor, stride=downsample_factor).to(device)
    downsampled_tensor = pool(tensor_in)

    # --- 4. 后处理：将输出张量恢复为原始格式 ---
    if is_grayscale:
        # (1, 1, H', W') -> (H', W')
        output_tensor = downsampled_tensor.squeeze(0).squeeze(0)
    else:
        # (1, C, H', W') -> (C, H', W') -> (H', W', C)
        output_tensor = downsampled_tensor.squeeze(0).permute(1, 2, 0)

    return output_tensor.cpu()

def downsample(arr:torch.Tensor,ds,use_cuda=False,show_detail=False,mode='mid'):
    """
    mode: mid or avg
    """
    if ds <= 0:
        return arr
    # if len(arr.shape) < 4:
    #     arr = arr.unsqueeze(-1)
    arr_ds = []
    if show_detail:
        pbar = tqdm(total = len(arr))
    for a in arr:
        if mode == 'mid':
            if len(a.shape) < 3:
                a = a.unsqueeze(-1)
            H,W = a.shape[:2]
            lines = np.arange(0,H - ds + 1,ds) + (ds - 1.) * 0.5
            samps = np.arange(0,W - ds + 1,ds) + (ds - 1.) * 0.5
            sample_idxs = np.stack(np.meshgrid(samps,lines,indexing='xy'),axis=-1).reshape(-1,2) # x,y
            a = torch.tensor(bilinear_interpolate(a,sample_idxs,use_cuda=use_cuda))
            arr_ds.append(a.reshape(len(lines),len(samps),-1).squeeze())
        elif mode == 'avg':
            a_ds = downsample_average(a,ds,use_cuda)
            arr_ds.append(a_ds)
        else:
            raise ValueError("downsample mode should either be mid or avg")
        if show_detail:
            pbar.update(1)
    arr_ds = torch.stack(arr_ds,dim=0)
    return arr_ds

def print_hwc_matrix(matrix: np.ndarray, precision:int = 2):
    """
    将一个形状为 (H, W, C) 的 NumPy 数组在终端中以 H*W 矩阵的格式打印出来。
    增加了对浮点数格式化的支持。

    Args:
        matrix (np.ndarray): 一个三维的 NumPy 数组，形状为 (H, W, C)。
        precision (Optional[int], optional): 
            当数组是浮点类型时，指定要保留的小数位数。
            如果为 None，则使用默认的字符串表示。默认为 None。
    """
    # 检查输入是否为三维 NumPy 数组
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 3:
        print("错误：输入必须是一个三维的 NumPy 数组 (H, W, C)。")
        return

    # 获取数组的维度
    H, W, C = matrix.shape

    # 如果数组为空，则不打印
    if H == 0 or W == 0:
        print("[]")
        return
    
    string_elements = []
    for h in range(H):
        row_elements = []
        for w in range(W):
            vector = matrix[h, w]
            string_element = ""
            if precision is not None:
                # 如果指定了精度，对向量中的每个数字进行格式化
                try:
                    # 使用 f-string 的嵌套格式化功能
                    formatted_numbers = [f"{num:.{precision}f}" for num in vector]
                    string_element = f"[{' '.join(formatted_numbers)}]"
                except (ValueError, TypeError):
                    # 如果格式化失败（例如，数组不是数字类型），则退回默认方式
                    string_element = str(vector)
            else:
                # 未指定精度，使用 NumPy 默认的字符串转换
                string_element = str(vector)
            
            row_elements.append(string_element)
        string_elements.append(row_elements)

    # 找到所有字符串化后的元素中的最大长度，用于对齐
    max_len = max([len(s) for row in string_elements for s in row] or [0])

    # 打印带边框的矩阵
    print("┌" + "─" * (W * (max_len + 2) - 2) + "┐")
    for row in string_elements:
        print("│", end="")
        for element in row:
            # 使用 ljust 方法填充空格，使每个元素占据相同的宽度
            print(f"{element:<{max_len}}", end="  ")
        print("│")
    print("└" + "─" * (W * (max_len + 2) - 2) + "┘")

def apply_polynomial(x, coefs):
    y = torch.zeros_like(x)
    for i, c in enumerate(coefs):
        y = y + c * (x ** (len(coefs) - 1 - i))
    return y

def get_map_coef(target:np.ndarray,bins=1000,deg=20):
    extend_bins = int(bins * 0.1)
    src = np.linspace(0,1,bins)
    tgt = np.quantile(target,src)
    tgt = np.concatenate([2 * tgt[0] - tgt[:extend_bins][::-1],tgt,2 * tgt[-1] - tgt[-extend_bins:][::-1]],axis=0)
    src = np.linspace(-1,1,bins + 2 * extend_bins)
    coefs = np.polyfit(src,tgt,deg = deg)
    return coefs

def resample_from_quad(
    source_image: np.ndarray,
    quad_coords: np.ndarray,
    target_shape: Tuple[int, int]
) -> Tuple[np.ndarray, np.ndarray]:
    """
    根据四边形的四个角点，在源图像(灰度或RGB)中进行重采样，得到校正后的矩形图像和坐标映射。

    Args:
        source_image (np.ndarray): 输入的源图像，形状为 (H, W) 的灰度图或 (H, W, 3) 的RGB/BGR图。
        quad_coords (np.ndarray): 四边形的四个角点坐标，形状为 (4, 2)，
                                  数据类型为 float 或 int。
                                  顺序为：左上、右上、右下、左下。
                                  坐标格式为 (row, col)，即 (行号, 列号)。
        target_shape (tuple[int, int]): 目标输出图像的尺寸 (h, w)。

    Returns:
        tuple[np.ndarray, np.ndarray]:
        - resampled_image (np.ndarray): 重采样后的矩形图像。
                                        如果输入是灰度图，形状为 (h, w)。
                                        如果输入是RGB图，形状为 (h, w, 3)。
        - coordinate_map (np.ndarray): 坐标映射矩阵，形状为 (h, w, 2)。
                                        map[y, x] = [row, col] 记录了新图(y,x)像素
                                        在原图中的浮点坐标。
    """
    # 1. 输入验证
    if source_image.ndim not in [2, 3]:
        raise ValueError("输入图像必须是二维 (灰度图) 或三维 (RGB/BGR图) 数组。")
    if quad_coords.shape != (4, 2):
        raise ValueError("角点坐标数组的形状必须是 (4, 2)。")

    h, w = target_shape
    
    # 2. 准备源坐标点和目标坐标点 
    # OpenCV 的函数期望坐标格式为 (x, y)，即 (列, 行)。
    src_points = quad_coords[:, ::-1].astype(np.float32) # 从 (row, col) 转换为 (col, row)

    # 定义目标矩形的四个角点坐标 (x, y)
    dst_points = np.array([
        [0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]
    ], dtype=np.float32)

    # 3. 计算透视变换矩阵 M 
    M, _ = cv2.findHomography(src_points, dst_points)

    # 4. 使用 M 对源图像进行透视变换
    # cv2.warpPerspective 可以自动处理单通道和多通道图像。
    # 如果 source_image 是 (H,W,3)，输出就是 (h,w,3)。
    resampled_image = cv2.warpPerspective(
        source_image,
        M,
        (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REPLICATE
    )

    # 5. 计算坐标映射矩阵 
    M_inv = np.linalg.inv(M)
    y_grid, x_grid = np.mgrid[0:h, 0:w]
    target_coords_homogeneous = np.stack(
        (x_grid.ravel(), y_grid.ravel(), np.ones(h * w)), axis=1
    )
    source_coords_homogeneous = target_coords_homogeneous @ M_inv.T
    source_coords_xy = source_coords_homogeneous[:, :2] / source_coords_homogeneous[:, 2, np.newaxis]
    source_coords_xy = source_coords_xy.reshape(h, w, 2)
    coordinate_map = source_coords_xy[:, :, ::-1]

    return resampled_image, coordinate_map