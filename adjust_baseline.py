import os
import argparse
import random
import time
import warnings
import torch
import torch.multiprocessing as mp
import numpy as np
import cv2
from tqdm import tqdm
from typing import List, Tuple, Dict, Optional
import rasterio
from scipy.optimize import least_squares
from scipy.interpolate import RegularGridInterpolator

# --- 依赖检查 ---
try:
    from kornia.feature import LoFTR
except ImportError:
    print("="*50)
    print("错误: 未找到 kornia 库。")
    print("请通过 'pip install kornia' 或 'pip install kornia kornia_moons' 安装。")
    print("="*50)
    exit(1)

try:
    # 假设 utils.py 在同一目录下
    # rs_image_1022.py 依赖: project_mercator, mercator2lonlat, bilinear_interpolate, resample_from_quad
    # adjust_test_1023.py 依赖: find_grids
    from utils import (
        project_mercator, mercator2lonlat,
        bilinear_interpolate, resample_from_quad,
        find_grids
    )
except ImportError:
    print("="*50)
    print("错误: 未找到 'utils.py' 文件。")
    print("此脚本严重依赖 utils.py 中的以下函数:")
    print(" 'project_mercator', 'mercator2lonlat', 'bilinear_interpolate', 'resample_from_quad', 'find_grids'")
    print("请确保 'utils.py' 文件与 'adjust_baseline.py' 位于同一目录。")
    print("="*50)
    exit(1)
# --- 依赖检查结束 ---


warnings.filterwarnings('ignore')

# -----------------------------------------------------------------
# 依赖的辅助函数 (从 adjust_test_1023.py 复制而来)
# -----------------------------------------------------------------

def format_time(seconds: float) -> str:
    """将秒数格式化为 HH:MM:SS """
    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    secs = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d}"

def haversine_distance(coords1: np.ndarray, coords2: np.ndarray) -> np.ndarray:
    """计算两组经纬度点之间的Haversine距离（米）"""
    R = 6371000  # 地球半径（米）
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

def check_pair_error(img_i: 'RSImage', img_j: 'RSImage') -> np.ndarray:
    """ 计算单对影像 (i, j) 之间的连接点误差（来自 tie_points.txt）"""

    if img_i.tie_points is None or img_j.tie_points is None:
        return np.array([])

    if len(img_i.tie_points) != len(img_j.tie_points):
        return np.array([])

    if len(img_i.tie_points) == 0:
        return np.array([])

    # 投影 img_i 的连接点
    lines_i = img_i.tie_points[:, 0]
    samps_i = img_i.tie_points[:, 1]
    # 使用 dem_interp 插值高程
    heights_i = img_i.dem_interp(img_i.tie_points[:, [1, 0]]) # dem_interp 需要 (samp, line) 顺序
    lats_i, lons_i = img_i.rpc.RPC_PHOTO2OBJ(samps_i, lines_i, heights_i, 'numpy')
    coords_i = np.stack([lats_i, lons_i], axis=-1)

    # 投影 img_j 的连接点
    lines_j = img_j.tie_points[:, 0]
    samps_j = img_j.tie_points[:, 1]
    heights_j = img_j.dem_interp(img_j.tie_points[:, [1, 0]])
    lats_j, lons_j = img_j.rpc.RPC_PHOTO2OBJ(samps_j, lines_j, heights_j, 'numpy')
    coords_j = np.stack([lats_j, lons_j], axis=-1)

    # 计算地理距离
    distances = haversine_distance(coords_i, coords_j)
    return distances

def check_all_pairs_error(images: List['RSImage'], overlapping_pairs: List[Tuple[int, int]]) -> np.ndarray:
    """在所有重叠对上计算并汇总误差（来自 tie_points.txt）"""
    all_distances = []
    # print("--- Global Error Report (Validation Tie Points) ---")
    for (i, j) in overlapping_pairs:
        distances = check_pair_error(images[i], images[j])
        if len(distances) > 0:
            all_distances.append(distances)
            print(f"Validation Pair ({i}, {j}) | Points: {len(distances)} | Mean Error: {distances.mean():.4f} m | Median Error: {np.median(distances):.4f} m")

    if not all_distances:
        print("No valid validation tie points found for any overlapping pair. Cannot generate report.")
        return np.array([0.0])

    all_distances = np.concatenate(all_distances)
    return all_distances

def find_overlapping_pairs(images: List['RSImage']) -> List[Tuple[int, int]]:
    """(复用) 通过检查地理坐标BBox，找出所有重叠的影像对。"""
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

    print(f"Found {len(pairs)} overlapping pairs.")
    return pairs

# -----------------------------------------------------------------
# 复制和修改的 RPCModelParameterTorch (来自 rpc.py)
# -----------------------------------------------------------------

class RPCModelParameterTorch:
    # --- (rpc.py 的全部内容开始) ---
    def __init__(self, data=torch.zeros(170,dtype=torch.double)):
        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LONG_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LONG_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]

        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]

        self.LATNUM = data[90:110]
        self.LATDEM = data[110:130]
        self.LONNUM = data[130:150]
        self.LONDEM = data[150:170]

        self.Clear_Adjust()

        self.device = self.LINE_OFF.device

    """Read orginal RPC File"""
    def load_from_file(self, filepath):
        if os.path.exists(filepath) is False:
            print("Error#001: cann't find " + filepath + " in the file system!")
            return

        with open(filepath, 'r') as f:
            all_the_text = f.read().splitlines()
            rfm_line = -1
            for line,text in enumerate(all_the_text):
                if "RFM_CORRECTION_PARAMETERS" in text:
                    rfm_line = line
                    break

        data = [np.float64(text.split()[1]) for text in (all_the_text[:rfm_line-1] if rfm_line > 0 else all_the_text)]
        data = torch.from_numpy(np.array(data, dtype=np.float64)).to(torch.double)

        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LONG_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LONG_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]
        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]
        
        if data.shape[0] >= 170:
            self.LATNUM = data[90:110]
            self.LATDEM = data[110:130]
            self.LONNUM = data[130:150]
            self.LONDEM = data[150:170]
        else:
            self.Calculate_Inverse_RPC()
        
        if rfm_line > 0:
            self.raw_adjust_params = [np.float64(text.split()[1]) for text in all_the_text[rfm_line + 1:]]
            self.Calculate_Adjust()
        else:
            self.raw_adjust_params = None

    def Create_Virtual_3D_Grid(self, xy_sample=30, z_sample=20):
        lat_max = self.LAT_OFF + self.LAT_SCALE
        lat_min = self.LAT_OFF - self.LAT_SCALE
        lon_max = self.LONG_OFF + self.LONG_SCALE
        lon_min = self.LONG_OFF - self.LONG_SCALE
        hei_max = self.HEIGHT_OFF + self.HEIGHT_SCALE
        hei_min = self.HEIGHT_OFF - self.HEIGHT_SCALE
        samp_max = self.SAMP_OFF + self.SAMP_SCALE
        samp_min = self.SAMP_OFF - self.SAMP_SCALE
        line_max = self.LINE_OFF + self.LINE_SCALE
        line_min = self.LINE_OFF - self.LINE_SCALE

        lat = torch.linspace(lat_min, lat_max, xy_sample).to(self.device,dtype=torch.double)
        lon = torch.linspace(lon_min, lon_max, xy_sample).to(self.device,dtype=torch.double)
        hei = torch.linspace(hei_min, hei_max, z_sample).to(self.device,dtype=torch.double)

        lat, lon, hei = torch.meshgrid(lat, lon, hei, indexing='ij') 

        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        samp, line = self.RPC_OBJ2PHOTO(lat, lon, hei)
        grid = torch.stack((samp, line, lat, lon, hei), dim=-1).to(self.device,dtype=torch.double)

        selected_grid = []
        for g in grid:
            flag = [g[0] < samp_min, g[0] > samp_max, g[1] < line_min, g[1] > line_max]
            if True in flag:
                continue
            else:
                selected_grid.append(g)

        if len(selected_grid) > 0:
            grid = torch.stack(selected_grid,dim=0).to(self.device,dtype=torch.double)
        else:
            print("警告: 虚拟格网点均在影像范围外。")
            grid = torch.empty(0, 5, device=self.device, dtype=torch.double)

        return grid

    def _solve_lstsq(self, ma, lv, x=None, k=1):
        assert ma.shape[0] == ma.shape[1], "ma with shape {} is not a square matrix.".format(ma.shape[0], ma.shape[1])
        
        if x is None:
            x = torch.zeros(ma.shape[0], dtype=torch.double, device=self.device)

        n = ma.shape[0]
        mak = ma.clone()
        mak += k * torch.eye(n).to(self.device,dtype=torch.double)
        lk = lv.clone()

        finish_time = 0

        for times in range(1000):
            try:
                x1 = torch.linalg.solve(mak,lk)
            except torch.linalg.LinAlgError:
                print("警告: 最小二乘解算中矩阵奇异，增加k值。")
                k *= 10
                mak = ma.clone() + k * torch.eye(n).to(self.device, dtype=torch.double)
                continue
                
            dif = torch.abs(x1 - x)
            maxdif = torch.max(dif)
            x = x1
            lk = lv + k * x

            finish_time = times + 1
            if maxdif < 1.0e-10:
                break
        return x, finish_time

    def Solve_Inverse_RPC(self, grid):
        samp, line, lat, lon, hei = torch.hsplit(grid,5)

        samp = samp.reshape(-1)
        line = line.reshape(-1)
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        samp = samp - self.SAMP_OFF
        samp = samp / self.SAMP_SCALE
        line = line - self.LINE_OFF
        line = line / self.LINE_SCALE

        lat = lat - self.LAT_OFF
        lat = lat / self.LAT_SCALE
        lon = lon - self.LONG_OFF
        lon = lon / self.LONG_SCALE
        hei = hei - self.HEIGHT_OFF
        hei = hei / self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(samp, line, hei)

        n_num = coef.shape[0]
        A = torch.zeros((n_num * 2, 78)).to(self.device,dtype=torch.double)
        A[0: n_num, 0:20] = - coef
        A[0: n_num, 20:39] = lat.reshape(-1, 1) * coef[:, 1:]
        A[n_num:, 39:59] = - coef
        A[n_num:, 59:78] = lon.reshape(-1, 1) * coef[:, 1:]

        l = torch.cat((lat, lon), -1)
        l = -l

        ATA = torch.matmul(A.T, A)
        ATl = torch.matmul(A.T, l)

        x, times = self._solve_lstsq(ATA, ATl)

        self.LATNUM = x[0:20]
        self.LATDEM[0] = 1.0
        self.LATDEM[1:20] = x[20:39]
        self.LONNUM = x[39:59]
        self.LONDEM[0] = 1.0
        self.LONDEM[1:20] = x[59:]

        return times

    def Calculate_Inverse_RPC(self):
        grid = self.Create_Virtual_3D_Grid()
        if grid.shape[0] == 0:
            print("错误: 无法创建虚拟格网，反向RPC计算失败。")
            return -1
        times = self.Solve_Inverse_RPC(grid)
        return times

    def Inverse_Adjust(self):
        R = self.adjust_params[:, :2]
        t = self.adjust_params[:, 2]   
        try:
            R_inv = torch.inverse(R) 
            t_new = -(R_inv @ t) 
            self.adjust_params_inv = torch.cat([R_inv, t_new.unsqueeze(1)], dim=1).to(torch.double)
        except torch.linalg.LinAlgError:
            print("警告: 仿射矩阵R不可逆。使用伪逆。")
            R_inv = torch.linalg.pinv(R)
            t_new = -(R_inv @ t)
            self.adjust_params_inv = torch.cat([R_inv, t_new.unsqueeze(1)], dim=1).to(torch.double)

    
    def Clear_Adjust(self):
        self.adjust_params = torch.tensor([
            [1.,0.,0.],
            [0.,1.,0.]
        ],dtype=torch.double,device=self.LNUM.device if hasattr(self, 'LNUM') else 'cpu')
        self.Inverse_Adjust()

    def Update_Adjust(self,new_adjust_params:torch.Tensor):
        new_adjust_params = new_adjust_params.to(self.adjust_params.device).to(torch.double)
        def merge_adjust(A:torch.Tensor,B:torch.Tensor) -> torch.Tensor:
            device = A.device
            dtype = A.dtype
            bottom_row = torch.tensor([[0.0, 0.0, 1.0]], dtype=dtype, device=device)
            A_h = torch.cat([A, bottom_row], dim=0)
            B_h = torch.cat([B, bottom_row], dim=0)
            C_h = B_h @ A_h
            return C_h[:2, :]
        self.adjust_params = merge_adjust(self.adjust_params,new_adjust_params).to(self.adjust_params.device).to(torch.double)
        self.Inverse_Adjust()
    
    def Calculate_Adjust(self):
        if self.raw_adjust_params is None: return
        corners = np.array([[0.,0.],[100.,0.],[0.,100.]],dtype=np.float32) #line samp
        offset_line = self.raw_adjust_params[0] + self.raw_adjust_params[1] * corners[:,1] + self.raw_adjust_params[2] * corners[:,0]
        offset_samp = self.raw_adjust_params[3] + self.raw_adjust_params[4] * corners[:,1] + self.raw_adjust_params[5] * corners[:,0]
        offset_corners = corners - np.stack([offset_line,offset_samp],axis=1) 
        af_trans = cv2.getAffineTransform(corners,offset_corners)
        self.Update_Adjust(torch.from_numpy(af_trans))

    def Merge_Adjust(self):
        identity_adjust = torch.tensor([
            [1.,0.,0.],
            [0.,1.,0.]
        ], dtype=torch.double, device=self.device)
        
        if torch.allclose(self.adjust_params, identity_adjust, atol=1e-8):
            print("Adjust parameters are already identity. No merge needed.")
            return

        print("Merging affine adjustment into RPC coefficients...")
        grid_obj = self.Create_Virtual_3D_Grid(xy_sample=50, z_sample=30) 
        if grid_obj.shape[0] == 0:
            print("错误: 无法创建用于合并的虚拟格网。操作中止。")
            return

        samp_target = grid_obj[:, 0]
        line_target = grid_obj[:, 1]
        lat = grid_obj[:, 2]
        lon = grid_obj[:, 3]
        hei = grid_obj[:, 4]

        P = (lat - self.LAT_OFF) / self.LAT_SCALE
        L = (lon - self.LONG_OFF) / self.LONG_SCALE
        H = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE
        line_target_norm = (line_target - self.LINE_OFF) / self.LINE_SCALE
        samp_target_norm = (samp_target - self.SAMP_OFF) / self.SAMP_SCALE

        coef = self.RPC_PLH_COEF(P, L, H)
        n_num = coef.shape[0]
        
        A_L = torch.zeros((n_num, 39), dtype=torch.double, device=self.device)
        A_L[:, 0:20] = coef
        A_L[:, 20:39] = -line_target_norm.unsqueeze(-1) * coef[:, 1:]
        l_L = line_target_norm * coef[:, 0]
        ATA_L = A_L.T @ A_L
        ATl_L = A_L.T @ l_L
        
        A_S = torch.zeros((n_num, 39), dtype=torch.double, device=self.device)
        A_S[:, 0:20] = coef
        A_S[:, 20:39] = -samp_target_norm.unsqueeze(-1) * coef[:, 1:]
        l_S = samp_target_norm * coef[:, 0]
        ATA_S = A_S.T @ A_S
        ATl_S = A_S.T @ l_S

        x_L, _ = self._solve_lstsq(ATA_L, ATl_L)
        x_S, _ = self._solve_lstsq(ATA_S, ATl_S)

        print("Updating direct model coefficients (LNUM, LDEM, SNUM, SDEM)...")
        self.LNUM = x_L[0:20].clone()
        self.LDEM[0] = 1.0
        self.LDEM[1:20] = x_L[20:39].clone()
        self.SNUM = x_S[0:20].clone()
        self.SDEM[0] = 1.0
        self.SDEM[1:20] = x_S[20:39].clone()

        print("Resetting adjustment parameters...")
        self.Clear_Adjust()

        print("Recalculating inverse RPC model...")
        times = self.Calculate_Inverse_RPC()
        print(f"Merge complete. Inverse RPC recalculated in {times} iterations.")

    def RPC_PLH_COEF(self, P, L, H):
        n_num = P.shape[0]
        coef = torch.zeros((n_num, 20),dtype=torch.double,device=P.device)
        coef[:, 0] = 1.0
        coef[:, 1] = L
        coef[:, 2] = P
        coef[:, 3] = H
        coef[:, 4] = L * P
        coef[:, 5] = L * H
        coef[:, 6] = P * H
        coef[:, 7] = L * L
        coef[:, 8] = P * P
        coef[:, 9] = H * H
        coef[:, 10] = P * L * H
        coef[:, 11] = L * L * L
        coef[:, 12] = L * P * P
        coef[:, 13] = L * H * H
        coef[:, 14] = L * L * P
        coef[:, 15] = P * P * P
        coef[:, 16] = P * H * H
        coef[:, 17] = L * L * H
        coef[:, 18] = P * P * H
        coef[:, 19] = H * H * H
        return coef
    
    def convert_tensor(self,arr,device):
        if isinstance(arr,torch.Tensor):
            return arr.to(dtype=torch.double,device=device)
        else:
            return torch.as_tensor(arr,dtype=torch.double,device=device)

    def RPC_OBJ2PHOTO(self, inlat, inlon, inhei, output_type='tensor'):
        lat = self.convert_tensor(inlat,self.device)
        lon = self.convert_tensor(inlon,self.device)
        hei = self.convert_tensor(inhei,self.device)
        
        is_batched = lat.dim() > 0
        if not is_batched:
            lat = lat.unsqueeze(0)
            lon = lon.unsqueeze(0)
            hei = hei.unsqueeze(0)

        lat_norm = (lat - self.LAT_OFF) / self.LAT_SCALE
        lon_norm = (lon - self.LONG_OFF) / self.LONG_SCALE
        hei_norm = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(lat_norm, lon_norm, hei_norm)

        samp_norm = torch.sum(coef * self.SNUM,dim=-1) / torch.sum(coef * self.SDEM,dim=-1)
        line_norm = torch.sum(coef * self.LNUM,dim=-1) / torch.sum(coef * self.LDEM,dim=-1)

        samp = samp_norm * self.SAMP_SCALE + self.SAMP_OFF
        line = line_norm * self.LINE_SCALE + self.LINE_OFF

        transformed_points = torch.stack([line,samp],dim=-1) @ self.adjust_params_inv[:,:2].T + self.adjust_params_inv[:,2]
        line_final = transformed_points[:,0]
        samp_final = transformed_points[:,1]

        if not is_batched:
            line_final = line_final.squeeze(0)
            samp_final = samp_final.squeeze(0)

        if output_type == 'numpy':
            samp_final = samp_final.cpu().numpy()
            line_final = line_final.cpu().numpy()

        return samp_final, line_final

    def RPC_PHOTO2OBJ(self, insamp, inline, inhei, output_type='tensor'):
        hei = self.convert_tensor(inhei,self.device)
        samp = self.convert_tensor(insamp,self.device)
        line = self.convert_tensor(inline,self.device)
 
        is_batched = samp.dim() > 0
        if not is_batched:
            samp = samp.unsqueeze(0)
            line = line.unsqueeze(0)
            hei = hei.unsqueeze(0)

        transformed_points = torch.stack([line,samp],dim=-1) @ self.adjust_params[:,:2].T + self.adjust_params[:,2]
        line_orig = transformed_points[:,0]
        samp_orig = transformed_points[:,1]

        samp_norm = (samp_orig - self.SAMP_OFF) / self.SAMP_SCALE
        line_norm = (line_orig - self.LINE_OFF) / self.LINE_SCALE
        hei_norm = (hei - self.HEIGHT_OFF) / self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(samp_norm, line_norm, hei_norm)

        lat_norm = torch.sum(coef * self.LATNUM, dim=-1) / torch.sum(coef * self.LATDEM, dim=-1)
        lon_norm = torch.sum(coef * self.LONNUM, dim=-1) / torch.sum(coef * self.LONDEM, dim=-1)

        lat = lat_norm * self.LAT_SCALE + self.LAT_OFF
        lon = lon_norm * self.LONG_SCALE + self.LONG_OFF
        
        if not is_batched:
            lat = lat.squeeze(0)
            lon = lon.squeeze(0)

        if output_type == 'numpy':
            lon = lon.cpu().numpy()
            lat = lat.cpu().numpy()

        return lat, lon
    
    def latlon2yx(self,latlon:torch.Tensor):
        r = 6378137.
        lon_rad = latlon[:,1] * torch.pi / 180.
        lat_rad = latlon[:,0] * torch.pi / 180.
        x = r * lon_rad
        y = r * torch.log(torch.tan(torch.pi / 4. + lat_rad / 2.))
        return torch.stack([y,x],dim=-1)

    def yx2latlon(self,yx:torch.Tensor):
        yx = self.convert_tensor(yx,self.device)
        r = 6378137.
        lon = (180. * yx[:,1]) / (torch.pi * r)
        lat = (2 * torch.atan(torch.exp(yx[:,0] / r)) - torch.pi * 0.5) * 180. / torch.pi
        return torch.stack([lat,lon],dim=-1)

    def RPC_XY2LINESAMP(self,x_in, y_in, h_in, output_type='tensor'):
        x = self.convert_tensor(x_in,self.device)
        y = self.convert_tensor(y_in,self.device)
        h = self.convert_tensor(h_in,self.device)
        
        is_batched = x.dim() > 0
        if not is_batched:
            x = x.unsqueeze(0)
            y = y.unsqueeze(0)
            h = h.unsqueeze(0)
            
        latlon = self.yx2latlon(torch.stack([y,x],dim=-1))
        samp,line = self.RPC_OBJ2PHOTO(latlon[:,0],latlon[:,1],h)
        
        if not is_batched:
            line = line.squeeze(0)
            samp = samp.squeeze(0)

        if output_type == 'numpy':
            line = line.cpu().numpy()
            samp = samp.cpu().numpy()
        
        return line,samp
    
    def RPC_LINESAMP2XY(self,line_in, samp_in, h_in, output_type='tensor'):
        line = self.convert_tensor(line_in,self.device)
        samp = self.convert_tensor(samp_in,self.device)
        h = self.convert_tensor(h_in,self.device)

        is_batched = line.dim() > 0
        if not is_batched:
            line = line.unsqueeze(0)
            samp = samp.unsqueeze(0)
            h = h.unsqueeze(0)

        lat,lon = self.RPC_PHOTO2OBJ(samp,line,h)
        yx = self.latlon2yx(torch.stack([lat,lon],dim=-1))
        y,x = yx[:,0],yx[:,1]
        
        if not is_batched:
            x = x.squeeze(0)
            y = y.squeeze(0)

        if output_type == 'numpy':
            x = x.cpu().numpy()
            y = y.cpu().numpy()

        return x,y
    
    def _project_xyh_to_linesamp_for_jacobian(self, xyh_tensor: torch.Tensor) -> torch.Tensor:
        x, y, h = xyh_tensor[..., 0], xyh_tensor[..., 1], xyh_tensor[..., 2]
        latlon = self.yx2latlon(torch.stack([y, x], dim=-1))
        samp, line = self.RPC_OBJ2PHOTO(latlon[:, 0], latlon[:, 1], h)
        return torch.stack([line, samp], dim=-1)

    def _vjp_projection_core(self, mu_xyh: torch.Tensor, sigma_xyh: torch.Tensor):
        mu_xyh.requires_grad_(True)
        if mu_xyh.grad is not None:
            mu_xyh.grad.zero_()

        line, samp = self.RPC_XY2LINESAMP(mu_xyh[:, 0], mu_xyh[:, 1], mu_xyh[:, 2])
        mu_linesamp = torch.stack([line, samp], dim=-1)

        grad_line, = torch.autograd.grad(
            outputs=line, inputs=mu_xyh,
            grad_outputs=torch.ones_like(line),
            create_graph=True, retain_graph=True,
        )
        grad_samp, = torch.autograd.grad(
            outputs=samp, inputs=mu_xyh,
            grad_outputs=torch.ones_like(samp),
            create_graph=False, retain_graph=False, 
        )
        
        var_xyh = sigma_xyh.pow(2)
        var_line = (grad_line.pow(2) * var_xyh).sum(dim=-1)
        var_samp = (grad_samp.pow(2) * var_xyh).sum(dim=-1)
        var_linesamp = torch.stack([var_line, var_samp], dim=-1)
            
        return mu_linesamp, var_linesamp

    def xy_distribution_to_linesamp(self, mu_xyh: torch.Tensor, sigma_xyh: torch.Tensor, chunk_size: int = 524288):
        is_batched = mu_xyh.dim() == 2
        if not is_batched:
            mu_xyh = mu_xyh.unsqueeze(0)
            sigma_xyh = sigma_xyh.unsqueeze(0)
        
        num_points = mu_xyh.shape[0]

        if num_points <= chunk_size:
            mu_linesamp, var_linesamp = self._vjp_projection_core(mu_xyh, sigma_xyh)
        else:
            print(f"警告: 点数 ({num_points}) 超过阈值 ({chunk_size})，自动启用分块计算。")
            mu_results = []
            var_results = []
            
            mu_chunks = torch.split(mu_xyh, chunk_size)
            sigma_chunks = torch.split(sigma_xyh, chunk_size)

            for mu_chunk, sigma_chunk in zip(mu_chunks, sigma_chunks):
                mu_chunk_out, var_chunk_out = self._vjp_projection_core(mu_chunk, sigma_chunk)
                mu_results.append(mu_chunk_out.detach())
                var_results.append(var_chunk_out.detach())

            mu_linesamp = torch.cat(mu_results, dim=0)
            var_linesamp = torch.cat(var_results, dim=0)
        
        sigma_linesamp = torch.sqrt(var_linesamp)
        if not is_batched:
            mu_linesamp = mu_linesamp.squeeze(0)
            sigma_linesamp = sigma_linesamp.squeeze(0)

        return mu_linesamp, sigma_linesamp
    
    def to_gpu(self,device = None):
        if device is None:
            if torch.cuda.is_available():
                device = 'cuda'
            else:
                print("CUDA not available, using CPU.")
                device = 'cpu'
        
        self.device = torch.device(device)
        
        self.LINE_OFF = self.LINE_OFF.to(self.device)
        self.SAMP_OFF = self.SAMP_OFF.to(self.device)
        self.LAT_OFF = self.LAT_OFF.to(self.device)
        self.LONG_OFF = self.LONG_OFF.to(self.device)
        self.HEIGHT_OFF = self.HEIGHT_OFF.to(self.device)
        self.LINE_SCALE = self.LINE_SCALE.to(self.device)
        self.SAMP_SCALE = self.SAMP_SCALE.to(self.device)
        self.LAT_SCALE = self.LAT_SCALE.to(self.device)
        self.LONG_SCALE = self.LONG_SCALE.to(self.device)
        self.HEIGHT_SCALE = self.HEIGHT_SCALE.to(self.device)

        self.LNUM = self.LNUM.to(self.device)
        self.LDEM = self.LDEM.to(self.device)
        self.SNUM = self.SNUM.to(self.device)
        self.SDEM = self.SDEM.to(self.device)

        self.LATNUM = self.LATNUM.to(self.device)
        self.LATDEM = self.LATDEM.to(self.device)
        self.LONNUM = self.LONNUM.to(self.device)
        self.LONDEM = self.LONDEM.to(self.device)

        self.adjust_params = self.adjust_params.to(self.device)
        self.adjust_params_inv = self.adjust_params_inv.to(self.device)

    def save_rpc_to_file(self, filepath):
        original_device = self.device
        if self.device.type != 'cpu':
            self.to_gpu('cpu')

        addition0 = ['LINE_OFF:', 'SAMP_OFF:', 'LAT_OFF:', 'LONG_OFF:', 'HEIGHT_OFF:', 'LINE_SCALE:', 'SAMP_SCALE:',
                     'LAT_SCALE:', 'LONG_SCALE:', 'HEIGHT_SCALE:', 'LINE_NUM_COEFF_1:', 'LINE_NUM_COEFF_2:',
                     'LINE_NUM_COEFF_3:', 'LINE_NUM_COEFF_4:', 'LINE_NUM_COEFF_5:', 'LINE_NUM_COEFF_6:',
                     'LINE_NUM_COEFF_7:', 'LINE_NUM_COEFF_8:', 'LINE_NUM_COEFF_9:', 'LINE_NUM_COEFF_10:',
                     'LINE_NUM_COEFF_11:', 'LINE_NUM_COEFF_12:', 'LINE_NUM_COEFF_13:', 'LINE_NUM_COEFF_14:',
                     'LINE_NUM_COEFF_15:', 'LINE_NUM_COEFF_16:', 'LINE_NUM_COEFF_17:', 'LINE_NUM_COEFF_18:',
                     'LINE_NUM_COEFF_19:', 'LINE_NUM_COEFF_20:', 'LINE_DEN_COEFF_1:', 'LINE_DEN_COEFF_2:',
                     'LINE_DEN_COEFF_3:', 'LINE_DEN_COEFF_4:', 'LINE_DEN_COEFF_5:', 'LINE_DEN_COEFF_6:',
                     'LINE_DEN_COEFF_7:', 'LINE_DEN_COEFF_8:', 'LINE_DEN_COEFF_9:', 'LINE_DEN_COEFF_10:',
                     'LINE_DEN_COEFF_11:', 'LINE_DEN_COEFF_12:', 'LINE_DEN_COEFF_13:', 'LINE_DEN_COEFF_14:',
                     'LINE_DEN_COEFF_15:', 'LINE_DEN_COEFF_16:', 'LINE_DEN_COEFF_17:', 'LINE_DEN_COEFF_18:',
                     'LINE_DEN_COEFF_19:', 'LINE_DEN_COEFF_20:', 'SAMP_NUM_COEFF_1:', 'SAMP_NUM_COEFF_2:',
                     'SAMP_NUM_COEFF_3:', 'SAMP_NUM_COEFF_4:', 'SAMP_NUM_COEFF_5:', 'SAMP_NUM_COEFF_6:',
                     'SAMP_NUM_COEFF_7:', 'SAMP_NUM_COEFF_8:', 'SAMP_NUM_COEFF_9:', 'SAMP_NUM_COEFF_10:',
                     'SAMP_NUM_COEFF_11:', 'SAMP_NUM_COEFF_12:', 'SAMP_NUM_COEFF_13:', 'SAMP_NUM_COEFF_14:',
                     'SAMP_NUM_COEFF_15:', 'SAMP_NUM_COEFF_16:', 'SAMP_NUM_COEFF_17:', 'SAMP_NUM_COEFF_18:',
                     'SAMP_NUM_COEFF_19:', 'SAMP_NUM_COEFF_20:', 'SAMP_DEN_COEFF_1:', 'SAMP_DEN_COEFF_2:',
                     'SAMP_DEN_COEFF_3:', 'SAMP_DEN_COEFF_4:', 'SAMP_DEN_COEFF_5:', 'SAMP_DEN_COEFF_6:',
                     'SAMP_DEN_COEFF_7:', 'SAMP_DEN_COEFF_8:', 'SAMP_DEN_COEFF_9:', 'SAMP_DEN_COEFF_10:',
                     'SAMP_DEN_COEFF_11:', 'SAMP_DEN_COEFF_12:', 'SAMP_DEN_COEFF_13:', 'SAMP_DEN_COEFF_14:',
                     'SAMP_DEN_COEFF_15:', 'SAMP_DEN_COEFF_16:', 'SAMP_DEN_COEFF_17:', 'SAMP_DEN_COEFF_18:',
                     'SAMP_DEN_COEFF_19:', 'SAMP_DEN_COEFF_20:', 'LAT_NUM_COEFF_1:', 'LAT_NUM_COEFF_2:',
                     'LAT_NUM_COEFF_3:', 'LAT_NUM_COEFF_4:', 'LAT_NUM_COEFF_5:', 'LAT_NUM_COEFF_6:',
                     'LAT_NUM_COEFF_7:', 'LAT_NUM_COEFF_8:', 'LAT_NUM_COEFF_9:', 'LAT_NUM_COEFF_10:',
                     'LAT_NUM_COEFF_11:', 'LAT_NUM_COEFF_12:', 'LAT_NUM_COEFF_13:', 'LAT_NUM_COEFF_14:',
                     'LAT_NUM_COEFF_15:', 'LAT_NUM_COEFF_16:', 'LAT_NUM_COEFF_17:', 'LAT_NUM_COEFF_18:',
                     'LAT_NUM_COEFF_19:', 'LAT_NUM_COEFF_20:', 'LAT_DEN_COEFF_1:', 'LAT_DEN_COEFF_2:',
                     'LAT_DEN_COEFF_3:', 'LAT_DEN_COEFF_4:', 'LAT_DEN_COEFF_5:', 'LAT_DEN_COEFF_6:',
                     'LAT_DEN_COEFF_7:', 'LAT_DEN_COEFF_8:', 'LAT_DEN_COEFF_9:', 'LAT_DEN_COEFF_10:',
                     'LAT_DEN_COEFF_11:', 'LAT_DEN_COEFF_12:', 'LAT_DEN_COEFF_13:', 'LAT_DEN_COEFF_14:',
                     'LAT_DEN_COEFF_15:', 'LAT_DEN_COEFF_16:', 'LAT_DEN_COEFF_17:', 'LAT_DEN_COEFF_18:',
                     'LAT_DEN_COEFF_19:', 'LAT_DEN_COEFF_20:', 'LONG_NUM_COEFF_1:', 'LONG_NUM_COEFF_2:',
                     'LONG_NUM_COEFF_3:', 'LONG_NUM_COEFF_4:', 'LONG_NUM_COEFF_5:', 'LONG_NUM_COEFF_6:',
                     'LONG_NUM_COEFF_7:', 'LONG_NUM_COEFF_8:', 'LONG_NUM_COEFF_9:', 'LONG_NUM_COEFF_10:',
                     'LONG_NUM_COEFF_11:', 'LONG_NUM_COEFF_12:', 'LONG_NUM_COEFF_13:', 'LONG_NUM_COEFF_14:',
                     'LONG_NUM_COEFF_15:', 'LONG_NUM_COEFF_16:', 'LONG_NUM_COEFF_17:', 'LONG_NUM_COEFF_18:',
                     'LONG_NUM_COEFF_19:', 'LONG_NUM_COEFF_20:', 'LONG_DEN_COEFF_1:', 'LONG_DEN_COEFF_2:',
                     'LONG_DEN_COEFF_3:', 'LONG_DEN_COEFF_4:', 'LONG_DEN_COEFF_5:', 'LONG_DEN_COEFF_6:',
                     'LONG_DEN_COEFF_7:', 'LONG_DEN_COEFF_8:', 'LONG_DEN_COEFF_9:', 'LONG_DEN_COEFF_10:',
                     'LONG_DEN_COEFF_11:', 'LONG_DEN_COEFF_12:', 'LONG_DEN_COEFF_13:', 'LONG_DEN_COEFF_14:',
                     'LONG_DEN_COEFF_15:', 'LONG_DEN_COEFF_16:', 'LONG_DEN_COEFF_17:', 'LONG_DEN_COEFF_18:',
                     'LONG_DEN_COEFF_19:', 'LONG_DEN_COEFF_20:']
        addition1 = ['pixels', 'pixels', 'degrees', 'degrees', 'meters', 'pixels', 'pixels', 'degrees', 'degrees',
                     'meters']
        text = ""
        text += addition0[0] + " " + str(self.LINE_OFF.item()) + " " + addition1[0] + "\n"
        text += addition0[1] + " " + str(self.SAMP_OFF.item()) + " " + addition1[1] + "\n"
        text += addition0[2] + " " + str(self.LAT_OFF.item()) + " " + addition1[2] + "\n"
        text += addition0[3] + " " + str(self.LONG_OFF.item()) + " " + addition1[3] + "\n"
        text += addition0[4] + " " + str(self.HEIGHT_OFF.item()) + " " + addition1[4] + "\n"
        text += addition0[5] + " " + str(self.LINE_SCALE.item()) + " " + addition1[5] + "\n"
        text += addition0[6] + " " + str(self.SAMP_SCALE.item()) + " " + addition1[6] + "\n"
        text += addition0[7] + " " + str(self.LAT_SCALE.item()) + " " + addition1[7] + "\n"
        text += addition0[8] + " " + str(self.LONG_SCALE.item()) + " " + addition1[8] + "\n"
        text += addition0[9] + " " + str(self.HEIGHT_SCALE.item()) + " " + addition1[9] + "\n"
        for i in range(10, 30):
            text += addition0[i] + " " + str(self.LNUM[i - 10].item()) + "\n"
        for i in range(30, 50):
            text += addition0[i] + " " + str(self.LDEM[i - 30].item()) + "\n"
        for i in range(50, 70):
            text += addition0[i] + " " + str(self.SNUM[i - 50].item()) + "\n"
        for i in range(70, 90):
            text += addition0[i] + " " + str(self.SDEM[i - 70].item()) + "\n"
        for i in range(90, 110):
            text += addition0[i] + " " + str(self.LATNUM[i - 90].item()) + "\n"
        for i in range(110, 130):
            text += addition0[i] + " " + str(self.LATDEM[i - 110].item()) + "\n"
        for i in range(130, 150):
            text += addition0[i] + " " + str(self.LONNUM[i - 130].item()) + "\n"
        for i in range(150, 170):
            text += addition0[i] + " " + str(self.LONDEM[i - 150].item()) + "\n"
        
        f = open(filepath, "w")
        f.write(text)
        f.close()
        
        if original_device.type != 'cpu':
            self.to_gpu(original_device)
    # --- (rpc.py 的全部内容结束) ---

    # --- (新增方法) ---
    def Set_Adjust(self, A: torch.Tensor):
        """
        (新增) 直接设置仿射变换矩阵，而不是累积它。
        专为scipy.optimize.least_squares解算器设计。
        """
        # 确保类型和设备正确
        self.adjust_params = A.to(device=self.device, dtype=torch.double)
        # 立即更新逆矩阵
        self.Inverse_Adjust()
    # --- (新增方法结束) ---


# -----------------------------------------------------------------
# 复制和修改的 RSImage (来自 rs_image_1022.py)
# -----------------------------------------------------------------

clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

class RSImage():
    def __init__(self, options, root: str, id: int, size_limit=0):
        """
        (已修改) root: path to folder which contains 'image.png','dem.npy','rpc.txt',
        (已修改) id: index of this image
        (已修改) __init__ 按需加载影像, 仅加载元数据
        """
        self.options = options
        self.root = root
        self.id = id
        
        # --- (修改: 影像按需加载) ---
        self.image_path = os.path.join(root, 'image.png')
        if not os.path.exists(self.image_path):
             raise IOError(f"影像文件未找到: {self.image_path}")
        self.image: Optional[np.ndarray] = None # 不在此处加载
        # --- (修改结束) ---
        
        dem_path = os.path.join(root, 'dem.npy')
        if not os.path.exists(dem_path):
             raise IOError(f"DEM文件未找到: {dem_path}")
        self.dem: np.ndarray = np.load(dem_path)
        
        tp_path = os.path.join(root, 'tie_points.txt')
        if os.path.exists(tp_path):
            self.tie_points = self.__load_tie_points__(tp_path)
        else:
            self.tie_points = None

        if size_limit > 0:
            # self.image = self.image[:size_limit,:size_limit] # Image 未加载
            self.dem = self.dem[:size_limit,:size_limit]
            print(f"警告: 已应用 size_limit={size_limit}。")

        # --- (修改: H, W的获取) ---
        # 尝试从DEM获取H,W
        if self.dem is not None and self.dem.size > 0:
             self.H, self.W = self.dem.shape[:2]
        else:
             # 如果DEM也没有，尝试读取影像尺寸 (这会减慢初始化速度)
             try:
                 # 使用 rasterio 或 cv2 高效读取尺寸
                 with rasterio.open(self.image_path) as src:
                     self.H = src.height
                     self.W = src.width
             except Exception as e:
                 print(f"警告: 无法从DEM获取H,W，尝试读取影像尺寸失败: {e}")
                 # 作为最后手段
                 temp_img = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
                 if temp_img is None:
                     raise IOError(f"无法读取影像 {self.image_path} 以获取尺寸")
                 self.H, self.W = temp_img.shape[:2]
                 del temp_img
        
        if size_limit > 0:
            self.H = min(self.H, size_limit)
            self.W = min(self.W, size_limit)
        # --- (修改结束) ---
                 
        rpc_path = os.path.join(root, 'rpc.txt')
        if not os.path.exists(rpc_path):
             raise IOError(f"RPC文件未找到: {rpc_path}")
        self.rpc = RPCModelParameterTorch()
        self.rpc.load_from_file(rpc_path)
        self.rpc.to_gpu()
        
        self.corner_xys = self.__get_corner_xys__() #[tl,tr,bl,br] [x,y]
        self.overlap_grids = []
        self.R = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        self.T = torch.tensor([0., 0.])
    
    # --- (新增方法) ---
    def load_image_data(self):
        """(新增) 按需加载影像数据"""
        if self.image is None:
            self.image = cv2.imread(self.image_path, cv2.IMREAD_GRAYSCALE)
            if self.image is None:
                raise IOError(f"cv2.imread 无法读取影像文件: {self.image_path}")
            
            # (新增) 如果应用了size_limit，在此处裁剪
            if self.H < self.image.shape[0] or self.W < self.image.shape[1]:
                 self.image = self.image[:self.H, :self.W]
                 
            # 统一为3通道，与DINO流一致，适配LoFTR输入
            self.image = np.stack([self.image] * 3, axis=-1)

    def release_image_data(self):
        """(新增) 按需释放影像数据"""
        if self.image is not None:
            del self.image
            self.image = None
    # --- (新增方法结束) ---

    # --- (rs_image_1022.py 的其余方法开始) ---
    def __load_image__(self,path) -> np.ndarray:
        print("Loading Image (Deprecated Method)")
        with rasterio.open(path) as src:
            data = src.read().astype(np.float32)
            for band in range(data.shape[0]):
                data[band] = (255. * data[band] / data[band].max())
            if data.ndim == 3:
                data = np.transpose(data, (1, 2, 0)).squeeze()
        return data[:self.H,:self.W] # 应用 H, W 限制

    def __load_tie_points__(self,path) -> np.ndarray:
        tie_points = np.loadtxt(path,dtype=int)
        if tie_points.ndim == 1:
            tie_points = tie_points.reshape(1,-1)
        elif tie_points.shape[1] != 2:
            print("tie points format error")
            return None
        return tie_points
    
    @torch.no_grad()
    def __get_corner_xys__(self):
        """
        return: [tl,tr,bl,br] [x,y] np.ndarray
        """
        # (修改) 确保使用正确的 H, W
        h_idx_max = self.H - 1
        w_idx_max = self.W - 1
        heights = [
            self.dem[0, 0], 
            self.dem[0, w_idx_max], 
            self.dem[h_idx_max, 0], 
            self.dem[h_idx_max, w_idx_max]
        ]
        samps = [0., w_idx_max, 0., w_idx_max]
        lines = [0., 0., h_idx_max, h_idx_max]
        
        latlons = torch.stack(self.rpc.RPC_PHOTO2OBJ(samps, lines, heights), dim=-1)
        xys = project_mercator(latlons)
        return xys.cpu().numpy()[:,[1,0]] # y,x -> x,y

    
    @torch.no_grad()
    def dem_interp(self,sampline:np.ndarray):
        """
        输入 sampline (N, 2) 格式为 [samp, line]
        """
        if sampline.ndim == 1:
            sampline = sampline[None]
        # bilinear_interpolate 需要 (line, samp) 顺序
        return bilinear_interpolate(self.dem, sampline[:, [1, 0]])
    
    @torch.no_grad()
    def xy_to_sampline(self,xy:np.ndarray,max_iter = 100):
        if xy.ndim == 1:
            xy = xy[None]
        latlon = mercator2lonlat(xy[:,[1,0]])
        
        # (修改) 使用H,W进行估算
        samp_init = self.W * (xy[:, 0] - self.corner_xys[0, 0]) / (self.corner_xys[3, 0] - self.corner_xys[0, 0])
        line_init = self.H * (xy[:, 1] - self.corner_xys[0, 1]) / (self.corner_xys[3, 1] - self.corner_xys[0, 1])
        sampline = np.stack([samp_init, line_init], axis=-1)

        dem = self.dem_interp(sampline)
        invalid_mask = np.full(dem.shape,True,dtype=bool)
        
        for iter in range(max_iter):
            if invalid_mask.sum() == 0:
                break
            sampline_new = np.stack(self.rpc.RPC_OBJ2PHOTO(latlon[invalid_mask,0],latlon[invalid_mask,1],dem[invalid_mask],'numpy'),axis=-1)
            dis = np.linalg.norm(sampline_new - sampline[invalid_mask],axis=-1)
            sampline[invalid_mask] = sampline_new
            
            # 更新DEM和掩码
            dem[invalid_mask] = self.dem_interp(sampline[invalid_mask])
            invalid_mask[invalid_mask] = dis > 1.
            
        return sampline.squeeze()

    @torch.no_grad()
    def get_image_by_sampline(self,tl_sampline:np.ndarray,br_sampline:np.ndarray,div_factor:int = 16):
        if self.image is None:
            raise RuntimeError("Image data is not loaded. Call load_image_data() first.")
        tl_sampline = np.array(tl_sampline)
        br_sampline = np.array(br_sampline)
        H = ((br_sampline[1] - tl_sampline[1]) // div_factor) * div_factor
        W = ((br_sampline[0] - tl_sampline[0]) // div_factor) * div_factor
        line_start = (br_sampline[1] - tl_sampline[1] - H) // 2 + tl_sampline[1]
        samp_start = (br_sampline[0] - tl_sampline[0] - W) // 2 + tl_sampline[0]
        tl_sampline = np.array([samp_start,line_start],dtype=int)
        br_sampline = np.array([samp_start + W,line_start + H],dtype=int)
        
        # (修改) 增加边界检查
        tl_sampline[0] = np.clip(tl_sampline[0], 0, self.W)
        tl_sampline[1] = np.clip(tl_sampline[1], 0, self.H)
        br_sampline[0] = np.clip(br_sampline[0], 0, self.W)
        br_sampline[1] = np.clip(br_sampline[1], 0, self.H)
        
        return self.image[tl_sampline[1]:br_sampline[1],tl_sampline[0]:br_sampline[0]]
    
    @torch.no_grad()
    def get_dem_by_sampline(self,tl_sampline:np.ndarray,br_sampline:np.ndarray,div_factor:int = 16):
        tl_sampline = np.array(tl_sampline)
        br_sampline = np.array(br_sampline)
        H = ((br_sampline[1] - tl_sampline[1]) // div_factor) * div_factor
        W = ((br_sampline[0] - tl_sampline[0]) // div_factor) * div_factor
        line_start = (br_sampline[1] - tl_sampline[1] - H) // 2 + tl_sampline[1]
        samp_start = (br_sampline[0] - tl_sampline[0] - W) // 2 + tl_sampline[0]
        tl_sampline = np.array([samp_start,line_start],dtype=int)
        br_sampline = np.array([samp_start + W,line_start + H],dtype=int)
        
        # (修改) 增加边界检查
        tl_sampline[0] = np.clip(tl_sampline[0], 0, self.W)
        tl_sampline[1] = np.clip(tl_sampline[1], 0, self.H)
        br_sampline[0] = np.clip(br_sampline[0], 0, self.W)
        br_sampline[1] = np.clip(br_sampline[1], 0, self.H)

        return self.dem[tl_sampline[1]:br_sampline[1],tl_sampline[0]:br_sampline[0]]

    @torch.no_grad()
    def get_image_by_xy(self,tlxy:np.ndarray,brxy:np.ndarray,div_factor:int = 16):
        if self.image is None:
            raise RuntimeError("Image data is not loaded. Call load_image_data() first.")
        tlxy = np.array(tlxy)
        brxy = np.array(brxy)
        tl_sampline = self.xy_to_sampline(tlxy)
        br_sampline = self.xy_to_sampline(brxy)
        return self.get_image_by_sampline(tl_sampline,br_sampline),tl_sampline,br_sampline

    @torch.no_grad()
    def get_dem_by_xy(self,tlxy:np.ndarray,brxy:np.ndarray,div_factor:int = 16):
        tlxy = np.array(tlxy)
        brxy = np.array(brxy)
        tl_sampline = self.xy_to_sampline(tlxy)
        br_sampline = self.xy_to_sampline(brxy)
        return self.get_dem_by_sampline(tl_sampline,br_sampline),tl_sampline,br_sampline

    @torch.no_grad()
    def resample_image_by_sampline(self,corner_samplines:np.ndarray,target_shape:Tuple[int,int],need_local:bool = False):
        if self.image is None:
            raise RuntimeError("Image data is not loaded. Call load_image_data() first.")
        img_resampled,local_hw2 = resample_from_quad(self.image,corner_samplines[:,[1,0]],target_shape)
        if need_local:
            return img_resampled,local_hw2
        else:
            return img_resampled
    
    @torch.no_grad()
    def resample_dem_by_sampline(self,corner_samplines:np.ndarray,target_shape:Tuple[int,int],need_local:bool = False):
        dem_resampled,local_hw2 = resample_from_quad(self.dem,corner_samplines[:,[1,0]],target_shape)
        if need_local:
            return dem_resampled,local_hw2
        else:
            return dem_resampled
        
    def vis_grid(self,diags:list[np.ndarray],output_path:str = None):
        if self.image is None:
            print("Warning: vis_grid requires image data. Loading...")
            self.load_image_data()
            
        vis_img = self.image.copy()
        for diag in diags:
            min_x,min_y,max_x,max_y = diag[:,0].min(),diag[:,1].min(),diag[:,0].max(),diag[:,1].max()
            corners = [self.xy_to_sampline(np.array([min_x,min_y])),
                       self.xy_to_sampline(np.array([max_x,min_y])),
                       self.xy_to_sampline(np.array([max_x,max_y])),
                       self.xy_to_sampline(np.array([min_x,max_y])),
                       self.xy_to_sampline(np.array([min_x,min_y]))]
            for i in range(len(corners) - 1):
                cv2.line(vis_img,(int(corners[i][0]),int(corners[i][1])),(int(corners[i+1][0]),int(corners[i+1][1])),(0,0,255),5)
        
        if not output_path is None:
            cv2.imwrite(output_path,vis_img)
        
        return vis_img
    # --- (rs_image_1022.py 的其余方法结束) ---


# -----------------------------------------------------------------
# 复制和修改的 load_imgs_bundle (来自 adjust_test_1023.py)
# -----------------------------------------------------------------
def load_imgs_bundle(args) -> List[RSImage]:
    """(已修改) 加载所有影像 (按需加载图像数据)。"""
    base_path = os.path.join(args.root, 'adjust_images')
    if not os.path.exists(base_path):
        print(f"错误: 路径 '{base_path}' 不存在。请检查 --root 参数。")
        exit(1)
        
    select_img_idxs = [int(i) for i in args.select_imgs.split(',')]
    
    try:
        img_folders_all = sorted([d for d in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, d))])
        img_folders = [img_folders_all[i] for i in select_img_idxs]
    except IndexError:
        print(f"错误: --select_imgs 索引超出范围。在 '{base_path}' 中只找到 {len(img_folders_all)} 个影像文件夹。")
        exit(1)
    except Exception as e:
        print(f"错误: 列出影像文件夹时出错: {e}")
        exit(1)

    images = []
    print(f"Found {len(img_folders)} image folders. Loading metadata...")
    for idx, folder in zip(select_img_idxs, img_folders): # (修改) 使用原始索引作为ID
        img_path = os.path.join(base_path, folder)
        try:
            # (修改) RSImage的__init__已被修改，不会加载图像
            # (修改) 传入原始索引 idx 作为 RSImage 的 id
            images.append(RSImage(args, img_path, idx))
            print(f"Loaded metadata for image {idx} from {folder}.")
        except Exception as e:
            print(f"!! 失败: 无法加载影像 {idx} (来自 {folder}): {e}")
            # raise e # 调试时可以打开
            
    print(f"Successfully loaded metadata for {len(images)} images.")
    return images

# -----------------------------------------------------------------
# Baseline 核心代码 (根据方案撰写)
# -----------------------------------------------------------------

# --- Kornia LoFTR 要求的图像预处理 ---
def preprocess_image_to_tensor(img_np: np.ndarray, device: torch.device):
    """
    将 (H, W, 3) BGR uint8 图像转换为 (1, 1, H, W) 灰度 float tensor
    """
    img = torch.from_numpy(img_np).permute(2, 0, 1).float().to(device) / 255.0 # (3,H,W)
    img_gray = torch.mean(img, dim=0, keepdim=True) # (1,H,W) 
    return img_gray.unsqueeze(0) # (1, 1, H, W)

# --- 阶段 1: 同名点提取 (并行) ---

# 全局变量，用于多进程共享（只读）
# 注意：这些变量将在 MP 工作进程中被重新初始化
g_images: List[RSImage] = []
g_loftr = None
g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
g_patch_size = 512
g_conf_threshold = 0.7

def init_worker(images_list: List[RSImage], patch_size: int, conf_threshold: float):
    """
    初始化多进程工作器（Worker）
    """
    global g_images, g_loftr, g_device, g_patch_size, g_conf_threshold
    
    # 1. 设置全局设备
    g_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 2. 加载 LoFTR 模型 (每个进程加载一次到各自的GPU)
    print(f"Worker {os.getpid()}: Loading LoFTR model to {g_device}...")
    g_loftr = LoFTR(pretrained='outdoor').to(g_device).eval()
    
    # 3. 存储其他全局配置
    g_images = images_list
    g_patch_size = patch_size
    g_conf_threshold = conf_threshold
    print(f"Worker {os.getpid()}: Initialized.")


def process_task_worker(task: Tuple[np.ndarray, int, int]) -> Optional[Tuple[int, int, np.ndarray, np.ndarray]]:
    """
    (核心) 单个进程的工作函数，用于提取一对影像在一个格网上的同名点。
    """
    global g_images, g_loftr, g_device, g_patch_size, g_conf_threshold
    
    diag, i, j = task
    
    # 获取 RSImage 对象
    # (注意: g_images 是主进程中对象的副本，修改它们不会影响主进程)
    img_i = g_images[i]
    img_j = g_images[j]
    
    try:
        # 1. 按需加载影像 (节省内存)
        img_i.load_image_data()
        img_j.load_image_data()

        # 2. 将地理格网(diag)重采样为像方块
        #    使用 np.array 构造4个角点
        corners_geo = np.array([
            diag[0],                         # [min_x, min_y]
            [diag[1, 0], diag[0, 1]],        # [max_x, min_y]
            diag[1],                         # [max_x, max_y]
            [diag[0, 0], diag[1, 1]]         # [min_x, max_y]
        ])
        
        corners_samp_i = img_i.xy_to_sampline(corners_geo)
        corners_samp_j = img_j.xy_to_sampline(corners_geo)

        # 3. 重采样影像和"local"坐标网格
        # local_i 是 (patch_size, patch_size, 2) 的网格，存储了每个像素在*完整影像*中的(line, samp)坐标
        patch_i, local_i = img_i.resample_image_by_sampline(corners_samp_i, (g_patch_size, g_patch_size), need_local=True)
        patch_j, local_j = img_j.resample_image_by_sampline(corners_samp_j, (g_patch_size, g_patch_size), need_local=True)

        # 4. 立即释放大影像内存
        img_i.release_image_data()
        img_j.release_image_data()

        # 5. 运行 LoFTR
        img_tensor_i = preprocess_image_to_tensor(patch_i, g_device)
        img_tensor_j = preprocess_image_to_tensor(patch_j, g_device)
        
        del patch_i, patch_j # 进一步释放内存
        
        with torch.no_grad():
            input_dict = {"image0": img_tensor_i, "image1": img_tensor_j}
            matches = g_loftr(input_dict)

        # 6. 筛选高质量匹配点 (LoFTR的kpts是 (x,y) 格式)
        conf = matches['confidence'].cpu().numpy()
        kpts_i_patch = matches['keypoints0'].cpu().numpy() # (N, 2) [x, y]
        kpts_j_patch = matches['keypoints1'].cpu().numpy() # (N, 2) [x, y]
        
        valid_idx = conf > g_conf_threshold
        num_valid = np.sum(valid_idx)
        
        if num_valid < 10: # 至少10个点
            return None
            
        kpts_i_patch = kpts_i_patch[valid_idx]
        kpts_j_patch = kpts_j_patch[valid_idx]

        # 7. (关键) 将"块坐标" (0-patch_size) 转换回 "全影像坐标"
        #    我们使用 local_i/local_j 网格进行插值
        
        # local 网格是 (H, W, 2)，存储 (line, samp)
        # resample_from_quad 返回 (line, samp)
        
        # kpts 是 (x, y)，对应 (W, H)
        # RegularGridInterpolator 需要 (y, x) 格式
        interp_coords_i_yx = kpts_i_patch[:, [1, 0]] # (N, 2) [y, x]
        interp_coords_j_yx = kpts_j_patch[:, [1, 0]] # (N, 2) [y, x]

        # 分别创建 line 和 samp 的插值器
        lines = np.linspace(0, g_patch_size - 1, g_patch_size)
        samps = np.linspace(0, g_patch_size - 1, g_patch_size)
        
        # 插值器 i
        interp_i_line = RegularGridInterpolator((lines, samps), local_i[..., 0])
        interp_i_samp = RegularGridInterpolator((lines, samps), local_i[..., 1])
        # 插值器 j
        interp_j_line = RegularGridInterpolator((lines, samps), local_j[..., 0])
        interp_j_samp = RegularGridInterpolator((lines, samps), local_j[..., 1])

        # 执行插值
        full_coords_i_line = interp_i_line(interp_coords_i_yx)
        full_coords_i_samp = interp_i_samp(interp_coords_i_yx)
        full_coords_j_line = interp_j_line(interp_coords_j_yx)
        full_coords_j_samp = interp_j_samp(interp_coords_j_yx)

        # (N, 2) [samp, line]
        full_coords_i = np.stack([full_coords_i_samp, full_coords_i_line], axis=-1) 
        full_coords_j = np.stack([full_coords_j_samp, full_coords_j_line], axis=-1)

        # 返回连接点
        # (重要) 返回原始影像索引 i 和 j
        return (i, j, full_coords_i, full_coords_j)

    except Exception as e:
        print(f"警告: Worker {os.getpid()} 任务 (diag, {i}, {j}) 失败: {e}")
        return None

# --- 阶段 2: 捆绑平差 (BA) ---

# 全局变量，用于BA解算器
# (在主进程中设置)
g_ba_images: List[RSImage] = []
g_ba_tie_points: List[Tuple[int, int, np.ndarray, np.ndarray]] = []
g_ba_device: Optional[torch.device] = None

def ba_residual_function(params_vec: np.ndarray) -> np.ndarray:
    """
    (核心) 捆绑平差的残差函数，用于 scipy.optimize.least_squares。
    """
    global g_ba_images, g_ba_tie_points, g_ba_device
    num_images = len(g_ba_images)
    
    # 1. 解包参数矢量
    # params_vec 包含 (N-1) * 6 个参数 (R[0,0], R[0,1], R[1,0], R[1,1], T[0], T[1])
    try:
        params = params_vec.reshape((num_images - 1, 6))
    except ValueError:
        print(f"错误: params_vec 形状不匹配。收到 {params_vec.shape}, 期望 {(num_images - 1) * 6}")
        return np.array([1e10]) # 返回一个巨大的误差

    all_residuals = []
    
    # 2. (关键) 将 *当前* 的仿射参数 *设置* 到RPC模型中
    #    注意：我们必须在每次函数调用时都这样做，因为scipy是无状态的
    original_states = [img.rpc.adjust_params.clone() for img in g_ba_images] # 保存原始状态
    
    try:
        # Image 0 (锚点) 保持不变 (单位阵)
        for k in range(num_images - 1):
            img_idx = k + 1 # 影响 images[1]...images[N-1]
            R_data = [params[k, 0], params[k, 1], params[k, 2], params[k, 3]]
            T_data = [params[k, 4], params[k, 5]]
            
            # (新增) 增加一个小的正则化，防止R退化
            R = torch.tensor([[R_data[0], R_data[1]], [R_data[2], R_data[3]]], device=g_ba_device)
            T = torch.tensor([T_data[0], T_data[1]], device=g_ba_device)
            
            A = torch.cat([R, T.unsqueeze(-1)], dim=-1)
            # 使用我们新增的 Set_Adjust
            g_ba_images[img_idx].rpc.Set_Adjust(A)

        # 3. 遍历所有同名点对，计算重投影误差
        #    (RPC计算在GPU上进行，非常快)
        with torch.no_grad():
            for (i, j, full_coords_i, full_coords_j) in g_ba_tie_points:
                
                # (N, 2) [samp, line]
                sl_i = torch.from_numpy(full_coords_i).to(g_ba_device)
                sl_j = torch.from_numpy(full_coords_j).to(g_ba_device)
                
                # --- 计算对称误差 (i -> j 和 j -> i) ---
                
                # (1) 投影 i -> j
                # 插值DEM高程 (DEM在__init__已加载)
                # dem_interp 需要 [samp, line]
                h_i = torch.from_numpy(g_ba_images[i].dem_interp(sl_i.cpu().numpy())).to(g_ba_device) 
                # 像 -> 地 (使用 image i 的 *已调整* RPC)
                lat_i, lon_i = g_ba_images[i].rpc.RPC_PHOTO2OBJ(sl_i[:, 0], sl_i[:, 1], h_i)
                # 地 -> 像 (使用 image j 的 *已调整* RPC)
                samp_j_pred, line_j_pred = g_ba_images[j].rpc.RPC_OBJ2PHOTO(lat_i, lon_i, h_i)
                
                # (2) 投影 j -> i
                h_j = torch.from_numpy(g_ba_images[j].dem_interp(sl_j.cpu().numpy())).to(g_ba_device)
                lat_j, lon_j = g_ba_images[j].rpc.RPC_PHOTO2OBJ(sl_j[:, 0], sl_j[:, 1], h_j)
                samp_i_pred, line_i_pred = g_ba_images[i].rpc.RPC_OBJ2PHOTO(lat_j, lon_j, h_j)

                # (3) 计算残差 (像素)
                # 误差 = 预测值 - 观测值
                res_i = torch.stack([samp_i_pred - sl_i[:, 0], line_i_pred - sl_i[:, 1]], dim=-1)
                res_j = torch.stack([samp_j_pred - sl_j[:, 0], line_j_pred - sl_j[:, 1]], dim=-1)
                
                all_residuals.append(res_i.cpu().numpy())
                all_residuals.append(res_j.cpu().numpy())

    except Exception as e:
        print(f"错误: 在残差计算中发生异常: {e}")
        # 恢复状态
        for k_img, state in enumerate(original_states):
            g_ba_images[k_img].rpc.Set_Adjust(state)
        return np.array([1e10]) # 返回一个巨大的误差
        
    finally:
        # 4. (关键) 恢复RPC到进入函数时的状态
        for k_img, state in enumerate(original_states):
            g_ba_images[k_img].rpc.Set_Adjust(state)

    if not all_residuals:
        return np.array([0.0]) # 没有点
        
    return np.concatenate(all_residuals, axis=0).ravel() # 返回1D残差向量


# -----------------------------------------------------------------
# 阶段 3: 主函数
# -----------------------------------------------------------------

def main(args):
    global g_ba_images, g_ba_tie_points, g_ba_device
    
    start_time_total = time.time()
    
    # 0. 设置BA计算设备
    g_ba_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"主进程 {os.getpid()}: 使用 {g_ba_device} 进行BA残差计算。")
    
    # 1. 加载数据元数据
    print("加载影像元数据 (RPCs, DEMs)...")
    # (修改) 将 args 传入 load_imgs_bundle
    g_ba_images = load_imgs_bundle(args) 
    if len(g_ba_images) < 2:
        print("错误: 至少需要2景影像才能进行平差。")
        return
    
    # (修改) 确保所有影像的RPC都在BA设备上
    for img in g_ba_images:
        img.rpc.to_gpu(g_ba_device)

    # 2. 初始精度报告 (复用)
    print("\n" + "="*50)
    print("--- 初始精度报告 (基于 Validation Tie Points) ---")
    overlapping_pairs = find_overlapping_pairs(g_ba_images)
    all_errors_init = check_all_pairs_error(g_ba_images, overlapping_pairs)
    if len(all_errors_init) > 0 and all_errors_init.mean() != 0.0:
        print("\n--- 初始全局误差 (Validation) 总结 ---")
        print(f"总点数: {len(all_errors_init)}")
        print(f"Mean Error:   {all_errors_init.mean():.4f} m")
        print(f"Median Error: {np.median(all_errors_init):.4f} m")
        print(f"RMSE:         {np.sqrt(np.mean(all_errors_init**2)):.4f} m")
    else:
        print("未找到用于验证的同名点。")
    print("="*50 + "\n")

    # 3. 准备同名点提取任务
    print("\n--- 阶段 1: 提取同名点 (LoFTR) ---")
    
    # 复用格网生成
    print("查找重叠格网...")
    all_corners = np.stack([img.corner_xys for img in g_ba_images], axis=0)
    all_common_diags = find_grids(all_corners, args.window_size, 
                                  offset_x=args.grid_offset_x, 
                                  offset_y=args.grid_offset_y)
    
    # 筛选格网
    if args.grid_num > 0 and len(all_common_diags) > args.grid_num:
        print(f"找到 {len(all_common_diags)} 个格网, 随机采样 {args.grid_num} 个。")
        selected_diags = random.sample(all_common_diags, args.grid_num)
    else:
        selected_diags = all_common_diags
        print(f"使用所有 {len(selected_diags)} 个格网。")

    # 创建任务列表 (格网, 影像i, 影像j)
    tasks = []
    # (修改) 我们需要真实索引，而不是列表索引
    pair_set = set(overlapping_pairs) # (i, j)
    # 创建一个 image_id -> RSImage 对象的映射
    images_map = {img.id: img for img in g_ba_images}
    
    for diag in selected_diags:
        # (简化逻辑：这里可以优化为只检查真正覆盖格网的影像)
        for (i, j) in pair_set:
            if i in images_map and j in images_map:
                 tasks.append((diag, i, j))
    print(f"共创建 {len(tasks)} 个 (格网, 像对) 提取任务。")
    
    if not tasks:
        print("错误: 未创建任何提取任务。检查影像是否重叠。")
        return

    # 4. 执行同名点提取 (并行)
    #    注意：LoFTR需要GPU，'spawn'启动方式是必须的
    try:
        mp.set_start_method('spawn', force=True) 
    except RuntimeError as e:
        print(f"警告: mp.set_start_method('spawn') 失败: {e}。可能已设置。")
        
    num_workers = min(max(1, args.num_workers), os.cpu_count()) # 限制
    
    tie_point_results = []
    start_time_loftr = time.time()
    print(f"使用 {num_workers} 个进程并行提取特征...")
    
    # (修改) 使用 initializer 来传递全局变量
    init_args = (g_ba_images, args.patch_size, args.conf_threshold)
    
    with mp.Pool(processes=num_workers, initializer=init_worker, initargs=init_args) as pool:
        # 使用 imap_unordered 来获取带tqdm的进度条
        for result in tqdm(pool.imap_unordered(process_task_worker, tasks), total=len(tasks), desc="提取同名点"):
            if result is not None:
                tie_point_results.append(result)

    # 整理结果到全局变量
    g_ba_tie_points = tie_point_results
    total_points = sum(len(tp[2]) for tp in g_ba_tie_points)
    elapsed_loftr = time.time() - start_time_loftr
    print(f"同名点提取完毕 (耗时: {format_time(elapsed_loftr)})。")
    print(f"共 {len(g_ba_tie_points)} 个连接 (来自不同格网/像对)，总计 {total_points} 个同名点。")

    if total_points == 0:
        print("错误：未能提取到任何同名点。退出。")
        return
        
    # 清理 LoFTR 显存 (工作进程已退出，显存自动释放)
    torch.cuda.empty_cache()

    # 5. 执行捆绑平差 (BA)
    print("\n--- 阶段 2: 运行稀疏捆绑平差 (SciPy) ---")
    start_time_ba = time.time()
    
    num_images = len(g_ba_images)
    num_params = (num_images - 1) * 6 # 6个仿射参数 (R:4, T:2)
    
    # 初始参数 (单位矩阵)
    # (R[0,0], R[0,1], R[1,0], R[1,1], T[0], T[1])
    init_param_single = np.array([1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    params_init = np.tile(init_param_single, num_images - 1)

    # 计算初始残差 (RMSE)
    print("计算初始残差 (基于 LoFTR 同名点)...")
    residuals_init = ba_residual_function(params_init)
    rmse_init = np.sqrt(np.mean(residuals_init**2))
    print(f"初始 RMSE (像素): {rmse_init:.4f}")

    # 运行解算器
    print("启动非线性最小二乘解算器 (scipy.optimize.least_squares)...")
    # loss='soft_l1' (鲁棒损失) 对LoFTR的离群点很有效
    # jac='2-point' (数值法) 避免了复杂的雅可比矩阵推导
    result = least_squares(
        ba_residual_function,
        params_init,
        jac='2-point',     # 使用2点法数值计算雅可比矩阵
        method='trf',      # (Trust Region Reflective) 适用于大规模稀疏问题
        loss='soft_l1',    # 鲁棒损失函数，减少离群点影响
        f_scale=1.0,       # 损失尺度 (像素)
        verbose=2          # 打印详细迭代过程 (0:无, 1:终止, 2:详细)
    )

    elapsed_ba = time.time() - start_time_ba
    print(f"\n解算完成 (耗时: {format_time(elapsed_ba)})。")

    # 计算最终残差 (RMSE)
    print("计算最终残差 (基于 LoFTR 同名点)...")
    residuals_final = result.fun # .fun 存储了最终的残差向量
    rmse_final = np.sqrt(np.mean(residuals_final**2))
    print(f"最终 RMSE (像素): {rmse_final:.4f} (从 {rmse_init:.4f})")

    # 6. 应用最终结果并生成报告
    print("\n--- 阶段 3: 应用调整并生成最终精度报告 ---")
    
    # (重要) 将解算出的*最佳*参数永久性地设置到RPC模型中
    best_params = result.x.reshape((num_images - 1, 6))
    for k in range(num_images - 1):
        img_idx = k + 1 # 影像 0 是锚点
        
        R_data = [best_params[k, 0], best_params[k, 1], best_params[k, 2], best_params[k, 3]]
        T_data = [best_params[k, 4], best_params[k, 5]]
            
        R = torch.tensor([[R_data[0], R_data[1]], [R_data[2], R_data[3]]], device=g_ba_device)
        T = torch.tensor([T_data[0], T_data[1]], device=g_ba_device)
        
        A = torch.cat([R, T.unsqueeze(-1)], dim=-1)
        
        # (修改) 找到正确的 RSImage 对象
        # g_ba_images[img_idx] 不一定是正确的，因为 g_ba_images[0] 的 id 可能是 0，也可能是其他
        # 我们必须按列表顺序应用
        g_ba_images[img_idx].rpc.Set_Adjust(A)
        print(f"应用最终仿射矩阵到影像 {g_ba_images[img_idx].id} (列表索引 {img_idx}):")
        print(A.cpu().numpy())


    # 7. 最终精度报告 (复用)
    #    (使用 validation tie points)
    print("\n" + "="*50)
    print("--- 最终精度报告 (基于 Validation Tie Points) ---")
    all_errors_final = check_all_pairs_error(g_ba_images, overlapping_pairs)
    
    if len(all_errors_final) > 0 and all_errors_final.mean() != 0.0:
        print("\n--- 最终全局误差 (Validation) 总结 ---")
        print(f"总点数: {len(all_errors_final)}")
        print(f"Mean Error:   {all_errors_final.mean():.4f} m")
        print(f"Median Error: {np.median(all_errors_final.mean()):.4f} m")
        print(f"RMSE:         {np.sqrt(np.mean(all_errors_final**2)):.4f} m")
    else:
        print("未找到用于验证的同名点。")
    print("="*50 + "\n")
    
    elapsed_total = time.time() - start_time_total
    print(f"Baseline 平差总耗时: {format_time(elapsed_total)}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="稀疏光束法平差 (LoFTR + SciPy) Baseline")
    
    # 复用 adjust_test_1023.py 的参数
    parser.add_argument('--root', type=str, required=True,
                        help='包含 "adjust_images" 文件夹的根路径')
    parser.add_argument('--window_size', type=int, default=2000,
                        help='用于查找重叠区域的地理格网大小 (米)')
    parser.add_argument('--select_imgs', type=str, default='0,1',
                        help='要处理的影像索引，以逗号分隔 (例如 "0,1,3")') 
    parser.add_argument('--grid_offset_x', type=float, default=0,
                        help='格网X方向偏移')
    parser.add_argument('--grid_offset_y', type=float, default=0,
                        help='格网Y方向偏移')
    parser.add_argument('--grid_num', type=int, default=20,
                        help="要随机采样的格网数量 (0 = 使用所有格网)")
    
    # Baseline 特有参数
    parser.add_argument('--patch_size', type=int, default=512,
                        help='LoFTR 匹配时重采样的影像块大小 (像素)')
    parser.add_argument('--conf_threshold', type=float, default=0.7,
                        help='LoFTR 匹配的置信度阈值')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='用于并行提取特征的工作进程数')
    
    args = parser.parse_args()
    
    main(args)
