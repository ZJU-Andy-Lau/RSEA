import os
import argparse
import itertools
import torch
import numpy as np
import cv2
from typing import List, Tuple, Dict
from tqdm import tqdm

# 导入 LoFTR (kornia)
try:
    from kornia.feature import LoFTR
except ImportError:
    print("="*50)
    print("错误: 未找到 'kornia' 库。")
    print("请通过 'pip install kornia' 安装 LoFTR。")
    print("="*50)
    exit()

# 导入辅助文件
from rs_image_1022 import RSImage
from rpc import RPCModelParameterTorch
from utils import find_grids, project_mercator, mercator2lonlat, bilinear_interpolate, resample_from_quad
from bba_helpers import load_imgs_bundle, find_overlapping_pairs, check_all_pairs_error

# 设置计算设备
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.set_default_dtype(torch.double) # BBA 需要双精度

class TraditionalBundleAdjuster:
    """
    使用传统法方程（高斯-牛顿法）和舒尔补（Schur Complement）
    实现的光束法区域网平差（BBA）解算器。
    """
    def __init__(self, images: List[RSImage], loftr_ckpt_path: str):
        self.images = images
        self.num_images = len(images)
        self.loftr = LoFTR(pretrained=None).to(DEVICE,dtype=torch.float).eval()
        try:
            self.loftr.load_state_dict(torch.load(loftr_ckpt_path)['state_dict'])
            print(f"成功加载 LoFTR 模型: {loftr_ckpt_path}")
        except Exception as e:
            print(f"错误: 无法加载 LoFTR 模型: {e}")
            print("请检查路径，或使用 'default' 来下载预训练模型。")
            if loftr_ckpt_path.lower() == 'default':
                 self.loftr = LoFTR(pretrained='outdoor').to(DEVICE).eval()
                 print("已加载 kornia 默认的 'outdoor' LoFTR 模型。")
            else:
                exit()

        self.matches = []
        self.observations = []
        self.tp_id_to_idx = {}
        self.img_id_to_idx = {}
        self.num_pts = 0
        self.num_img_params = 6 * (self.num_images - 1)

        self.ground_points = None # (N_pts, 3) [Lat, Lon, H]
        self.image_affines = None # (N_img-1, 6) [a1,a2,a0, b1,b2,b0]

    def _find_overlapping_grids(self, window_size=1000.0, offset_x=0, offset_y=0):
        """ 1. 查找所有重叠格网 """
        print("Step 1: 查找重叠格网...")
        all_corners = np.stack([img.corner_xys for img in self.images], axis=0)
        return find_grids(all_corners, window_size, offset_x, offset_y)

    @torch.no_grad()
    def _extract_tie_points(self, grids: List[np.ndarray]):
        """ 2. 使用LoFTR在格网上提取同名点 """
        print("Step 2: 提取同名点 (Tie Points)...")
        
        loftr_res = 832 # LoFTR 期望的输入分辨率
        
        for diag in tqdm(grids, desc="Matching Grids"):
            diag_tl, diag_br = diag[0], diag[1]
            overlapping_imgs = []
            
            # 找到所有覆盖此格网的影像
            for img in self.images:
                corners_geo = np.array([
                    diag_tl, [diag_br[0], diag_tl[1]],
                    diag_br, [diag_tl[0], diag_br[1]]
                ])
                # corners_samp 已经是 [samp, line] 格式
                corners_samp = img.xy_to_sampline(corners_geo)
                
                if (corners_samp.min() < 0 or 
                    corners_samp[:, 0].max() > img.W or 
                    corners_samp[:, 1].max() > img.H):
                    continue
                
                # resample_image_by_sampline 期望 [line, samp] 格式的角点
                img_patch = img.resample_image_by_sampline(corners_samp[:, [1, 0]], (loftr_res, loftr_res), need_local=False)
                overlapping_imgs.append({
                    'img_id': img.id,
                    'patch_gray': torch.from_numpy(cv2.cvtColor(img_patch, cv2.COLOR_BGR2GRAY)).float().to(DEVICE)[None, None] / 255.0,
                    'corners_samp': corners_samp # 存储 [samp, line] 格式的角点
                })
            
            # 在所有重叠对上运行LoFTR
            for pair in itertools.combinations(overlapping_imgs, 2):
                img_i_data = pair[0]
                img_j_data = pair[1]
                
                batch = {'image0': img_i_data['patch_gray'], 'image1': img_j_data['patch_gray']}
                with torch.no_grad():
                    results = self.loftr(batch)
                
                # mkpts_i 是 (N, 2) [x, y] -> [samp, line]
                mkpts_i = results['keypoints0'].cpu().numpy() 
                mkpts_j = results['keypoints1'].cpu().numpy()
                conf = results['confidence'].cpu().numpy()

                print(len(mkpts_i),conf.mean(),conf.max().conf.min())
                
                if len(mkpts_i) == 0:
                    continue
                
                # 筛选高置信度的点
                valid = conf > 0.5
                mkpts_i, mkpts_j = mkpts_i[valid], mkpts_j[valid]
                
                if len(mkpts_i) < 10:
                    continue

                # --- [修改开始] ---
                
                # 1. (已删除) 不再需要翻转 LoFTR 的输出
                # mkpts_i_ls = mkpts_i[:, [1, 0]] <-- 错误, 已删除
                # mkpts_j_ls = mkpts_j[:, [1, 0]] <-- 错误, 已删除
                
                # 2. 计算变换矩阵 M
                # cv2.getPerspectiveTransform 期望 src 和 dst 都是 (x, y) -> [samp, line]
                
                # src 角点 (LoFTR patch) [samp, line]
                src_cv2_corners = np.array([[0,0], [loftr_res-1,0], [loftr_res-1,loftr_res-1], [0,loftr_res-1]], dtype=np.float32)
                
                # dst 角点 (原始影像) [samp, line]
                dst_i_cv2_corners = img_i_data['corners_samp'].astype(np.float32)
                dst_j_cv2_corners = img_j_data['corners_samp'].astype(np.float32)

                M_i = cv2.getPerspectiveTransform(src_cv2_corners, dst_i_cv2_corners)
                M_j = cv2.getPerspectiveTransform(src_cv2_corners, dst_j_cv2_corners)
                
                # 3. 应用变换
                # cv2.perspectiveTransform 期望输入 (N, 1, 2) 且为 (x, y) -> [samp, line]
                # mkpts_i 已经是 (N, 2) [samp, line]
                
                # 添加一个维度 (N, 2) -> (N, 1, 2)
                mkpts_i_cv2 = mkpts_i[:, None, :].astype(np.float32)
                mkpts_j_cv2 = mkpts_j[:, None, :].astype(np.float32)

                # full_pts_i_sl 的格式是 (N, 2) [samp, line]
                full_pts_i_sl = cv2.perspectiveTransform(mkpts_i_cv2, M_i).squeeze(1)
                full_pts_j_sl = cv2.perspectiveTransform(mkpts_j_cv2, M_j).squeeze(1)

                # 4. 存储匹配
                # 后续代码期望 (line, samp) 格式
                for k in range(len(full_pts_i_sl)):
                    pt_i_sl = full_pts_i_sl[k] # [samp, line]
                    pt_j_sl = full_pts_j_sl[k] # [samp, line]
                    
                    # 存储为 (line, samp)
                    self.matches.append(
                        (img_i_data['img_id'], (pt_i_sl[1], pt_i_sl[0]), 
                            img_j_data['img_id'], (pt_j_sl[1], pt_j_sl[0]))
                    )
                # --- [修改结束] ---
                    
            
        print(f"提取了 {len(self.matches)} 个两两匹配对。")

    def _build_connected_components(self):
        """ 3. 构建全局连接点 """
        print("Step 3: 构建全局连接点...")
        
        point_to_tpid = {}  # Dict[Tuple(img_id, l_hash, s_hash), tp_id]
        global_tie_points = {} # Dict[tp_id, List[Tuple(img_id, line, samp)]]
        next_tp_id = 0
        
        def hash_coord(c):
            # 将浮点坐标哈希，以便在字典中查找
            return int(round(c * 10)) 

        for (img_id_i, (l_i, s_i), img_id_j, (l_j, s_j)) in self.matches:
            
            key_i = (img_id_i, hash_coord(l_i), hash_coord(s_i))
            key_j = (img_id_j, hash_coord(l_j), hash_coord(s_j))
            
            id_i = point_to_tpid.get(key_i)
            id_j = point_to_tpid.get(key_j)
            
            if id_i is None and id_j is None:
                # Case 1: 新点
                new_id = next_tp_id
                global_tie_points[new_id] = [
                    (img_id_i, l_i, s_i), (img_id_j, l_j, s_j)
                ]
                point_to_tpid[key_i] = new_id
                point_to_tpid[key_j] = new_id
                next_tp_id += 1
            elif id_i is not None and id_j is None:
                # Case 2: 扩展点 i
                global_tie_points[id_i].append((img_id_j, l_j, s_j))
                point_to_tpid[key_j] = id_i
            elif id_i is None and id_j is not None:
                # Case 3: 扩展点 j
                global_tie_points[id_j].append((img_id_i, l_i, s_i))
                point_to_tpid[key_i] = id_j
            elif id_i != id_j:
                # Case 4: 合并点
                id_keep, id_remove = min(id_i, id_j), max(id_i, id_j)
                for obs in global_tie_points[id_remove]:
                    global_tie_points[id_keep].append(obs)
                    key_obs = (obs[0], hash_coord(obs[1]), hash_coord(obs[2]))
                    point_to_tpid[key_obs] = id_keep
                del global_tie_points[id_remove]
        
        # 筛选，只保留至少在2张影像上观测到的点
        final_global_tie_points = {
            tp_id: obs for tp_id, obs in global_tie_points.items() if len(obs) >= 2
        }
        
        # 构建最终的 BBA 观测列表和索引
        self.tp_id_to_idx = {tp_id: idx for idx, tp_id in enumerate(final_global_tie_points.keys())}
        self.img_id_to_idx = {img.id: i for i, img in enumerate(self.images) if img.id != 0}
        self.num_pts = len(self.tp_id_to_idx)
        
        for tp_id, obs_list in final_global_tie_points.items():
            for (img_id, line, samp) in obs_list:
                self.observations.append((img_id, tp_id, line, samp))
        
        print(f"构建了 {self.num_pts} 个全局连接点。")
        print(f"总观测数: {len(self.observations)}。")
        print(f"待解算影像参数: {self.num_img_params} (来自 {self.num_images - 1} 张影像)。")
        print(f"待解算物方点参数: {self.num_pts * 3}。")

    def _initialize_unknowns(self):
        """ 4. 为所有未知数设置初值 """
        print("Step 4: 初始化未知数初值...")
        
        # 1. 影像仿射参数
        self.image_affines = torch.tensor(
            [[1., 0., 0., 0., 1., 0.] for _ in range(self.num_images - 1)], 
            dtype=torch.double, device=DEVICE
        )
        
        # 2. 物方点三维坐标
        self.ground_points = torch.zeros((self.num_pts, 3), dtype=torch.double, device=DEVICE)
        
        # 记录哪些点已初始化
        initialized_mask = [False] * self.num_pts
        
        with torch.no_grad():
            for (img_id, tp_id, line, samp) in self.observations:
                j_idx = self.tp_id_to_idx[tp_id]
                if initialized_mask[j_idx]:
                    continue
                
                img = self.images[img_id]
                img.rpc.to_gpu(DEVICE) # 确保RPC在GPU上
                
                try:
                    # 使用dem_interp获取高程
                    h = img.dem_interp(np.array([[samp, line]]))
                    h_tensor = torch.tensor(h, dtype=torch.double, device=DEVICE)
                    
                    # 反算 [Lat, Lon]
                    lat, lon = img.rpc.RPC_PHOTO2OBJ(
                        torch.tensor(samp, dtype=torch.double, device=DEVICE),
                        torch.tensor(line, dtype=torch.double, device=DEVICE),
                        h_tensor
                    )
                    
                    self.ground_points[j_idx, 0] = lat
                    self.ground_points[j_idx, 1] = lon
                    self.ground_points[j_idx, 2] = h_tensor
                    initialized_mask[j_idx] = True
                    
                except Exception as e:
                    print(f"Warning: 初始化点 {tp_id} 失败: {e}")
                    
        if not all(initialized_mask):
            print("Warning: 部分连接点未能成功初始化坐标。")

    # --- 核心投影函数 ---
    
    def _project_raw(self, rpc: RPCModelParameterTorch, lat: torch.Tensor, lon: torch.Tensor, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        核心投影逻辑：(Lat, Lon, H) -> (line_raw, samp_raw)
        [重要] 此函数 *不* 应用RPC对象内部的仿射修正。
        """
        lat_norm = (lat - rpc.LAT_OFF) / rpc.LAT_SCALE
        lon_norm = (lon - rpc.LONG_OFF) / rpc.LONG_SCALE
        h_norm = (h - rpc.HEIGHT_OFF) / rpc.HEIGHT_SCALE
        
        coef = rpc.RPC_PLH_COEF(lat_norm.unsqueeze(0), lon_norm.unsqueeze(0), h_norm.unsqueeze(0)).squeeze(0)
        
        samp_norm = torch.sum(coef * rpc.SNUM) / torch.sum(coef * rpc.SDEM)
        line_norm = torch.sum(coef * rpc.LNUM) / torch.sum(coef * rpc.LDEM)
        
        samp_raw = samp_norm * rpc.SAMP_SCALE + rpc.SAMP_OFF
        line_raw = line_norm * rpc.LINE_SCALE + rpc.LINE_OFF
        
        return line_raw, samp_raw

    def _apply_affine(self, raw_coords: torch.Tensor, affine_params: torch.Tensor) -> torch.Tensor:
        """
        应用 6 参数仿射变换： (line_raw, samp_raw) -> (line_pred, samp_pred)
        affine_params: [a1, a2, a0, b1, b2, b0]
        """
        line_raw, samp_raw = raw_coords
        
        # 对应于 [l, s, 1] @ [[a1, b1], [a2, b2], [a0, b0]]
        line_pred = affine_params[0] * line_raw + affine_params[1] * samp_raw + affine_params[2]
        samp_pred = affine_params[3] * line_raw + affine_params[4] * samp_raw + affine_params[5]
        
        return torch.stack([line_pred, samp_pred])

    def _get_jacobian_and_residual(self, P_j: torch.Tensor, A_i: torch.Tensor, rpc_i: RPCModelParameterTorch, obs_coord: torch.Tensor, is_anchor: bool):
        """
        计算单次观测的雅可比矩阵 J 和残差 l。
        """
        
        # 1. 定义局部投影函数（用于自动求导）
        def full_projection_on_P(p):
            raw = self._project_raw(rpc_i, p[0], p[1], p[2])
            return self._apply_affine(raw, A_i)
            
        def full_projection_on_A(a):
            # P_j 在这里是常量
            raw = self._project_raw(rpc_i, P_j[0], P_j[1], P_j[2])
            return self._apply_affine(raw, a)

        # 2. 计算预测值和残差
        pred_coord = full_projection_on_P(P_j)
        l_k = obs_coord - pred_coord # (2,)
        
        # 3. 计算雅可比矩阵
        # J_k_Pj: d(pred) / d(P) = d(pred) / d(raw) * d(raw) / d(P)
        J_k_Pj = torch.autograd.functional.jacobian(full_projection_on_P, P_j) # (2, 3)
        
        J_k_Ai = None
        if not is_anchor:
            # J_k_Ai: d(pred) / d(A)
            J_k_Ai = torch.autograd.functional.jacobian(full_projection_on_A, A_i) # (2, 6)
            
        return l_k, J_k_Pj, J_k_Ai

    def run_adjustment(self, max_iter=20, lambda_damping=1e-4, convergence_threshold=1e-6):
        """ 5. 运行BBA解算 """
        print("Step 5: 开始BBA迭代解算...")
        
        # 锚点影像 (img_id == 0) 的固定仿射参数
        anchor_affine = torch.tensor([1., 0., 0., 0., 1., 0.], dtype=torch.double, device=DEVICE)
        
        for iter in range(max_iter):
            
            # --- A. 初始化法方程组件 ---
            # V 和 C_A 直接构建为稠密
            V = torch.zeros((self.num_img_params, self.num_img_params), dtype=torch.double, device=DEVICE)
            C_A = torch.zeros((self.num_img_params, 1), dtype=torch.double, device=DEVICE)
            
            # U, W, C_P 是稀疏的，按点存储
            U_list = [torch.zeros((3, 3), dtype=torch.double, device=DEVICE) for _ in range(self.num_pts)]
            W_list = [[] for _ in range(self.num_pts)] # List[List[Tuple(i_idx, W_ji(3x6))]]
            Cp_list = [torch.zeros((3, 1), dtype=torch.double, device=DEVICE) for _ in range(self.num_pts)]
            
            total_error = 0.0

            # --- B. 遍历观测，构建 N 和 C ---
            print(f"Iter {iter+1}/{max_iter}: 正在构建法方程...")
            pbar = tqdm(self.observations)
            for (img_id, tp_id, obs_line, obs_samp) in pbar:
                
                j_idx = self.tp_id_to_idx[tp_id]
                i_idx = self.img_id_to_idx.get(img_id) # if img_id==0, i_idx is None
                
                P_j = self.ground_points[j_idx].clone().requires_grad_(True)
                rpc_i = self.images[img_id].rpc
                obs_coord = torch.tensor([obs_line, obs_samp], dtype=torch.double, device=DEVICE)
                
                is_anchor = (i_idx is None)
                A_i = anchor_affine if is_anchor else self.image_affines[i_idx].clone().requires_grad_(True)
                
                try:
                    l_k, J_k_Pj, J_k_Ai = self._get_jacobian_and_residual(P_j, A_i, rpc_i, obs_coord, is_anchor)
                except Exception as e:
                    print(f"Warning: 雅可比计算失败 for pt {tp_id} on img {img_id}. {e}")
                    continue

                J_P_T = J_k_Pj.T # (3, 2)
                
                # 累加 U 和 C_P
                U_list[j_idx] += J_P_T @ J_k_Pj
                Cp_list[j_idx] += J_P_T @ l_k.view(-1, 1)
                
                if not is_anchor:
                    J_A_T = J_k_Ai.T # (6, 2)
                    i_start, i_end = 6 * i_idx, 6 * (i_idx + 1)
                    
                    # 累加 V 和 C_A
                    V[i_start:i_end, i_start:i_end] += J_A_T @ J_k_Ai
                    C_A[i_start:i_end] += J_A_T @ l_k.view(-1, 1)
                    
                    # 存储 W 块
                    W_ji = J_P_T @ J_k_Ai # (3, 6)
                    W_list[j_idx].append((i_idx, W_ji))
                
                total_error += torch.dot(l_k, l_k)
            
            if len(self.observations) == 0:
                print("错误：观测列表为空，无法计算误差。")
                break
            
            print(f"Iter {iter+1}: 总误差 (RMSE): {torch.sqrt(total_error / len(self.observations)):.4f} 像素")

            # --- C. 解算舒尔补系统 ---
            print(f"Iter {iter+1}: 正在解算舒尔补...")
            
            N_reduced = V + torch.eye(self.num_img_params, device=DEVICE) * lambda_damping
            C_reduced = C_A.clone()
            
            U_inv_list = [None] * self.num_pts
            U_inv_Cp_list = [None] * self.num_pts

            # 1. 计算 U 的逆 和 U_inv * C_P
            for j in range(self.num_pts):
                U_j_damped = U_list[j] + torch.eye(3, device=DEVICE) * lambda_damping
                try:
                    U_j_inv = torch.linalg.inv(U_j_damped)
                    U_inv_list[j] = U_j_inv
                    U_inv_Cp_list[j] = U_j_inv @ Cp_list[j] # (3, 1)
                except torch.linalg.LinAlgError:
                    print(f"Warning: 点 {j} 的 U 矩阵奇异。")
                    U_inv_list[j] = torch.zeros((3, 3), device=DEVICE) # 贡献为0
                    U_inv_Cp_list[j] = torch.zeros((3, 1), device=DEVICE)
            
            # 2. 计算 N_reduced 和 C_reduced
            for j in range(self.num_pts):
                U_j_inv = U_inv_list[j]
                U_inv_Cp_j = U_inv_Cp_list[j]
                W_j_blocks = W_list[j] # List[Tuple(i_idx, W_ji(3x6))]
                
                for (i_idx_a, W_ji_a) in W_j_blocks:
                    i_start_a, i_end_a = 6*i_idx_a, 6*(i_idx_a+1)
                    W_ji_a_T = W_ji_a.T # (6, 3)
                    
                    W_T_U_inv = W_ji_a_T @ U_j_inv # (6, 3)
                    
                    # C_reduced = C_A - W^T * U_inv * C_P
                    C_reduced[i_start_a:i_end_a] -= W_T_U_inv @ Cp_list[j]
                    
                    # N_reduced = V - W_a^T * U_inv * W_b
                    for (i_idx_b, W_ji_b) in W_j_blocks:
                        i_start_b, i_end_b = 6*i_idx_b, 6*(i_idx_b+1)
                        N_reduced[i_start_a:i_end_a, i_start_b:i_end_b] -= W_T_U_inv @ W_ji_b

            # --- D. 求解线性方程 ---
            try:
                delta_A = torch.linalg.solve(N_reduced, C_reduced) # (N_img_params, 1)
            except torch.linalg.LinAlgError:
                print(f"Iter {iter+1}: 错误: 缩减后的法方程 (N_reduced) 奇异。平差失败。")
                break
            
            # --- E. 反向代入求解 delta_P ---
            delta_P = torch.zeros((self.num_pts, 3), dtype=torch.double, device=DEVICE)
            for j in range(self.num_pts):
                W_j_delta_A = torch.zeros((3, 1), dtype=torch.double, device=DEVICE)
                for (i_idx, W_ji) in W_list[j]:
                    i_start, i_end = 6 * i_idx, 6 * (i_idx + 1)
                    W_j_delta_A += W_ji @ delta_A[i_start:i_end]
                
                # delta_P = U_inv * (C_P - W * delta_A)
                delta_P[j] = (U_inv_Cp_list[j] - U_inv_list[j] @ W_j_delta_A).squeeze()

            # --- F. 更新参数 & 检查收敛 ---
            with torch.no_grad():
                self.ground_points += delta_P
                self.image_affines += delta_A.view(self.num_images - 1, 6)
            
            delta_norm = torch.norm(delta_A) + torch.norm(delta_P)
            print(f"Iter {iter+1}: 改正数范数 (Delta Norm): {delta_norm.item()}")
            if delta_norm < convergence_threshold:
                print(f"收敛于第 {iter+1} 次迭代。")
                break
        
        print("BBA 迭代解算完成。")

    def _bake_results(self):
        """ 6. 将解算的仿射参数烘焙回 RPC 对象 """
        print("Step 6: 烘焙平差结果到 RPC 模型...")
        with torch.no_grad():
            for img_id, i_idx in self.img_id_to_idx.items():
                final_affine_params = self.image_affines[i_idx].cpu() # 移至CPU
                
                # A_matrix (2, 3)
                A_matrix = torch.tensor([
                    [final_affine_params[0], final_affine_params[1], final_affine_params[2]],
                    [final_affine_params[3], final_affine_params[4], final_affine_params[5]]
                ], dtype=torch.double)
                
                # Update_Adjust 期望 (2, 3) 矩阵
                self.images[img_id].rpc.Update_Adjust(A_matrix)
        print("烘焙完成。")

    def run(self, window_size=1000.0, max_iter=20):
        """ 运行完整的BBA流程 """
        grids = self._find_overlapping_grids(window_size=window_size)
        
        # 检查 grids 是否为空。
        # 使用 len(grids) == 0 同时兼容 list 和 ndarray。
        if len(grids) == 0:
            print("未找到重叠格网，无法提取连接点。")
            return
            
        self._extract_tie_points(grids)
        if len(self.matches) == 0:
            print("未提取到连接点，平差中止。")
            return
            
        self._build_connected_components()
        if self.num_pts == 0:
            print("未构建任何全局连接点，平差中止。")
            return
            
        self._initialize_unknowns()
        self.run_adjustment(max_iter=max_iter)
        self._bake_results()

    def check_accuracy(self, overlapping_pairs: List[Tuple[int, int]]):
        """ 7. 使用独立的检查点评定精度 """
        print("Step 7: 评定最终精度 (使用 check_points)...")
        check_all_pairs_error(self.images, overlapping_pairs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="传统光束法区域网平差 (BBA)")
    
    parser.add_argument('--root', type=str, required=True,
                        help='包含 "adjust_images" 文件夹的根目录')
    
    parser.add_argument('--select_imgs', type=str, required=True,
                        help='要平差的影像索引，以逗号分隔, e.g., "0,1,2"')
    
    parser.add_argument('--loftr_path', type=str, required=True,
                        help='LoFTR 预训练权重(.ckpt)的路径，或输入 "default" 使用kornia预训练模型')
    
    parser.add_argument('--window_size', type=float, default=1000.0,
                        help='用于查找重叠区域的格网大小 (米)')

    parser.add_argument('--max_iter', type=int, default=20,
                        help='BBA 迭代解算的最大次数')
    
    args = parser.parse_args()

    if DEVICE.type == 'cpu':
        print("="*50)
        print("警告: 未检测到 CUDA。BBA 计算将在 CPU 上运行，")
        print("这将会非常缓慢。强烈建议在有 GPU 的环境上运行。")
        print("="*50)

    # 1. 加载影像
    images = load_imgs_bundle(args.root, args.select_imgs)
    if len(images) < 2:
        print("错误: 至少需要两张影像才能进行平差。")
        exit()
        
    # 2. 查找用于 *精度检查* 的重叠对
    overlapping_pairs = find_overlapping_pairs(images)

    # 3. 运行平差前的初始精度检查
    print("\n" + "="*50)
    print("--- 初始精度检查 (平差前) ---")
    check_all_pairs_error(images, overlapping_pairs)
    print("="*50 + "\n")

    # 4. 初始化并运行BBA解算器
    try:
        adjuster = TraditionalBundleAdjuster(images, args.loftr_path)
        adjuster.run(window_size=args.window_size, max_iter=args.max_iter)
    except Exception as e:
        print(f"\n !!! BBA 过程遭遇致命错误: {e}")
        import traceback
        traceback.print_exc()
        exit()

    # 5. 运行平差后的最终精度检查
    print("\n" + "="*50)
    print("--- 最终精度检查 (平差后) ---")
    # 检查结果已经烘焙到 'images' 列表中的 RPC 对象里
    adjuster.check_accuracy(overlapping_pairs)
    print("="*50 + "\n")

    print("平差流程执行完毕。")

