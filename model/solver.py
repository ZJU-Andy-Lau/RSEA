import torch
import torch.nn as nn

class AffineFitter:
    """
    使用概率方法拟合一个2D仿射变换。

    该类通过最小化加权的最小二乘损失（等价于负对数似然）来寻找
    一个最佳的仿射变换，将源点映射到目标分布。权重由预测的标准差确定。
    """

    def __init__(self, verbose: bool = True):
        """
        初始化拟合器。

        Args:
            learning_rate (float): 优化器的学习率。
            num_iterations (int): 梯度下降的迭代次数。
            verbose (bool): 是否在拟合过程中打印损失信息。
        """
        self.verbose = verbose
        
        # 最终得到的仿射变换矩阵，(2, 3)
        self.transformation_matrix = None

    def fit(self, 
            source_points: torch.Tensor, 
            pred_means: torch.Tensor, 
            pred_stds: torch.Tensor,
            return_res: bool = False) -> torch.Tensor:
        """
        执行仿射变换的拟合过程。

        Args:
            source_points (torch.Tensor): 原始点坐标，形状为 (N, 2)。
            pred_means (torch.Tensor): 预测的目标点坐标均值，形状为 (N, 2)。
            pred_stds (torch.Tensor): 预测的目标点坐标标准差，形状为 (N, 2)。

        Returns:
            torch.Tensor: 拟合得到的 (2, 3) 仿射变换矩阵。
        """
        if self.verbose:
            print("开始求解仿射变换")

        # --- 1. 数据校验与准备 ---
        if not (source_points.shape == pred_means.shape == pred_stds.shape and source_points.dim() == 2 and source_points.shape[1] == 2):
            raise ValueError("所有输入张量的形状必须为 (N, 2)。")
        
        device = source_points.device
        dtype = source_points.dtype
        num_points = source_points.shape[0]

        source_homogeneous = torch.cat(
            [source_points, torch.ones(num_points, 1, device=device, dtype=dtype)], 
            dim=1
        )

        # --- 2. 计算权重 ---
        weights = 1.0 / (pred_stds.pow(2) + 1e-8)
        w_x = weights[:, 0].to(dtype=dtype)
        w_y = weights[:, 1].to(dtype=dtype)
        
        mu_x = pred_means[:, 0].to(dtype=dtype)
        mu_y = pred_means[:, 1].to(dtype=dtype)

        # --- 3. 构建线性方程组 Hp = b (向量化版本) ---
        
        # --- 计算 Hessian 矩阵 H ---
        # H 是一个 6x6 的块矩阵:
        # H = [[H_xx,  0   ],
        #      [ 0  ,  H_yy]]
        # 其中 H_xx = sum(w_xi * p_i * p_i^T)
        # H_yy = sum(w_yi * p_i * p_i^T)
        # 这里的 p_i 是齐次坐标 [x_i, y_i, 1]
        
        # 使用矩阵乘法高效计算 H_xx 和 H_yy
        # (P.T @ (w * P)) 等价于 sum(w_i * p_i * p_i^T)
        H_xx = source_homogeneous.T @ (w_x.unsqueeze(1) * source_homogeneous)
        H_yy = source_homogeneous.T @ (w_y.unsqueeze(1) * source_homogeneous)
        
        H = torch.zeros((6, 6), device=device, dtype=dtype)
        H[0:3, 0:3] = H_xx
        H[3:6, 3:6] = H_yy

        # --- 计算向量 b ---
        # b 是一个 6x1 的块向量:
        # b = [[b_x],
        #      [b_y]]
        # 其中 b_x = sum(w_xi * mu_xi * p_i)
        # b_y = sum(w_yi * mu_yi * p_i)
        

        # 使用矩阵-向量乘法高效计算 b_x 和 b_y
        b_x = source_homogeneous.T @ (w_x * mu_x)
        b_y = source_homogeneous.T @ (w_y * mu_y)
        
        b = torch.cat([b_x, b_y]).unsqueeze(1) # 拼接成 6x1 的列向量

        # --- 4. 求解线性方程组 ---
        if self.verbose:
            print("线性方程组构建完成，正在求解 Hp = b ...")
        
        try:
            # 使用torch.linalg.solve求解器，它比手动求逆更稳定、更高效
            params_vec = torch.linalg.solve(H, b, out=None) # PyTorch 1.10+
            # 对于旧版本PyTorch，可以使用: params_vec, _ = torch.solve(b, H)
        except torch.linalg.LinAlgError as e:
            print("错误：矩阵H是奇异的或接近奇异的，无法求解。")
            print("这通常发生在输入点共线或点数量过少的情况下。")
            raise e

        # --- 5. 格式化并保存结果 ---
        # 将解向量 p (6x1) 变形为 2x3 的仿射矩阵
        self.transformation_matrix = params_vec.reshape(2, 3)
        
        
        if self.verbose:
            print("求解完成！")
            ori_dis = torch.norm(source_points - pred_means,dim=-1).mean()
            trans_points = self.transform(source_points)
            trans_dis = torch.norm(trans_points - pred_means,dim=-1).mean()
            print(f"初始误差: {ori_dis.item():.2f} \t 变换后误差: {trans_dis.item():.2f}")
           
            
        if return_res:
            return self.transformation_matrix,trans_dis
        else:
            return self.transformation_matrix

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """
        使用拟合好的变换矩阵来变换新的点。

        Args:
            points (torch.Tensor): 需要变换的点，形状为 (M, 2)。

        Returns:
            torch.Tensor: 变换后的点，形状为 (M, 2)。
        """
        if self.transformation_matrix is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合，然后才能进行变换。")
        
        device = points.device
        num_points = points.shape[0]
        
        points_homogeneous = torch.cat(
            [points, torch.ones(num_points, 1, device=device)],
            dim=1
        )
        
        # 使用存储的矩阵进行变换
        transformed_points = points_homogeneous @ self.transformation_matrix.T
        
        return transformed_points

class HomographyFitter:
    """
    使用非线性优化方法（L-BFGS）来拟合一个2D单应变换。

    该类通过最小化加权的最小二乘损失（等价于负对数似然）来寻找
    最佳的单应变换。由于单应变换对于其参数是非线性的，我们不能
    使用直接解法，而是采用像L-BFGS这样的迭代优化器。
    """

    def __init__(self, max_iterations: int = 2000, lr: float = 1e-3,
                 patience: int = 50, tolerance: float = 1e-7,
                 weight_decay: float = 1e-4, verbose: bool = True):
        """
        初始化拟合器。

        Args:
            max_iterations (int): 优化的最大迭代次数。如果小于等于0，则启用早停策略。
            lr (float): AdamW优化器的学习率。
            patience (int): 早停策略的“耐心值”。
            tolerance (float): 用于判断损失是否“显著下降”的阈值。
            weight_decay (float): AdamW的权重衰减系数。
            verbose (bool): 是否在拟合过程中打印信息。
        """
        self.max_iterations = max_iterations
        self.lr = lr
        self.patience = patience
        self.tolerance = tolerance
        self.weight_decay = weight_decay
        self.verbose = verbose
        # 最终得到的单应变换矩阵，3x3
        self.transformation_matrix = None

    def _get_normalization_matrix(self, points: torch.Tensor) -> torch.Tensor:
        """计算将点归一化的变换矩阵。"""
        mean = points.mean(dim=0)
        cx, cy = mean[0], mean[1]

        # 将点移动到以原点为中心
        centered_points = points - mean
        
        # 计算平均距离，并缩放使其约为sqrt(2)
        avg_dist = (centered_points**2).sum(dim=1).sqrt().mean()
        # 修正: 确保新创建的张量与输入张量有相同的dtype和device
        scale = torch.sqrt(torch.tensor(2.0, dtype=points.dtype, device=points.device)) / (avg_dist + 1e-8)

        # 构建归一化矩阵
        # T = [s, 0, -s*cx]
        #     [0, s, -s*cy]
        #     [0, 0, 1    ]
        T = torch.eye(3, device=points.device, dtype=points.dtype)
        T[0, 0] = T[1, 1] = scale
        T[0, 2] = -scale * cx
        T[1, 2] = -scale * cy
        return T

    def fit(self,
            source_points: torch.Tensor,
            pred_means: torch.Tensor,
            pred_stds: torch.Tensor) -> torch.Tensor:
        """
        执行单应变换的拟合过程。
        """
        if self.verbose:
            print("开始使用AdamW和坐标归一化拟合单应变换...")
            if self.max_iterations <= 0:
                print(f"早停已启用: patience={self.patience}, tolerance={self.tolerance}")

        device = source_points.device
        dtype = source_points.dtype
        
        # --- 1. 坐标归一化 (鲁棒性关键步骤) ---
        T_source = self._get_normalization_matrix(source_points)
        T_target = self._get_normalization_matrix(pred_means)
        T_target_inv = torch.linalg.inv(T_target)

        source_h = torch.cat([source_points, torch.ones(source_points.shape[0], 1, device=device, dtype=dtype)], dim=1)
        pred_means_h = torch.cat([pred_means, torch.ones(pred_means.shape[0], 1, device=device, dtype=dtype)], dim=1)

        # 应用归一化
        norm_source_h = (T_source @ source_h.T).T
        norm_pred_means_h = (T_target @ pred_means_h.T).T
        
        # 清理: 移除未使用的变量 norm_source_points
        norm_pred_means = norm_pred_means_h[:, :2] / norm_pred_means_h[:, 2].unsqueeze(1)
        
        # --- 2. 初始化变换参数和优化器 ---
        initial_params = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], device=device, dtype=dtype)
        self.params = nn.Parameter(initial_params)
        optimizer = torch.optim.AdamW([self.params], lr=self.lr, weight_decay=self.weight_decay)

        # --- 3. 计算损失权重 ---
        # 权重不需要归一化
        weights = 1.0 / (pred_stds.pow(2) + 1e-8)

        # --- 4. 在归一化坐标上进行优化循环 ---
        iteration = 0
        best_loss = float('inf')
        patience_counter = 0

        while True:
            iteration += 1
            optimizer.zero_grad()
            
            h_matrix_norm = torch.cat([self.params, torch.tensor([1.0], device=device, dtype=dtype)]).reshape(3, 3)
            
            # 在归一化空间中进行变换
            transformed_h_norm = norm_source_h @ h_matrix_norm.T
            w_norm = transformed_h_norm[:, 2].unsqueeze(1)
            transformed_points_norm = transformed_h_norm[:, :2] / (w_norm + 1e-8)
            
            # 在归一化空间中计算损失
            error = transformed_points_norm - norm_pred_means
            weighted_squared_error = error.pow(2) * weights
            loss = weighted_squared_error.sum()

            if torch.isnan(loss) or torch.isinf(loss):
                if self.verbose:
                    print(f"迭代 {iteration:4d}, 损失变为无效值(NaN/Inf)，优化失败。")
                self.transformation_matrix = torch.eye(3, device=device, dtype=dtype)
                return self.transformation_matrix

            loss.backward()
            optimizer.step()

            if self.verbose and (iteration % 200 == 0 or iteration == 1):
                print(f"迭代 {iteration:4d}, 损失: {loss.item():.6f}，最小：{best_loss:.6f}")

            if best_loss - loss.item() > self.tolerance:
                best_loss = loss.item()
                patience_counter = 0
            else:
                patience_counter += 1

            if self.max_iterations > 0 and iteration >= self.max_iterations:
                if self.verbose:
                    print(f"达到最大迭代次数上限: {self.max_iterations}。")
                break
            
            if self.max_iterations <= 0 and patience_counter >= self.patience:
                if self.verbose:
                    print(f"\n损失在 {self.patience} 次迭代内没有显著下降，提前停止于第 {iteration} 次迭代。")
                break

        # --- 5. 反归一化并保存结果 ---
        final_h_norm = torch.cat([self.params.detach(), torch.tensor([1.0], device=device, dtype=dtype)]).reshape(3, 3)
        self.transformation_matrix = T_target_inv.to(torch.float) @ final_h_norm.to(torch.float) @ T_source.to(torch.float)
        
        # --- 6. 计算并输出最终结果 ---
        if self.verbose:
            # 新增: 计算最终的平均残差 (Reprojection Error)
            final_transformed_points = self.transform(source_points)
            residuals = torch.sqrt(((final_transformed_points - pred_means)**2).sum(dim=1))
            avg_residual = residuals.mean()
            print(f"拟合完成于第 {iteration} 次迭代。最终损失: {best_loss:.6f}, 平均残差: {avg_residual.item():.6f} 像素")
            
        return self.transformation_matrix

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """
        使用拟合好的变换矩阵来变换新的点。
        """
        if self.transformation_matrix is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合，然后才能进行变换。")
        
        device = points.device
        dtype = points.dtype
        num_points = points.shape[0]
        
        points_homogeneous = torch.cat(
            [points, torch.ones(num_points, 1, device=device, dtype=dtype)],
            dim=1
        )
        
        transformed_homogeneous = points_homogeneous @ self.transformation_matrix.T
        
        w = transformed_homogeneous[:, 2].unsqueeze(1)
        transformed_points = transformed_homogeneous[:, :2] / (w + 1e-8)
        
        return transformed_points

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """
        使用拟合好的变换矩阵来变换新的点。
        """
        if self.transformation_matrix is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合，然后才能进行变换。")
        
        device = points.device
        dtype = points.dtype
        num_points = points.shape[0]
        
        points_homogeneous = torch.cat(
            [points, torch.ones(num_points, 1, device=device, dtype=dtype)],
            dim=1
        )
        
        transformed_homogeneous = points_homogeneous @ self.transformation_matrix.T
        
        w = transformed_homogeneous[:, 2].unsqueeze(1)
        transformed_points = transformed_homogeneous[:, :2] / (w + 1e-8)
        
        return transformed_points

    def transform(self, points: torch.Tensor) -> torch.Tensor:
        """
        使用拟合好的变换矩阵来变换新的点。
        """
        if self.transformation_matrix is None:
            raise RuntimeError("必须先调用 .fit() 方法进行拟合，然后才能进行变换。")
        
        device = points.device
        dtype = points.dtype
        num_points = points.shape[0]
        
        points_homogeneous = torch.cat(
            [points, torch.ones(num_points, 1, device=device, dtype=dtype)],
            dim=1
        )
        
        transformed_homogeneous = points_homogeneous @ self.transformation_matrix.T
        
        w = transformed_homogeneous[:, 2].unsqueeze(1)
        transformed_points = transformed_homogeneous[:, :2] / (w + 1e-8)
        
        return transformed_points