import torch
from pykeops.torch import LazyTensor
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List, Tuple, Dict

# 假设的外部依赖
from rpc import RPCModelParameterTorch
from rs_image_1022 import RSImage

# 从同一核心模块导入
from adjustment_core.validation import calculate_error_report
from adjustment_core.utils import TqdmLogger
# [FIXED] 移除此处的循环导入
# from adjustment_core.data import SharedGrid # 仅用于类型提示


def warp_local(local:torch.Tensor,dem:torch.Tensor,rpc_src:RPCModelParameterTorch,rpc_dst:RPCModelParameterTorch,affine_matrix:torch.Tensor):
    # RPC内部计算是float64，但affine_matrix是float32，需要转换
    affine_matrix_double = affine_matrix.to(torch.double)
    ones = torch.ones(local.shape[0],1).to(device=local.device,dtype=local.dtype)
    local_homo = torch.cat([local,ones],dim=-1)
    
    # 转换为double进行RPC计算
    trans_local = local_homo.to(torch.double) @ affine_matrix_double.T

    lats,lons = rpc_src.RPC_PHOTO2OBJ(trans_local[:,1],trans_local[:,0],dem)
    samps,lines = rpc_dst.RPC_OBJ2PHOTO(lats,lons,dem)
    warped_local = torch.stack([lines,samps],dim=-1).to(torch.float32) # 输出转回float32
    return warped_local

def feature_sampling(local:torch.Tensor, query:torch.Tensor, k = 4):
    """
    [已修正]
    在 'local' (像方坐标) 中为 'query' (warp 后的像方坐标) 查找 K 个最近邻。
    
    1. 使用 PyKeOps (no_grad) 高效查找 K-NN 索引 (idxs) 和用于过滤的距离 (dists_no_grad)。
    2. 使用 PyTorch (with_grad) 重新计算空间距离 (dists_valid)，以确保梯度可以反向传播。

    Args:
        local (torch.Tensor): (N_base, 2) 基础点云 (window_i.local)
        query (torch.Tensor): (N_query, 2) 查询点云 (warp_j_to_i)
        k (int): K近邻的数量

    Returns:
        tuple:
            - dists_valid (torch.Tensor): [N_valid, k] 每个有效查询点的 K 个空间距离 (带梯度)
            - idxs_valid (torch.Tensor): [N_valid, k] 每个有效查询点的 K 个邻近点索引
            - valid_mask (torch.Tensor): [N_query]布尔掩码，标记哪些查询点有效
    """
    
    # 步骤 1: 使用 PyKeOps 在无梯度上下文中查找索引和过滤用距离
    with torch.no_grad():
        point_base = LazyTensor(local.contiguous().unsqueeze(0))
        query_lazy = LazyTensor(query.contiguous().unsqueeze(1))
        # (N_query, N_base)
        dist_ij_sq_no_grad:LazyTensor = ((query_lazy - point_base) ** 2).sum(-1)
        
        # dists_sq_no_grad: [N_query, k], idxs: [N_query, k]
        dists_sq_no_grad, idxs = dist_ij_sq_no_grad.Kmin_argKmin(K = k, dim=1) 
        
        # [N_query, k] (无梯度)
        dists_no_grad = torch.sqrt(dists_sq_no_grad)

    # 步骤 2: 过滤掉距离太远的点 (例如，大于 64 像素)
    # [N_query]
    valid_mask = (dists_no_grad.min(dim=1).values < 64) 

    # 步骤 3: 过滤索引和 *需要计算梯度* 的查询点
    # [N_valid, k]
    idxs_valid = idxs[valid_mask]
    # [N_valid, 2] (此张量连接着梯度)
    query_valid = query[valid_mask] 
    
    if query_valid.shape[0] == 0: # 如果没有有效的点
        return None, None, valid_mask

    # 步骤 4: 使用索引 gather K 个邻近点的坐标
    # local: [N_base, 2]
    # locals_kmin: [N_valid, k, 2]
    locals_kmin = local[idxs_valid] 

    # 步骤 5: [核心] 使用标准 PyTorch 重新计算空间距离，以保留梯度
    
    # 扩展 query_valid 以便广播: [N_valid, 2] -> [N_valid, 1, 2]
    query_valid_expanded = query_valid.unsqueeze(1)
    
    # [N_valid, k, 2]
    diff = query_valid_expanded - locals_kmin
    
    # [N_valid, k] (此张量 *包含* 梯度)
    dists_valid = torch.norm(diff, dim=-1, p=2) 
    
    # 返回 K 近邻的 (带梯度)空间距离、索引和有效掩码
    return dists_valid, idxs_valid, valid_mask

def fit_affine_bundle(args,
                      local_shared_grids, # [FIXED] 使用字符串前向引用
                      images: List[RSImage], 
                      model_ddp: DDP, 
                      optimizer_r: torch.optim.Adam, 
                      optimizer_t: torch.optim.Adam, 
                      scheduler_r, 
                      scheduler_t, 
                      local_rank:int, 
                      world_size:int,
                      patience: int,           
                      overlapping_pairs: List[Tuple[int, int]],
                      current_level: int 
                      ) -> List[Dict[str, torch.Tensor]]: 
    """
    (未修改)
    使用DDP并行计算损失并优化仿射矩阵，支持基于loss或error的早停。
    (Refactored) 使用 TqdmLogger 统一处理日志记录。
    (Refactored) 使用 calculate_error_report 统一处理误差计算。
    """
    
    num_images = len(images)
    if num_images < 2 and local_rank == 0 and not args.auto:
        print("Error: Need at least 2 images for bundle adjustment.")
        return [] 
    
    # --- [Refactored] 初始化 Logger ---
    logger = TqdmLogger(args, args.max_iter, current_level, local_rank)
    
    # ---初始化早停和最佳模型变量 ---
    best_model_state = [] 
    if local_rank == 0:
        min_metric_val = float('inf') 
        patience_counter = 0
        criterion = args.stop_criterion
        loss_threshold = args.min_loss_threshold
        error_threshold = args.min_error_threshold
        
        if not args.auto:
            print(f"Starting optimization with criterion='{criterion}', patience={patience}.")
            if criterion == 'loss':
                print(f"Using min_loss_threshold={loss_threshold}")
            else: # criterion == 'error'
                print(f"Using min_error_threshold={error_threshold}m")
                # (check_error_during_train 已经在主文件中被强制启用)
                       
    stop_signal = torch.tensor(0.0, device=local_rank)

    # 4. 迭代优化
    for iter in range(args.max_iter):
        
        # [FIXED] 检查优化器是否存在
        if optimizer_r:
            optimizer_r.zero_grad()
        if optimizer_t:
            optimizer_t.zero_grad()
        
        local_total_loss = torch.tensor(0.0, device=local_rank)
        num_valid_grids = 0 
        
        # 5. 只在 *本地* 的格网子集上循环
        if len(local_shared_grids) == 0:
            pass 
        else:
            for grid in local_shared_grids:
                grid_avg_loss = grid.calculate_all_pairs_loss(model_ddp, images, local_rank)
                
                if not torch.isnan(grid_avg_loss) and not torch.isinf(grid_avg_loss) and grid_avg_loss > 0:
                    local_total_loss = local_total_loss + grid_avg_loss
                    num_valid_grids += 1
                else:
                    if grid_avg_loss > 0: 
                        print(f"[Rank{local_rank}]: Detect invalid loss:{grid_avg_loss.item()} in Grid {grid.id}")

            if num_valid_grids > 0:
                local_total_loss = local_total_loss / num_valid_grids
            
        # 7. 反向传播
        if local_total_loss > 0: # [FIXED] 仅在loss有效时反向传播
            local_total_loss.backward()
        
        if optimizer_r:
            optimizer_r.step()
        if optimizer_t:
            optimizer_t.step()
            
        # 1. 获取全局平均损失 (所有进程都需要)
        global_loss_sum = local_total_loss.clone().detach()
        dist.all_reduce(global_loss_sum, op=dist.ReduceOp.SUM)
        global_avg_loss = (global_loss_sum / world_size).item() 
        
        # 2. Rank 0 进行决策
        metric_log_dict = {} # 用于存储本轮日志信息
        
        if local_rank == 0:
            
            mean_err = 0.0 # 初始化
            
            # 确定是否需要在本轮计算 error
            # (check_error_during_train 已在主函数中根据 stop_criterion 自动设置)
            should_calculate_error = args.check_error_during_train and (iter + 1) % 10 == 0
            
            # 计算 error (如果需要)
            if should_calculate_error:
                # 精度检查逻辑
                original_params_list = [img.rpc.adjust_params.clone() for img in images]
                original_params_inv_list = [img.rpc.adjust_params_inv.clone() for img in images]
                try:
                    with torch.no_grad():
                        for i in range(1, num_images): 
                            current_A_i = model_ddp.module.get_affine(i).detach()
                            images[i].rpc.Update_Adjust(current_A_i) 
                    
                    # --- [Refactored] 调用统一的验证函数 ---
                    # (仅在训练中调用，不需要详细打印，所以 verbose=False)
                    error_report = calculate_error_report(images, overlapping_pairs, verbose=False)
                    mean_err = error_report['mean']
                    metric_log_dict['err(m)'] = mean_err # 存入日志
                    # ---
                finally:
                    with torch.no_grad():
                        for i in range(num_images):
                            images[i].rpc.adjust_params = original_params_list[i]
                            images[i].rpc.adjust_params_inv = original_params_list[i]

            # 确定本轮用于判断的指标和阈值
            current_metric_val = 0.0
            current_threshold = 0.0
            perform_check_this_iter = False 
            
            if args.stop_criterion == 'loss':
                current_metric_val = global_avg_loss
                current_threshold = args.min_loss_threshold
                perform_check_this_iter = True # loss 每轮都检查
                metric_log_dict['min_met'] = f"{min_metric_val:.4f}"
            elif args.stop_criterion == 'error' and should_calculate_error: # 只有计算了 error 的轮次才检查
                current_metric_val = mean_err 
                current_threshold = args.min_error_threshold
                perform_check_this_iter = True
                metric_log_dict['min_met'] = f"{min_metric_val:.2f}m"
            elif args.stop_criterion == 'error':
                metric_log_dict['min_met'] = f"{min_metric_val:.2f}m" # 保持显示
            
            # 执行判断 (仅在 perform_check_this_iter 为 True 时)
            if perform_check_this_iter and current_metric_val > 0: # 增加 > 0 检查，防止 error 为 0 时误判
                # 检查是否有显著改善 (注意: error 是越小越好)
                if (min_metric_val - current_metric_val) > current_threshold:
                    min_metric_val = current_metric_val
                    patience_counter = 0
                    
                    best_model_state = []
                    for sub_model in model_ddp.module.models: 
                        best_model_state.append({
                            'R': sub_model.R.data.clone(), 
                            'T': sub_model.T.data.clone()
                        })
                else:
                    # 没有显著改善
                    patience_counter += 1
            elif args.stop_criterion == 'error' and not should_calculate_error:
                 # 如果是 error 标准，但本轮未计算 error，则不增加 patience 计数器
                 pass
            elif perform_check_this_iter and current_metric_val <= 0 and args.stop_criterion == 'error':
                if not args.auto:
                    print(f"  Warning: Mean error is {current_metric_val:.4f}. Skipping best model check for this iteration.")


            # 检查是否需要早停
            if patience_counter >= patience:
                if not args.auto:
                    print(f"--- Early stopping triggered at iter {iter+1} based on '{args.stop_criterion}' ---")
                    if args.stop_criterion == 'loss':
                        print(f"Loss ({global_avg_loss:.4f}) did not improve by {args.min_loss_threshold} for {patience} iterations. Min loss: {min_metric_val:.4f}")
                    else: # error
                         print(f"Mean Error ({current_metric_val:.4f}m) did not improve by {args.min_error_threshold}m for {patience} check intervals. Min error: {min_metric_val:.4f}m")
                stop_signal.fill_(1.0) 

            # --- [Refactored] 日志记录 ---
            # 更新学习率 (非 auto 模式下需要)
            if not args.auto:
                 metric_log_dict['lr_t'] = scheduler_t.get_last_lr()[0] if scheduler_t else 0
                 metric_log_dict['lr_r'] = scheduler_r.get_last_lr()[0] if scheduler_r else 0
            
            # 调用 logger
            logger.update(iter, global_avg_loss, metric_log_dict, patience_counter, patience)

        
        # 3.广播停止信号
        dist.broadcast(stop_signal, src=0)

        # 4.检查停止信号
        if stop_signal.item() == 1.0:
            if local_rank == 0 and not args.auto:
                print(f"Rank {local_rank}: Received stop signal. Breaking optimization loop.")
            break 
        
        if scheduler_r:
            scheduler_r.step()
        if scheduler_t:
            scheduler_t.step()

    # 优化循环结束
    logger.close() # [Refactored] 关闭 logger
    
    if local_rank == 0 and not args.auto:
        print("Bundle adjustment optimization finished for this level.")

    return best_model_state

