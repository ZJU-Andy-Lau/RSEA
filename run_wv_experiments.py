#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_experiments.py: 自动化实验管理器

该脚本用于自动化调用 adjust_test_1023.py, 实现以下功能:
1. 对指定参数 (root, max_lr, window_size, grid_num) 进行网格搜索。
2. 根据 window_size 动态设置 num_levels。
3. 自动传递 --auto 标志。
4. 为每次实验生成唯一ID (experiment_id)，并以此命名输出文件夹。
5. 捕获 adjust_test_1023.py 的成功、失败或崩溃状态。
6. (新) 从 adjust_test_1023.py 生成的 final_results.json 文件中读取所有精度指标。
7. 将所有参数和结果实时记录到主日志文件 (master_experiment_log.csv)。
8. 实现断点续跑：自动跳过日志文件中已存在的实验。
"""

import os
import subprocess
import pandas as pd
import itertools
import hashlib
import re
import numpy as np
from tqdm import tqdm
import sys
import time
import json # <--- [新导入]

# --- [用户必须配置] ---

# 1. DDP 启动器和GPU数量
# (示例) 使用 torchrun 和 4 个 GPU。请根据您的环境修改。
# 例如: ['python', '-m', 'torch.distributed.launch', '--nproc_per_node=4']
DDP_LAUNCHER = ['torchrun', '--nproc_per_node=8']

# 2. adjust_test_1023.py 的固定参数
# (修改) --root 已被移除，--auto 已被添加
FIXED_ARGS = [
    # '--root', '/path/to/your/data_root',  # <--- !! 已移除, 将在网格搜索中定义
    '--auto', # <--- !! 新增: 强制开启 auto 模式
    '--dino_path', 'weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth', # <--- 检查路径
    '--encoder_path', 'weights/encoder_dino_1024_d100_b2_h20', # <--- 检查路径
    '--select_imgs', '0,1,2',
    '--patience', '1000',
    '--conf_threshold', '0.5',
    '--max_iter', '15000',
    '--kmin_k', '4',
    '--vis_resolution', '0.5',
    '--stop_criterion', 'error', # 或 'error'
    '--min_loss_threshold', '1e-4',
    # '--check_error_during_train', # 如果使用, 请取消注释
    # ... 在这里添加您其他所有固定的参数 ...
    # 例如: '--grid_offset_x', '0',
    # 例如: '--select_grid_by_conf',
]

# 3. 主日志文件
# 脚本将自动创建此文件 (如果不存在)
MASTER_LOG_CSV = './log/wv_experiments_log.csv'

# --- [配置结束] ---


def get_param_grid():
    """
    生成所有实验的参数组合, 应用条件逻辑, 并创建唯一的实验ID。
    """
    print("正在生成参数网格...")
    
    # 1. 定义搜索空间
    param_space = {
        'root': ['./datasets/wv_test_error_5', './datasets/wv_test_error_10'], # <--- !! [修改] 新增, 请填入您的路径
        'max_lr': [0.1, 0.05, 0.01],
        'window_size': [2000, 1000],
        'grid_num': [8, 16, 24, 32]
    }
    
    # 2. 创建所有笛卡尔积组合
    keys, values = zip(*param_space.items())
    all_combinations = [dict(zip(keys, v)) for v in itertools.product(*values)]
    
    processed_experiments = []
    
    # 3. 应用条件逻辑 和 生成ID
    for params in all_combinations:
        # 应用条件逻辑: window_size -> num_levels
        if params['window_size'] == 2000:
            params['num_levels'] = 3
        elif params['window_size'] == 1000:
            params['num_levels'] = 2
        
        # 4. 生成可复现的 Experiment ID
        # [修改] 添加 root 的 basename, 使 ID 更具信息量且唯一
        root_basename = os.path.basename(params['root'].rstrip('/')) # 获取路径的最后一部分
        param_string = (f"root={root_basename}_" # <--- 新增
                        f"lr={params['max_lr']}_"
                        f"ws={params['window_size']}_"
                        f"gn={params['grid_num']}_"
                        f"nl={params['num_levels']}")
        
        # 使用 sha256 避免潜在的哈希碰撞, 取前12位
        params['experiment_id'] = hashlib.sha256(param_string.encode()).hexdigest()[:12]
        
        processed_experiments.append(params)
        
    print(f"成功生成 {len(processed_experiments)} 个实验配置。")
    return processed_experiments

def load_completed_experiments(log_file):
    """
    加载主日志文件, 创建 (如果不存在), 并返回已记录的 experiment_id 集合。
    """
    # [修改] 定义日志文件的所有列
    columns = [
        'experiment_id', 'root', 'max_lr', 'window_size', 'grid_num', 'num_levels', 
        'status', 'run_time_seconds', 
        'mean_error', 'median_error', 'rmse', 'max_error', 
        '<1m', '<3m', '<5m', 'total_tie_points'
    ]
    
    if not os.path.exists(log_file):
        try:
            # 如果日志不存在, 创建一个空的 DataFrame 并写入表头
            df = pd.DataFrame(columns=columns)
            df.to_csv(log_file, index=False)
            print(f"已创建新的日志文件: {log_file}")
            return set()
        except IOError as e:
            print(f"[!!] 致命错误: 无法创建日志文件 {log_file}。请检查权限。错误: {e}")
            sys.exit(1)
            
    try:
        # 如果日志存在, 读取它
        df = pd.read_csv(log_file)
        # 验证表头是否匹配
        if list(df.columns) != columns:
            print(f"[!!] 警告: {log_file} 的表头与预期不符。")
            print(f"    预期: {columns}")
            print(f"    实际: {list(df.columns)}")
            print("    [!!] 脚本将尝试继续, 但可能导致日志格式错乱。")
            
        # 返回所有已记录的 experiment_id 集合
        # 无论 'status' 是 'completed' 还是 'failed', 都算作已运行
        completed_ids = set(df['experiment_id'].unique())
        print(f"从 {log_file} 加载了 {len(completed_ids)} 个已运行的实验记录。")
        return completed_ids
        
    except pd.errors.EmptyDataError:
        print(f"日志文件 {log_file} 为空。将视为新文件处理。")
        # 文件为空, 但已存在, 用正确的表头覆盖它
        df = pd.DataFrame(columns=columns)
        df.to_csv(log_file, index=False)
        return set()
    except Exception as e:
        print(f"[!!] 致命错误: 无法读取日志文件 {log_file}。错误: {e}")
        sys.exit(1)

def load_results_from_json(root_path, experiment_id):
    """
    (新) 从 final_results.json 文件加载实验结果。
    取代 parse_results_from_stdout
    """
    # 定义所有期望的指标, 默认为 nan
    results_keys = [
        'mean_error', 'median_error', 'rmse', 'max_error', 
        '<1m', '<3m', '<5m', 'total_tie_points'
    ]
    default_results = {k: np.nan for k in results_keys}
    default_results['total_tie_points'] = 0 # 默认为 0
    
    try:
        # 路径格式: {root}/output_{experiment_id}/final_results.json
        json_path = os.path.join(root_path, f'output_{experiment_id}', 'final_results.json')
        
        if not os.path.exists(json_path):
            print(f"    [!] 错误: 结果文件未找到 (File Not Found): {json_path}")
            return default_results

        with open(json_path, 'r') as f:
            data_from_json = json.load(f)
        
        # 更新默认字典, 确保所有键都存在
        # 这可以防止 JSON 文件万一缺少某个键时出错
        default_results.update(data_from_json)
        
        # 只返回我们关心的键
        final_results = {k: default_results.get(k) for k in results_keys}
        return final_results

    except json.JSONDecodeError:
        print(f"    [!] 错误: 无法解析 JSON 结果文件 (JSON Decode Error): {json_path}")
        return default_results
    except Exception as e:
        print(f"    [!] 错误: 加载结果 JSON 时发生未知异常: {e}")
        return default_results

def log_experiment(log_file, log_entry):
    """
    将单次实验的结果 (一个字典) 追加到主 CSV 日志文件。
    这是一个独立函数, 确保文件I/O的原子性 (追加操作)。
    """
    try:
        # 将日志条目转换为单行 DataFrame
        new_row_df = pd.DataFrame([log_entry])
        
        # 以追加模式 (mode='a') 写入, 并且不写入表头 (header=False)
        new_row_df.to_csv(log_file, mode='a', header=False, index=False)
        
    except IOError as e:
        print(f"\n    [!!] 致命错误: 无法写入主日志 {log_file}！错误: {e}")
        print(f"    [!!] 实验 {log_entry.get('experiment_id')} 的数据可能已丢失！")
        print("    [!!] 请检查文件权限或磁盘空间。脚本将终止。")
        sys.exit(1)
    except Exception as e:
        print(f"\n    [!!] 写入日志时发生未知错误: {e}")
        # 决定是否终止
        # sys.exit(1)

def main():
    """
    主执行函数: 循环、调用、容灾、记录
    """
    print("--- [自动化实验管理器启动] ---")
    
    # 1. 获取所有实验配置
    all_experiments = get_param_grid()
    
    # 2. 加载已完成的实验, 实现断点续跑
    completed_ids = load_completed_experiments(MASTER_LOG_CSV)
    
    # 过滤出未完成的实验
    experiments_to_run = [exp for exp in all_experiments if exp['experiment_id'] not in completed_ids]
    
    if not experiments_to_run:
        print("\n--- [所有实验均已完成] ---")
        return
        
    print(f"总共 {len(all_experiments)} 个实验, {len(experiments_to_run)} 个待运行。")
    
    # 3. 循环执行所有待运行的实验
    try:
        with tqdm(experiments_to_run, desc="总实验进度") as pbar:
            for params in pbar:
                exp_id = params['experiment_id']
                pbar.set_description(f"运行中: {exp_id}")
                
                print(f"\n--- [开始实验: {exp_id}] ---")
                print(f"    参数: {params}")
                
                # 4. 构建 DDP 调用命令
                cmd = list(DDP_LAUNCHER) 
                cmd.append('adjust_test_1023.py')
                cmd.extend(FIXED_ARGS)
                
                # 添加本次实验的动态参数
                for key, value in params.items():
                    # 将 Python 的 True/False 转换为空标志 (如果需要)
                    if isinstance(value, bool) and value:
                        cmd.append(f'--{key}')
                    elif not (isinstance(value, bool) and not value):
                        cmd.append(f'--{key}')
                        cmd.append(str(value))
                        
                # print(f"    命令: {' '.join(cmd)}") # 调试时取消注释

                # 5. 执行与容灾
                start_time = time.time()
                status = 'failed' # 默认为 'failed'
                
                # [修改] 定义默认的 nan 结果字典
                results_keys = [
                    'mean_error', 'median_error', 'rmse', 'max_error', 
                    '<1m', '<3m', '<5m', 'total_tie_points'
                ]
                results = {k: np.nan for k in results_keys}
                results['total_tie_points'] = 0
                
                try:
                    # 执行子进程。
                    # capture_output=True 捕获 stdout/stderr
                    # text=True         使用系统默认编码 (通常是 utf-8)
                    # check=False       我们手动检查返回码, 不让它在失败时抛出异常
                    # encoding='utf-8'  显式指定编码, 避免 Windows 上的 GBK 问题
                    result = subprocess.run(cmd, 
                                            capture_output=True, 
                                            text=True, 
                                            check=False, 
                                            encoding='utf-8')
                    
                    end_time = time.time()
                    run_time = end_time - start_time
                    
                    # 检查返回码
                    if result.returncode == 0:
                        # 成功！
                        print(f"    [✓] 实验 {exp_id} 成功。 (耗时: {run_time:.2f} 秒)")
                        status = 'completed'
                        
                        # [修改] 从 JSON 加载结果, 而不是解析 stdout
                        results = load_results_from_json(params['root'], exp_id)
                        print(f"    [i] 结果: mean_error={results.get('mean_error', 'N/A'):.4f} m, <1m={results.get('<1m', 'N/A'):.2f} %")
                        
                    else:
                        # 失败！
                        print(f"    [X] 实验 {exp_id} 失败 (Return Code: {result.returncode})。 (耗时: {run_time:.2f} 秒)")
                        status = 'failed'
                        # 记录 stderr 以便调试
                        print("--- [STDERR (最后 1000 字符)] ---")
                        print(result.stderr[-1000:])
                        print("--- [END STDERR] ---")
                        
                except KeyboardInterrupt:
                    print(f"\n[!!] 检测到用户中断 (Ctrl+C)。")
                    print("    [i] 正在终止当前实验并安全退出...")
                    # 不记录本次实验, 直接退出循环
                    sys.exit(0) # 正常退出
                    
                except Exception as e:
                    # 捕获更严重的错误 (例如 subprocess 启动失败, OOM Kill, 命令本身错误)
                    end_time = time.time()
                    run_time = end_time - start_time
                    print(f"    [X] 实验 {exp_id} 遭遇灾难性错误: {e}")
                    status = 'catastrophic_failure'

                # 6. 持久化日志 (无论成功与否)
                log_entry = params.copy()
                log_entry['status'] = status
                log_entry['run_time_seconds'] = round(run_time, 2)
                log_entry.update(results) # [修改] 合并从 JSON 加载的 results 字典
                
                # [修改] 确保日志条目的键顺序与表头一致
                ordered_log_entry = {col: log_entry.get(col) for col in [
                    'experiment_id', 'root', 'max_lr', 'window_size', 'grid_num', 'num_levels', 
                    'status', 'run_time_seconds', 
                    'mean_error', 'median_error', 'rmse', 'max_error', 
                    '<1m', '<3m', '<5m', 'total_tie_points'
                ]}
                
                log_experiment(MASTER_LOG_CSV, ordered_log_entry)
                
                # 更新内存中的已完成集合
                completed_ids.add(exp_id)
                print(f"    [i] 实验 {exp_id} 已记录到 {MASTER_LOG_CSV}。")
    
    except KeyboardInterrupt:
        print("\n[!!] 用户在实验循环间隙中断。脚本将退出。")
        
    print("\n--- [所有待运行的实验已处理完毕] ---")

if __name__ == "__main__":
    # 检查 adjust_test_1023.py 是否存在
    if not os.path.exists('adjust_test_1023.py'):
        print("[!!] 错误: 'adjust_test_1023.py' 未在当前目录找到。")
        print("[!!] 请将此脚本与 'adjust_test_1023.py' 放在同一目录下。")
        sys.exit(1)
        
    main()

