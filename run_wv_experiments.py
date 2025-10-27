#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
run_experiments.py: 自动化实验管理器 (已修改)

该脚本用于自动化调用 adjust_test_1023.py, 实现以下功能:
1. 对指定参数 (root, max_lr, window_size, grid_num) 进行网格搜索。
2. 根据 window_size 动态设置 num_levels。
3. 自动传递 --auto 标志。
4. 为每次实验生成唯一ID (experiment_id)，并以此命名输出文件夹。
5. 捕获 adjust_test_1023.py 的成功、失败或崩溃状态。
6. (新) 从 adjust_test_1023.py 生成的 final_results.json 文件中读取所有精度指标。
7. (修改) 将所有参数和结果实时记录到主日志 (master_experiment_log.csv)，并覆盖旧记录。
8. (修改) 实现断点续跑：自动跳过 *仅已完成 (completed)* 的实验，重跑失败或中断的实验。
"""

import os
import subprocess
import pandas as pd
import itertools
import hashlib
import re
import numpy as np
# from tqdm import tqdm # <--- [修改] 移除总进度条
import sys
import time
import json 

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
    '--dino_path', 'weights', # <--- 检查路径
    '--encoder_path', 'weights/encoder_dino_1024_d100_b2_h20', # <--- 检查路径
    '--select_imgs', '0,1,2',
    '--patience', '1000',
    '--conf_threshold', '0.5',
    '--max_iter', '15000',
    '--kmin_k', '4',
    '--vis_resolution', '0.5',
    '--stop_criterion', 'error', # 或 'error'
    '--min_loss_threshold', '1e-4',
    '--max_grid_num','32',
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
        'grid_num': [8, 16, 24],
        'seed':[9,13,17,27,32]
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

def load_log_and_completed_ids(log_file):
    """
    (修改) 加载主日志文件, 创建 (如果不存在)。
    返回:
        df_log (pd.DataFrame): 完整的日志内容。
        completed_ids (set): *仅包含* status == 'completed' 的 experiment_id 集合。
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
            df_log = pd.DataFrame(columns=columns)
            df_log.to_csv(log_file, index=False)
            print(f"已创建新的日志文件: {log_file}")
            return df_log, set()
        except IOError as e:
            print(f"[!!] 致命错误: 无法创建日志文件 {log_file}。请检查权限。错误: {e}")
            sys.exit(1)
            
    try:
        # 如果日志存在, 读取它
        df_log = pd.read_csv(log_file)
        
        # 验证表头是否匹配
        if list(df_log.columns) != columns:
            print(f"[!!] 警告: {log_file} 的表头与预期不符。")
            print(f"    预期: {columns}")
            print(f"    实际: {list(df_log.columns)}")
            print("    [!!] 脚本将尝试使用预期表头继续, 这可能导致数据错位或丢失。")
            
            # 尝试用标准列重新加载，丢弃不匹配的
            df_log = pd.read_csv(log_file, names=columns, header=0, usecols=lambda c: c in columns)
            # 确保所有标准列都存在
            for col in columns:
                if col not in df_log:
                    df_log[col] = np.nan
        
        # (关键修改) 只筛选 'completed' 状态的
        completed_ids = set(df_log[df_log['status'] == 'completed']['experiment_id'].unique())
        
        print(f"从 {log_file} 加载了 {len(df_log)} 条日志记录, 其中 {len(completed_ids)} 个实验已'completed'。")
        return df_log, completed_ids
        
    except pd.errors.EmptyDataError:
        print(f"日志文件 {log_file} 为空。将视为新文件处理。")
        # 文件为空, 但已存在, 用正确的表头覆盖它
        df_log = pd.DataFrame(columns=columns)
        df_log.to_csv(log_file, index=False)
        return df_log, set()
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

def update_log_file(log_file: str, df_log: pd.DataFrame, log_entry: dict) -> pd.DataFrame:
    """
    (新) 将单次实验结果 (字典) 更新或追加到内存中的 DataFrame,
    然后 *原子地* 将整个 DataFrame 覆写回磁盘。
    返回:
        updated_df_log (pd.DataFrame): 更新后的 DataFrame。
    """
    try:
        exp_id = log_entry['experiment_id']
        
        # 查找该 experiment_id 是否已存在
        index_to_update = df_log.index[df_log['experiment_id'] == exp_id].tolist()
        
        if index_to_update:
            # --- 存在, 执行覆盖 ---
            idx = index_to_update[0]
            # print(f"    [i] 覆盖日志 (ID: {exp_id}, Index: {idx})") # 调试时使用
            for col, val in log_entry.items():
                df_log.at[idx, col] = val
        else:
            # --- 不存在, 执行追加 ---
            # print(f"    [i] 追加新日志 (ID: {exp_id})") # 调试时使用
            new_row_df = pd.DataFrame([log_entry])
            df_log = pd.concat([df_log, new_row_df], ignore_index=True)

        # --- 原子写入磁盘 ---
        # 写入临时文件
        temp_file = log_file + '.tmp'
        df_log.to_csv(temp_file, index=False)
        # 原子替换 (os.replace 在 POSIX 和 Windows 上都是原子的)
        os.replace(temp_file, log_file)
        
        return df_log # 返回更新后的 DataFrame

    except IOError as e:
        print(f"\n    [!!] 致命错误: 无法写入主日志 {log_file}！错误: {e}")
        print(f"    [!!] 实验 {log_entry.get('experiment_id')} 的数据可能已丢失！")
        print("    [!!] 请检查文件权限或磁盘空间。脚本将终止。")
        sys.exit(1)
    except Exception as e:
        print(f"\n    [!!] 写入日志时发生未知错误: {e}")
        # 决定是否终止
        sys.exit(1)


def main():
    """
    (修改) 主执行函数: 循环、调用、容灾、记录
    """
    print("--- [自动化实验管理器启动] ---")
    
    # 1. 获取所有实验配置
    all_experiments = get_param_grid()
    
    # 2. (修改) 加载日志 DataFrame 和 *已完成* 的ID
    df_log, completed_ids = load_log_and_completed_ids(MASTER_LOG_CSV)
    
    # 3. (修改) 过滤出未完成的实验
    # (此行逻辑不变, 但由于 completed_ids 的定义改变, 行为已变为 "跳过已完成")
    experiments_to_run = [exp for exp in all_experiments if exp['experiment_id'] not in completed_ids]
    
    if not experiments_to_run:
        print("\n--- [所有实验均已完成 ('completed' 状态)] ---")
        return
        
    print(f"总共 {len(all_experiments)} 个实验, {len(experiments_to_run)} 个待运行 (已跳过 {len(completed_ids)} 个 'completed')。")
    
    # 4. (修改) 循环执行所有待运行的实验 (移除 tqdm)
    try:
        num_to_run = len(experiments_to_run)
        for i, params in enumerate(experiments_to_run):
            exp_id = params['experiment_id']
            
            # (修改) 打印清晰的实验编号和总数
            print(f"\n--- [开始实验 {i+1}/{num_to_run}: {exp_id}] ---")
            print(f"    参数: {params}")
                
            # 5. 构建 DDP 调用命令
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

            # 6. 执行与容灾
            start_time = time.time()
            status = 'failed' # 默认为 'failed'
            run_time = 0.0 # 初始化
            
            # [修改] 定义默认的 nan 结果字典
            results_keys = [
                'mean_error', 'median_error', 'rmse', 'max_error', 
                '<1m', '<3m', '<5m', 'total_tie_points'
            ]
            results = {k: np.nan for k in results_keys}
            results['total_tie_points'] = 0
            
            try:
                # [!! 核心修改 !!]
                # 1. 复制当前的环境变量
                my_env = os.environ.copy()
                
                # 2. 设置 PYTHONUNBUFFERED=1
                #    这将强制子进程(torchrun及其worker)不缓冲stdout,
                #    允许tqdm进度条(如 mean_error)实时刷新到本终端。
                my_env["PYTHONUNBUFFERED"] = "1"

                # 3. 将修改后的 'env' 传入 subprocess.run
                #    (并保持上一版的 stdout=None, stderr=subprocess.PIPE)
                result = subprocess.run(cmd, 
                                        stdout=None,              # (保持) 允许 stdout 传递到终端
                                        stderr=subprocess.PIPE,   # (保持) 仅捕获 stderr
                                        text=True, 
                                        check=False, 
                                        encoding='utf-8',
                                        env=my_env)               # <--- [!! 新增此行 !!]
                
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
                    # 记录 stderr 以便调试 (仍然有效)
                    if result.stderr:
                        print("--- [STDERR (最后 1000 字符)] ---")
                        print(result.stderr[-1000:]) # 只打印最后1000字符
                        print("--- [END STDERR] ---")
                    
            except KeyboardInterrupt:
                print(f"\n[!!] 检测到用户中断 (Ctrl+C)。")
                print("    [i] 正在终止当前实验并安全退出...")
                # 记录中断状态并退出
                status = 'interrupted'
                end_time = time.time()
                run_time = end_time - start_time
                
                # (新) 即使中断, 也尝试记录日志条目
                log_entry = params.copy()
                log_entry['status'] = status
                log_entry['run_time_seconds'] = round(run_time, 2)
                log_entry.update(results) # results 此时应为默认的 nan
                
                ordered_log_entry = {col: log_entry.get(col) for col in [
                    'experiment_id', 'root', 'max_lr', 'window_size', 'grid_num', 'num_levels', 
                    'status', 'run_time_seconds', 
                    'mean_error', 'median_error', 'rmse', 'max_error', 
                    '<1m', '<3m', '<5m', 'total_tie_points'
                ]}
                
                df_log = update_log_file(MASTER_LOG_CSV, df_log, ordered_log_entry)
                print(f"    [i] 已将实验 {exp_id} 标记为 'interrupted' 并保存日志。")
                sys.exit(0) # 正常退出
                
            except Exception as e:
                # 捕获更严重的错误 (例如 subprocess 启动失败, OOM Kill, 命令本身错误)
                end_time = time.time()
                run_time = end_time - start_time
                print(f"    [X] 实验 {exp_id} 遭遇灾难性错误: {e}")
                status = 'catastrophic_failure'

            # 7. 持久化日志 (无论成功与否)
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
            
            # (修改) 持久化日志 (调用新函数)
            df_log = update_log_file(MASTER_LOG_CSV, df_log, ordered_log_entry)
            
            # (修改) 更新内存中的已完成集合
            if status == 'completed':
                completed_ids.add(exp_id)
            
            print(f"    [i] 实验 {exp_id} 已记录 (状态: {status}) 到 {MASTER_LOG_CSV}。")
    
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

