import sys
import time
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import argparse
import os

def get_elevation(image_path: str, dem_path: str, output_path: str):
    """
    使用 Rasterio，根据遥感影像的RPC模型，在DEM上采样高程。

    Args:
        image_path (str): 带有RPC元数据的遥感影像路径 (TIF)。
        dem_path (str): DEM数据路径 (TIF)。
        output_path (str): 输出高程文件的路径 (TIF)。
    """
    try:
        # 1. --- 数据和环境设置 (使用上下文管理器) ---
        print("开始处理 (使用 Rasterio)...")
        with rasterio.open(image_path) as image_src, \
             rasterio.open(dem_path) as dem_src:

            print(f"源影像 (RPC): {image_path}")
            print(f"源DEM: {dem_path}")
            print(f"输出高程图: {output_path}")

            # 检查RPC元数据
            rpcs = image_src.rpcs
            if not rpcs:
                print("错误: 源影像中未找到RPC元数据。")
                sys.exit(1)

            # 获取影像尺寸和元数据 profile
            img_width = image_src.width
            img_height = image_src.height
            
            # 准备输出文件的元数据
            profile = image_src.profile
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw',
                tiled=True
            )

            # 将DEM数据一次性读入内存以加速采样
            # 注意：如果DEM文件非常巨大，可能需要分块读取DEM
            print("正在加载DEM到内存...")
            dem_data = dem_src.read(1)
            dem_nodata = dem_src.nodata or -9999.0 # 获取DEM的无效值

            # 获取DEM的平均高程作为初始值
            valid_dem_data = dem_data[dem_data != dem_nodata]
            if valid_dem_data.size > 0:
                initial_elevation = valid_dem_data.mean()
                print(f"使用DEM平均高程 {initial_elevation:.2f} 米作为初始值。")
            else:
                print("警告: DEM中无有效数据，使用0作为初始高程。")
                initial_elevation = 0


            # 2. --- 创建输出文件并分块处理 ---
            with rasterio.open(output_path, 'w', **profile) as output_ds:
                # 定义块大小
                block_size_x = 512
                block_size_y = 512
                
                print("开始分块计算高程...")
                pbar = tqdm(total=img_width * img_height, desc="处理像素")

                for y_offset in range(0, img_height, block_size_y):
                    for x_offset in range(0, img_width, block_size_x):
                        current_block_y = min(block_size_y, img_height - y_offset)
                        current_block_x = min(block_size_x, img_width - x_offset)

                        # 创建当前块的影像坐标网格 (row/col)
                        # 使用'ij'索引，使输出的meshgrid形状与块大小匹配
                        cols, rows = np.meshgrid(
                            np.arange(x_offset, x_offset + current_block_x),
                            np.arange(y_offset, y_offset + current_block_y)
                        )
                        
                        # 初始化高程块
                        elevations = np.full((current_block_y, current_block_x), initial_elevation, dtype=np.float64)

                        # --- 迭代求解高程 ---
                        for i in range(3):
                            # RPC反向变换: (影像坐标+高程) -> (经度, 纬度)
                            # Rasterio的RPC接口直接接受展平的numpy数组
                            lons, lats = rpcs.back_project(cols.flatten(), rows.flatten(), elevations.flatten())
                            
                            # 地理坐标 -> DEM像素坐标 (使用Rasterio的便捷方法)
                            try:
                                dem_rows, dem_cols = dem_src.index(lons, lats)
                            except Exception: # 如果坐标超出DEM范围，会抛出异常
                                continue

                            # 将坐标转换为numpy数组以便进行索引和边界检查
                            dem_rows = np.array(dem_rows)
                            dem_cols = np.array(dem_cols)

                            # --- 从DEM采样新高程 (矢量化) ---
                            # 创建一个掩码，标记出在DEM范围内的点
                            valid_mask = (dem_rows >= 0) & (dem_rows < dem_src.height) & \
                                         (dem_cols >= 0) & (dem_cols < dem_src.width)

                            # 初始化新高程数组为无效值
                            new_elevations = np.full_like(elevations.flatten(), dem_nodata)
                            
                            # 对所有在范围内的点，一次性从内存中的DEM数据采样
                            valid_dem_rows = dem_rows[valid_mask]
                            valid_dem_cols = dem_cols[valid_mask]
                            sampled_values = dem_data[valid_dem_rows, valid_dem_cols]
                            
                            new_elevations[valid_mask] = sampled_values
                            
                            # 更新高程值，对于无效采样点，保留上一次迭代的值
                            final_valid_mask = new_elevations != dem_nodata
                            elevations.flat[final_valid_mask] = new_elevations[final_valid_mask]

                        # 将最终计算出的高程块写入输出文件
                        window = Window(x_offset, y_offset, current_block_x, current_block_y)
                        output_ds.write(elevations.astype(np.float32), window=window)
                        pbar.update(current_block_x * current_block_y)

                pbar.close()

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        sys.exit(1)

    print("高程计算完成。")
    print(f"成功创建高程文件: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str)
    args = parser.parse_args()
    root = args.root

    start_time = time.time()
    names = [i.split('.')[0] for i in os.listdir(root) if 'rpc' in i and 'PAN' in i]
    for name in names:
        get_elevation(os.path.join(root,f'{name}.tiff'),os.path.join(root,'dem.tif'),os.path.join(root,f'{name}_dem.tif'))
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")