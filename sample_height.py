import sys
import time
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import argparse
import os
from rpc import RPCModelParameterTorch

def get_elevation(image_path: str, dem_path: str, output_path: str, rpc: 'RPCModelParameterTorch'):
    """
    使用 Rasterio，根据遥感影像的RPC模型，在DEM上采样高程。
    此版本使用【矢量化】操作进行坐标转换，以实现最高性能。

    参数:
        image_path (str): 带有RPC元数据的遥感影像路径 (TIF)。
        dem_path (str): DEM数据路径 (TIF)。
        output_path (str): 输出高程文件的路径 (TIF)。
        rpc: 传入的RPC模型对象。
    """
    try:
        # --- 1. 使用 Rasterio 打开文件进行 I/O 操作 ---
        print("开始处理 (使用矢量化的高性能方法)...")
        with rasterio.open(image_path) as image_src, \
             rasterio.open(dem_path) as dem_src:

            print(f"源影像: {image_path}")
            print(f"源DEM: {dem_path}")
            print(f"输出高程图: {output_path}")

            # --- 2. 准备元数据和初始值 ---
            img_width = image_src.width
            img_height = image_src.height
            
            # 【优化】获取DEM的逆仿射变换矩阵，用于矢量化坐标转换
            inv_transform = ~dem_src.transform
            
            block_size = 512

            profile = image_src.profile
            dem_nodata = dem_src.nodata or -9999.0
            profile.update(
                dtype=rasterio.float32,
                count=1,
                compress='lzw',
                tiled=True,
                blockxsize=block_size,
                blockysize=block_size,
                nodata=dem_nodata
            )

            print("正在加载DEM到内存...")
            dem_data = dem_src.read(1)
            valid_dem_data = dem_data[dem_data != dem_nodata]
            initial_elevation = valid_dem_data.mean() if valid_dem_data.size > 0 else 0
            print(f"使用DEM平均高程 {initial_elevation:.2f} 米作为初始值。")

            # --- 3. 创建输出文件并分块处理 ---
            with rasterio.open(output_path, 'w', **profile) as output_ds:
                print("开始分块计算高程...")
                pbar = tqdm(total=img_width * img_height, desc="处理像素")

                for y_offset in range(0, img_height, block_size):
                    for x_offset in range(0, img_width, block_size):
                        current_block_y = min(block_size, img_height - y_offset)
                        current_block_x = min(block_size, img_width - x_offset)

                        samps, lines = np.meshgrid(
                            np.arange(x_offset, x_offset + current_block_x),
                            np.arange(y_offset, y_offset + current_block_y)
                        )
                        
                        samps_center = samps.astype(np.float64) + 0.5
                        lines_center = lines.astype(np.float64) + 0.5
                        
                        elevations = np.full((current_block_y, current_block_x), initial_elevation, dtype=np.float64)

                        # --- 迭代求解高程 ---
                        for i in range(3):
                            lats, lons = rpc.RPC_PHOTO2OBJ(
                                samps_center.flatten(),
                                lines_center.flatten(),
                                elevations.flatten(),
                                'numpy'
                            )
                            
                            # --- 矢量化采样逻辑 (核心性能优化) ---
                            # 使用逆变换矩阵，一次性将所有经纬度坐标转换为DEM的像素行列号
                            dem_cols, dem_rows = inv_transform * (lons, lats)
                            
                            # 将浮点型的像素坐标转换为整型，用于数组索引
                            dem_rows = dem_rows.astype(np.int64)
                            dem_cols = dem_cols.astype(np.int64)

                            # [删除] 不再需要效率低下的 Python 循环
                            
                            # 创建一个掩膜，标记出所有在DEM范围内的有效坐标
                            valid_mask = (dem_rows >= 0) & (dem_rows < dem_src.height) & \
                                         (dem_cols >= 0) & (dem_cols < dem_src.width)
                            
                            # 初始化新高程数组，并从内存中的 dem_data 数组采样
                            new_elevations = np.full_like(elevations.flatten(), dem_nodata)
                            sampled_values = dem_data[dem_rows[valid_mask], dem_cols[valid_mask]]
                            new_elevations[valid_mask] = sampled_values
                            
                            # 再次检查采样值，确保不使用DEM中的nodata值
                            final_valid_mask = new_elevations != dem_nodata
                            elevations.flat[final_valid_mask] = new_elevations[final_valid_mask]
                            # --- 采样逻辑结束 ---

                        window = Window(x_offset, y_offset, current_block_x, current_block_y)
                        output_ds.write(elevations.astype(np.float32), 1, window=window)
                        pbar.update(current_block_x * current_block_y)

                pbar.close()

    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

    print("\n高程计算完成。")
    print(f"成功创建高程文件: {output_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str)
    args = parser.parse_args()
    root = args.root

    start_time = time.time()
    names = [i.split('.rpc')[0] for i in os.listdir(root) if 'rpc' in i and 'PAN' in i]
    for name in names:
        rpc = RPCModelParameterTorch()
        rpc.load_from_file(os.path.join(root,f'{name}.rpc'))
        rpc.to_gpu()
        get_elevation(os.path.join(root,f'{name}.tiff'),os.path.join(root,'dtm_egm.tif'),os.path.join(root,f'{name}_height.tif'),rpc)
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")