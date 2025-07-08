import sys
import time
import rasterio
from rasterio.windows import Window
import numpy as np
from tqdm import tqdm
import argparse
import os
from rpc import RPCModelParameterTorch

def get_elevation(image_path: str, dem_path: str, output_path: str, rpc:RPCModelParameterTorch):
    """
    使用 Rasterio，根据遥感影像的RPC模型，在DEM上采样高程。

    Args:
        image_path (str): 带有RPC元数据的遥感影像路径 (TIF)。
        dem_path (str): DEM数据路径 (TIF)。
        output_path (str): 输出高程文件的路径 (TIF)。
    """
    try:
        # --- 1. 使用 Rasterio 打开文件进行 I/O 操作 ---
        print("开始处理 (使用自定义RPC类和Rasterio)...")
        with rasterio.open(image_path) as image_src, \
             rasterio.open(dem_path) as dem_src:

            print(f"源影像: {image_path}")
            print(f"源DEM: {dem_path}")
            print(f"输出高程图: {output_path}")

            # --- 2. 准备元数据和初始值 ---
            img_width = image_src.width
            img_height = image_src.height
            profile = image_src.profile
            profile.update(dtype=rasterio.float32, count=1, compress='lzw', tiled=True)

            print("正在加载DEM到内存...")
            dem_data = dem_src.read(1)
            dem_nodata = dem_src.nodata or -9999.0
            valid_dem_data = dem_data[dem_data != dem_nodata]
            initial_elevation = valid_dem_data.mean() if valid_dem_data.size > 0 else 0
            print(f"使用DEM平均高程 {initial_elevation:.2f} 米作为初始值。")

            # --- 3. 创建输出文件并分块处理 ---
            with rasterio.open(output_path, 'w', **profile) as output_ds:
                block_size_x, block_size_y = 512, 512
                
                print("开始分块计算高程...")
                pbar = tqdm(total=img_width * img_height, desc="处理像素")

                for y_offset in range(0, img_height, block_size_y):
                    for x_offset in range(0, img_width, block_size_x):
                        current_block_y = min(block_size_y, img_height - y_offset)
                        current_block_x = min(block_size_x, img_width - x_offset)

                        samps, lines = np.meshgrid(
                            np.arange(x_offset, x_offset + current_block_x),
                            np.arange(y_offset, y_offset + current_block_y)
                        )
                        
                        # 假设RPC模型使用像素中心坐标，这通常能提供更高精度
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
                            # 使用 Rasterio 的方法将地理坐标转为DEM像素索引
                            dem_rows, dem_cols = dem_src.index(lons, lats)
                            dem_rows, dem_cols = np.array(dem_rows), np.array(dem_cols)

                            valid_mask = (dem_rows >= 0) & (dem_rows < dem_src.height) & \
                                         (dem_cols >= 0) & (dem_cols < dem_src.width)
                            
                            new_elevations = np.full_like(elevations.flatten(), dem_nodata)
                            sampled_values = dem_data[dem_rows[valid_mask], dem_cols[valid_mask]]
                            new_elevations[valid_mask] = sampled_values
                            
                            final_valid_mask = new_elevations != dem_nodata
                            elevations.flat[final_valid_mask] = new_elevations[final_valid_mask]

                        window = Window(x_offset, y_offset, current_block_x, current_block_y)
                        output_ds.write(elevations.astype(np.float32),1, window=window)
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
        get_elevation(os.path.join(root,f'{name}.tiff'),os.path.join(root,'dem_egm.tif'),os.path.join(root,f'{name}_dem.tif'),rpc)
    end_time = time.time()
    print(f"总耗时: {end_time - start_time:.2f} 秒")