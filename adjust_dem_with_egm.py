import rasterio
import numpy as np
import argparse
import os

from rasterio.warp import reproject, Resampling

def adjust_dem_with_gravity(dem_path, global_gravity_path, output_path):
    """
    使用全球重力异常数据调整局部DEM。
    该函数会自动将全球数据重投影到DEM的范围和分辨率上。

    参数:
    dem_path (str): 输入的局部DEM高程数据路径 (TIF格式)。
    global_gravity_path (str): 输入的全球重力异常数据路径 (TIF格式)。
    output_path (str): 输出调整后DEM数据的路径 (TIF格式)。
    """
    try:
        # 1. 以读取模式打开DEM文件，它将作为我们的目标模板
        with rasterio.open(dem_path) as dem_src:
            # 获取DEM的元数据和数据
            dem_meta = dem_src.meta.copy()
            dem_data = dem_src.read(1)
            
            # 打开全球重力异常数据
            with rasterio.open(global_gravity_path) as gravity_src:
                
                # 2. 创建一个空的数组，用于存放重投影后的重力数据
                # 它的尺寸、CRS、Transform都将与DEM完全一致
                reprojected_gravity_data = np.zeros_like(dem_data)

                print("正在将全球重力数据重投影到DEM范围...")
                
                # 3. 执行重投影
                # 这个函数是关键，它处理了所有的裁剪、重采样和坐标变换
                reproject(
                    source=rasterio.band(gravity_src, 1),
                    destination=reprojected_gravity_data,
                    src_transform=gravity_src.transform,
                    src_crs=gravity_src.crs,
                    dst_transform=dem_src.transform,
                    dst_crs=dem_src.crs,
                    resampling=Resampling.bilinear  # 为连续数据（如重力）选择双线性插值
                )
                
                print("重投影完成。")

                # 4. 逐像素相加
                # 处理DEM中的nodata值
                nodata_value = dem_src.nodata
                mask = np.zeros(dem_data.shape, dtype=bool)
                if nodata_value is not None:
                    mask = dem_data == nodata_value
                    dem_data[mask] = 0  # 避免nodata值参与计算
                    reprojected_gravity_data[mask] = 0 # 对应区域也置零

                print("正在计算新的高程...")
                adjusted_dem_data = dem_data + reprojected_gravity_data
                
                # 将结果中的nodata区域重新设为nodata值
                if nodata_value is not None:
                    adjusted_dem_data[mask] = nodata_value

                # 5. 更新元数据并写入新的TIF文件
                # 输出文件的元数据与输入DEM完全一致
                output_meta = dem_src.meta.copy()
                output_meta.update({"dtype": adjusted_dem_data.dtype})

                print(f"正在将结果写入到: {output_path}")
                with rasterio.open(output_path, 'w', **output_meta) as dest:
                    dest.write(adjusted_dem_data, 1)
                
                print("任务完成！")

    except FileNotFoundError as e:
        print(f"错误：找不到文件 - {e.filename}")
    except Exception as e:
        print(f"发生了一个错误: {e}")
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dem_path',type=str)
    parser.add_argument('--egm_path',type=str)
    args = parser.parse_args()

    # --- 请在这里修改您的文件名 ---
    dem_file = args.dem_path  # 替换为您的DEM文件名
    gravity_file = args.egm_path  # 替换为您的重力异常数据文件名
    output_file = args.dem_path.replace('dem','dem_egm') # 定义输出文件名
    # --------------------------------

    adjust_dem_with_gravity(dem_file, gravity_file, output_file)