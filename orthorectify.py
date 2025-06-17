import numpy as np
import rasterio
from rasterio.transform import from_origin
from rasterio.enums import Resampling
from pyproj import CRS
from tqdm import tqdm
from scipy.interpolate import RegularGridInterpolator
from rpc import RPCModelParameterTorch

def orthorectify_image(image, dem, rpc:RPCModelParameterTorch, output_path, resolution=0.5):
    """
    基于RPC模型与DEM正射校正遥感影像

    参数：
        image (ndarray): 原始影像 (H, W)
        dem (ndarray): DEM数据 (H, W)，与影像同覆盖范围，单位为米
        rpc: RPC对象，包含RPC_PHOTO2OBJ和RPC_OBJ2PHOTO方法
        output_path (str): 输出正射影像的tif路径
        resolution (float): 输出影像的空间分辨率，单位米
    """
    H, W = image.shape

    # 影像行列号索引
    lines = np.arange(H)
    samps = np.arange(W)

    # DEM范围，假设DEM与影像一一对应
    dem_min = dem.min()
    dem_max = dem.max()

    # 影像四角像素坐标
    corner_lines = np.array([0, 0, H-1, H-1])
    corner_samps = np.array([0, W-1, 0, W-1])
    corner_dems = np.array([dem_min, dem_min, dem_max, dem_max])

    # 四角地理坐标
    corner_lats, corner_lons = rpc.RPC_PHOTO2OBJ(corner_samps, corner_lines, corner_dems,'numpy')

    # 输出影像地理范围
    lat_min = corner_lats.min()
    lat_max = corner_lats.max()
    lon_min = corner_lons.min()
    lon_max = corner_lons.max()

    # 输出影像尺寸
    out_H = int(np.ceil((lat_max - lat_min) / (resolution / 111320)))  # 1 deg ≈ 111.32 km
    out_W = int(np.ceil((lon_max - lon_min) / (resolution / (40075000 * np.cos(np.deg2rad((lat_max + lat_min) / 2)) / 360))))

    # 生成输出影像格网地理坐标
    out_lat_grid = lat_max - np.arange(out_H) * (resolution / 111320)
    out_lon_grid = lon_min + np.arange(out_W) * (resolution / (40075000 * np.cos(np.deg2rad((lat_max + lat_min) / 2)) / 360))

    # DEM插值器
    dem_lines = np.arange(H)
    dem_samps = np.arange(W)
    dem_interpolator = RegularGridInterpolator((dem_lines, dem_samps), dem, bounds_error=False, fill_value=dem_min)

    # 原始影像插值器（双线性）
    image_interpolator = RegularGridInterpolator((lines, samps), image, method='linear', bounds_error=False, fill_value=0)

    # 定义输出影像
    ortho_image = np.zeros((out_H, out_W), dtype=image.dtype)

    # 网格化计算，分块处理提高内存效率
    block_size = 1024
    for i in tqdm(range(0, out_H, block_size)):
        i_end = min(i + block_size, out_H)
        for j in range(0, out_W, block_size):
            j_end = min(j + block_size, out_W)

            grid_lats = out_lat_grid[i:i_end].reshape(-1, 1)
            grid_lons = out_lon_grid[j:j_end].reshape(1, -1)

            grid_latlon = np.stack([np.repeat(grid_lats, j_end-j, axis=1),
                                    np.repeat(grid_lons, i_end-i, axis=0)], axis=-1)

            # DEM值（最近邻插值）
            dem_points = np.stack([
                (H-1) * (lat_max - grid_latlon[..., 0]) / (lat_max - lat_min),
                (W-1) * (grid_latlon[..., 1] - lon_min) / (lon_max - lon_min)
            ], axis=-1)
            dem_vals = dem_interpolator(dem_points.reshape(-1, 2)).reshape(dem_points.shape[:-1])

            # 投影到影像像素行列号
            samps_pred, lines_pred = rpc.RPC_OBJ2PHOTO(grid_latlon[..., 0].ravel(),
                                                       grid_latlon[..., 1].ravel(),
                                                       dem_vals.ravel(),'numpy')

            photo_points = np.stack([lines_pred, samps_pred], axis=-1)
            pixel_vals = image_interpolator(photo_points).reshape(grid_latlon.shape[:-1])

            ortho_image[i:i_end, j:j_end] = pixel_vals.astype(image.dtype)

    # 写入GeoTIFF
    transform = from_origin(out_lon_grid[0], out_lat_grid[0],
                            (out_lon_grid[1] - out_lon_grid[0]),
                            (out_lat_grid[0] - out_lat_grid[1]))

    with rasterio.open(
        output_path, 'w',
        driver='GTiff',
        height=out_H,
        width=out_W,
        count=1,
        dtype=image.dtype,
        crs=CRS.from_epsg(4326),
        transform=transform
    ) as dst:
        dst.write(ortho_image, 1)

    print(f"正射影像已保存至 {output_path}")