import rasterio
import numpy as np
import os
from itertools import combinations

def read_as_panchromatic(image_path):
    """
    读取多光谱影像并将其转换为全色影像（简单地取所有波段的平均值）。
    """
    with rasterio.open(image_path) as src:
        # 读取所有波段
        data = src.read()
        # 计算所有波段的平均值作为全色影像
        panchromatic_data = np.mean(data, axis=0).astype(src.profile['dtype'])
        # 复制原始影像的地理参考信息和变换信息
        profile = src.profile
        profile.update(
            count=1,  # 全色影像只有一个波段
            dtype=panchromatic_data.dtype  # 更新数据类型
        )
        return panchromatic_data, profile

def get_bounding_box(profile):
    """
    从影像的profile中获取其边界框 (minx, miny, maxx, maxy)。
    """
    transform = profile['transform']
    width = profile['width']
    height = profile['height']

    minx = transform.c
    maxy = transform.f
    maxx = transform.c + width * transform.a + height * transform.b
    miny = transform.f + width * transform.d + height * transform.e

    return minx, miny, maxx, maxy

def find_overlap(bbox1, bbox2, min_overlap_size_m, transform1, transform2):
    """
    计算两个边界框的重叠区域，并检查是否满足最小尺寸要求。
    返回重叠区域的地理坐标 (min_overlap_x, min_overlap_y, max_overlap_x, max_overlap_y)
    以及像素尺寸。
    """
    minx1, miny1, maxx1, maxy1 = bbox1
    minx2, miny2, maxx2, maxy2 = bbox2

    # 计算重叠区域的地理坐标
    overlap_minx = max(minx1, minx2)
    overlap_miny = max(miny1, miny2)
    overlap_maxx = min(maxx1, maxx2)
    overlap_maxy = min(maxy1, maxy2)

    # 检查是否有重叠
    if overlap_minx >= overlap_maxx or overlap_miny >= overlap_maxy:
        return None, None

    # 计算重叠区域的宽度和高度（米）
    overlap_width_m = overlap_maxx - overlap_minx
    overlap_height_m = overlap_maxy - overlap_miny

    # 将米转换为像素
    # 由于不同影像可能分辨率不同，这里取第一个影像的像素尺寸进行粗略估计
    pixel_width_1 = abs(transform1.a)
    pixel_height_1 = abs(transform1.e)
    overlap_width_pixels = int(overlap_width_m / pixel_width_1)
    overlap_height_pixels = int(overlap_height_m / pixel_height_1)

    # 检查重叠区域尺寸是否满足要求
    if overlap_width_m >= min_overlap_size_m and overlap_height_m >= min_overlap_size_m:
        return (overlap_minx, overlap_miny, overlap_maxx, overlap_maxy), (overlap_width_pixels, overlap_height_pixels)
    else:
        return None, None

def extract_and_save_overlap(image_path, overlap_bbox, output_folder, output_suffix, original_profile):
    """
    从原始影像中提取重叠区域并保存为新的全色GeoTIFF文件。
    """
    minx, miny, maxx, maxy = overlap_bbox

    with rasterio.open(image_path) as src:
        # 使用窗口读取重叠区域数据
        # rasterio.windows.from_bounds 能够根据地理坐标计算出对应的像素窗口
        window = src.window(minx, miny, maxx, maxy)
        data = src.read(window=window)
        
        # 将多光谱数据转换为全色
        panchromatic_data = np.mean(data, axis=0).astype(original_profile['dtype'])

        # 更新profile以匹配重叠区域
        transform, width, height = rasterio.windows.calculate_default_transform(
            src.crs, src.transform, src.width, src.height, window=window
        )
        
        output_profile = original_profile.copy()
        output_profile.update({
            'height': height,
            'width': width,
            'transform': transform,
            'count': 1,
            'dtype': panchromatic_data.dtype
        })

        output_filename = os.path.join(output_folder, f"{os.path.basename(image_path).replace('.tif', '')}_{output_suffix}.tif")
        with rasterio.open(output_filename, 'w', **output_profile) as dst:
            dst.write(panchromatic_data, 1)
        print(f"Saved overlapping region to: {output_filename}")


def process_images(input_folder, output_folder_pairs, output_folder_triples):
    """
    处理文件夹中的所有影像，查找两两和三三重叠区域。
    """
    if not os.path.exists(output_folder_pairs):
        os.makedirs(output_folder_pairs)
    if not os.path.exists(output_folder_triples):
        os.makedirs(output_folder_triples)

    image_files = [os.path.join(input_folder, f) for f in os.listdir(input_folder) if f.endswith('.tif')]

    # 存储影像信息： (全色数据, profile, 原始路径)
    image_data = []
    for img_path in image_files:
        try:
            pan_data, profile = read_as_panchromatic(img_path)
            image_data.append({'path': img_path, 'pan_data': pan_data, 'profile': profile})
            print(f"Successfully read and converted {os.path.basename(img_path)} to panchromatic.")
        except rasterio.errors.RasterioIOError as e:
            print(f"Error reading {os.path.basename(img_path)}: {e}. Skipping this file.")
            continue

    # --- 寻找两两重叠区域 ---
    print("\n--- Searching for pairwise overlapping regions (>= 5000m x 5000m) ---")
    processed_pairs = set() # To avoid processing the same pair twice (img1, img2) vs (img2, img1)
    for i in range(len(image_data)):
        for j in range(i + 1, len(image_data)):
            img1_info = image_data[i]
            img2_info = image_data[j]

            # Use a tuple of sorted paths to ensure uniqueness
            pair_key = tuple(sorted((img1_info['path'], img2_info['path'])))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)

            bbox1 = get_bounding_box(img1_info['profile'])
            bbox2 = get_bounding_box(img2_info['profile'])

            overlap_bbox, overlap_pixels = find_overlap(bbox1, bbox2, 5000, img1_info['profile']['transform'], img2_info['profile']['transform'])

            if overlap_bbox:
                print(f"Found significant overlap between {os.path.basename(img1_info['path'])} and {os.path.basename(img2_info['path'])}")
                print(f"  Overlap BBox (geographic): {overlap_bbox}")
                print(f"  Approximate Overlap Size (pixels): {overlap_pixels[0]}x{overlap_pixels[1]}")

                # 提取并保存重叠区域
                extract_and_save_overlap(img1_info['path'], overlap_bbox, output_folder_pairs, f"overlap_{os.path.basename(img2_info['path']).replace('.tif', '')}", img1_info['profile'])
                extract_and_save_overlap(img2_info['path'], overlap_bbox, output_folder_pairs, f"overlap_{os.path.basename(img1_info['path']).replace('.tif', '')}", img2_info['profile'])
            else:
                print(f"No significant overlap found between {os.path.basename(img1_info['path'])} and {os.path.basename(img2_info['path'])}")

    # --- 尝试寻找三张影像重叠区域 ---
    print("\n--- Searching for triple overlapping regions (>= 1000m x 1000m) ---")
    processed_triples = set()
    for img1_info, img2_info, img3_info in combinations(image_data, 3):
        # Create a unique key for the triple
        triple_key = tuple(sorted((img1_info['path'], img2_info['path'], img3_info['path'])))
        if triple_key in processed_triples:
            continue
        processed_triples.add(triple_key)

        bboxes = [get_bounding_box(img1_info['profile']),
                  get_bounding_box(img2_info['profile']),
                  get_bounding_box(img3_info['profile'])]
        
        # Calculate the intersection of all three bounding boxes
        overlap_minx = max(bbox[0] for bbox in bboxes)
        overlap_miny = max(bbox[1] for bbox in bboxes)
        overlap_maxx = min(bbox[2] for bbox in bboxes)
        overlap_maxy = min(bbox[3] for bbox in bboxes)

        if overlap_minx >= overlap_maxx or overlap_miny >= overlap_maxy:
            print(f"No triple overlap found for {os.path.basename(img1_info['path'])}, {os.path.basename(img2_info['path'])}, {os.path.basename(img3_info['path'])}")
            continue

        overlap_width_m = overlap_maxx - overlap_minx
        overlap_height_m = overlap_maxy - overlap_miny

        # For triple overlap, we'll use the first image's resolution for pixel estimation
        pixel_width_1 = abs(img1_info['profile']['transform'].a)
        pixel_height_1 = abs(img1_info['profile']['transform'].e)
        overlap_width_pixels = int(overlap_width_m / pixel_width_1)
        overlap_height_pixels = int(overlap_height_m / pixel_height_1)

        if overlap_width_m >= 1000 and overlap_height_m >= 1000:
            overlap_bbox_triple = (overlap_minx, overlap_miny, overlap_maxx, overlap_maxy)
            print(f"Found significant triple overlap for {os.path.basename(img1_info['path'])}, {os.path.basename(img2_info['path'])}, {os.path.basename(img3_info['path'])}")
            print(f"  Overlap BBox (geographic): {overlap_bbox_triple}")
            print(f"  Approximate Overlap Size (pixels): {overlap_pixels[0]}x{overlap_pixels[1]}")

            # 提取并保存三张影像的重叠区域
            extract_and_save_overlap(img1_info['path'], overlap_bbox_triple, output_folder_triples, f"triple_overlap_{os.path.basename(img2_info['path']).replace('.tif', '')}_{os.path.basename(img3_info['path']).replace('.tif', '')}", img1_info['profile'])
            extract_and_save_overlap(img2_info['path'], overlap_bbox_triple, output_folder_triples, f"triple_overlap_{os.path.basename(img1_info['path']).replace('.tif', '')}_{os.path.basename(img3_info['path']).replace('.tif', '')}", img2_info['profile'])
            extract_and_save_overlap(img3_info['path'], overlap_bbox_triple, output_folder_triples, f"triple_overlap_{os.path.basename(img1_info['path']).replace('.tif', '')}_{os.path.basename(img2_info['path']).replace('.tif', '')}", img3_info['profile'])
        else:
            print(f"No significant triple overlap found for {os.path.basename(img1_info['path'])}, {os.path.basename(img2_info['path'])}, {os.path.basename(img3_info['path'])}")


if __name__ == "__main__":
    # 配置输入和输出文件夹
    input_directory = "./data/dadukou/raw" # 替换为你的影像文件夹路径
    output_directory_pairs = "./data/dadukou/pairs"
    output_directory_triples = "./data/dadukou/triples"

    process_images(input_directory, output_directory_pairs, output_directory_triples)

    print("\n处理完成！请查看输出文件夹中的结果。")