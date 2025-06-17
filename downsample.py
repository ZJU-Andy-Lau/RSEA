import os
import re
import argparse
from PIL import Image
import numpy as np
import cv2
from scipy import ndimage
import matplotlib.pyplot as plt
from skimage.transform import resize, downscale_local_mean
from tqdm import tqdm

def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='遥感影像下采样工具')
    parser.add_argument('--input', '-i', required=True, help='输入PNG影像文件夹')
    parser.add_argument('--output', '-o', required=True, help='输出PNG影像文件夹')
    parser.add_argument('--method', '-m', choices=['nearest', 'bilinear', 'bicubic', 'mean', 'gaussian'], 
                        default='bicubic', help='下采样方法 (默认: bicubic)')
    parser.add_argument('--factor', '-f', type=int,default=2,help='下采样倍率')
    # parser.add_argument('--visualize', '-v', action='store_true', help='可视化结果对比')
    return parser.parse_args()

def nearest_neighbor_downsample(img, factor=2):
    """最近邻抽取下采样"""
    h, w = img.shape[:2]
    return img[::factor, ::factor]

def bilinear_downsample(img, factor=2):
    """双线性插值下采样"""
    h, w = img.shape[:2]
    return cv2.resize(img, (w//factor, h//factor), interpolation=cv2.INTER_LINEAR)

def bicubic_downsample(img, factor=2):
    """双三次插值下采样"""
    h, w = img.shape[:2]
    return cv2.resize(img, (w//factor, h//factor), interpolation=cv2.INTER_CUBIC)

def mean_downsample(img, factor=2):
    """均值下采样"""
    return downscale_local_mean(img, (factor, factor)).astype(np.uint8)

def gaussian_downsample(img, factor=2):
    """高斯滤波后下采样"""
    # 高斯滤波，sigma = 0.5*factor 以匹配抗混叠标准
    img_smooth = ndimage.gaussian_filter(img, sigma=0.5*factor)
    return img_smooth[::factor, ::factor]

def process_rpc(input_file, output_file, processors):
    try:
        with open(input_file, 'r') as f:
            lines = f.readlines()
        
        modified_lines = []
        for line in lines:
            processed = False
            for pattern, processor in processors.items():
                match = re.search(pattern, line)
                if match:
                    try:
                        value = float(match.group(1))
                        processed_value = processor(value)
                        modified_line = line.replace(match.group(1), f"{processed_value:.2f}")
                        modified_lines.append(modified_line)
                        processed = True
                        break
                    except (ValueError, IndexError) as e:
                        print(f"Warning: Failed to process line '{line.strip()}': {e}")
            if not processed:
                modified_lines.append(line)
        
        with open(output_file, 'w') as f:
            f.writelines(modified_lines)
        
        # print(f"成功处理rpc，结果已保存至: {output_file}")
    except Exception as e:
        print(f"处理rpc时出错: {e}")

def downsample_res(matrix, k, method='mean'):
    """
    对含NaN值的矩阵进行k倍下采样
    
    参数:
    matrix (np.ndarray): 输入的(N,M)矩阵
    k (int): 下采样倍数
    method (str): 处理方法，可选'mean'（默认）或'median'
    
    返回:
    np.ndarray: 下采样后的矩阵
    """
    N, M = matrix.shape
    
    # 计算下采样后的尺寸
    new_N = N // k
    new_M = M // k
    
    # 重塑矩阵以便于分块处理
    reshaped = matrix[:new_N*k, :new_M*k].reshape(new_N, k, new_M, k)
    
    # 根据指定方法进行下采样
    if method == 'mean':
        # 计算每个块的均值，忽略NaN
        result = np.nanmean(reshaped, axis=(1, 3))
    else:
        # 计算每个块的中位数，忽略NaN
        result = np.nanmedian(reshaped, axis=(1, 3))
    
    return result

def main():
    args = parse_arguments()
    
    # 检查输入文件是否存在
    if not os.path.exists(args.input):
        print(f"错误: 输入路径 '{args.input}' 不存在")
        return
    
    processors = {
        r'LINE_SCALE:\s+([\d.]+)\s+pixels': lambda x: x / args.factor,  
        r'SAMP_SCALE:\s+([\d.]+)\s+pixels': lambda x: x / args.factor,  
        r'LINE_OFF:\s+([\d.]+)\s+pixels': lambda x: x / args.factor,
        r'SAMP_OFF:\s+([\d.]+)\s+pixels': lambda x: x / args.factor,           
    }

    # 选择下采样方法
    method_mapping = {
        'nearest': nearest_neighbor_downsample,
        'bilinear': bilinear_downsample,
        'bicubic': bicubic_downsample,
        'mean': mean_downsample,
        'gaussian': gaussian_downsample,
        # 'wavelet': wavelet_downsample
    }

    names = [i.split('.')[0] for i in os.listdir(args.input) if 'png' in i]
    for name in tqdm(names):
        img = cv2.imread(os.path.join(args.input,f'{name}.png'),cv2.IMREAD_GRAYSCALE)
        dem = np.load(os.path.join(args.input,f'{name}_dem.npy'))
        res = np.load(os.path.join(args.input,f'{name}_res.npy'))

        downsampled_img = method_mapping[args.method](img,args.factor)
        downsampled_dem = bilinear_downsample(img,args.factor)
        downsampled_res = downsample_res(res,args.factor)

        cv2.imwrite(os.path.join(args.output,f'{name}.png'),downsampled_img)
        np.save(os.path.join(args.output,f'{name}_dem.npy'),downsampled_dem)
        np.save(os.path.join(args.output,f'{name}_res.npy'),downsampled_res)

        process_rpc(os.path.join(args.input,f'{name}.rpc'),os.path.join(args.output,f'{name}.rpc'),processors)


if __name__ == "__main__":
    main()    