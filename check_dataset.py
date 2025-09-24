import numpy as np
import h5py
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
from utils import get_current_time
from dataloader import residual_average
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default=None)
    parser.add_argument('--img_idx',type=int,default=None)
    parser.add_argument('--output_folder',type=str,default=None)
    args = parser.parse_args()

    data = h5py.File(args.dataset_path,'r')
    keys = list(data.keys())

    if args.img_idx is None:
        img_idx = np.random.randint(0,len(keys)-1)
    else:
        img_idx = args.img_idx

    view_num = len(data[keys[img_idx]]['images'].keys())
    print(f"数据集中共有 {len(keys)} 个样本，当前选择key: {keys[img_idx]}，包含 {view_num} 个视角。")
    os.makedirs(args.output_folder,exist_ok=True)
    for view_idx in range(view_num):
        img = data[keys[img_idx]]['images'][f'image_{view_idx}'][:]
        cv2.imwrite(os.path.join(args.output_folder,f'image_{keys[img_idx]}_{view_idx}.png'),img)
    
    # img = data[keys[img_idx]]['images'][f'image_{args.view_idx}'][:]
    # residual_raw = data[keys[img_idx]]['residuals'][f'residual_{args.view_idx}'][:]
    # H,W = img.shape[:2]
    # img = img[H // 2 - 512 : H // 2 + 512, W // 2 - 512 : W // 2 + 512]
    # residual_raw = residual_raw[H // 2 - 512 : H // 2 + 512, W // 2 - 512 : W // 2 + 512]
    # img = np.stack([img] * 3,axis=-1)
    # residual_raw = clamp_res(residual_raw)

    
    # vis_raw(img,residual_raw,args.output_folder)
    # vis_mask(img,residual_average(residual_raw,16),args.output_folder)



    
