import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
import torch
from rpc import RPCModelParameterTorch
import os
import argparse

def read_tif(tif_path,tl,br):
    window = Window(tl[1],tl[0],br[1]-tl[1],br[0]-tl[0])
    with rasterio.open(tif_path) as src:
        data = src.read(window = window)
        pan_data = np.mean(data,axis=0).astype(src.profile('dtype'))
        return pan_data

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str)
    args = parser.parse_args()

    root = args.root
    names = [i.split('.')[0] for i in os.listdir(root) if 'rpc' in i and 'PAN' in i]
    src1 = rasterio.open(os.path.join(root,f'{names[0]}.tiff'))
    src2 = rasterio.open(os.path.join(root,f'{names[1]}.tiff'))
    rpc1 = RPCModelParameterTorch()
    rpc2 = RPCModelParameterTorch()
    rpc1.load_from_file(os.path.join(root,f'{names[0]}.rpc'))
    rpc2.load_from_file(os.path.join(root,f'{names[1]}.rpc'))
    H1,W1 = src1.height,src1.width
    H2,W2 = src2.height,src2.width
    

        