import numpy as np
import torch
from rpc import RPCModelParameterTorch,load_rpc
import os
import argparse
from tqdm import tqdm,trange
from typing import List

def load_tie_points(path):
    tie_points = np.loadtxt(path,dtype=int)
    if tie_points.ndim == 1:
        tie_points = tie_points.reshape(1,-1)
    elif tie_points.shape[1] != 2:
        print("tie points format error")
        return None
    return tie_points.astype(np.float32)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str)
    args = parser.parse_args()

    root = args.root
    view_num = len(os.listdir(root))
    rpc_paths = [os.path.join(root,i,'rpc.txt') for i in os.listdir(root)]
    tie_point_paths = [os.path.join(root,i,'tie_points.txt') for i in os.listdir(root)]
    dem_paths = [os.path.join(root,i,'dem.npy') for i in os.listdir(root)]

    rpcs:List[RPCModelParameterTorch] = [load_rpc(path,to_gpu=True) for path in rpc_paths]
    tie_points:List[np.ndarray] = [load_tie_points(path) for path in tie_point_paths]
    dems:List[np.ndarray] = [np.load(path,mmap_mode='r') for path in dem_paths]

    xys = []
    for view_idx in trange(view_num):
        points = tie_points[view_idx]
        h = dems[view_idx][points[:,0].astype(int),points[:,1].astype(int)]
        x,y = rpcs[view_idx].RPC_LINESAMP2XY(points[:,0],points[:,1],h,'numpy')
        xys.append(np.stack([x,y],axis=-1))
    
    for i in range(view_num - 1):
        for j in range(i+1,view_num):
            dis = np.linalg.norm(xys[i] - xys[j],axis=-1)
            print(f"==================Error of Image {i} and Image {j}==================")
            print(f"max:{dis.max()}\nmin:{dis.min()}\nmean:{dis.mean()}\nmedian:{np.median(dis)}\n<1m:{(dis < 1.).sum() * 1. / len(dis)}\n<3m:{(dis < 3.).sum() * 1. / len(dis)}\n<5m:{(dis < 5.).sum() * 1. / len(dis)}")
            print("====================================================================")
