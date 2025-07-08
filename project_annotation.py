import numpy as np
from rpc import RPCModelParameterTorch
import os
import argparse
from tqdm import tqdm
import cv2

def load_points(txt_path):
    points = []
    with open(txt_path, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) == 2:
                row, col = int(parts[0]), int(parts[1])
                points.append([row,col])
    return np.array(points,dtype=int)

def save_points(txt_path,points:np.ndarray):
    with open(txt_path,'w') as f:
        output = ""
        for point in points:
            output += f"{int(point[0])},{int(point[1])}\n"
        f.write(output)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str)
    args = parser.parse_args()
    root = args.root
    output = os.path.join(root,'vis')
    os.makedirs(output,exist_ok=True)

    dem = np.load(os.path.join(root,'img1_dem.npy'))
    img1 = cv2.imread(os.path.join(root,'img1.png'))
    img2 = cv2.imread(os.path.join(root,'img2.png'))

    rpc1 = RPCModelParameterTorch()
    rpc2 = RPCModelParameterTorch()
    rpc1.load_from_file(os.path.join(root,'img1.rpc'))
    rpc2.load_from_file(os.path.join(root,'img2.rpc'))
    rpc1.to_gpu()
    rpc2.to_gpu()
    
    tie_points1 = load_points(os.path.join(root,'img1.txt'))
    heights = dem[tie_points1[:,0],tie_points1[:,1]]
    lats,lons = rpc1.RPC_PHOTO2OBJ(tie_points1[:,1],tie_points1[:,0],heights,'numpy')
    tie_points2 = np.stack(rpc2.RPC_OBJ2PHOTO(lats,lons,heights,'numpy'),axis=-1)[:,[1,0]].astype(int)

    save_points(os.path.join(root,'img2.txt'),tie_points2)

    for point in tqdm(tie_points1):
        cv2.circle(img1,(int(point[1]),int(point[0])),2,(0,255,0),-1)
    
    for point in tqdm(tie_points2):
        cv2.circle(img2,(int(point[1]),int(point[0])),2,(0,255,0),-1)

    cv2.imwrite(os.path.join(output,'vis1.png'),img1)
    cv2.imwrite(os.path.join(output,'vis2.png'),img2)


    

