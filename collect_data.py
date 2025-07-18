import numpy as np
import h5py
import cv2
import os
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--root',type=str)
    parser.add_argument('--output_path',type=str)
    args = parser.parse_args()

    root = args.root
    output_path = args.output_path
    os.makedirs(output_path,exist_ok=True)
    
    file_paths = os.listdir(root)[:30]
    pbar = tqdm(total=len(file_paths))
    with h5py.File(os.path.join(output_path,'train_data.h5'),'w') as f:
        for idx,file in enumerate(file_paths):
            path = os.path.join(root,file)
            grp = f.create_group(f'{idx}')
            img_paths = [os.path.join(path,i) for i in os.listdir(path) if 'png' in i]
            res_paths = [os.path.join(path,i) for i in os.listdir(path) if 'res' in i]
            if len(img_paths) != len(res_paths):
                raise ValueError(f"image num should be equal to residual num, but get {len(img_paths)} images and {len(res_paths)} residuals")
            imgs = [cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) for img_path in img_paths]
            residuals = [np.load(res_path) for res_path in res_paths]
            obj = np.load(os.path.join(path,'obj.npy')).astype(np.float32)
            
            img_grp = grp.create_group('images')
            res_grp = grp.create_group('residuals')
            for img_idx,img in enumerate(imgs):
                img_grp.create_dataset(name=f"image_{img_idx}",data=img)
            for res_idx,res in enumerate(residuals):
                res_grp.create_dataset(name=f'residual_{res_idx}',data=res)
            grp.create_dataset(name="obj",data=obj)

            pbar.update(1)
                