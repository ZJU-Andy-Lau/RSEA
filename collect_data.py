import numpy as np
import h5py
import cv2
import os
from tqdm import tqdm
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # parser.add_argument('--root',type=str)
    parser.add_argument('--output_path',type=str)
    parser.add_argument('--file_num',type=int,default=-1)
    args = parser.parse_args()

    roots = ['../RSEA/datasets/beijing_2000','../RSEA/datasets/guangzhou_n_2000','../RSEA/datasets/guangzhou_s_2000','../RSEA/datasets/wv_2000'] #
    count = 0
    f = h5py.File(os.path.join(args.output_path,'train_data.h5'),'w')
    for root in roots:
        # root = args.root
        file_num = args.file_num
        os.makedirs(args.output_path,exist_ok=True)
        
        file_paths = os.listdir(root)
        if file_num > 0:
            file_paths = file_paths[:file_num]
        pbar = tqdm(total=len(file_paths))
        for idx,file in enumerate(file_paths):
            path = os.path.join(root,file)
            grp = f.create_group(f'{count}')
            img_paths = [os.path.join(path,i) for i in os.listdir(path) if 'png' in i]
            res_paths = [os.path.join(path,i) for i in os.listdir(path) if 'res' in i]
            if len(img_paths) != len(res_paths):
                raise ValueError(f"image num should be equal to residual num, but get {len(img_paths)} images and {len(res_paths)} residuals")
            imgs = [cv2.imread(img_path,cv2.IMREAD_GRAYSCALE) for img_path in img_paths]
            residuals = [np.load(res_path) for res_path in res_paths]
            for res in residuals:
                res[res < 0] = np.nan
            obj = np.load(os.path.join(path,'obj.npy')).astype(np.float32)
            
            img_grp = grp.create_group('images')
            res_grp = grp.create_group('residuals')
            for img_idx,img in enumerate(imgs):
                img_grp.create_dataset(name=f"image_{img_idx}",data=img)
            for res_idx,res in enumerate(residuals):
                res_grp.create_dataset(name=f'residual_{res_idx}',data=res)
            grp.create_dataset(name="obj",data=obj)

            pbar.update(1)
            count += 1
                