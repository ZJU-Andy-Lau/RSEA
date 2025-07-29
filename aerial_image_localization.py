import warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import str2bool,get_coord_mat
from model_new import Encoder,HomographyFitter
from tqdm import tqdm
from scheduler import MultiStageOneCycleLR
import cv2
import os
from rpc import RPCModelParameterTorch,load_rpc
from torchvision import transforms
from grid import Grid
from rs_image import RSImage
import argparse

cfg_large = {
        'input_channels':3,
        'patch_feature_channels':512,
        'global_feature_channels':256,
        'img_size':1024,
        'window_size':16,
        'embed_dim':192,
        'depth':[2,2,18],
        'num_heads':[6, 12, 24],
        'drop_path_rate':.2,
        'pretrain_window_size':[12, 12, 12],
        'unfreeze_backbone_modules':[]
    }

def overlay_image_with_homography(img_A, img_B, H):
    """
    根据单应性矩阵 H 将图像 B 变换并镶嵌到图像 A 上。

    参数:
    img_A (numpy.ndarray): 背景图像（大图）。
    img_B (numpy.ndarray): 需要变换的前景图像（小图）。
    H (numpy.ndarray): 3x3 的单应性变换矩阵，从 B 的坐标系变换到 A 的坐标系。

    返回:
    numpy.ndarray: B 镶嵌在 A 上的结果图像。
    """
    # 获取背景图像 A 的尺寸
    height_A, width_A = img_A.shape[:2]

    # --- 1. 使用 H 对小图 B 进行透视变换 ---
    # 输出图像的尺寸 (dsize) 必须是背景图像 A 的尺寸
    warped_B = cv2.warpPerspective(img_B, H, (width_A, height_A))

    # --- 2. 创建变换后区域的掩码 (Mask) ---
    # 创建一个与小图 B 等大的全白图像
    mask_B = np.ones_like(img_B, dtype=np.uint8) * 255
    # 对该白色图像进行与 B 完全相同的变换，得到目标区域的掩码
    mask_warped = cv2.warpPerspective(mask_B, H, (width_A, height_A))
    # 将掩码转为单通道灰度图
    mask_final = cv2.cvtColor(mask_warped, cv2.COLOR_BGR2GRAY)
    # 为确保掩码是纯黑白的，进行二值化处理
    _, mask_final = cv2.threshold(mask_final, 1, 255, cv2.THRESH_BINARY)

    # --- 3. 融合图像 ---
    # 创建掩码的反转，用于在 A 中“挖洞”
    mask_inv = cv2.bitwise_not(mask_final)

    # 使用反转的掩码，将大图 A 中需要被覆盖的区域变黑
    img_A_holed = cv2.bitwise_and(img_A, img_A, mask=mask_inv)

    # 使用原始掩码，提取出变换后 B 中的有效像素
    img_B_extracted = cv2.bitwise_and(warped_B, warped_B, mask=mask_final)

    # 将“挖洞”后的 A 和提取出的 B 相加，完成镶嵌
    result = cv2.add(img_A_holed, img_B_extracted)

    return result

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')

    parser.add_argument('--encoder_path', type=str, default='weights/pretrain_swt_cnn_r2_0409_large/backbone.pth',
                        help='file containing pre-trained encoder weights')
    
    parser.add_argument('--create_grids',type=str2bool,default=True)
    
    parser.add_argument('--crop_size', type=int, default=1024,
                        help='size of input data')
    
    parser.add_argument('--crop_step', type=int, default=0,
                        help='step length of sliding window when cropping input data')
    
    parser.add_argument('--grid_size', type=int, default=2000,
                        help='step length of sliding window when cropping input data')
    
    parser.add_argument('--max_buffer_size', type=int, default=270000,
                        help='max patch number in buffer')

    parser.add_argument('--mapper_blocks_num', type=int, default=5,
                        help='depth of the regression head, defines the map size')
    
    parser.add_argument('--batch_size', type=int, default=8,
                        help='number of input images when extracting features')
    
    parser.add_argument('--patches_per_batch', type=int, default=2 ** 13,
                        help='number of patches in a batch')
    
    parser.add_argument('--use_global_feature',type=str2bool,default=False)

    parser.add_argument('--grid_train_lr_max', type=float, default=0.001,
                        help='highest learning rate')
    
    parser.add_argument('--grid_train_lr_min', type=float, default=0.0001,
                        help='lowest learning rate')
    
    parser.add_argument('--grid_training_iters', type=int, default=10000,
                        help='number of epochs through the finetune mapper')
    
    parser.add_argument('--grid_warmup_iters', type=int, default=200,
                        help='number of epochs for lr climbing to lr_max')
    
    parser.add_argument('--grid_summit_hold_iters', type=int, default=8800,
                        help='number of epochs for lr staying lr_max after warmup')
    
    parser.add_argument('--grid_cooldown_iters', type=int, default=1000,
                        help='number of epochs for lr staying lr_max after warmup')
    
    parser.add_argument('--resume_training',type=str2bool,default=False)

    parser.add_argument('--nearest_neighbor_num',type=int,default=3)
    
    parser.add_argument('--conf_threshold', type=float, default=0.7,
                        help='minimum confidence to filter reliable patches')
    
    parser.add_argument('--grid_path',type=str,default='./datasets/wv_al')

    parser.add_argument('--localize_image_path',type=str,required=True)
    
    options = parser.parse_args()

    encoder = Encoder(cfg_large,output_global_feature=options.use_global_feature)
    encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(options.encoder_path).items()})
    encoder.eval()

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    device = 'cuda'

    if options.create_grids:
        os.makedirs(options.grid_path)
        diag = np.array([
            [-1.3580166e+07,4.4906410e+06],
            [-1.3578166e+07,4.4886410e+06]
        ])

        grid = Grid(options,
                    encoder = encoder,
                    output_path = options.grid_path,
                    diag = diag,
                    device=device)
        
        ref_images_root = os.path.join(options.root,'ref_images')
        for image_folder in os.listdir(ref_images_root):
            img = RSImage(options,os.path.join(ref_images_root,image_folder))
            grid.add_img(img)
        grid.create_elements()
        grid.train_mapper()
    
    else:
        grid = Grid(options = options,
                    encoder = encoder,
                    output_path = options.grid_path,
                    grid_path = options.grid_path)
    
    align_image_path = [os.path.join(options.root,'ref_images',i) for i in os.listdir(os.path.join(options.root,'ref_images'))][-1]
    align_image = RSImage(options,align_image_path,id=0)

    image_rgb = cv2.imread(options.localize_image_path)
    image_gray = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)
    image_gray = np.stack([image_gray] * 3,axis=-1)
    H,W = image_gray.shape[:2]
    local_hw2 = get_coord_mat(H,W)
    
    pred_res = grid.pred_xyh(image_gray,local_hw2)
    mu_linesamp,sigma_linesamp = align_image.rpc.xy_distribution_to_linesamp(pred_res['mu_xyh_P3'],pred_res['sigma_xyh_P3'])
    local_linesamp = pred_res['locals_P2']
    conf = pred_res['confs_P1']
    valid_score = pred_res['valid_score_P1']

    fitter = HomographyFitter(max_epochs=100,lr=0.01)
    valid_mask = valid_score > .5
    mu_linesamp = mu_linesamp[valid_mask]
    sigma_linesamp = sigma_linesamp[valid_mask]
    local_linesamp = local_linesamp[valid_mask]

    H = fitter.fit(local_linesamp,mu_linesamp,sigma_linesamp)

    mix_img = overlay_image_with_homography(align_image.image,image_rgb,H)

    cv2.imwrite(os.path.join(options.grid_path,'mix_img.png'),mix_img)





        




