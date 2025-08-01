import warnings
warnings.filterwarnings("ignore")
from matplotlib import axis
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from utils import str2bool,get_coord_mat
from model_new import Encoder,AffineFitter,HomographyFitter
from tqdm import tqdm
from scheduler import MultiStageOneCycleLR
import cv2
import os
from rpc import RPCModelParameterTorch,load_rpc
from torchvision import transforms
from grid import Grid
from rs_image import RSImage
import argparse
from copy import deepcopy
from utils import estimate_affine_ransac

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

def overlay_image_with_homography(img_A, img_B, H, H_is_for_xy=True):
    """
    根据单应性矩阵 H 将图像 B 变换并镶嵌到图像 A 上。
    此函数可以处理两种坐标系的单应性矩阵。

    参数:
    img_A (numpy.ndarray): 背景图像（大图）。
    img_B (numpy.ndarray): 需要变换的前景图像（小图）。
    H (numpy.ndarray): 3x3 的单应性变换矩阵。
    H_is_for_xy (bool): 
        - True (默认): H 是为标准的 OpenCV 坐标系 (x, y) 计算的，其中 x=列, y=行。
        - False: H 是为 NumPy 索引类的坐标系 (行, 列) / (line, samp) 计算的。
                 函数将在内部将其转换为 (x, y) 格式。

    返回:
    numpy.ndarray: B 镶嵌在 A 上的结果图像。
    """
    # 获取背景图像 A 的尺寸
    height_A, width_A = img_A.shape[:2]
    
    H_cv = np.copy(H)
    # 如果 H 是为 (行, 列) 坐标系定义的，需要转换它以适配 OpenCV 的 (x, y) 坐标系
    if not H_is_for_xy:
        # 转换方法是交换矩阵的第一行和第二行，然后交换第一列和第二列
        # 这相当于 H_cv = S * H_rc * S，其中 S 是一个交换前两个坐标的矩阵
        H_cv[[0, 1], :] = H_cv[[1, 0], :] # 交换行
        H_cv[:, [0, 1]] = H_cv[:, [1, 0]] # 交换列

    # --- 1. 使用 H 对小图 B 进行透视变换 ---
    # 输出图像的尺寸 (dsize) 必须是背景图像 A 的尺寸
    warped_B = cv2.warpPerspective(img_B, H_cv, (width_A, height_A))

    # --- 2. 创建变换后区域的掩码 (Mask) ---
    # 创建一个与小图 B 等大的全白图像
    mask_B = np.ones_like(img_B, dtype=np.uint8) * 255
    # 对该白色图像进行与 B 完全相同的变换，得到目标区域的掩码
    mask_warped = cv2.warpPerspective(mask_B, H_cv, (width_A, height_A))
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

def overlay_image_with_affine(img_A, img_B, M, M_is_for_xy=True):
    """
    根据仿射变换矩阵 M 将图像 B 变换并镶嵌到图像 A 上。

    参数:
    img_A (numpy.ndarray): 背景图像（大图）。
    img_B (numpy.ndarray): 需要变换的前景图像（小图）。
    M (numpy.ndarray): 2x3 的仿射变换矩阵。
    M_is_for_xy (bool): 
        - True (默认): M 是为标准的 OpenCV 坐标系 (x, y) 计算的，其中 x=列, y=行。
        - False: M 是为 NumPy 索引类的坐标系 (行, 列) / (line, samp) 计算的。
                 函数将在内部将其转换为 (x, y) 格式。

    返回:
    numpy.ndarray: B 镶嵌在 A 上的结果图像。
    """
    # 获取背景图像 A 的尺寸
    height_A, width_A = img_A.shape[:2]
    
    M_cv = np.copy(M)
    # 如果 M 是为 (行, 列) 坐标系定义的，需要转换它以适配 OpenCV 的 (x, y) 坐标系
    if not M_is_for_xy:
        # 原始映射:
        # row' = m00*row + m01*col + m02
        # col' = m10*row + m11*col + m12
        # OpenCV 需要 (x=col, y=row):
        # x' = m11*x + m10*y + m12
        # y' = m01*x + m00*y + m02
        m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
        m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
        M_cv = np.array([
            [m11, m10, m12],
            [m01, m00, m02]
        ], dtype=np.float32)

    # --- 1. 使用 M 对小图 B 进行仿射变换 ---
    # 输出图像的尺寸 (dsize) 必须是背景图像 A 的尺寸
    warped_B = cv2.warpAffine(img_B, M_cv, (width_A, height_A))

    # --- 2. 创建变换后区域的掩码 (Mask) ---
    # 创建一个与小图 B 等大的全白图像
    # 注意：仿射变换没有3D效果，所以我们创建一个单通道的掩码就足够了
    mask_B = np.ones((img_B.shape[0], img_B.shape[1]), dtype=np.uint8) * 255
    
    # 对该白色图像进行与 B 完全相同的变换，得到目标区域的掩码
    mask_final = cv2.warpAffine(mask_B, M_cv, (width_A, height_A))
    # warpAffine 直接输出单通道，无需转换，但为保险起见可以二值化
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
    
    parser.add_argument('--max_buffer_size', type=int, default=1000000,
                        help='max patch number in buffer')

    parser.add_argument('--mapper_blocks_num', type=int, default=5,
                        help='depth of the regression head, defines the map size')
    
    parser.add_argument('--batch_size', type=int, default=12,
                        help='number of input images when extracting features')
    
    parser.add_argument('--patches_per_batch', type=int, default=2 ** 14,
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

    print("==============================options==============================")
    for k,v in vars(options).items():
        print(f"{k}:{v}")
    print("===================================================================")

    encoder = Encoder(cfg_large,output_global_feature=options.use_global_feature)
    encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(options.encoder_path).items()})
    encoder.eval()

    print("Encoder Loaded")

    transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])
    
    device = 'cuda'

    if options.create_grids:
        print("Creating Grid")
        os.makedirs(options.grid_path,exist_ok=True)
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
        print("Loading Ref Images")
        for image_folder in tqdm(os.listdir(ref_images_root)):
            img = RSImage(options,os.path.join(ref_images_root,image_folder),id=0)
            grid.add_img(img)
        grid.create_elements()
        grid.train_mapper(save_checkpoint=False)
    
    else:
        print("Loading Grid")
        grid = Grid(options = options,
                    encoder = encoder,
                    output_path = options.grid_path,
                    grid_path = options.grid_path)
    
    align_image_path = [os.path.join(options.root,'ref_images',i) for i in os.listdir(os.path.join(options.root,'ref_images'))][-1]
    align_image = RSImage(options,align_image_path,id=0)

    print("Align Image loaded")

    # whole_img = deepcopy(align_image.image)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    image_rgb = cv2.imread(options.localize_image_path)
    print(f"Localize Image Loaded, shape:{image_rgb.shape}")
    image_gray = cv2.cvtColor(image_rgb,cv2.COLOR_RGB2GRAY)
    image_gray = clahe.apply(image_gray)
    image_gray = np.stack([image_gray] * 3,axis=-1)
    H,W = image_gray.shape[:2]
    local_hw2 = get_coord_mat(H,W)
    
    print("Predictiong XYH")

    pred_res = grid.pred_dense_xyh(image_gray,local_hw2)
    mu_linesamp,sigma_linesamp = align_image.rpc.xy_distribution_to_linesamp(pred_res['mu_xyh_P3'],pred_res['sigma_xyh_P3'])
    mu_linesamp,sigma_linesamp = mu_linesamp.detach().cpu().numpy(),sigma_linesamp.detach().cpu().numpy()
    local_linesamp = pred_res['locals_P2'].detach().cpu().numpy()
    conf = pred_res['confs_P1'].detach().cpu().numpy()
    valid_score = pred_res['valid_score_P1'].detach().cpu().numpy()
    conf_score = np.linalg.norm(sigma_linesamp,axis=-1)

    fitter = AffineFitter()
    # fitter = HomographyFitter(max_iterations=-1,lr=1e-4,patience=10000)
    valid_mask = (valid_score > .1) 

    # for point in mu_linesamp[~valid_mask]:
    #     cv2.circle(whole_img,point[[1,0]],2,(0,0,255),-1)
    for point in local_linesamp[~valid_mask]:
        cv2.circle(image_gray,point[[1,0]].astype(int),1,(0,0,255),-1)

    mu_linesamp = mu_linesamp[valid_mask]
    sigma_linesamp = sigma_linesamp[valid_mask]
    local_linesamp = local_linesamp[valid_mask]
    conf_score = conf_score[valid_mask]

    #计算threshold
    offset = mu_linesamp.mean(axis=0) - local_linesamp.mean(axis=0)
    threshold = np.linalg.norm(local_linesamp + offset[None] - mu_linesamp).mean()

    print(f"avg sigma:{conf_score.mean()}")
    print(f"threshold:{threshold}")

    _,mask = cv2.findHomography(local_linesamp,mu_linesamp,cv2.RANSAC,ransacReprojThreshold=threshold)
    # _,mask = cv2.estimateAffine2D(local_linesamp.cpu().numpy(),mu_linesamp.cpu().numpy(),cv2.RANSAC,ransacReprojThreshold=conf_score.mean().item())
    inliers = mask.ravel() == 1
    outliers = mask.ravel() == 0
    print(f"inlier: {inliers.sum()}/{len(inliers)}")

    # for point in mu_linesamp[outliers]:
    #     cv2.circle(whole_img,point[[1,0]],2,(92,92,235),-1)
    for point in local_linesamp[outliers]:
        cv2.circle(image_gray,point[[1,0]].astype(int),1,(255,0,0),-1)
    for point in local_linesamp[inliers]:
        cv2.circle(image_gray,point[[1,0]].astype(int),1,(0,255,0),-1)

    mu_linesamp = mu_linesamp[inliers]
    sigma_linesamp = sigma_linesamp[inliers]
    local_linesamp = local_linesamp[inliers]

    # M,_ = cv2.estimateAffine2D(local_linesamp,mu_linesamp,cv2.RANSAC,ransacReprojThreshold=conf_score.mean())
    M = fitter.fit(torch.from_numpy(local_linesamp[:,[1,0]]).cuda(),torch.from_numpy(mu_linesamp[:,[1,0]]).cuda(),torch.from_numpy(sigma_linesamp[:,[1,0]]).cuda()).cpu().numpy()


    # pred_points = mu_linesamp.cpu().numpy().astype(int)
    # print(f"H 矩阵：\n{H}")
    # print(f"仿射变换：\n{M}")
    # print(f"inlier: {inlier_num}/{len(local_linesamp)}")


    # mix_img = overlay_image_with_homography(align_image.image,image_rgb,H,True)
    mix_img = overlay_image_with_affine(align_image.image,image_rgb,M,True)
    
    # for point in pred_points:
    #     cv2.circle(whole_img,point,5,(0,255,0),-1)

    cv2.imwrite(os.path.join(options.grid_path,'mix_img.png'),mix_img)
    # cv2.imwrite(os.path.join(options.grid_path,'whole_img.png'),whole_img)
    cv2.imwrite(os.path.join(options.grid_path,'point_img.png'),image_gray)
    





        




