import os
import logging

from utils import str2bool
os.environ['BITSANDBYTES_NOWELCOME'] = '1'
logging.basicConfig(level=logging.ERROR)
import warnings
warnings.filterwarnings("ignore")
import argparse
from distutils.util import strtobool
from copy import deepcopy
from rsea_dino_1012 import RSEA


def _strtobool(x):
    return bool(strtobool(x))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    # =============================通用与路径参数=============================
    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')
    
    parser.add_argument('--ref_image_num',type=int,default=-1)

    parser.add_argument('--dino_path', type=str, default='weights',
                        help='file containing pre-trained encoder weights')

    parser.add_argument('--encoder_path', type=str, default='weights/pretrain_swt_cnn_r2_0409_large/backbone.pth',
                        help='file containing pre-trained encoder weights')
    
    parser.add_argument('--create_grids',type=_strtobool,default=True)
    
    parser.add_argument('--crop_size', type=int, default=1024,
                        help='size of input data')
    
    parser.add_argument('--crop_num_h', type=int, default=8,
                        help='number of uniform crops along the height')
    
    parser.add_argument('--crop_num_w', type=int, default=8,
                        help='number of uniform crops along the width')
    
    parser.add_argument('--grid_size', type=int, default=3000,
                        help='step length of sliding window when cropping input data')
    
    parser.add_argument('--block_size', type=int,default=500)
    
    parser.add_argument('--max_buffer_size', type=int, default=270000,
                        help='max patch number in buffer')

    parser.add_argument('--mapper_blocks_num', type=int, default=5,
                        help='depth of the regression head, defines the map size')
    
    parser.add_argument('--grid_num',type=int,default=-1)

    parser.add_argument('--digit_num',type=int,default=3)

    parser.add_argument('--grid_offset_x',type=float,default=0)

    parser.add_argument('--grid_offset_y',type=float,default=0)
    
    parser.add_argument('--sample_factor', type=int, default=16,
                        help='Downsampling factor of the encoder feature map.')
    
    # ============================= 坐标先验与损失控制参数 =============================
    parser.add_argument('--prior_noise_min', type=float, default=1.0,
                        help='课程学习中，坐标先验噪声的初始最小值 (单位:米).')
    
    parser.add_argument('--prior_noise_max', type=float, default=100.0,
                        help='课程学习中，坐标先验噪声的最终最大值 (单位:米).')
    
    parser.add_argument('--validation_noise_std', type=float, default=10.0,
                        help='验证时，为坐标先验注入的固定噪声水平 (单位:米).')
    
    parser.add_argument('--feature_noise_level', type=float, default=0.1,
                        help='为特征向量添加的正交噪声强度，以确保余弦相似度不低于0.9。')
    
    # [新功能]: 添加高程噪声比例参数
    parser.add_argument('--height_noise_ratio', type=float, default=0.1,
                        help='定义最大高程噪声相对于当前block高程范围的比例。')

    #=============================Element Training Params=============================

    parser.add_argument('--element_train_lr_max', type=float, default=0.001,
                        help='highest learning rate')
    
    parser.add_argument('--element_train_lr_min', type=float, default=0.0001,
                        help='lowest learning rate')
    
    parser.add_argument('--element_training_iters', type=int, default=5000,
                        help='number of epochs through the training mapper')
    
    parser.add_argument('--element_warmup_iters', type=int, default=200,
                        help='number of epochs for lr climbing to lr_max')
    
    parser.add_argument('--element_summit_hold_iters', type=int, default=3800,
                        help='number of epochs for lr staying lr_max after warmup')
    

    #=============================Element Finetune Params=============================

    
    parser.add_argument('--element_finetune_lr_max', type=float, default=0.0001,
                        help='highest learning rate')
    
    parser.add_argument('--element_finetune_lr_min', type=float, default=0.000001,
                        help='lowest learning rate')
    
    parser.add_argument('--element_finetune_iters', type=int, default=1000,
                        help='number of epochs through the finetune mapper')
    
    parser.add_argument('--finetune_warmup_iters', type=int, default=50,
                        help='number of epochs for lr climbing to lr_max')
    
    parser.add_argument('--finetune_summit_hold_iters', type=int, default=150,
                        help='number of epochs for lr staying lr_max after warmup')
    
    #=============================Grid Training Params=============================

    # --- [核心修改] 更新训练参数体系为Epoch制 ---
    parser.add_argument('--num_epochs', type=int, default=10,
                        help='训练的总轮数 (epochs)。一个epoch代表模型完整看过一次所有数据。')
    
    # --- [核心修改] 区分不同阶段的batch_size ---
    parser.add_argument('--encoder_batch_size', type=int, default=8,
                        help='特征提取阶段（处理大图）的批次大小。')
    
    parser.add_argument('--mapper_batch_size', type=int, default=128,
                        help='Mapper训练阶段（处理patches）在梯度累积每一步的批次大小。')

    parser.add_argument('--validation_epoch_interval', type=int, default=1,
                        help='每隔多少个训练轮数 (epochs) 执行一次验证。')

    parser.add_argument('--grid_train_lr_max', type=float, default=0.001,
                        help='highest learning rate')
    
    parser.add_argument('--grid_train_lr_min', type=float, default=0.0001,
                        help='lowest learning rate')
    
    parser.add_argument('--grid_warmup_epochs', type=float, default=0.5,
                        help='学习率预热阶段所占的Epoch数。')
    
    parser.add_argument('--grid_cooldown_epochs', type=float, default=2.0,
                        help='学习率冷却阶段所占的Epoch数。')
    
    parser.add_argument('--consistency_weight', type=float, default=50.0,
                        help='一阶平滑损失 (loss_consistency) 的权重.')
    
    parser.add_argument('--affine_weight', type=float, default=1.0,
                        help='像方仿射一致性损失 (loss_affine) 的权重.')

    # [新功能]: 添加可视化频率控制参数
    parser.add_argument('--visualization_epoch_interval', type=int, default=0,
                        help='每隔多少个epoch输出一次可视化散点图。设置为0则禁用此功能。')

    parser.add_argument('--resume_training',type=str2bool,default=False)

    parser.add_argument('--nearest_neighbor_num',type=int,default=3)

    parser.add_argument('--save_checkpoints',type=str2bool,default=True)

    #=============================Grid Finetune Params=============================

    
    parser.add_argument('--grid_finetune_lr_max', type=float, default=0.0001,
                        help='highest learning rate')
    
    parser.add_argument('--grid_finetune_lr_min', type=float, default=0.0001,
                        help='lowest learning rate')
    
    parser.add_argument('--grid_finetune_iters', type=int, default=1000,
                        help='number of epochs through the finetune mapper')
    
    parser.add_argument('--grid_finetune_warmup_iters', type=int, default=50,
                        help='number of epochs for lr climbing to lr_max')
    
    parser.add_argument('--grid_finetune_cooldown_iters', type=int, default=500,
                        help='number of epochs for lr staying lr_max after warmup')
    
    parser.add_argument('--conf_threshold', type=float, default=0.7,
                        help='minimum confidence to filter reliable patches')
    
    parser.add_argument('--ransac_threshold', type=int, default=20,
                        help='default threshold for ransac')
    
    parser.add_argument('--residual_threshold', type=int, default=50)
    
    parser.add_argument('--ransac_iters_num', type=int, default=10000,
                        help='iterations of ransac')
    
    parser.add_argument('--use_gpu', type=_strtobool, default=True,
                        help='Use GPU for accelerating')
    
    parser.add_argument('--use_clahe', type=_strtobool, default=False)
    
    parser.add_argument('--log_postfix', type=str, default='',
                        help='log_postfix')

    # =============================新增的诊断功能参数=============================
    parser.add_argument('--run_diagnostics', type=_strtobool, default=False,
                        help='是否运行特征分布诊断. 若为True, 将不会执行adjust流程.')

    parser.add_argument('--source_image_path', type=str, default='',
                        help='[诊断用] 源域(训练)影像的文件夹路径.')
    
    parser.add_argument('--target_image_path', type=str, default='',
                        help='[诊断用] 目标域(新)影像的文件夹路径.')

    options = parser.parse_args()

    rsea = RSEA(options)

    ref_images_root = os.path.join(options.root,'ref_images')
    adjust_images_root = os.path.join(options.root,'adjust_images')
    grid_root = os.path.join(options.root,'grids')

    ref_image_folders = os.listdir(ref_images_root)
    if options.ref_image_num > 0:
        ref_image_folders = ref_image_folders[:options.ref_image_num]

    if options.create_grids:
        # 基于ref_images创建网格
        for image_folder in ref_image_folders:
            rsea.add_image(os.path.join(ref_images_root,image_folder))
        rsea.create_grids(grid_size=options.grid_size,max_grid_num=options.grid_num)

    rsea.load_grids(grid_root)

    # 根据命令行参数决定执行诊断还是调整
    if options.run_diagnostics:
        print("\n========================= 运行特征分布诊断 =========================")
        if not options.source_image_path or not options.target_image_path:
            print("错误: 运行诊断需要提供 --source_image_path 和 --target_image_path 参数。")
        else:
            rsea.visualize_feature_distribution(
                source_image_folder=options.source_image_path,
                target_image_folder=options.target_image_path
            )
        print("=========================== 诊断流程结束 ===========================")
    else:
        print("\n========================= 运行影像几何调整 =========================")
        # 基于网格对adjust_images平差
        rsea.adjust([os.path.join(adjust_images_root,i) for i in os.listdir(adjust_images_root)])
        print("=========================== 调整流程结束 ===========================")

