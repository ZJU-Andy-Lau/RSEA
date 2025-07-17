import warnings
warnings.filterwarnings("ignore")
import argparse
from distutils.util import strtobool
import os
from copy import deepcopy
from rsea import RSEA


def _strtobool(x):
    return bool(strtobool(x))

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--root', type=str,
                        help='path to all images needed adjustment in a folder')

    parser.add_argument('--encoder_path', type=str, default='weights/pretrain_swt_cnn_r2_0409_large/backbone.pth',
                        help='file containing pre-trained encoder weights')
    
    parser.add_argument('--create_grids',type=_strtobool,default=True)
    
    parser.add_argument('--crop_size', type=int, default=1024,
                        help='size of input data')
    
    parser.add_argument('--crop_step', type=int, default=0,
                        help='step length of sliding window when cropping input data')
    
    parser.add_argument('--grid_size', type=int, default=3000,
                        help='step length of sliding window when cropping input data')
    
    parser.add_argument('--max_buffer_size', type=int, default=270000,
                        help='max patch number in buffer')

    parser.add_argument('--mapper_blocks_num', type=int, default=5,
                        help='depth of the regression head, defines the map size')
    
    parser.add_argument('--batch_size', type=int, default=16,
                        help='number of input images when extracting features')
    
    parser.add_argument('--patches_per_batch', type=int, default=2 ** 13,
                        help='number of patches in a batch')
    

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
    
    parser.add_argument('--grid_cool_down_iters', type=int, default=1000,
                        help='number of epochs for lr staying lr_max after warmup')
    
    # parser.add_argument('--lr_decay_per_100_epochs', type=float, default=0.85,
    #                     help='factor of lr decay in every 10 epochs')
    
    parser.add_argument('--conf_threshold', type=float, default=0.7,
                        help='minimum confidence to filter reliable patches')
    
    parser.add_argument('--ransac_threshold', type=int, default=20,
                        help='default threshold for ransac')
    
    parser.add_argument('--ransac_iters_num', type=int, default=10000,
                        help='iterations of ransac')
    
    parser.add_argument('--use_gpu', type=_strtobool, default=True,
                        help='Use GPU for accelerating')
    
    parser.add_argument('--log_postfix', type=str, default='',
                        help='log_postfix')
    

    options = parser.parse_args()

    rsea = RSEA(options)

    ref_images_root = os.path.join(options.root,'ref_images')
    adjust_images_root = os.path.join(options.root,'adjust_images')
    grid_root = os.path.join(options.root,'grids')

    if options.create_grids:
        # 基于ref_images创建网格
        for image_folder in os.listdir(ref_images_root):
            rsea.add_image(os.path.join(ref_images_root,image_folder))
        rsea.create_grids(grid_size=options.grid_size)
    else:
        #加载网格
        rsea.load_grids(grid_root)

    # 基于网格对adjust_images平差
    rsea.adjust([os.path.join(adjust_images_root,i) for i in os.listdir(adjust_images_root)])

    # adjust_images = rsea.adjust([os.path.join(options.root,i) for i in image_folders],options)
    # rsea.check_error(os.path.join(options.root,f'log_{options.log_postfix}.csv'),adjust_images)


   
    
    