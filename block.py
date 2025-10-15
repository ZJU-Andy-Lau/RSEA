from enum import Enum
import warnings

import scheduler
warnings.filterwarnings('ignore')
import argparse
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from model_new import Encoder,Decoder
import os
import cv2
from datetime import datetime,timedelta
import time
from utils import get_coord_mat,project_mercator,mercator2lonlat,downsample,bilinear_interpolate,apply_polynomial,get_map_coef

from rpc import RPCModelParameterTorch
from tqdm import tqdm,trange
from scheduler import MultiStageOneCycleLR
from torch.optim import AdamW,lr_scheduler
from criterion_0927 import CriterionTrainOneImg,CriterionTrainElement,CriterionTrainGrid
import torch.nn.functional as F
from orthorectify import orthorectify_image
import rasterio
from scipy.interpolate import RegularGridInterpolator
from copy import deepcopy
from torchvision import transforms
import kornia.augmentation as K
from matplotlib import pyplot as plt
import random
from typing import List,Dict

from rs_image import RSImage
from element_1012 import Element
from utils import Status


class Block():
    def __init__(self,options,diag:np.ndarray,diag_ratio:np.ndarray,map_coeffs):
        """
        diag:[[tl_x,tl_y],[br_x,br_y]]
        diag_pix:[[tl_line_ratio,tl_samp_ratio],[br_line_ratio,br_samp_ratio]]
        """
        self.options = options
        self.diag = diag
        self.diag_ratio = diag_ratio 
        
        # --- [核心修改] 增加Mapper的输入通道数以容纳坐标先验 ---
        # 坐标先验 (x, y, h) 增加了3个输入通道
        prior_channels = 3
        mapper_total_input_channels = options.mapper_input_channel + prior_channels
        
        self.mapper = Decoder(in_channels=mapper_total_input_channels, block_num=options.mapper_blocks_num)
        
        self.border = np.array([self.diag[:,0].min(),self.diag[:,1].min(),self.diag[:,0].max(),self.diag[:,1].max()])#[min_x,min_y,max_x,max_y]
        
        # --- [核心修改] 初始化map_coeffs以包含高程范围 ---
        if map_coeffs is None:
            self.map_coeffs = {
                'x': None, 'y': None, 'h': None,
                'h_min': None, 'h_max': None
            }
        else:
            self.map_coeffs = map_coeffs
            if 'h_min' not in self.map_coeffs:
                self.map_coeffs['h_min'] = None
            if 'h_max' not in self.map_coeffs:
                self.map_coeffs['h_max'] = None

        self.status = Status.NOT_INIT
    
    def get_block_state_dict(self):
        """
        [核心修改] 在状态字典中增加高程范围的保存
        """
        state_dict = {
            'mapper':self.mapper.state_dict(),
            'diag':torch.from_numpy(self.diag),
            'diag_ratio':torch.from_numpy(self.diag_ratio),
            'map_coeffs_x':torch.from_numpy(self.map_coeffs['x']),
            'map_coeffs_y':torch.from_numpy(self.map_coeffs['y']),
            'map_coeffs_h':torch.from_numpy(self.map_coeffs['h']),
            'map_coeffs_h_min': torch.tensor(self.map_coeffs['h_min']),
            'map_coeffs_h_max': torch.tensor(self.map_coeffs['h_max']),
            'status':self.status
        }
        return state_dict
