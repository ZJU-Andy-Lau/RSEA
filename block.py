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
from criterion import CriterionTrainOneImg,CriterionTrainElement,CriterionTrainGrid
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
from pykeops.torch import LazyTensor

from rs_image import RSImage
from element import Element
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
        self.mapper = Decoder(in_channels=options.mapper_input_channel,digit_num=options.digit_num,block_num=options.mapper_blocks_num)
        # self.optimizer = AdamW(self.mapper.parameters(),lr=self.options.grid_train_lr_max)
        # self.scheduler = MultiStageOneCycleLR(optimizer=self.optimizer,
        #                                         total_steps=self.options.grid_training_iters,
        #                                         warmup_ratio=self.options.grid_warmup_iters / self.options.grid_training_iters,
        #                                         cooldown_ratio=self.options.grid_cooldown_iters / self.options.grid_training_iters)
        self.border = np.array([self.diag[:,0].min(),self.diag[:,1].min(),self.diag[:,0].max(),self.diag[:,1].max()])#[min_x,min_y,max_x,max_y]
        self.map_coeffs = map_coeffs
        self.status = Status.NOT_INIT
    
    def get_block_state_dict(self):
        state_dict = {
            'mapper':self.mapper.state_dict(),
            'diag':torch.from_numpy(self.diag),
            'diag_ratio':torch.from_numpy(self.diag_ratio),
            'map_coeffs_x':torch.from_numpy(self.map_coeffs['x']),
            'map_coeffs_y':torch.from_numpy(self.map_coeffs['y']),
            'map_coeffs_h':torch.from_numpy(self.map_coeffs['h']),
            'status':self.status
        }
        return state_dict