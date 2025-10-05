import os
import warnings
warnings.filterwarnings("ignore")
import math
import h5py
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
from model.encoder_dino_0927 import EncoderDino
from model.decoders import DecoderFinetune
from utils import apply_polynomial,get_map_coef,bilinear_interpolate,visualize_subset_points,get_current_time
from tqdm import tqdm
from scheduler import MultiStageOneCycleLR
import cv2
from torchvision import transforms

@torch.no_grad()
def extract_features(args,img_tensor:torch.Tensor):
    """
    input: img_tensor 1,3,H,W
    output: features 1,D,h,w
    """
    encoder = EncoderDino(dino_weight_path=args.dino_weight_path)
    encoder.load_adapter(os.path.join(args.encoder_path,'adapter.pth'))
    encoder = encoder.cuda().eval()

    upsample_times = int(math.log2(encoder.SAMPLE_FACTOR) - math.log2(args.downsample))

    img_tensor = img_tensor.cuda()
    features = []

    for idx in range(img_tensor.shape[0]):
        feature,_ = encoder(img_tensor[idx:idx+1])
        features.append(feature)
    
    features = torch.concatenate(features,dim=0)

    for _ in range(upsample_times):
        features = F.interpolate(features,scale_factor = 2,mode = 'bilinear')

    return features

def crop_train_data(image, k):
    """
    在每条边上取k等分点，依次将每个点作为裁切图像的左上角，
    其他边上也取对应点作为其他顶点，裁切一圈的影像。

    参数:
    image (np.ndarray): 输入的OpenCV图像 (H, W, C)。
    k (int): 每条边的分段数。k=3 表示每条边有2个等分点+2个角点。

    返回:
    list: 包含所有裁切和变换后的图像的列表。
    """
    def get_point_on_perimeter(index, k, W, H):
        if not k > 0:
            raise ValueError("k 必须是正整数")

        # 上边 (0 <= index < k)
        if index < k:
            # t 是在当前边上的进度 (0.0 到 1.0)
            t = index / k
            return (W * t, 0)
        
        # 右边 (k <= index < 2*k)
        elif index < 2 * k:
            t = (index - k) / k
            return (W, H * t)

        # 下边 (2*k <= index < 3*k)
        elif index < 3 * k:
            t = (index - 2 * k) / k
            # 从右向左移动，所以是 1-t
            return (W * (1 - t), H)
            
        # 左边 (3*k <= index < 4*k)
        else:
            t = (index - 3 * k) / k
            # 从下向上移动，所以是 1-t
            return (0, H * (1 - t))
    if not isinstance(k, int) or k <= 0:
        print("参数 k 必须是一个正整数。")
        return []

    H, W = image.shape[:2]
    total_steps = 4 * k
    cropped_images = []

    # 目标图像的四个角点是固定的
    dst_points = np.float32([
        [0, 0],      # 左上角
        [W - 1, 0],      # 右上角
        [W - 1, H - 1],      # 右下角
        [0, H - 1]       # 左下角
    ])

    for i in range(total_steps):
        # 计算当前裁切四边形的四个源顶点
        # p1 是左上角顶点
        # p2 是右上角顶点, 领先 k 步
        # p3 是右下角顶点, 领先 2k 步
        # p4 是左下角顶点, 领先 3k 步
        p1_idx = i
        p2_idx = (i + k) % total_steps
        p3_idx = (i + 2 * k) % total_steps
        p4_idx = (i + 3 * k) % total_steps

        p1 = get_point_on_perimeter(p1_idx, k, W - 1, H - 1)
        p2 = get_point_on_perimeter(p2_idx, k, W - 1, H - 1)
        p3 = get_point_on_perimeter(p3_idx, k, W - 1, H - 1)
        p4 = get_point_on_perimeter(p4_idx, k, W - 1, H - 1)

        src_points = np.float32([p1, p2, p3, p4])

        # 计算并应用透视变换
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        warped_image = cv2.warpPerspective(image, M, (W, H))
        
        cropped_images.append(warped_image)

    return cropped_images

def crop_test_data(image):
    H, W = image.shape[:2]

    src_points = np.float32([
        [W / 2, 0],      # 上边中点
        [W - 1, H / 2],      # 右边中点
        [W / 2, H - 1],      # 下边中点
        [0, H / 2]       # 左边中点
    ])

    dst_points = np.float32([
        [0, 0],          # 左上角
        [W - 1, 0],          # 右上角
        [W - 1, H - 1],          # 右下角
        [0, H - 1]           # 左下角
    ])
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    warped_image = cv2.warpPerspective(image, M, (W, H))

    return warped_image

def centerize_obj(obj:np.ndarray):
    x = obj[...,0]
    y = obj[...,1]
    h = obj[...,2]
    print(f"x:{x.shape} \t x_max:{x.max()} \t x_min:{x.min()} \t mean:{(x - (x.max() + x.min()) * .5).mean()}")
    x = x - (x.max() + x.min()) * .5
    y = y - (y.max() + y.min()) * .5
    return np.stack([x,y,h],axis=-1)

def warp_by_poly(raw,coefs):
    x = (raw[:,0] + 1.) * .5 * (coefs['x'][1] - coefs['x'][0]) + coefs['x'][0]
    y = (raw[:,1] + 1.) * .5 * (coefs['y'][1] - coefs['y'][0]) + coefs['y'][0]
    h = apply_polynomial(raw[:,2],coefs['h'])
    warped = torch.stack([x,y,h],dim=-1)
    return warped

def downsample(arr,ds):
    if ds <= 0:
        return arr
    H,W = arr.shape[:2]
    lines = np.arange(0,H - ds + 1,ds) + (ds - 1.) * 0.5
    samps = np.arange(0,W - ds + 1,ds) + (ds - 1.) * 0.5
    sample_idxs = np.stack(np.meshgrid(samps,lines,indexing='xy'),axis=-1).reshape(-1,2) # x,y
    arr_ds = bilinear_interpolate(arr,sample_idxs)
    arr_ds = arr_ds.reshape(len(lines),len(samps),-1).squeeze()
    return arr_ds


def train(args,features,gt_objs,test_features,test_obj,map_coeffs):
    decoder = DecoderFinetune(in_channels=features.shape[1],block_num=args.decoder_block_num)
    decoder = decoder.cuda()
    
    epochs = args.epochs
    optimizer = optim.Adam(decoder.parameters(), lr=args.lr)
    scheduler = MultiStageOneCycleLR(optimizer,
                                     total_steps=epochs,
                                     warmup_ratio=.1,
                                     cooldown_ratio=.7)
    
    gt_objs = gt_objs.flatten(0,2).cuda()
    # criterion = nn.MSELoss()
    min_loss = 1e9
    for epoch in range(epochs):
        optimizer.zero_grad()
        output = decoder(features)
        output = output.permute(0,2,3,1).flatten(0,2)
        pred_obj = warp_by_poly(output,map_coeffs)
        loss = torch.norm(pred_obj - gt_objs,dim=1).mean()
        loss.backward()
        optimizer.step()
        scheduler.step()
        if (epoch + 1) % (epochs // 10) == 0: # 打印10轮日志
            print(f"Epoch [{epoch+1}/{epochs}] | Loss: {loss:.4f} | min Loss: {min_loss:.4f}")
            evaluate(args,decoder,test_features,test_obj,map_coef,epochs // 10)

        if loss < min_loss:
            best_state_dict = decoder.state_dict()
            min_loss = loss
    
    decoder.load_state_dict(best_state_dict)
    return decoder

@torch.no_grad()
def evaluate(args,decoder:DecoderFinetune,features,gt_objs,map_coeffs,idx=0):
    output = decoder(features)
    output = output.permute(0,2,3,1).flatten(0,2)
    pred_obj = warp_by_poly(output,map_coeffs)
    gt_objs = gt_objs.flatten(0,1).cuda()
    # print(f"pred:{pred_obj.mean(dim=0)} \n gt:{gt_objs.mean(dim=0)}")
    dis = torch.norm(pred_obj - gt_objs,dim=1)
    print(f"| eval | dis: mean:{dis.mean().item():.2f} \t median:{dis.median().item():.2f} \t min:{dis.min().item():.2f} \t max:{dis.max().item():.2f}")
    visualize_subset_points(pred_obj.cpu().numpy(),gt_objs.cpu().numpy(),os.path.join(args.output_path,f"{args.test_name}_res_{idx}.png"),point_radius=2)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="perfomance test")
    parser.add_argument('--encoder_path',type=str,default=None)
    parser.add_argument('--dino_weight_path',type=str,default='./weights/dinov3_vitl16_pretrain_sat493m-eadcf0ff.pth')
    parser.add_argument('--img_path',type=str,default='./performance_test_data/0')
    parser.add_argument('--output_path',type=str,default='./datasets/performance_test_res')
    parser.add_argument('--test_name',type=str,default=None)
    # parser.add_argument('--dataset_num',type=int,default=None)
    # parser.add_argument('--output_dir', type=str, default='./trained_decoders', help='保存训练好的Decoder权重的目录')
    parser.add_argument('--window_size', type=int, default=1024, help='Encoder的输入窗口大小')
    parser.add_argument('--epochs', type=int, default=1000, help='每个Decoder的训练轮数')
    parser.add_argument('--lr', type=float, default=1e-4, help='学习率')
    # parser.add_argument('--batch_size', type=int, default=4, help='训练时的批量大小')
    parser.add_argument('--decoder_block_num',type=int,default=1)
    parser.add_argument('--downsample',type=int,default=16)
    args = parser.parse_args()

    DOWNSAMPLE = args.downsample
    os.makedirs(args.output_path,exist_ok = True)
    if args.test_name is None:
        args.test_name = get_current_time()

    img = cv2.imread(os.path.join(args.img_path,'image.png'))
    imgs_train = crop_train_data(img,k=3)
    img_test = crop_test_data(img)


    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),(0.229, 0.224, 0.225))
                ])
    
    img_train_tensor = torch.stack([transform(img_train) for img_train in imgs_train],dim=0)
    img_test_tensor = transform(img_test)[None]

    obj = np.load(os.path.join(args.img_path,'obj.npy'))
    obj = centerize_obj(obj)

    objs_train = crop_train_data(obj,k = 3)
    obj_test = crop_test_data(obj)

    map_coef = {
            'x':np.array([obj[:,:,0].min(),obj[:,:,0].max()]),
            'y':np.array([obj[:,:,1].min(),obj[:,:,1].max()]),
            'h':get_map_coef(obj[:,:,2].reshape(-1))
        }
    


    obj_train_downsample = torch.stack([torch.from_numpy(downsample(obj_train,DOWNSAMPLE)) for obj_train in objs_train],dim=0)
    obj_test_downsample = torch.from_numpy(downsample(obj_test,DOWNSAMPLE))

    train_features = extract_features(args,img_train_tensor)
    test_features = extract_features(args,img_test_tensor)

    decoder = train(args,train_features,obj_train_downsample,test_features,obj_test_downsample,map_coef)
    evaluate(args,decoder,test_features,obj_test_downsample,map_coef,'final')
    

    

    