import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel
import torch.distributed as dist
import numpy as np
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from criterion import CriterionFinetune
from model_new import Encoder,ProjectHead,Decoder

import datetime
import time
from tqdm import tqdm,trange
from skimage.transform import AffineTransform
from skimage.measure import ransac
from torch.cuda.amp import autocast, GradScaler
from dataloader import PretrainDataset,ImageSampler
from utils import TableLogger,kaiming_init_weights,str2bool,warp_by_extend
import random
import warnings
warnings.filterwarnings("ignore")
from scheduler import MultiStageOneCycleLR

from functools import partial
import cv2

def print_on_main(msg, rank):
    if rank == 0:
        print(msg)


cfg_base = {
            'input_channels':3,
            'patch_feature_channels':512,
            'global_feature_channels':256,
            'img_size':256,
            'window_size':8,
            'embed_dim':128,
            'depth':[2,2,18],
            'num_heads':[4,8,16],
            'drop_path_rate':.5,
            'unfreeze_backbone_modules':['head','norm','layers.2.blocks.16','layers.2.blocks.17']
        }
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
        'unfreeze_backbone_modules':['head','norm','layers.2.blocks.14','layers.2.blocks.15','layers.2.blocks.16','layers.2.blocks.17']
    }
def apply_polynomial(x, coefs):
    y = torch.zeros_like(x)
    for i, c in enumerate(coefs):
        y = y + c * (x ** (len(coefs) - 1 - i))
    return y

def warp_by_poly(raw,coefs):
    # raw[:,0] = .5 * (raw[:,0] + 1.) * (bbox['x_max'] - bbox['x_min']) + bbox['x_min']
    # raw[:,1] = .5 * (raw[:,1] + 1.) * (bbox['y_max'] - bbox['y_min']) + bbox['y_min']
    # raw[:,2] = .5 * (raw[:,2] + 1.) * (bbox['h_max'] - bbox['h_min']) + bbox['h_min']
    x = apply_polynomial(raw[:,0],coefs['x'])
    y = apply_polynomial(raw[:,1],coefs['y'])
    h = apply_polynomial(raw[:,2],coefs['h'])
    warped = torch.stack([x,y,h],dim=-1)
    return warped

def distibute_model(model:nn.Module,local_rank):
    model = DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank,broadcast_buffers=False)
    return model

def output_img(imgs_raw:torch.Tensor,output_path:str,name:str):
    os.makedirs(output_path,exist_ok=True)
    for idx,img in enumerate(imgs_raw):
        img = img.permute(1,2,0).cpu().numpy()[:,:,0]
        img = 255 * (img - img.min()) / (img.max() - img.min())
        cv2.imwrite(f'{output_path}/{name}_{idx}.png',img.astype(np.uint8))

def print_hwc_matrix(matrix: np.ndarray, precision:int = 2):
    """
    将一个形状为 (H, W, C) 的 NumPy 数组在终端中以 H*W 矩阵的格式打印出来。
    增加了对浮点数格式化的支持。

    Args:
        matrix (np.ndarray): 一个三维的 NumPy 数组，形状为 (H, W, C)。
        precision (Optional[int], optional): 
            当数组是浮点类型时，指定要保留的小数位数。
            如果为 None，则使用默认的字符串表示。默认为 None。
    """
    # 检查输入是否为三维 NumPy 数组
    if not isinstance(matrix, np.ndarray) or matrix.ndim != 3:
        print("错误：输入必须是一个三维的 NumPy 数组 (H, W, C)。")
        return

    # 获取数组的维度
    H, W, C = matrix.shape

    # 如果数组为空，则不打印
    if H == 0 or W == 0:
        print("[]")
        return
    
    string_elements = []
    for h in range(H):
        row_elements = []
        for w in range(W):
            vector = matrix[h, w]
            string_element = ""
            if precision is not None:
                # 如果指定了精度，对向量中的每个数字进行格式化
                try:
                    # 使用 f-string 的嵌套格式化功能
                    formatted_numbers = [f"{num:.{precision}f}" for num in vector]
                    string_element = f"[{' '.join(formatted_numbers)}]"
                except (ValueError, TypeError):
                    # 如果格式化失败（例如，数组不是数字类型），则退回默认方式
                    string_element = str(vector)
            else:
                # 未指定精度，使用 NumPy 默认的字符串转换
                string_element = str(vector)
            
            row_elements.append(string_element)
        string_elements.append(row_elements)

    # 找到所有字符串化后的元素中的最大长度，用于对齐
    max_len = max([len(s) for row in string_elements for s in row] or [0])

    # 打印带边框的矩阵
    print("┌" + "─" * (W * (max_len + 2) - 2) + "┐")
    for row in string_elements:
        print("│", end="")
        for element in row:
            # 使用 ljust 方法填充空格，使每个元素占据相同的宽度
            print(f"{element:<{max_len}}", end="  ")
        print("│")
    print("└" + "─" * (W * (max_len + 2) - 2) + "┘")


def pretrain(args):
    pprint = partial(print_on_main, rank=dist.get_rank())
    pprint("Loading Dataset")
    rank = dist.get_rank()

    dataset = PretrainDataset(root = args.dataset_path,
                              dataset_num = args.dataset_num,
                              batch_size = args.batch_size,
                              downsample = 16,
                              input_size = 1024,
                              mode='train')
    sampler = ImageSampler(dataset,shuffle=True)
    dataloader = DataLoader(dataset,sampler=sampler,batch_size=1,num_workers=8,drop_last=False,pin_memory=True,shuffle=False)
    dataset_num = dataset.dataset_num
    pprint("Building Encoder")

    cfg = cfg_large
    # cfg['unfreeze_backbone_modules'] = ['head','norm',*[f'layers.2.blocks.{i}' for i in unfreeze_blocks]]

    encoder = Encoder(cfg)
    projector = ProjectHead(encoder.patch_feature_channels,128)
    if not args.encoder_path is None:
        encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.encoder_path,'backbone.pth'),map_location='cpu').items()},strict=True)
        pprint('Encoder Loaded')
    os.makedirs(os.path.dirname(args.encoder_output_path),exist_ok=True)

    encoder_optimizer = optim.AdamW(params=list(encoder.get_unfreeze_parameters()) + list(projector.parameters()),lr = args.lr_encoder_max)

    encoder_scheduler = MultiStageOneCycleLR(optimizer=encoder_optimizer,
                                             max_lr=args.lr_encoder_max,
                                             steps_per_epoch=dataset_num,
                                             n_epochs_per_stage=args.max_epoch,
                                             summit_hold=0,
                                             gamma=.63 ** (1. / (200 * dataset_num)),
                                             pct_start=200. / args.max_epoch)
    

    patch_feature_channels = encoder.patch_feature_channels
    output_channels = encoder.output_channels


    encoder = encoder.to(args.device)
    projector = projector.to(args.device)
    num_gpus = dist.get_world_size()
    pprint(f"Using {num_gpus} GPUS")
    if num_gpus > 1:
        encoder = distibute_model(encoder,args.local_rank)
        projector = distibute_model(projector,args.local_rank)
    
    pprint("Building Decoders")

    if args.decoder_output_path is not None:
        os.makedirs(args.decoder_output_path,exist_ok=True)

    decoders = []
    optimizers = []
    schedulers = []
    for dataset_idx in trange(dataset_num):
        decoder = Decoder(in_channels=output_channels,block_num=1,use_bn=False)

        if not args.decoder_path is None and os.path.exists(os.path.join(args.decoder_path,f'decoder_{dataset_idx}.pth')):
            decoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.encoder_path,f'decoder_{dataset_idx}.pth')).items()})

        decoder = decoder.to(args.device)
        if num_gpus > 1:
            decoder = distibute_model(decoder,args.local_rank)

        decoder.train()
        optimizer = optim.AdamW(params=decoder.parameters(),lr = args.lr_decoder_max)
        scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                            max_lr=args.lr_decoder_max,
                                            steps_per_epoch=1,
                                            n_epochs_per_stage=args.max_epoch,
                                            summit_hold=0,
                                            gamma=.63 ** (1. / 200),
                                            pct_start=100. / args.max_epoch)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        decoders.append(decoder)

    min_loss = args.min_loss
    last_loss = None
    start_time = time.perf_counter()
    criterion = CriterionFinetune()
    os.makedirs('./log',exist_ok=True)
    logger = TableLogger('./log',['epoch','loss','loss_obj','loss_height','loss_conf','loss_feat','loss_dis','k','lr_encoder','lr_decoder'],'finetune_log')
    
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.max_epoch):
        pprint(f'\nEpoch:{epoch}')
        sampler.set_epoch(epoch)
        # valid(epoch)
        total_loss = 0
        total_loss_obj = 0
        total_loss_dis = 0
        total_loss_height = 0
        total_loss_conf = 0
        total_loss_feat = 0
        count = 0
        encoder.train()
        for iter_idx,data in enumerate(dataloader):
            img1,img2,obj1,obj2,residual1,residual2,overlap1,overlap2,dataset_idx = data
            dataset_idx = dataset_idx.item()
            img1 = img1.squeeze(0).to(args.device)
            img2 = img2.squeeze(0).to(args.device)
            obj1 = obj1.squeeze(0).to(args.device)
            obj2 = obj2.squeeze(0).to(args.device)
            residual1 = residual1.squeeze(0).to(args.device)
            residual2 = residual2.squeeze(0).to(args.device)
            overlap1 = overlap1.squeeze(0).to(args.device)
            overlap2 = overlap2.squeeze(0).to(args.device)
            B,H,W = obj1.shape[:3]

            # Info = ""
            # Info += "\n===========================DEBUG INFO===========================\n"
            # Info += f"Rank{dist.get_rank()}\n"
            # Info += f"img shape:{img1.shape}\n"
            # Info += f"obj shape:{obj.shape}\n"
            # Info += f"residual shape:{residual1.shape}\n"
            # Info += f"dataset idx:{dataset_idx}\n"
            # Info += "================================================================\n"

            # print(Info)
            

            decoder = decoders[dataset_idx]
            decoder_optimizer = optimizers[dataset_idx]
            decoder.train()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()

            # print(f"\n2---------debug:{dist.get_rank()}\n")
            # with autocast():
            feat1,conf1 = encoder(img1)
            feat2,conf2 = encoder(img2)
            # dist.barrier()

            patch_feat1,global_feat1 = feat1[:,:patch_feature_channels],feat1[:,patch_feature_channels:]
            patch_feat2,global_feat2 = feat2[:,:patch_feature_channels],feat2[:,patch_feature_channels:]

            project_feat1 = projector(patch_feat1)
            project_feat2 = projector(patch_feat2)

            patch_feat_noise_amp1 = torch.rand(patch_feat1.shape[0],1,patch_feat1.shape[2],patch_feat1.shape[3]).to(args.device) * .3
            patch_feat_noise_amp2 = torch.rand(patch_feat2.shape[0],1,patch_feat2.shape[2],patch_feat2.shape[3]).to(args.device) * .3
            global_feat_noise_amp1 = torch.rand(global_feat1.shape[0],1,global_feat1.shape[2],global_feat1.shape[3]).to(args.device) * .8
            global_feat_noise_amp2 = torch.rand(global_feat2.shape[0],1,global_feat2.shape[2],global_feat2.shape[3]).to(args.device) * .8
            patch_feat_noise1 = F.normalize(torch.normal(mean=0.,std=patch_feat1.std().item(),size=patch_feat1.shape),dim=1).to(args.device) * patch_feat_noise_amp1
            patch_feat_noise2 = F.normalize(torch.normal(mean=0.,std=patch_feat2.std().item(),size=patch_feat2.shape),dim=1).to(args.device) * patch_feat_noise_amp2
            global_feat_noise1 = F.normalize(torch.normal(mean=0.,std=global_feat1.std().item(),size=global_feat1.shape),dim=1).to(args.device) * global_feat_noise_amp1
            global_feat_noise2 = F.normalize(torch.normal(mean=0.,std=global_feat2.std().item(),size=global_feat2.shape),dim=1).to(args.device) * global_feat_noise_amp2

            feat_input1 = torch.concatenate([F.normalize(patch_feat1 + patch_feat_noise1,dim=1),F.normalize(global_feat1 + global_feat_noise1,dim=1)],dim=1)
            feat_input2 = torch.concatenate([F.normalize(patch_feat2 + patch_feat_noise2,dim=1),F.normalize(global_feat2 + global_feat_noise2,dim=1)],dim=1)
            # feat_input1 = feat1
            # feat_input2 = feat2

            pred1_P3 = []
            pred2_P3 = []
            pred_skip_1_P3 = []
            pred_skip_2_P3 = []
            
            output1_B3hw = decoder(feat_input1)
            output2_B3hw = decoder(feat_input2)
            output1_P3 = output1_B3hw.permute(0,2,3,1).flatten(0,2)
            output2_P3 = output2_B3hw.permute(0,2,3,1).flatten(0,2)

            # decoder.requires_grad_(False)
            # output_skip_1_B3hw = decoder(feat1[n * B : (n+1) * B])
            # output_skip_2_B3hw = decoder(feat2[n * B : (n+1) * B])
            # output_skip_1_P3 = output_skip_1_B3hw.permute(0,2,3,1).flatten(0,2)
            # output_skip_2_P3 = output_skip_2_B3hw.permute(0,2,3,1).flatten(0,2)
            # decoder.requires_grad_(True)
            
            obj_map_coef = dataset.obj_map_coefs[dataset_idx]

            # print(obj.shape)
            # print("bbox:",obj_bbox)

            pred1_P3.append(warp_by_poly(output1_P3,obj_map_coef))
            pred2_P3.append(warp_by_poly(output2_P3,obj_map_coef))
            # pred_skip_1_P3.append(warp_by_bbox(output_skip_1_P3,obj_bbox))
            # pred_skip_2_P3.append(warp_by_bbox(output_skip_2_P3,obj_bbox))
            
            pred1_P3 = torch.concatenate(pred1_P3,dim=0)
            pred2_P3 = torch.concatenate(pred2_P3,dim=0)
            # print_hwc_matrix(torch.concatenate([obj,pred1_P3.reshape(obj.shape)],dim=-1)[0,28:36,28:36].detach().cpu().numpy(),2)
            # print_hwc_matrix(obj[0,28:36,28:36].detach().cpu().numpy(),2)
            # print_hwc_matrix(pred1_P3.reshape(obj.shape)[0,28:36,28:36].detach().cpu().numpy(),2)
            # pred_skip_1_P3 = torch.concatenate(pred_skip_1_P3,dim=0)
            # pred_skip_2_P3 = torch.concatenate(pred_skip_2_P3,dim=0)

            # print("1:",torch.isnan(pred1_P3).any() & torch.isnan(pred2_P3).any() & torch.isnan(pred_skip_1_P3).any() & torch.isnan(pred_skip_2_P3).any())

            project_feat1_PD = project_feat1.permute(0,2,3,1).flatten(0,2)
            project_feat2_PD = project_feat2.permute(0,2,3,1).flatten(0,2)
            conf1_P = conf1.permute(0,2,3,1).reshape(-1)
            conf2_P = conf2.permute(0,2,3,1).reshape(-1)
            obj1_P3 = obj1.flatten(0,2)
            obj2_P3 = obj2.flatten(0,2)
            residual1_P = residual1.reshape(-1).detach()
            residual2_P = residual2.reshape(-1).detach()
            conf_mean = .5 * conf1_P.clone().detach().mean() + .5 * conf2_P.clone().detach().mean()
                # print(f"\n3---------debug:{dist.get_rank()}\n")

            # if rank == 0:
            #     print(torch.stack([pred1_P3,pred2_P3,obj_P3],dim=1))

            loss,loss_obj,loss_height,loss_conf,loss_feat,loss_dis,k = criterion(epoch,
                                                                                project_feat1_PD,project_feat2_PD,
                                                                                pred1_P3,pred2_P3,
                                                                                conf1_P,conf2_P,
                                                                                obj1_P3,obj2_P3,
                                                                                residual1_P,residual2_P,
                                                                                overlap1,overlap2,
                                                                                H,W)

                
            # loss_dis,dis_obj,dis_height = criterion_dis(pred_skip_1_P3,pred_skip_2_P3,residual1_P,residual2_P,k)    
            # loss_dis,dis_obj,dis_height = criterion_dis(pred1_P3,pred2_P3,residual1_P,residual2_P,k)

            # dummy_loss = 0.0
            # dummy_input = torch.zeros_like(feat_input1[:1],device=loss_normal.device,dtype=loss_normal.dtype)
            # for decoder in decoders:
            #     dummy_output = decoder(dummy_input)
            #     dummy_loss = dummy_loss + dummy_output.sum() * 0.0
                    
            if torch.isnan(loss_feat):
                print(f"nan feat loss in rank{dist.get_rank()},exit")
                exit()

                # print(f"\n4---------debug:{dist.get_rank()}\n")
            
            loss.backward()
            # scaler.scale(loss).backward()

            encoder_optimizer.step()
            decoder_optimizer.step()
            # scaler.step(encoder_optimizer)
            # for idx in dataset_idxs:
            #     scaler.step(optimizers[idx])
            # scaler.update()

            encoder_scheduler.step()
            
            loss_rec = loss.clone().detach()
            loss_obj_rec = loss_obj.clone().detach()
            loss_dis_rec = loss_dis.clone().detach()
            loss_height_rec = loss_height.clone().detach()
            loss_conf_rec = loss_conf.clone().detach()
            loss_feat_rec = loss_feat.clone().detach()

            total_loss += loss_rec
            total_loss_obj += loss_obj_rec
            total_loss_dis += loss_dis_rec
            total_loss_height += loss_height_rec
            total_loss_conf += loss_conf_rec
            total_loss_feat += loss_feat_rec
            count += 1
            # print(f"\n6---------debug:{dist.get_rank()}\n")


            dist.all_reduce(loss_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_obj_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_dis_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_height_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_conf_rec,dist.ReduceOp.AVG)
            dist.all_reduce(loss_feat_rec,dist.ReduceOp.AVG)
            dist.all_reduce(conf_mean,dist.ReduceOp.AVG)
            dist.barrier()

            if dist.get_rank() == 0:
                curtime = time.perf_counter()
                curstep = epoch * dataset_num + count
                remain_step = args.max_epoch  * dataset_num - curstep
                cost_time = curtime - start_time
                remain_time = remain_step * cost_time / curstep

                print(f"epoch:{epoch} iter:{iter_idx+1}/{dataset_num}\t l_obj:{loss_obj_rec.item():.2f} \t l_dis:{loss_dis_rec.item():.2f} \t l_h:{loss_height_rec.item():.2f} \t l_conf:{loss_conf_rec.item():.2f} \t cm:{conf_mean.item():.2f} \t k:{k:.2f} \t l_f:{loss_feat_rec.item():.2f} \t en_lr:{encoder_optimizer.param_groups[0]['lr']:.2e}  de_lr:{optimizers[0].param_groups[0]['lr']:.2e} \t time:{str(datetime.timedelta(seconds=round(cost_time)))}  ETA:{str(datetime.timedelta(seconds=round(remain_time)))}")


        for scheduler in schedulers:
            scheduler.step()            
        
        # print(f"\n7---------debug:{dist.get_rank()}\n")
        total_loss /= count
        total_loss_obj /= count
        total_loss_dis /= count
        total_loss_height /= count
        total_loss_conf /= count
        total_loss_feat /= count

        dist.all_reduce(total_loss,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_obj,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_dis,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_height,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_conf,dist.ReduceOp.AVG)
        dist.all_reduce(total_loss_feat,dist.ReduceOp.AVG)

        total_loss = total_loss.item()
        total_loss_obj = total_loss_obj.item()
        total_loss_dis = total_loss_dis.item()
        total_loss_height = total_loss_height.item()
        total_loss_conf = total_loss_conf.item()
        total_loss_feat = total_loss_feat.item()
        # print(f"\n8---------debug:{dist.get_rank()}\n")


        if dist.get_rank() == 0:
            if last_loss is None:
                print(f'total_loss:{total_loss} \t min_loss:{min_loss} \t obj:{total_loss_obj:.2f} \t dis:{total_loss_dis:.2f} \t height:{total_loss_height:.2f} \t conf:{total_loss_conf:.4f} \t feat:{total_loss_feat:.4f}')
            else:
                print(f"total_loss:{total_loss} \t diff:{'+' if total_loss - last_loss > 0 else ''}{total_loss - last_loss} \t min_loss:{min_loss} \t obj:{total_loss_obj:.2f} \t dis:{total_loss_dis:.2f} \t height:{total_loss_height:.2f} \t conf:{total_loss_conf:.4f} \t feat:{total_loss_feat:.4f}")
            last_loss = total_loss

            # torch.save(encoder.state_dict(),os.path.join(os.path.join(args.encoder_output_path,f'backbone_{epoch}.pth')))
            
            if total_loss_obj < min_loss:
                min_loss = total_loss_obj
                encoder_state_dict = {k:v.detach().cpu() for k,v in encoder.state_dict().items()}
                torch.save(encoder_state_dict,os.path.join(os.path.join(args.encoder_output_path,'backbone.pth')))
                # torch.save(encoder.state_dict(),args.encoder_output_path)
                # if not args.freeze_decoder:
                for dataset_idx in range(dataset_num):
                    torch.save(decoders[dataset_idx].state_dict(),os.path.join(args.encoder_output_path,f'decoder_{dataset_idx}.pth'))
                print('best updated')
            logger.update({
                'epoch':epoch,
                'loss':total_loss,
                'loss_obj':total_loss_obj,
                'loss_height':total_loss_height,
                'loss_dis':total_loss_dis,
                'loss_conf':total_loss_conf,
                'loss_feat':total_loss_feat,
                'k':k.item(),
                'lr_encoder':f"{encoder_optimizer.param_groups[0]['lr']:.7f}",
                'lr_decoder':f"{optimizers[0].param_groups[0]['lr']:.7f}"
            })
        # print(f"\n9---------debug:{dist.get_rank()}\n")
        dist.barrier()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--dataset_num',type=int,default=None)
    parser.add_argument('--encoder_path',type=str,default=None)
    parser.add_argument('--decoder_path',type=str,default=None)
    parser.add_argument('--encoder_output_path',type=str,default='./weights/encoder_finetune.pth')
    parser.add_argument('--decoder_output_path',type=str,default=None)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--data_batch_size',type=int,default=4)
    parser.add_argument('--use_gpu',type=bool,default=True)
    parser.add_argument('--max_epoch',type=int,default=200)
    parser.add_argument('--lr_encoder_min',type=float,default=1e-7)
    parser.add_argument('--lr_encoder_max',type=float,default=1e-4)
    parser.add_argument('--lr_decoder_min',type=float,default=1e-7)
    parser.add_argument('--lr_decoder_max',type=float,default=1e-3) #1e-3
    parser.add_argument('--min_loss',type=float,default=1e8)
    parser.add_argument("--local_rank", default=os.getenv('LOCAL_RANK', -1), type=int)

    args = parser.parse_args()
    # gpus = os.environ['CUDA_VISIBLE_DEVICES']
    # args.multi_gpu = len(gpus.split(',')) > 1

    if args.local_rank != -1:
        torch.cuda.set_device(args.local_rank)
        torch.cuda.empty_cache()
        args.device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    # if args.batch_size % 4 != 0:
    #     raise ValueError("Batch size must be divisible by 4")
    # args.batch_size = args.batch_size // 4

    pprint = partial(print_on_main, rank=dist.get_rank())

    if not os.path.exists(args.encoder_output_path):
        os.mkdir(args.encoder_output_path)

    if not os.path.exists(args.decoder_output_path):
        os.mkdir(args.decoder_output_path)

    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    pprint("==============================configs==============================")
    for k,v in vars(args).items():
        pprint(f"{k}:{v}")
    pprint("===================================================================")
    pretrain(args)