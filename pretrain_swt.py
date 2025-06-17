import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import torch.nn.functional as F
import torch.nn.init as init
import numpy as np
from torch.utils.data import Dataset,DataLoader
from criterion import CriterionFinetune
from model_new import Encoder0324,Encoder0409,Decoder

import cv2
import datetime
import time
from tqdm import tqdm
from skimage.transform import AffineTransform
from skimage.measure import ransac
from PIL import Image
from torch.cuda.amp import autocast, GradScaler
from dataloader import PretrainDataset
from utils import TableLogger,kaiming_init_weights,str2bool,warp_by_extend
import random
import warnings
warnings.filterwarnings("ignore")
from scheduler import MultiStageOneCycleLR

from torch.amp import autocast, GradScaler
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
def pretrain(args):
    dataset_num = args.dataset_num
    dataset = PretrainDataset(args.dataset_path,dataset_num,args.view_num,downsample=16,mode='train')
    valid_dataset = PretrainDataset(f'{args.dataset_path}_valid',dataset_num,args.view_num,downsample=16,mode='valid')
    dataloader = DataLoader(dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last=True)
    valid_dataloader = DataLoader(valid_dataset,batch_size=args.batch_size,shuffle=True,num_workers=4,drop_last=True)
    batch_num = len(dataloader)
    # unfreeze_blocks = np.arange(16,18,dtype=int)
    cfg = cfg_large
    # cfg['unfreeze_backbone_modules'] = ['head','norm',*[f'layers.2.blocks.{i}' for i in unfreeze_blocks]]
    encoder = Encoder0409(cfg)
    if not args.encoder_path is None:
        encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.encoder_path,'backbone.pth')).items()},strict=True)
        print('Encoder Loaded')
    if not os.path.exists(os.path.dirname(args.encoder_output_path)):
        os.mkdir(os.path.dirname(args.encoder_output_path))

    encoder_optimizer = optim.AdamW(params=encoder.get_unfreeze_parameters(),lr = args.lr_encoder_max)

    encoder_scheduler = MultiStageOneCycleLR(optimizer=encoder_optimizer,
                                             max_lr=args.lr_encoder_max,
                                             steps_per_epoch=batch_num,
                                             n_epochs_per_stage=args.max_epoch,
                                             summit_hold=0,
                                             gamma=.63 ** (1. / (50 * batch_num )),
                                             pct_start=30. / args.max_epoch)
    

    patch_feature_channels = encoder.patch_feature_channels
    output_channels = encoder.output_channels

    if args.use_gpu:
        encoder = encoder.cuda()
        if args.multi_gpu:
            encoder = nn.DataParallel(encoder)
    
    if args.decoder_output_path is not None and not os.path.exists(args.decoder_output_path):
        os.mkdir(args.decoder_output_path)
    decoders = []
    optimizers = []
    schedulers = []
    for dataset_idx in range(dataset_num):
        decoder = Decoder(in_channels=output_channels,block_num=5,use_bn=False)
        # if args.decoder_path is not None:
        if not args.decoder_path is None and os.path.exists(os.path.join(args.decoder_path,f'decoder_{dataset_idx}.pth')):
            decoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.encoder_path,f'decoder_{dataset_idx}.pth')).items()})
            print(f"Decoder {os.path.join(args.encoder_path,f'decoder_{dataset_idx}.pth')} Loaded")
        #     else:
        #         print(f"Decoder {args.decoder_paths[dataset_idx]} Not Exist!")

        if args.use_gpu:
            decoder = decoder.cuda()
            if args.multi_gpu:
                decoder = nn.DataParallel(decoder)

        decoder.train()
        optimizer = optim.AdamW(params=decoder.parameters(),lr = args.lr_decoder_max)
        scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                            max_lr=args.lr_decoder_max,
                                            steps_per_epoch=batch_num,
                                            n_epochs_per_stage=args.max_epoch,
                                            summit_hold=0,
                                            gamma=.63 ** (1. / (30 * batch_num)),
                                            pct_start=3. / args.max_epoch)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        decoders.append(decoder)

    @torch.no_grad()
    def valid(epoch):
        encoder.eval()
        total_loss = 0
        total_loss_obj = 0
        total_loss_height = 0
        # total_loss_photo = 0
        # total_loss_bias = 0
        # total_loss_reg = 0
        total_loss_conf = 0
        total_loss_dis = 0
        total_loss_adj = 0
        total_loss_feat = 0
        total_sp = 0
        total_sn = 0
        total_sa = 0
        total_loss_strcut = 0
        count = 0
        for batch_idx,items in enumerate(tqdm(valid_dataloader)):
            imgs,locals,objs,residuals,view_idxs = items
            for dataset_idx in range(dataset_num):
                img1 = imgs[dataset_idx]['v1'].contiguous()
                img2 = imgs[dataset_idx]['v2'].contiguous()
                obj1 = objs[dataset_idx]['v1'].contiguous()
                obj2 = objs[dataset_idx]['v2'].contiguous()
                local1 = locals[dataset_idx]['v1'].contiguous()
                local2 = locals[dataset_idx]['v2'].contiguous()
                residual1 = residuals[dataset_idx]['v1'].contiguous()
                residual2 = residuals[dataset_idx]['v2'].contiguous()
                v1 = view_idxs[dataset_idx]['v1']
                v2 = view_idxs[dataset_idx]['v2']

                rpcs = dataset.datasets[dataset_idx]['rpcs']
                extend = dataset.datasets[dataset_idx]['extend']

                decoder= decoders[dataset_idx]
                decoder.eval()

                if args.use_gpu:
                    img1 = img1.cuda()
                    img2 = img2.cuda()
                    obj1 = obj1.cuda()
                    obj2 = obj2.cuda()
                    local1 = local1.cuda()
                    local2 = local2.cuda()
                    residual1 = residual1.cuda()
                    residual2 = residual2.cuda()
                    for rpc in rpcs:
                        rpc.to_gpu()
                    extend = extend.cuda()
                
                B,H,W = local1.shape[0],local1.shape[1],local1.shape[2]
                
                with autocast('cuda'):
                    feat1,conf1 = encoder(img1)
                    feat2,conf2 = encoder(img2)
                    output1_B3hw = decoder(feat1)
                    output2_B3hw = decoder(feat2)
                    output1_P3 = output1_B3hw.permute(0,2,3,1).flatten(0,2)
                    output2_P3 = output2_B3hw.permute(0,2,3,1).flatten(0,2)
                    pred1_P3 = warp_by_extend(output1_P3,extend)
                    pred2_P3 = warp_by_extend(output2_P3,extend)
                    feat1_PD = feat1[:,:patch_feature_channels].permute(0,2,3,1).flatten(0,2)
                    feat2_PD = feat2[:,:patch_feature_channels].permute(0,2,3,1).flatten(0,2)
                    conf1_P = conf1.permute(0,2,3,1).reshape(-1)
                    conf2_P = conf2.permute(0,2,3,1).reshape(-1)
                    obj1_P3 = obj1.flatten(0,2)
                    obj2_P3 = obj2.flatten(0,2)
                    local1_P2 = local1.flatten(0,2)
                    local2_P2 = local2.flatten(0,2)
                    residual1_P = residual1.reshape(-1)
                    residual2_P = residual2.reshape(-1)

                loss,loss_obj,loss_dis,loss_adj,loss_height,loss_conf,loss_feat,sp,sn,sa = criterion(epoch,args.max_epoch,dataset_idx,
                                                                                                feat1_PD,feat2_PD,
                                                                                                pred1_P3,pred2_P3,
                                                                                                conf1_P,conf2_P,
                                                                                                obj1_P3,obj2_P3,
                                                                                                local1_P2,local2_P2,
                                                                                                residual1_P,residual2_P,
                                                                                                rpcs,v1,v2,
                                                                                                B,H,W,
                                                                                                )

                total_loss += loss.item()
                total_loss_obj += loss_obj.item()
                total_loss_dis += loss_dis.item()
                total_loss_adj += loss_adj.item()
                total_loss_height += loss_height.item()
                # total_loss_photo += loss_photo.item()
                # total_loss_bias += loss_bias.item()
                # total_loss_reg += loss_reg.item()
                total_loss_conf += loss_conf.item()
                total_loss_feat += loss_feat.item()
                total_sp += sp.item()
                total_sn += sn.item()
                total_sa += sa.item()
                # total_loss_strcut += loss_struct.item()
                count += 1

        total_loss /= count
        total_loss_obj /= count
        total_loss_height /= count
        # total_loss_photo /= count
        # total_loss_bias /= count
        # total_loss_reg /= count
        total_loss_conf /= count
        total_loss_dis /= count
        total_loss_adj /= count
        total_loss_feat /= count
        total_sp /= count
        total_sn /= count
        total_sa /= count
        total_loss_strcut /= count

        print(f"valid: loss:{total_loss} \t obj:{total_loss_obj} \t dis:{total_loss_dis} \t adj:{total_loss_adj} \t height:{total_loss_height}  \t conf:{total_loss_conf} \t feat:{total_loss_feat} \t sp:{total_sp} \t sn:{total_sn} \t sa:{total_sa} ") # \t photo:{loss_photo} \t bias:{loss_bias} \t reg:{loss_reg}

        return total_loss_obj

    min_loss = args.min_loss
    last_loss = None
    start_time = time.perf_counter()
    criterion = CriterionFinetune(dataset_num)
    scaler = GradScaler()
    logger = TableLogger('./log',['epoch','loss','loss_obj','loss_obj_valid','loss_height','loss_dis','sp','lr_encoder','lr_decoder'],'finetune_log')
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch}')
        # valid(epoch)
        total_loss = 0
        total_loss_obj = 0
        total_loss_dis = 0
        total_loss_adj = 0
        total_loss_height = 0
        # total_loss_photo = 0
        # total_loss_bias = 0
        # total_loss_reg = 0
        total_loss_conf = 0
        total_loss_feat = 0
        total_sp = 0
        total_sn = 0
        total_sa = 0
        total_loss_strcut = 0
        count = 0
        encoder.train()
        for batch_idx,items in enumerate(dataloader):
            imgs,locals,objs,residuals,view_idxs = items
            print(view_idxs)
            for dataset_idx in range(dataset_num):
                t0 = time.perf_counter()
                img1 = imgs[dataset_idx]['v1'].contiguous()
                img2 = imgs[dataset_idx]['v2'].contiguous()
                obj1 = objs[dataset_idx]['v1'].contiguous()
                obj2 = objs[dataset_idx]['v2'].contiguous()
                local1 = locals[dataset_idx]['v1'].contiguous()
                local2 = locals[dataset_idx]['v2'].contiguous()
                residual1 = residuals[dataset_idx]['v1'].contiguous()
                residual2 = residuals[dataset_idx]['v2'].contiguous()
                v1 = view_idxs[dataset_idx]['v1']
                v2 = view_idxs[dataset_idx]['v2']

                rpcs = dataset.datasets[dataset_idx]['rpcs']
                extend = dataset.datasets[dataset_idx]['extend']

                decoder= decoders[dataset_idx]
                decoder.train()
                encoder_optimizer.zero_grad()
                decoder_optimizer = optimizers[dataset_idx]
                decoder_optimizer.zero_grad()

                if args.use_gpu:
                    img1 = img1.cuda().contiguous()
                    img2 = img2.cuda().contiguous()
                    obj1 = obj1.cuda()
                    obj2 = obj2.cuda()
                    local1 = local1.cuda()
                    local2 = local2.cuda()
                    residual1 = residual1.cuda()
                    residual2 = residual2.cuda()
                    for rpc in rpcs:
                        rpc.to_gpu()
                    extend = extend.cuda()
                
                B,H,W = local1.shape[0],local1.shape[1],local1.shape[2]

                with autocast('cuda'):
                    feat1,conf1 = encoder(img1)
                    feat2,conf2 = encoder(img2)
                    patch_feat1,global_feat1 = feat1[:,:patch_feature_channels],feat1[:,patch_feature_channels:]
                    patch_feat2,global_feat2 = feat2[:,:patch_feature_channels],feat2[:,patch_feature_channels:]

                    patch_feat_noise_amp1 = torch.rand(patch_feat1.shape[0],1,patch_feat1.shape[2],patch_feat1.shape[3]).cuda() * .3
                    patch_feat_noise_amp2 = torch.rand(patch_feat2.shape[0],1,patch_feat2.shape[2],patch_feat2.shape[3]).cuda() * .3
                    global_feat_noise_amp1 = torch.rand(global_feat1.shape[0],1,global_feat1.shape[2],global_feat1.shape[3]).cuda() * .8
                    global_feat_noise_amp2 = torch.rand(global_feat2.shape[0],1,global_feat2.shape[2],global_feat2.shape[3]).cuda() * .8
                    patch_feat_noise1 = F.normalize(torch.normal(mean=0.,std=patch_feat1.std().item(),size=patch_feat1.shape),dim=1).cuda() * patch_feat_noise_amp1
                    patch_feat_noise2 = F.normalize(torch.normal(mean=0.,std=patch_feat2.std().item(),size=patch_feat2.shape),dim=1).cuda() * patch_feat_noise_amp2
                    global_feat_noise1 = F.normalize(torch.normal(mean=0.,std=global_feat1.std().item(),size=global_feat1.shape),dim=1).cuda() * global_feat_noise_amp1
                    global_feat_noise2 = F.normalize(torch.normal(mean=0.,std=global_feat2.std().item(),size=global_feat2.shape),dim=1).cuda() * global_feat_noise_amp2

                    feat_input1 = torch.concatenate([F.normalize(patch_feat1 + patch_feat_noise1,dim=1),F.normalize(global_feat1 + global_feat_noise1,dim=1)],dim=1)
                    feat_input2 = torch.concatenate([F.normalize(patch_feat2 + patch_feat_noise2,dim=1),F.normalize(global_feat2 + global_feat_noise2,dim=1)],dim=1)

                    output1_B3hw = decoder(feat_input1)
                    output2_B3hw = decoder(feat_input2)
                    output1_P3 = output1_B3hw.permute(0,2,3,1).flatten(0,2)
                    output2_P3 = output2_B3hw.permute(0,2,3,1).flatten(0,2)
                    pred1_P3 = warp_by_extend(output1_P3,extend)
                    pred2_P3 = warp_by_extend(output2_P3,extend)
                    feat1_PD = feat1[:,:patch_feature_channels].permute(0,2,3,1).flatten(0,2)
                    feat2_PD = feat2[:,:patch_feature_channels].permute(0,2,3,1).flatten(0,2)
                    conf1_P = conf1.permute(0,2,3,1).reshape(-1)
                    conf2_P = conf2.permute(0,2,3,1).reshape(-1)
                    obj1_P3 = obj1.flatten(0,2)
                    obj2_P3 = obj2.flatten(0,2)
                    local1_P2 = local1.flatten(0,2)
                    local2_P2 = local2.flatten(0,2)
                    residual1_P = residual1.reshape(-1)
                    residual2_P = residual2.reshape(-1)
                t1 = time.perf_counter()
                loss,loss_obj,loss_dis,loss_adj,loss_height,loss_conf,loss_feat,sp,sn,sa = criterion(epoch,args.max_epoch,dataset_idx,
                                                                                                feat1_PD,feat2_PD,
                                                                                                pred1_P3,pred2_P3,
                                                                                                conf1_P,conf2_P,
                                                                                                obj1_P3,obj2_P3,
                                                                                                local1_P2,local2_P2,
                                                                                                residual1_P,residual2_P,
                                                                                                rpcs,v1,v2,
                                                                                                B,H,W,
                                                                                                )
                if torch.isnan(loss_feat):
                    print("nan feat loss,continue")
                    continue
                conf_mean = .5 * conf1_P.mean() + .5 * conf2_P.mean()

                total_loss += loss.item()
                total_loss_obj += loss_obj.item()
                total_loss_dis += loss_dis.item()
                total_loss_adj += loss_adj.item()
                total_loss_height += loss_height.item()
                # total_loss_photo += loss_photo.item()
                # total_loss_bias += loss_bias.item()
                # total_loss_reg += loss_reg.item()
                total_loss_conf += loss_conf.item()
                total_loss_feat += loss_feat.item()
                total_sp += sp.item()
                total_sn += sn.item()
                total_sa += sa.item()
                # total_loss_strcut += loss_struct.item()
                count += 1

                # loss.backward()
                # encoder_optimizer.step()
                scaler.scale(loss).backward()
                scaler.step(encoder_optimizer)
                scaler.step(decoder_optimizer)
                scaler.update()

                curtime = time.perf_counter()
                curstep = epoch * batch_num * dataset_num + count
                remain_step = args.max_epoch * batch_num * dataset_num - curstep
                cost_time = curtime - start_time
                remain_time = remain_step * cost_time / curstep
                t2 = time.perf_counter()
                # print(t1 - t0,t2 - t1)
                print(f"epoch:{epoch}  batch:{batch_idx}/{batch_num}  dataset:{dataset_idx} \t l_obj:{loss_obj.item():.2f} \t l_dis:{loss_dis.item():.2f} \t l_adj:{loss_adj.item():.2f} \t l_height:{loss_height.item():.2f} \t l_conf:{loss_conf.item():.2f} \t cm:{conf_mean.item():.2f} \t t:{criterion.residual_thresholds[dataset_idx]:.2f} \t l_feat:{loss_feat.item():.2f} \t sp:{sp.item():.4f} \t sn:{sn.item():.4f} \t sa:{sa.item():.4f} \t en_lr:{encoder_optimizer.param_groups[0]['lr']:.2e}  de_lr:{decoder_optimizer.param_groups[0]['lr']:.2e} \t time:{str(datetime.timedelta(seconds=round(cost_time)))}  ETA:{str(datetime.timedelta(seconds=round(remain_time)))}")

            encoder_scheduler.step()
            for scheduler in schedulers:
                scheduler.step()
            
        
        total_loss /= count
        total_loss_obj /= count
        total_loss_dis /= count
        total_loss_adj /= count
        total_loss_height /= count
        # total_loss_photo /= count
        # total_loss_bias /= count
        # total_loss_reg /= count
        total_loss_conf /= count
        total_loss_feat /= count
        total_sp /= count
        total_sn /= count
        total_sa /= count
        total_loss_strcut /= count
        if last_loss is None:
            print(f'total_loss:{total_loss} \t min_loss:{min_loss} \t obj:{total_loss_obj:.2f} \t dis:{total_loss_dis:.2f} \t adj:{total_loss_adj:.2f} \t height:{total_loss_height:.2f} \t conf:{total_loss_conf:.4f} \t feat:{total_loss_feat:.4f} \t sp:{total_sp:.4f} \t sn:{total_sn:.4f} \t sa:{total_sa:.4f}')
        else:
            print(f"total_loss:{total_loss} \t diff:{'+' if total_loss - last_loss > 0 else ''}{total_loss - last_loss} \t min_loss:{min_loss} \t obj:{total_loss_obj:.2f} \t dis:{total_loss_dis:.2f} \t adj:{total_loss_adj:.2f} \t height:{total_loss_height:.2f} \t conf:{total_loss_conf:.4f} \t feat:{total_loss_feat:.4f}  \t sp:{total_sp:.4f} \t sn:{total_sn:.4f} \t sa:{total_sa:.4f}")
        last_loss = total_loss

        total_loss_obj_valid = valid(epoch)
        # total_loss_obj_valid = 0

        torch.save(encoder.state_dict(),os.path.join(os.path.join(args.encoder_output_path,f'backbone_{epoch}.pth')))
        
        if total_loss_obj < min_loss:
            min_loss = total_loss_obj
            torch.save(encoder.state_dict(),os.path.join(os.path.join(args.encoder_output_path,'backbone.pth')))
            # torch.save(encoder.state_dict(),args.encoder_output_path)
            # if not args.freeze_decoder:
            for dataset_idx in range(dataset_num):
                torch.save(decoders[dataset_idx].state_dict(),os.path.join(args.encoder_output_path,f'decoder_{dataset_idx}.pth'))
            print('best updated')
        logger.update({
            'epoch':epoch,
            'loss':total_loss,
            'loss_obj':total_loss_obj,
            'loss_obj_valid':total_loss_obj_valid,
            'loss_height':total_loss_height,
            'loss_dis':total_loss_dis,
            'sp':total_sp,
            'lr_encoder':f"{encoder_optimizer.param_groups[0]['lr']:.7f}",
            'lr_decoder':f"{decoder_optimizer.param_groups[0]['lr']:.7f}"
        })

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path',type=str,default='./datasets')
    parser.add_argument('--encoder_path',type=str,default=None)
    parser.add_argument('--decoder_path',type=str,default=None)
    parser.add_argument('--encoder_output_path',type=str,default='./weights/encoder_finetune.pth')
    parser.add_argument('--decoder_output_path',type=str,default=None)
    parser.add_argument('--batch_size',type=int,default=128)
    parser.add_argument('--use_gpu',type=bool,default=True)
    parser.add_argument('--max_epoch',type=int,default=200)
    parser.add_argument('--lr_encoder_min',type=float,default=1e-7)
    parser.add_argument('--lr_encoder_max',type=float,default=1e-4)
    parser.add_argument('--lr_decoder_min',type=float,default=1e-7)
    parser.add_argument('--lr_decoder_max',type=float,default=1e-3) #1e-3
    parser.add_argument('--min_loss',type=float,default=1e8)

    args = parser.parse_args()
    gpus = os.environ['CUDA_VISIBLE_DEVICES']
    args.multi_gpu = len(gpus.split(',')) > 1
    args.dataset_num = 1#len(set([int(i.split('_')[0].split('d')[1]) for i in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path,i))]))
    args.view_num = len(set([int(i.split('_')[1].split('v')[1]) for i in os.listdir(args.dataset_path) if os.path.isdir(os.path.join(args.dataset_path,i))]))

    if not os.path.exists(args.encoder_output_path):
        os.mkdir(args.encoder_output_path)

    if not os.path.exists(args.decoder_output_path):
        os.mkdir(args.decoder_output_path)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)

    print('=========================configs=========================')
    print(args)
    print('=========================================================')
    pretrain(args)