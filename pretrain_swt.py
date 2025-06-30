import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.parallel import DistributedDataParallel
import numpy as np
from torch.utils.data import Dataset,DataLoader,DistributedSampler
from criterion import CriterionFinetuneNormal,CriterionFinetuneDis
from model_new import Encoder,ProjectHead,Decoder

import datetime
import time
from tqdm import tqdm,trange
from skimage.transform import AffineTransform
from skimage.measure import ransac
from torch.cuda.amp import autocast, GradScaler
from dataloader import PretrainDataset
from utils import TableLogger,kaiming_init_weights,str2bool,warp_by_extend
import random
import warnings
warnings.filterwarnings("ignore")
from scheduler import MultiStageOneCycleLR
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

def warp_by_bbox(raw,bbox):
    # raw[:,0] = .5 * (raw[:,0] + 1.) * (bbox['x_max'] - bbox['x_min']) + bbox['x_min']
    # raw[:,1] = .5 * (raw[:,1] + 1.) * (bbox['y_max'] - bbox['y_min']) + bbox['y_min']
    # raw[:,2] = .5 * (raw[:,2] + 1.) * (bbox['h_max'] - bbox['h_min']) + bbox['h_min']
    x = .5 * (raw[:,0] + 1.) * (bbox['x_max'] - bbox['x_min']) + bbox['x_min']
    y = .5 * (raw[:,1] + 1.) * (bbox['y_max'] - bbox['y_min']) + bbox['y_min']
    h = .5 * (raw[:,2] + 1.) * (bbox['h_max'] - bbox['h_min']) + bbox['h_min']
    warped = torch.stack([x,y,h],dim=-1)
    return warped

def distibute_model(model:nn.Module,local_rank):
    model = DistributedDataParallel(model,device_ids=[local_rank],output_device=local_rank)
    return model

def pretrain(args):
    print("Loading Dataset")
    dataset = PretrainDataset(root = args.dataset_path,
                              dataset_num = args.dataset_num,
                              batch_size = args.batch_size,
                              downsample = 16,
                              input_size = 1024,
                              mode='train')
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset,sampler=sampler,batch_size=args.data_batch_size,num_workers=4,drop_last=False,pin_memory=True)
    dataset_num = dataset.dataset_num
    data_batch_num = len(dataloader)
    print("Building Encoder")

    cfg = cfg_large
    # cfg['unfreeze_backbone_modules'] = ['head','norm',*[f'layers.2.blocks.{i}' for i in unfreeze_blocks]]

    encoder = Encoder(cfg)
    projector = ProjectHead(encoder.patch_feature_channels,128)
    if not args.encoder_path is None:
        encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.encoder_path,'backbone.pth')).items()},strict=True)
        print('Encoder Loaded')
    os.makedirs(os.path.dirname(args.encoder_output_path),exist_ok=True)

    encoder_optimizer = optim.AdamW(params=list(encoder.get_unfreeze_parameters()) + list(projector.parameters()),lr = args.lr_encoder_max)

    encoder_scheduler = MultiStageOneCycleLR(optimizer=encoder_optimizer,
                                             max_lr=args.lr_encoder_max,
                                             steps_per_epoch=data_batch_num,
                                             n_epochs_per_stage=args.max_epoch,
                                             summit_hold=0,
                                             gamma=.63 ** (1. / (50 * data_batch_num)),
                                             pct_start=30. / args.max_epoch)
    

    patch_feature_channels = encoder.patch_feature_channels
    output_channels = encoder.output_channels


    encoder = encoder.to(args.device)
    projector = projector.to(args.device)
    num_gpus = torch.cuda.device_count()
    print(f"Using {num_gpus} GPUS")
    if num_gpus > 1:
        encoder = distibute_model(encoder,args.local_rank)
        projector = distibute_model(projector,args.local_rank)
    
    print("Building Decoders")

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
                                            gamma=.63 ** (1. / 30),
                                            pct_start=5. / args.max_epoch)
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        decoders.append(decoder)

    min_loss = args.min_loss
    last_loss = None
    start_time = time.perf_counter()
    criterion_normal = CriterionFinetuneNormal()
    criterion_dis = CriterionFinetuneDis()
    scaler = GradScaler()
    os.makedirs('./log',exist_ok=True)
    logger = TableLogger('./log',['epoch','loss','loss_obj','loss_height','loss_conf','loss_feat','loss_dis','k','lr_encoder','lr_decoder'],'finetune_log')
    
    torch.autograd.set_detect_anomaly(True)
    for epoch in range(args.max_epoch):
        print(f'Epoch:{epoch}')
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
        for data_batch_idx,data in enumerate(dataloader):
            img1,img2,obj,residual1,residual2,dataset_idxs = data
            N,B,H,W = obj.shape[:4]
            img1 = img1.reshape(N*B,-1,img1.shape[-2],img1.shape[-1])
            img2 = img2.reshape(N*B,-1,img2.shape[-2],img2.shape[-1])
            obj = obj.reshape(N*B,H,W,-1)
            residual1 = residual1.reshape(N*B,H,W)
            residual2 = residual2.reshape(N*B,H,W)
            
            for idx in dataset_idxs:
                decoder = decoders[idx]
                decoder.train()
                encoder_optimizer.zero_grad()
                decoder_optimizer = optimizers[idx]
                decoder_optimizer.zero_grad()
            
            # if args.use_gpu:
            #     img1 = img1.cuda().contiguous()
            #     img2 = img2.cuda().contiguous()
            #     obj = obj.cuda()
            #     residual1 = residual1.cuda()
            #     residual2 = residual2.cuda()
            img1 = img1.to(args.device)
            img2 = img2.to(args.device)
            obj = obj.to(args.device)
            residual1 = residual1.to(args.device)
            residual2 = residual2.to(args.device)
            
            with autocast():
                feat1,conf1 = encoder(img1)
                feat2,conf2 = encoder(img2)

                patch_feat1,global_feat1 = feat1[:,:patch_feature_channels],feat1[:,patch_feature_channels:]
                patch_feat2,global_feat2 = feat2[:,:patch_feature_channels],feat2[:,patch_feature_channels:]

                project_feat1 = projector(patch_feat1)
                project_feat2 = projector(patch_feat2)

                # patch_feat_noise_amp1 = torch.rand(patch_feat1.shape[0],1,patch_feat1.shape[2],patch_feat1.shape[3]).cuda() * .3
                # patch_feat_noise_amp2 = torch.rand(patch_feat2.shape[0],1,patch_feat2.shape[2],patch_feat2.shape[3]).cuda() * .3
                # global_feat_noise_amp1 = torch.rand(global_feat1.shape[0],1,global_feat1.shape[2],global_feat1.shape[3]).cuda() * .8
                # global_feat_noise_amp2 = torch.rand(global_feat2.shape[0],1,global_feat2.shape[2],global_feat2.shape[3]).cuda() * .8
                # patch_feat_noise1 = F.normalize(torch.normal(mean=0.,std=patch_feat1.std().item(),size=patch_feat1.shape),dim=1).cuda() * patch_feat_noise_amp1
                # patch_feat_noise2 = F.normalize(torch.normal(mean=0.,std=patch_feat2.std().item(),size=patch_feat2.shape),dim=1).cuda() * patch_feat_noise_amp2
                # global_feat_noise1 = F.normalize(torch.normal(mean=0.,std=global_feat1.std().item(),size=global_feat1.shape),dim=1).cuda() * global_feat_noise_amp1
                # global_feat_noise2 = F.normalize(torch.normal(mean=0.,std=global_feat2.std().item(),size=global_feat2.shape),dim=1).cuda() * global_feat_noise_amp2

                # feat_input1 = torch.concatenate([F.normalize(patch_feat1 + patch_feat_noise1,dim=1),F.normalize(global_feat1 + global_feat_noise1,dim=1)],dim=1)
                # feat_input2 = torch.concatenate([F.normalize(patch_feat2 + patch_feat_noise2,dim=1),F.normalize(global_feat2 + global_feat_noise2,dim=1)],dim=1)
                feat_input1 = feat1
                feat_input2 = feat2

                pred1_P3 = []
                pred2_P3 = []
                pred_skip_1_P3 = []
                pred_skip_2_P3 = []
                
                for n,idx in enumerate(dataset_idxs):
                    decoder = decoders[idx]
                    output1_B3hw = decoder(feat_input1[n * B : (n+1) * B])
                    output2_B3hw = decoder(feat_input2[n * B : (n+1) * B])
                    output1_P3 = output1_B3hw.permute(0,2,3,1).flatten(0,2)
                    output2_P3 = output2_B3hw.permute(0,2,3,1).flatten(0,2)

                    decoder.requires_grad_(False)
                    output_skip_1_B3hw = decoder(feat1[n * B : (n+1) * B])
                    output_skip_2_B3hw = decoder(feat2[n * B : (n+1) * B])
                    output_skip_1_P3 = output_skip_1_B3hw.permute(0,2,3,1).flatten(0,2)
                    output_skip_2_P3 = output_skip_2_B3hw.permute(0,2,3,1).flatten(0,2)
                    decoder.requires_grad_(True)

                    output1_P3,output2_P3,output_skip_1_P3,output_skip_2_P3 = \
                        output1_P3.to(torch.float32),output2_P3.to(torch.float32),output_skip_1_P3.to(torch.float32),output_skip_2_P3.to(torch.float32)
                    
                    obj_bbox = dataset.obj_bboxs[idx]

                    pred1_P3.append(warp_by_bbox(output1_P3,obj_bbox))
                    pred2_P3.append(warp_by_bbox(output2_P3,obj_bbox)) 
                    pred_skip_1_P3.append(warp_by_bbox(output_skip_1_P3,obj_bbox))
                    pred_skip_2_P3.append(warp_by_bbox(output_skip_2_P3,obj_bbox))
                
                pred1_P3 = torch.concatenate(pred1_P3,dim=0)
                pred2_P3 = torch.concatenate(pred2_P3,dim=0)
                pred_skip_1_P3 = torch.concatenate(pred_skip_1_P3,dim=0)
                pred_skip_2_P3 = torch.concatenate(pred_skip_2_P3,dim=0)

                # print("1:",torch.isnan(pred1_P3).any() & torch.isnan(pred2_P3).any() & torch.isnan(pred_skip_1_P3).any() & torch.isnan(pred_skip_2_P3).any())

                project_feat1_PD = project_feat1.permute(0,2,3,1).flatten(0,2)
                project_feat2_PD = project_feat2.permute(0,2,3,1).flatten(0,2)
                conf1_P = conf1.permute(0,2,3,1).reshape(-1)
                conf2_P = conf2.permute(0,2,3,1).reshape(-1)
                obj_P3 = obj.flatten(0,2)
                residual1_P = residual1.reshape(-1).detach()
                residual2_P = residual2.reshape(-1).detach()

                loss_normal,loss_obj,loss_height,loss_conf,loss_feat,k = criterion_normal(epoch,
                                                                                    project_feat1_PD,project_feat2_PD,
                                                                                    pred1_P3,pred2_P3,
                                                                                    conf1_P,conf2_P,
                                                                                    obj_P3,
                                                                                    residual1_P,residual2_P,
                                                                                    H,W)

                    
                    
                loss_dis,dis_obj,dis_height = criterion_dis(pred_skip_1_P3,pred_skip_2_P3,residual1_P,residual2_P,k)
                        
                if torch.isnan(loss_feat):
                    print("nan feat loss,continue")
                    continue

                loss = loss_normal + loss_dis * max(min(1.,epoch / 20. - 1.),0.)
            # loss.backward()
            # encoder_optimizer.step()
            # for idx in dataset_idxs:
            #     optimizers[idx].step()
            print("loss:",loss)
            scaler.scale(loss).backward()
            scaler.step(encoder_optimizer)
            for idx in dataset_idxs:
                scaler.step(optimizers[idx])
            scaler.update()

            conf_mean = .5 * conf1_P.mean() + .5 * conf2_P.mean()

            total_loss += loss.item()
            total_loss_obj += loss_obj.item()
            total_loss_dis += loss_dis.item()
            total_loss_height += loss_height.item()
            total_loss_conf += loss_conf.item()
            total_loss_feat += loss_feat.item()

            count += 1
          

            curtime = time.perf_counter()
            curstep = epoch * data_batch_num + count
            remain_step = args.max_epoch  * data_batch_num - curstep
            cost_time = curtime - start_time
            remain_time = remain_step * cost_time / curstep

            print(f"epoch:{epoch}  batch:{data_batch_idx}/{data_batch_num} \t l_obj:{loss_obj.item():.2f} \t l_dis:{loss_dis.item():.2f} \t l_height:{loss_height.item():.2f} \t l_conf:{loss_conf.item():.2f} \t cm:{conf_mean.item():.2f} \t k:{k:.2f} \t l_feat:{loss_feat.item():.2f} \t en_lr:{encoder_optimizer.param_groups[0]['lr']:.2e}  de_lr:{decoder_optimizer.param_groups[0]['lr']:.2e} \t time:{str(datetime.timedelta(seconds=round(cost_time)))}  ETA:{str(datetime.timedelta(seconds=round(remain_time)))}")

            encoder_scheduler.step()

        for scheduler in schedulers:
            scheduler.step()            
        
        total_loss /= count
        total_loss_obj /= count
        total_loss_dis /= count
        total_loss_height /= count
        total_loss_conf /= count
        total_loss_feat /= count
        if last_loss is None:
            print(f'total_loss:{total_loss} \t min_loss:{min_loss} \t obj:{total_loss_obj:.2f} \t dis:{total_loss_dis:.2f} \t height:{total_loss_height:.2f} \t conf:{total_loss_conf:.4f} \t feat:{total_loss_feat:.4f}')
        else:
            print(f"total_loss:{total_loss} \t diff:{'+' if total_loss - last_loss > 0 else ''}{total_loss - last_loss} \t min_loss:{min_loss} \t obj:{total_loss_obj:.2f} \t dis:{total_loss_dis:.2f} \t height:{total_loss_height:.2f} \t conf:{total_loss_conf:.4f} \t feat:{total_loss_feat:.4f}")
        last_loss = total_loss

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
            'loss_height':total_loss_height,
            'loss_dis':total_loss_dis,
            'loss_conf':total_loss_conf,
            'loss_feat':total_loss_feat,
            'k':k.item(),
            'lr_encoder':f"{encoder_optimizer.param_groups[0]['lr']:.7f}",
            'lr_decoder':f"{optimizers[0].param_groups[0]['lr']:.7f}"
        })

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
        args.device=torch.device("cuda", args.local_rank)
        torch.distributed.init_process_group(backend="nccl", init_method='env://')

    if args.batch_size % 4 != 0:
        raise ValueError("Batch size must be divisible by 4")
    args.batch_size = args.batch_size // 4

    if not os.path.exists(args.encoder_output_path):
        os.mkdir(args.encoder_output_path)

    if not os.path.exists(args.decoder_output_path):
        os.mkdir(args.decoder_output_path)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    print("==============================configs==============================")
    for k,v in vars(args).items():
        print(f"{k}:{v}")
    print("===================================================================")
    pretrain(args)