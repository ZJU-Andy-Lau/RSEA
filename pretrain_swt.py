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
from model_new import Encoder,ProjectHead,DecoderFinetune
import h5py
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
from copy import deepcopy

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

def sample_features(features: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    H, W = features.shape[-2:]
    
    y_coords = coords[:, :, 0]
    x_coords = coords[:, :, 1]

    x_normalized = 2.0 * x_coords / W - 1.0
    y_normalized = 2.0 * y_coords / H - 1.0
    
    normalized_grid = torch.stack([x_normalized, y_normalized], dim=2)

    grid_for_sampling = normalized_grid.unsqueeze(1)

    sampled_features = F.grid_sample(
        features,
        grid_for_sampling,
        mode='bilinear',
        padding_mode='border',
        align_corners=True
    )
    
    final_output = sampled_features.squeeze(2)
    return final_output

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

def compute_loss(args,epoch,data,encoder:Encoder,decoder:DecoderFinetune,projector:ProjectHead,criterion:nn.Module):
    img1 = data['img1'].squeeze(0).to(args.device)
    img2 = data['img2'].squeeze(0).to(args.device)
    obj1 = data['obj1'].squeeze(0).to(args.device)
    obj2 = data['obj2'].squeeze(0).to(args.device)
    residual1 = data['residual1'].squeeze(0).to(args.device)
    residual2 = data['residual2'].squeeze(0).to(args.device)
    overlap1 = data['overlap1'].squeeze(0).to(args.device)
    overlap2 = data['overlap2'].squeeze(0).to(args.device)
    obj_map_coef = data['obj_map_coef']
    B,H,W = obj1.shape[:3]

    feat1,conf1 = encoder(img1)
    feat2,conf2 = encoder(img2)

    patch_feat1,global_feat1 = feat1[:,:args.patch_feature_channels],feat1[:,args.patch_feature_channels:]
    patch_feat2,global_feat2 = feat2[:,:args.patch_feature_channels],feat2[:,args.patch_feature_channels:]

    feat1_sample = sample_features(feat1,overlap1).unsqueeze(-1) # B,D,N,1
    feat2_sample = sample_features(feat2,overlap2).unsqueeze(-1)

    project_feat1 = projector(feat1_sample[:,:args.patch_feature_channels])
    project_feat2 = projector(feat2_sample[:,:args.patch_feature_channels])

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
    
    output1_B3hw = decoder(feat_input1)
    output2_B3hw = decoder(feat_input2)
    output1_P3 = output1_B3hw.permute(0,2,3,1).flatten(0,2)
    output2_P3 = output2_B3hw.permute(0,2,3,1).flatten(0,2)
    pred1_P3 = warp_by_poly(output1_P3,obj_map_coef)
    pred2_P3 = warp_by_poly(output2_P3,obj_map_coef)

    decoder_freeze = deepcopy(decoder)
    for params in decoder_freeze.parameters():
        params.requires_grad_ = False
    decoder_freeze.eval()
    output1_freeze_B3hw = decoder_freeze(feat1_sample)
    output2_freeze_B3hw = decoder_freeze(feat2_sample)
    output1_freeze_P3 = output1_freeze_B3hw.permute(0,2,3,1).flatten(0,2)
    output2_freeze_P3 = output2_freeze_B3hw.permute(0,2,3,1).flatten(0,2)
    pred1_freeze_P3 = warp_by_poly(output1_freeze_P3,obj_map_coef)
    pred2_freeze_P3 = warp_by_poly(output2_freeze_P3,obj_map_coef)

    project_feat1_PD = project_feat1.permute(0,2,3,1).flatten(0,2)
    project_feat2_PD = project_feat2.permute(0,2,3,1).flatten(0,2)
    conf1_P = conf1.permute(0,2,3,1).reshape(-1)
    conf2_P = conf2.permute(0,2,3,1).reshape(-1)
    obj1_P3 = obj1.flatten(0,2)
    obj2_P3 = obj2.flatten(0,2)
    residual1_P = residual1.reshape(-1).detach()
    residual2_P = residual2.reshape(-1).detach()
    conf_mean = .5 * conf1_P.clone().detach().mean() + .5 * conf2_P.clone().detach().mean()

    loss,loss_obj,loss_height,loss_conf,loss_feat,k = criterion(epoch,
                                                                project_feat1_PD,project_feat2_PD,
                                                                pred1_P3,pred2_P3,
                                                                conf1_P,conf2_P,
                                                                obj1_P3,obj2_P3,
                                                                residual1_P,residual2_P,
                                                                H,W)
    
    loss_dis = torch.norm(pred1_freeze_P3 - pred2_freeze_P3,dim=-1).mean()
    loss = loss + loss_dis * max(min(1.,epoch / 5. - 1.),0.)

    return loss,loss_obj,loss_height,loss_conf,loss_feat,loss_dis,k,conf_mean

def pretrain(args):
    os.makedirs('./log',exist_ok=True)
    os.makedirs(args.encoder_output_path,exist_ok=True)
    os.makedirs(args.checkpoints_path,exist_ok=True)
    pprint = partial(print_on_main, rank=dist.get_rank())
    num_gpus = dist.get_world_size()
    pprint(f"Using {num_gpus} GPUS")
    pprint("Loading Dataset")
    rank = dist.get_rank()
    
    if args.resume_training:
        training_configs = torch.load(os.path.join(args.checkpoints_path,'training_configs.pth'))
        min_loss = training_configs['min_loss']
        last_loss = training_configs['last_loss']
        epoch = training_configs['epoch']
        dataset_indices = training_configs['dataset_indices'].to(args.device)
        log_name = training_configs['log_name']
        if rank == 0:
            logger = TableLogger('./log',['epoch','loss','loss_obj','loss_height','loss_conf','loss_feat','loss_dis','k','lr_encoder','lr_decoder'],name = log_name)
        else:
            logger = None
    else:
        training_configs = None
        min_loss = args.min_loss
        last_loss = None
        epoch = 0
        if rank == 0:
            logger = TableLogger('./log',['epoch','loss','loss_obj','loss_height','loss_conf','loss_feat','loss_dis','k','lr_encoder','lr_decoder'],prefix = f'{args.log_prefix}_finetune_log')
        else:
            logger = None

    if not args.resume_training:
        dataset_indices = torch.empty(args.dataset_num,dtype=torch.long,device=args.device)
        if rank == 0:
            with h5py.File(os.path.join(args.dataset_path,'train_data.h5'),'r') as f:
                total_num = len(f.keys())
            dataset_indices = torch.randperm(total_num)[:args.dataset_num].to(args.device)
            indices_str = [str(idx) for idx in dataset_indices.cpu().numpy()]
            indices_str = " ".join(indices_str)
            with open(os.path.join('./log',f'{args.log_prefix}_dataset_idxs_log.txt'),'a') as f:
                f.write(f"{indices_str}\n")

    dist.barrier()
    dist.broadcast(dataset_indices,src=0)
    dataset_indices = dataset_indices.cpu().numpy()

    dataset = PretrainDataset(root = args.dataset_path,
                              dataset_idxs=dataset_indices,
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
    encoder_optimizer = optim.AdamW(params=list(encoder.get_unfreeze_parameters()) + list(projector.parameters()),lr = args.lr_encoder_max)

    encoder_scheduler = MultiStageOneCycleLR(optimizer=encoder_optimizer,
                                             total_steps=dataset_num * args.max_epoch,
                                             warmup_ratio=100. / args.max_epoch,
                                             cooldown_ratio=.5)
    
    args.patch_feature_channels = encoder.patch_feature_channels
    args.output_channels = encoder.patch_feature_channels + encoder.global_feature_channels
    
    if args.resume_training:
        encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.checkpoints_path,'encoder.pth'),map_location='cpu').items()},strict=True)
        projector.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.checkpoints_path,'projector.pth'),map_location='cpu').items()})
        encoder_optimizer.load_state_dict(torch.load(os.path.join(args.checkpoints_path,'encoder_optimizer.pth')))
        encoder_scheduler.load_state_dict(torch.load(os.path.join(args.checkpoints_path,'encoder_scheduler.pth')))
        
    elif not args.encoder_path is None:
        encoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.encoder_path,'backbone.pth'),map_location='cpu').items()},strict=True)
        pprint('Encoder Loaded')

    encoder = encoder.to(args.device)
    projector = projector.to(args.device)
    for state in encoder_optimizer.state.values():
        for k, v in state.items():
            if isinstance(v, torch.Tensor):
                state[k] = v.to(args.device)
    if num_gpus > 1:
        encoder = distibute_model(encoder,args.local_rank)
        projector = distibute_model(projector,args.local_rank)

    

    pprint("Building Decoders")     

    decoders = []
    optimizers = []
    schedulers = []
    for dataset_idx in trange(dataset_num):
        decoder = DecoderFinetune(in_channels=args.output_channels,block_num=args.decoder_block_num,use_bn=False)
        optimizer = optim.AdamW(params=decoder.parameters(),lr = args.lr_decoder_max)
        scheduler = MultiStageOneCycleLR(optimizer=optimizer,
                                        total_steps=args.max_epoch,
                                        warmup_ratio=100. / args.max_epoch,
                                        cooldown_ratio=.5)
        
        if args.resume_training:
            decoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.checkpoints_path,f'decoder_{dataset_idx}.pth'),map_location='cpu').items()})
            optimizer.load_state_dict(torch.load(os.path.join(args.checkpoints_path,f'decoder_optimizer_{dataset_idx}.pth')))
            scheduler.load_state_dict(torch.load(os.path.join(args.checkpoints_path,f'decoder_scheduler_{dataset_idx}.pth')))

        decoder = decoder.to(args.device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(args.device)
        if num_gpus > 1:
            decoder = distibute_model(decoder,args.local_rank)

        decoder.train()

        # elif not args.decoder_path is None and os.path.exists(os.path.join(args.decoder_path,f'decoder_{dataset_idx}.pth')):
        #     decoder.load_state_dict({k.replace("module.",""):v for k,v in torch.load(os.path.join(args.encoder_path,f'decoder_{dataset_idx}.pth')).items()})
        
        optimizers.append(optimizer)
        schedulers.append(scheduler)
        decoders.append(decoder)

    
    start_time = time.perf_counter()
    criterion = CriterionFinetune()
    step_count = 0
    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(epoch,args.max_epoch):
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
            # output_img(img1[0],'./img_check',f'epoch_{epoch}_img1_rank_{rank}_idx_{dataset_idx}')
            # output_img(img2[0],'./img_check',f'epoch_{epoch}_img2_rank_{rank}_idx_{dataset_idx}')

            decoder = decoders[dataset_idx]
            decoder_optimizer = optimizers[dataset_idx]
            decoder.train()
            encoder_optimizer.zero_grad()
            decoder_optimizer.zero_grad()
            
            compose_data = {
                "img1":img1,
                'img2':img2,
                "obj1":obj1,
                "obj2":obj2,
                "residual1":residual1,
                "residual2":residual2,
                "overlap1":overlap1,
                "overlap2":overlap2,
                "obj_map_coef":dataset.obj_map_coefs[dataset_idx]
            }

            loss,loss_obj,loss_height,loss_conf,loss_feat,loss_dis,k,conf_mean = compute_loss(args,epoch,compose_data,encoder,decoder,projector,criterion)

            # if rank == 1 and epoch == 1 and iter_idx == 1:
            #     loss = torch.tensor(torch.nan,device=loss.device)
            

            loss_is_nan = not torch.isfinite(loss).all()

            loss_status_tensor = torch.tensor([loss_is_nan], dtype=torch.float32, device=rank)

            dist.all_reduce(loss_status_tensor, op=dist.ReduceOp.SUM)

            if loss_status_tensor.item() > 0:
                pprint(f"--- 检测到 NaN！Epoch {epoch}, iter {iter_idx+1}, dataset {dataset_idx}. 所有进程将一起跳过此次更新。---")
                del loss,loss_obj,loss_height,loss_conf,loss_feat,loss_dis,conf_mean
                encoder_scheduler.step()
                continue 
            
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
            step_count += 1
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
                curstep = step_count
                remain_step = (args.max_epoch - epoch)  * dataset_num - count
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
                torch.save(encoder_state_dict,os.path.join(args.encoder_output_path,'backbone.pth'))
                # torch.save(encoder.state_dict(),args.encoder_output_path)
                # if not args.freeze_decoder:
                # for dataset_idx in range(dataset_num):
                #     torch.save(decoders[dataset_idx].state_dict(),os.path.join(args.decoder_output_path,f'decoder_{dataset_idx}.pth'))
                print('best updated')
            
            if epoch % 5 == 0:
                path = args.checkpoints_path
                encoder_state_dict = {k:v.detach().cpu() for k,v in encoder.state_dict().items()}
                encoder_optimizer_state_dict = encoder_optimizer.state_dict()
                encoder_scheduler_state_dict = encoder_scheduler.state_dict()
                projector_state_dict = projector.state_dict()
                torch.save(encoder_state_dict,os.path.join(path,'encoder.pth'))
                torch.save(encoder_optimizer_state_dict,os.path.join(path,'encoder_optimizer.pth'))
                torch.save(encoder_scheduler_state_dict,os.path.join(path,'encoder_scheduler.pth'))
                torch.save(projector_state_dict,os.path.join(path,'projector.pth'))
                for i in range(dataset_num):
                    decoder_state_dict = {k:v.detach().cpu() for k,v in decoders[i].state_dict().items()}
                    decoder_optimizer_state_dict = optimizers[i].state_dict()
                    decoder_scheduler_state_dict = schedulers[i].state_dict()
                    torch.save(decoder_state_dict,os.path.join(path,f'decoder_{i}.pth'))
                    torch.save(decoder_optimizer_state_dict,os.path.join(path,f'decoder_optimizer_{i}.pth'))
                    torch.save(decoder_scheduler_state_dict,os.path.join(path,f'decoder_scheduler_{i}.pth'))
                training_configs = {
                    'dataset_indices':torch.from_numpy(dataset_indices),
                    'epoch':torch.tensor(epoch),
                    'min_loss':torch.tensor(min_loss),
                    'last_loss':torch.tensor(last_loss),
                    'log_name':logger.file_name
                }
                torch.save(training_configs,os.path.join(path,'training_configs.pth'))

        
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
    parser.add_argument('--encoder_output_path',type=str,default='./weights/encoder_finetune.pth')
    parser.add_argument('--checkpoints_path',type=str,default=None)
    parser.add_argument('--batch_size',type=int,default=8)
    parser.add_argument('--decoder_block_num',type=int,default=1)
    parser.add_argument('--resume_training',type=str2bool,default=False)
    parser.add_argument('--max_epoch',type=int,default=200)
    parser.add_argument('--lr_encoder_min',type=float,default=1e-7)
    parser.add_argument('--lr_encoder_max',type=float,default=1e-4)
    parser.add_argument('--lr_decoder_min',type=float,default=1e-7)
    parser.add_argument('--lr_decoder_max',type=float,default=1e-3) #1e-3
    parser.add_argument('--min_loss',type=float,default=1e8)
    parser.add_argument('--log_prefix',type=str,default='')
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

    # torch.manual_seed(42)
    # np.random.seed(42)
    # random.seed(42)

    warnings.filterwarnings("ignore", category=UserWarning)
    warnings.filterwarnings("ignore", category=FutureWarning)
    pprint("==============================configs==============================")
    for k,v in vars(args).items():
        pprint(f"{k}:{v}")
    pprint("===================================================================")
    pretrain(args)