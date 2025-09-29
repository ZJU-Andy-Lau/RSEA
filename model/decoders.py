import torch
import torch.nn as nn
import torch.nn.functional as F

class DecoderFinetune(nn.Module):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.
    """

    def get_block(self,channels):
        return nn.Sequential(
            nn.Conv2d(channels,channels * 2,1,1,0),
            nn.ReLU(),
            nn.Conv2d(channels * 2,channels * 2,1,1,0),
            nn.ReLU(),
            nn.Conv2d(channels * 2,channels,1,1,0)
        )

    def __init__(self,in_channels=512,block_num=5,use_bn=False):
        super().__init__()
        block_num = max(block_num,1)
        self.use_bn = use_bn
        self.blocks = nn.ModuleList([self.get_block(in_channels) for _ in range(block_num)])
        self.output_x = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,1,1,1,0),
            nn.Tanh()
        )
        self.output_y = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,1,1,1,0),
            nn.Tanh()
        )
        self.output_height = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,1,1,1,0),
            nn.Tanh()
        )
        # self.bn = bnac(in_channels)


    def forward(self, res):
        # res = res / torch.norm(res,dim=1,keepdim=True)
        # if self.use_bn:
        #     res = self.bn(res)
        for block in self.blocks:
            x = block(res)
            res = res + x
        x_res = self.output_x(res)
        y_res = self.output_y(res)
        height_res = self.output_height(res)
        return torch.cat([x_res,y_res,height_res],dim=1)
    
class Decoder(nn.Module):
    def get_block(self,channels):
        return nn.Sequential(
            nn.Conv2d(channels,channels * 2,1,1,0),
            nn.ReLU(),
            nn.Conv2d(channels * 2,channels * 2,1,1,0),
            nn.ReLU(),
            nn.Conv2d(channels * 2,channels,1,1,0)
        )

    def __init__(self,in_channels=512,block_num=5,use_bn=False):
        super().__init__()
        block_num = max(block_num,1)
        self.use_bn = use_bn
        self.blocks = nn.ModuleList([self.get_block(in_channels) for _ in range(block_num)])
        self.output_xy = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,4,1,1,0),
        )
        self.output_height = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,2,1,1,0),
        )
        self.score_head = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,1,1,1,0),
            nn.Sigmoid()
        )
        # self.bn = bnac(in_channels)


    def forward(self, res):
        valid_score = self.score_head(res)
        for block in self.blocks:
            x = block(res)
            res = res + x
        xy_res = self.output_xy(res)
        height_res = self.output_height(res)
        
        mu_xy = F.tanh(xy_res[:,:2])
        log_sigma_xy = F.tanh(xy_res[:,2:]) * 5.
        mu_h = F.tanh(height_res[:,:1])
        log_sigma_h = F.tanh(height_res[:,1:]) * 5.

        return torch.cat([mu_xy,mu_h,log_sigma_xy,log_sigma_h],dim=1),valid_score
    
    def forward_valid(self,res):
        valid_score = self.score_head(res)
        return valid_score


class Decoder_Modulate(nn.Module):
    """
    MLP network predicting per-pixel scene coordinates given a feature vector. All layers are 1x1 convolutions.
    """

    def get_block(self,channels):
        return nn.Sequential(
            nn.Conv2d(channels,channels * 2,1,1,0),
            nn.ReLU(),
            nn.Conv2d(channels * 2,channels * 2,1,1,0),
            nn.ReLU(),
            nn.Conv2d(channels * 2,channels,1,1,0)
        )

    def __init__(self,in_channels=512,block_num=5,digit_num=3,use_bn=False):
        super().__init__()
        block_num = max(block_num,1)
        self.digit_num = max(digit_num,1)
        self.use_bn = use_bn
        self.blocks = nn.ModuleList([self.get_block(in_channels) for _ in range(block_num)])
        self.init_xy = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,4,1,1,0)
        )
        self.init_height = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,2,1,1,0)
        )
        
        self.modulate_xy = nn.Sequential(
            nn.Conv2d(in_channels + 4,in_channels,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,4,1,1,0),
        )
        self.modulate_height = nn.Sequential(
            nn.Conv2d(in_channels + 2,in_channels,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,2,1,1,0),
        )

        self.score_head = nn.Sequential(
            nn.Conv2d(in_channels,in_channels // 16,1,1,0),
            nn.ReLU(),
            nn.Conv2d(in_channels // 16,1,1,1,0),
            nn.Sigmoid()
        )
        # self.bn = bnac(in_channels)


    def forward(self, res, per_digit = False):
        # res = res / torch.norm(res,dim=1,keepdim=True)
        # if self.use_bn:
        #     res = self.bn(res)
        valid_score = self.score_head(res)
        for block in self.blocks:
            x = block(res)
            res = res + x

        xy_res = self.init_xy(res)
        height_res = self.init_height(res)
        logit = torch.cat([xy_res[:,:2],height_res[:,:1],xy_res[:,2:],height_res[:,1:]],dim=1)# mu_x,mu_y,mu_h,s_x,s_y,s_h
        digit = F.tanh(logit)
        digit = torch.cat([digit[:,:3],digit[:,3:] * 5.],dim=1)
        # digit[:,3:] = digit[:,3:] * 5.
        
        digit_list = [digit]

        for i in range(self.digit_num - 1):
            modulate_xy_input = torch.cat([digit[:,:2].detach(),digit[:,3:5].detach() / 5.,res],dim=1)
            modulate_h_input = torch.cat([digit[:,2:3].detach(),digit[:,5:].detach() / 5.,res] ,dim=1)
            delta_xy = self.modulate_xy(modulate_xy_input)
            delta_h = self.modulate_height(modulate_h_input)
            delta_logit = torch.cat([delta_xy[:,:2],delta_h[:,:1],delta_xy[:,2:],delta_h[:,1:]],dim=1)
            logit = logit + delta_logit
            digit = F.tanh(logit)
            digit = torch.cat([digit[:,:3],digit[:,3:] * 5.],dim=1)
            # digit[:,3:] = digit[:,3:] * 5.
            # delta_mu_xy = F.tanh(delta_xy[:,:2])
            # delta_log_sigma_xy = F.tanh(delta_xy[:,2:])
            # delta_mu_h = F.tanh(delta_h[:,:1])
            # delta_log_sigma_h = F.tanh(delta_h[:,1:])
            # delta = torch.cat([delta_mu_xy,delta_mu_h,delta_log_sigma_xy,delta_log_sigma_h],dim=1)
            # digit = digit + delta
            digit_list.append(digit)


            # xy_res = self.output_xy_list[i](res)
            # height_res = self.output_height_list[i](res)
            # mu_xy = F.tanh(xy_res[:,:2])
            # log_sigma_xy = F.tanh(xy_res[:,2:])
            # mu_h = F.tanh(height_res[:,:1])
            # log_sigma_h = F.tanh(height_res[:,1:])
            # digit = torch.cat([mu_xy,mu_h,log_sigma_xy,log_sigma_h],dim=1)
            # if digit_total is None:
            #     digit_total = digit
            # else:
            #     digit_total = digit_total + digit

            # if i < self.digit_num - 1:

            # digit_list.append(digit_total)

        # xy_res = self.output_xy(res)
        # height_res = self.output_height(res)
        
        # mu_xy = F.tanh(xy_res[:,:2])
        # log_sigma_xy = F.tanh(xy_res[:,2:]) * 10.
        # mu_h = F.tanh(height_res[:,:1])
        # log_sigma_h = F.tanh(height_res[:,1:]) * 10.
        if per_digit:
            return digit_list,valid_score
        else:
            return digit_list[-1],valid_score
    
    def forward_valid(self,res):
        valid_score = self.score_head(res)
        return valid_score