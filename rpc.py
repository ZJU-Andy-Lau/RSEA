import numpy as np
import os
import torch
import cv2


class RPCModelParameterTorch:
    def __init__(self, data=torch.zeros(170,dtype=torch.double)):
        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LONG_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LONG_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]

        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]

        self.LATNUM = data[90:110]
        self.LATDEM = data[110:130]
        self.LONNUM = data[130:150]
        self.LONDEM = data[150:170]

        self.Clear_Adjust()

        self.device = self.LINE_OFF.device

    """Read orginal RPC File"""
    def load_from_file(self, filepath):
        """
        Here, we define:
            direct rpc: from object (lat, lon, hei) to photo (sample, line)
            inverse rpc: from photo (sample, line, hei) to object (lat, lon)
        Function: Read direct rpc from file and then calculate the inverse rpc
        """
        if os.path.exists(filepath) is False:
            print("Error#001: cann't find " + filepath + " in the file system!")
            return

        with open(filepath, 'r') as f:
            all_the_text = f.read().splitlines()
            rfm_line = -1
            for line,text in enumerate(all_the_text):
                if "RFM_CORRECTION_PARAMETERS" in text:
                    rfm_line = line
                    break

        data = [np.float64(text.split()[1]) for text in (all_the_text[:rfm_line-1] if rfm_line > 0 else all_the_text)]
        data = torch.from_numpy(np.array(data, dtype=np.float64)).to(torch.double)

        self.LINE_OFF = data[0]
        self.SAMP_OFF = data[1]
        self.LAT_OFF = data[2]
        self.LONG_OFF = data[3]
        self.HEIGHT_OFF = data[4]
        self.LINE_SCALE = data[5]
        self.SAMP_SCALE = data[6]
        self.LAT_SCALE = data[7]
        self.LONG_SCALE = data[8]
        self.HEIGHT_SCALE = data[9]
        self.LNUM = data[10:30]
        self.LDEM = data[30:50]
        self.SNUM = data[50:70]
        self.SDEM = data[70:90]
        
        if data.shape[0] >= 170:
            self.LATNUM = data[90:110]
            self.LATDEM = data[110:130]
            self.LONNUM = data[130:150]
            self.LONDEM = data[150:170]
        else:
            self.Calculate_Inverse_RPC()
        
    
        # if data.shape[0] == 96:
        #     self.raw_adjust_params = data[90:96]
        # else:
        #     self.raw_adjust_params = None
        
        if rfm_line > 0:
            self.raw_adjust_params = [np.float64(text.split()[1]) for text in all_the_text[rfm_line + 1:]]
            self.Calculate_Adjust()
        else:
            self.raw_adjust_params = None

    def Create_Virtual_3D_Grid(self, xy_sample=30, z_sample=20):
        """
        Create_Virtual 3D control grid in the object space
        :return: grid (N, 5)
        """
        lat_max = self.LAT_OFF + self.LAT_SCALE
        lat_min = self.LAT_OFF - self.LAT_SCALE
        lon_max = self.LONG_OFF + self.LONG_SCALE
        lon_min = self.LONG_OFF - self.LONG_SCALE
        hei_max = self.HEIGHT_OFF + self.HEIGHT_SCALE
        hei_min = self.HEIGHT_OFF - self.HEIGHT_SCALE
        samp_max = self.SAMP_OFF + self.SAMP_SCALE
        samp_min = self.SAMP_OFF - self.SAMP_SCALE
        line_max = self.LINE_OFF + self.LINE_SCALE
        line_min = self.LINE_OFF - self.LINE_SCALE

        lat = torch.linspace(lat_min, lat_max, xy_sample).to(self.device,dtype=torch.double) #np.linspace(lat_min, lat_max, xy_sample)
        lon = torch.linspace(lon_min, lon_max, xy_sample).to(self.device,dtype=torch.double)
        hei = torch.linspace(hei_min, hei_max, z_sample).to(self.device,dtype=torch.double)

        lat, lon, hei = torch.meshgrid(lat, lon, hei) #np.meshgrid(lat, lon, hei)

        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        samp, line = self.RPC_OBJ2PHOTO(lat, lon, hei)
        grid = torch.stack((samp, line, lat, lon, hei), dim=-1).to(self.device,dtype=torch.double)

        selected_grid = []
        for g in grid:
            flag = [g[0] < samp_min, g[0] > samp_max, g[1] < line_min, g[1] > line_max]
            if True in flag:
                continue
            else:
                selected_grid.append(g)

        # grid = np.array(selected_grid)
        grid = torch.stack(selected_grid,dim=0).to(self.device,dtype=torch.double)
        return grid
    
    def Solve_Inverse_RPC(self, grid):
        def solve(ma, lv, x=0, k=1):
            """
            :param lv:
            :param ma: the Normal matrix
            :param x: init value
            :param k:
            :return:
            """
            assert ma.shape[0] == ma.shape[1], "ma with shape () is not a square matrix.".format(ma.shape[0], ma.shape[1])

            n = ma.shape[0]
            mak = ma.clone() #np.copy(ma)
            mak += k * torch.eye(n).to(self.device,dtype=torch.double)
            lk = lv.clone() #np.copy(lv)

            finish_time = 0

            for times in range(1000):
                x1 = torch.linalg.solve(mak,lk) #np.linalg.solve(mak, lk)
                dif = torch.abs(x1 - x)#np.fabs(x1 - x)
                maxdif = torch.max(dif)
                x = x1
                lk = lv + k * x

                finish_time = times + 1
                # print(finish_time, maxdif)
                if maxdif < 1.0e-10:
                    break
            return x, finish_time
        samp, line, lat, lon, hei = torch.hsplit(grid,5) #np.hsplit(grid.copy(), 5)

        samp = samp.reshape(-1)
        line = line.reshape(-1)
        lat = lat.reshape(-1)
        lon = lon.reshape(-1)
        hei = hei.reshape(-1)

        # 归一化
        samp = samp - self.SAMP_OFF
        samp = samp / self.SAMP_SCALE
        line = line - self.LINE_OFF
        line = line / self.LINE_SCALE

        lat = lat - self.LAT_OFF
        lat = lat / self.LAT_SCALE
        lon = lon - self.LONG_OFF
        lon = lon / self.LONG_SCALE
        hei = hei - self.HEIGHT_OFF
        hei = hei / self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(samp, line, hei)

        n_num = coef.shape[0]
        A = torch.zeros((n_num * 2, 78)).to(self.device,dtype=torch.double)
        A[0: n_num, 0:20] = - coef
        A[0: n_num, 20:39] = lat.reshape(-1, 1) * coef[:, 1:]
        A[n_num:, 39:59] = - coef
        A[n_num:, 59:78] = lon.reshape(-1, 1) * coef[:, 1:]

        l = torch.cat((lat, lon), -1) #np.concatenate((lat, lon), -1)
        l = -l

        ATA = torch.matmul(A.T, A)

        ATl = torch.matmul(A.T, l)

        x, times = solve(ATA, ATl)

        self.LATNUM = x[0:20]
        self.LATDEM[0] = 1.0
        self.LATDEM[1:20] = x[20:39]
        self.LONNUM = x[39:59]
        self.LONDEM[0] = 1.0
        self.LONDEM[1:20] = x[59:]

        return times
    def Calculate_Inverse_RPC(self):
        
        grid = self.Create_Virtual_3D_Grid()
        times = self.Solve_Inverse_RPC(grid)
        return times

    def Inverse_Adjust(self):
        R = self.adjust_params[:, :2]
        t = self.adjust_params[:, 2]   
        R_inv = torch.inverse(R) 
        t_new = -(R_inv @ t) 
        self.adjust_params_inv = torch.cat([R_inv, t_new.unsqueeze(1)], dim=1).to(torch.double)
    
    def Clear_Adjust(self):
        self.adjust_params = torch.tensor([
            [1.,0.,0.],
            [0.,1.,0.]
        ],dtype=torch.double,device=self.LNUM.device)
        self.Inverse_Adjust()

    def Update_Adjust(self,adjust_params:torch.Tensor):
        self.adjust_params = adjust_params.to(self.adjust_params.device).to(torch.double)
        self.Inverse_Adjust()
    
    def Calculate_Adjust(self):
        corners = np.array([[0.,0.],[100.,0.],[0.,100.]],dtype=np.float32) #line samp
        offset_line = self.raw_adjust_params[0] + self.raw_adjust_params[1] * corners[:,1] + self.raw_adjust_params[2] * corners[:,0]
        offset_samp = self.raw_adjust_params[3] + self.raw_adjust_params[4] * corners[:,1] + self.raw_adjust_params[5] * corners[:,0]
        offset_corners = corners - np.stack([offset_line,offset_samp],axis=1)
        af_trans = cv2.getAffineTransform(corners,offset_corners)
        self.Update_Adjust(torch.from_numpy(af_trans))

    def Merge_Adjust(self):
        self.rectify_rpc(self.adjust_params.reshape(-1))
        self.Calculate_Inverse_RPC()
        self.Clear_Adjust()

    def RPC_PLH_COEF(self, P, L, H):
        n_num = P.shape[0]
        coef = torch.zeros((n_num, 20),dtype=torch.double,device=P.device)
        coef[:, 0] = 1.0
        coef[:, 1] = L
        coef[:, 2] = P
        coef[:, 3] = H
        coef[:, 4] = L * P
        coef[:, 5] = L * H
        coef[:, 6] = P * H
        coef[:, 7] = L * L
        coef[:, 8] = P * P
        coef[:, 9] = H * H
        coef[:, 10] = P * L * H
        coef[:, 11] = L * L * L
        coef[:, 12] = L * P * P
        coef[:, 13] = L * H * H
        coef[:, 14] = L * L * P
        coef[:, 15] = P * P * P
        coef[:, 16] = P * H * H
        coef[:, 17] = L * L * H
        coef[:, 18] = P * P * H
        coef[:, 19] = H * H * H

        return coef
    
    def convert_tensor(self,arr,device):
        if isinstance(arr,torch.Tensor):
            return arr.to(dtype=torch.double,device=device)
        else:
            return torch.as_tensor(arr,dtype=torch.double,device=device)

    def RPC_OBJ2PHOTO(self, inlat, inlon, inhei, output_type='tensor'):
        """
        From (lat, lon, hei) to (samp, line) using the direct rpc
        rpc: RPC_MODEL_PARAMETER
        lat, lon, hei (n)
        """
        lat = self.convert_tensor(inlat,self.device)
        lon = self.convert_tensor(inlon,self.device)
        hei = self.convert_tensor(inhei,self.device)

        lat = lat - self.LAT_OFF
        lat = lat / self.LAT_SCALE

        lon = lon - self.LONG_OFF
        lon = lon / self.LONG_SCALE

        hei = hei - self.HEIGHT_OFF
        hei = hei / self.HEIGHT_SCALE

        coef = self.RPC_PLH_COEF(lat, lon, hei)

        # # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        samp = torch.sum(coef * self.SNUM,dim=-1) / torch.sum(coef * self.SDEM,dim=-1)
        line = torch.sum(coef * self.LNUM,dim=-1) / torch.sum(coef * self.LDEM,dim=-1)

        samp *= self.SAMP_SCALE
        samp += self.SAMP_OFF

        line *= self.LINE_SCALE
        line += self.LINE_OFF

        transformed_points = torch.stack([line,samp],dim=-1) @ self.adjust_params_inv[:,:2].T + self.adjust_params_inv[:,2]
        line = transformed_points[:,0]
        samp = transformed_points[:,1]

        if output_type == 'numpy':
            samp = samp.cpu().numpy()
            line = line.cpu().numpy()

        # if not self.adjust_params is None:
        #     adjust_offset_line = self.adjust_params[0] + self.adjust_params[1] * samp + self.adjust_params[2] * line
        #     adjust_offset_samp = self.adjust_params[3] + self.adjust_params[4] * samp + self.adjust_params[5] * line

        #     samp = samp + adjust_offset_samp
        #     line = line + adjust_offset_line

        return samp, line

    def RPC_PHOTO2OBJ(self, insamp, inline, inhei, output_type='tensor'):
        """
        From (samp, line, hei) to (lat, lon) using the inverse rpc
        rpc: RPC_MODEL_PARAMETER
        lat, lon, hei (n)
        """
        # samp = torch.tensor(insamp,dtype=torch.double,device=self.device)
        # line = torch.tensor(inline,dtype=torch.double,device=self.device)
        hei = self.convert_tensor(inhei,self.device)
        samp = self.convert_tensor(insamp,self.device)
        line = self.convert_tensor(inline,self.device)
 
        transformed_points = torch.stack([line,samp],dim=-1) @ self.adjust_params[:,:2].T + self.adjust_params[:,2]
        line = transformed_points[:,0]
        samp = transformed_points[:,1]

        samp = samp - self.SAMP_OFF
        samp = samp / self.SAMP_SCALE

        line = line - self.LINE_OFF
        line = line / self.LINE_SCALE

        hei = hei - self.HEIGHT_OFF
        hei = hei / self.HEIGHT_SCALE


        coef = self.RPC_PLH_COEF(samp, line, hei)

        # rpc.SNUM: (20), coef: (n, 20) out_pts: (n, 2)
        lat = torch.sum(coef * self.LATNUM, dim=-1) / torch.sum(coef * self.LATDEM, dim=-1)
        lon = torch.sum(coef * self.LONNUM, dim=-1) / torch.sum(coef * self.LONDEM, dim=-1)

        # print(self.LATDEM)
        lat *= self.LAT_SCALE
        lat += self.LAT_OFF

        lon *= self.LONG_SCALE
        lon += self.LONG_OFF

        if output_type == 'numpy':
            lon = lon.cpu().numpy()
            lat = lat.cpu().numpy()

        return lat, lon
    
    def latlon2yx(self,latlon:torch.Tensor):
        """
        (lat,lon) -> (y,x) N,2
        """
        r = 6378137.
        lon_rad = latlon[:,1] * torch.pi / 180.
        lat_rad = latlon[:,0] * torch.pi / 180.
        x = r * lon_rad
        y = r * torch.log(torch.tan(torch.pi / 4. + lat_rad / 2.))
        return torch.stack([y,x],dim=-1)

    def yx2latlon(self,yx:torch.Tensor):
        """
        (y,x) -> (lat,lon) N,2
        """
        yx = self.convert_tensor(yx,self.device)
        r = 6378137.
        lon = (180. * yx[:,1]) / (torch.pi * r)
        lat = (2 * torch.atan(torch.exp(yx[:,0] / r)) - torch.pi * 0.5) * 180. / torch.pi
        return torch.stack([lat,lon],dim=-1)

    def RPC_XY2LINESAMP(self,x_in, y_in, h_in, output_type='tensor'):
        x = self.convert_tensor(x_in,self.device)
        y = self.convert_tensor(y_in,self.device)
        h = self.convert_tensor(h_in,self.device)
        
        latlon = self.yx2latlon(torch.stack([y,x],dim=-1))
        samp,line = self.RPC_OBJ2PHOTO(latlon[:,0],latlon[:,1],h)

        if output_type == 'numpy':
            line = line.cpu().numpy()
            samp = samp.cpu().numpy()
        
        return line,samp
    
    def RPC_LINESAMP2XY(self,line_in, samp_in, h_in, output_type='tensor'):
        line = self.convert_tensor(line_in,self.device)
        samp = self.convert_tensor(samp_in,self.device)
        h = self.convert_tensor(h_in,self.device)

        lat,lon = self.RPC_PHOTO2OBJ(samp,line,h)
        yx = self.latlon2yx(torch.stack([lat,lon],dim=-1))
        y,x = yx[:,0],yx[:,1]

        if output_type == 'numpy':
            x = x.cpu().numpy()
            y = y.cpu().numpy()

        return x,y
    
    def _project_xyh_to_linesamp_for_jacobian(self, xyh_tensor: torch.Tensor) -> torch.Tensor:
        """
        A helper function that takes a single (N, 3) tensor for xyh
        and returns a (N, 2) tensor for line/samp.
        This format is required by torch.autograd.functional.jacobian.
        """
        x, y, h = xyh_tensor[..., 0], xyh_tensor[..., 1], xyh_tensor[..., 2]
        
        # Chain the transformations
        latlon = self.yx2latlon(torch.stack([y, x], dim=-1))
        samp, line = self.RPC_OBJ2PHOTO(latlon[:, 0], latlon[:, 1], h)
        
        return torch.stack([line, samp], dim=-1)

    def _vjp_projection_core(self, mu_xyh: torch.Tensor, sigma_xyh: torch.Tensor):
        """
        【私有核心函数】执行VJP投影计算。
        假定输入的张量块能完全载入显存。
        """
        # 确保 mu_xyh 可以追踪梯度，这是 autograd.grad 的要求
        mu_xyh.requires_grad_(True)

        # --- 均值传播 ---
        line, samp = self.RPC_XY2LINESAMP(mu_xyh[:, 0], mu_xyh[:, 1], mu_xyh[:, 2])
        mu_linesamp = torch.stack([line, samp], dim=-1)

        # --- 方差传播 (VJP方法) ---
        grad_line, = torch.autograd.grad(
            outputs=line, inputs=mu_xyh,
            grad_outputs=torch.ones_like(line),
            create_graph=True, retain_graph=True,
        )
        grad_samp, = torch.autograd.grad(
            outputs=samp, inputs=mu_xyh,
            grad_outputs=torch.ones_like(samp),
            create_graph=True,
        )
        
        var_xyh = sigma_xyh.pow(2)
        var_line = (grad_line.pow(2) * var_xyh).sum(dim=-1)
        var_samp = (grad_samp.pow(2) * var_xyh).sum(dim=-1)
        var_linesamp = torch.stack([var_line, var_samp], dim=-1)
            
        return mu_linesamp, var_linesamp

    def xy_distribution_to_linesamp(self, mu_xyh: torch.Tensor, sigma_xyh: torch.Tensor, chunk_size: int = 524288):
        """
        【最终版公开接口】一个鲁棒且高效的投影函数，集成了VJP方法和自动分块机制。

        Args:
            mu_xyh (torch.Tensor): 形状为 (N, 3) 或 (3,) 的张量，表示 (x, y, h) 的均值。
            sigma_xyh (torch.Tensor): 形状为 (N, 3) 或 (3,) 的张量，表示 (x, y, h) 的标准差。
            chunk_size (int, optional): 处理块的大小。如果输入点的总数 N 超过此值，
                                        将自动启用分块计算。默认为 524288 (512 * 1024)。
                                        您可以根据您的GPU显存大小调整此值。

        Returns:
            tuple[torch.Tensor, torch.Tensor]:
                - mu_linesamp (torch.Tensor): 投影后的像方均值 (line, samp)，形状为 (N, 2) 或 (2,)。
                - var_linesamp (torch.Tensor): 投影后的像方方差 (line, samp)，形状为 (N, 2) 或 (2,)。
        """
        # 确保输入是批处理格式
        is_batched = mu_xyh.dim() == 2
        if not is_batched:
            mu_xyh = mu_xyh.unsqueeze(0)
            sigma_xyh = sigma_xyh.unsqueeze(0)
        
        num_points = mu_xyh.shape[0]

        # --- 调度逻辑：根据输入点数决定是否分块 ---
        if num_points <= chunk_size:
            # 点数不多，直接调用核心VJP函数一次性计算，效率最高
            # print(f"点数 ({num_points}) 未超过阈值 ({chunk_size})，执行直接计算。")
            mu_linesamp, var_linesamp = self._vjp_projection_core(mu_xyh, sigma_xyh)
        else:
            # 点数过多，启用分块计算以保证稳定性
            print(f"警告: 点数 ({num_points}) 超过阈值 ({chunk_size})，自动启用分块计算。")
            mu_results = []
            var_results = []
            
            # 使用 torch.split 进行安全分块
            mu_chunks = torch.split(mu_xyh, chunk_size)
            sigma_chunks = torch.split(sigma_xyh, chunk_size)

            for mu_chunk, sigma_chunk in zip(mu_chunks, sigma_chunks):
                # 对每个块调用核心VJP函数
                mu_chunk_out, var_chunk_out = self._vjp_projection_core(mu_chunk, sigma_chunk)
                
                # 收集结果
                mu_results.append(mu_chunk_out)
                var_results.append(var_chunk_out)

            # 将所有块的结果拼接起来
            # PyTorch的autograd会自动处理拼接操作的梯度回传
            mu_linesamp = torch.cat(mu_results, dim=0)
            var_linesamp = torch.cat(var_results, dim=0)
        
        sigma_linesamp = torch.sqrt(var_linesamp)
        # 如果原始输入不是批处理格式，则恢复其形状
        if not is_batched:
            mu_linesamp = mu_linesamp.squeeze(0)
            sigma_linesamp = sigma_linesamp.squeeze(0)

        return mu_linesamp, sigma_linesamp

    # def adjust(self, insamp, inline):
    #     samp = torch.tensor(insamp,dtype=torch.double,device=self.device)
    #     line = torch.tensor(inline,dtype=torch.double,device=self.device)
    #     offset_line = self.adjust_params[0] + self.adjust_params[1] * samp + self.adjust_params[2] * line
    #     offset_samp = self.adjust_params[3] + self.adjust_params[4] * samp + self.adjust_params[5] * line
    #     samp += offset_samp
    #     line += offset_line
    #     return samp,line
    

    def to_gpu(self,device = None):
        if device is None:
            device = 'cuda'
        self.LINE_OFF = self.LINE_OFF.to(device)
        self.SAMP_OFF = self.SAMP_OFF.to(device)
        self.LAT_OFF = self.LAT_OFF.to(device)
        self.LONG_OFF = self.LONG_OFF.to(device)
        self.HEIGHT_OFF = self.HEIGHT_OFF.to(device)
        self.LINE_SCALE = self.LINE_SCALE.to(device)
        self.SAMP_SCALE = self.SAMP_SCALE.to(device)
        self.LAT_SCALE = self.LAT_SCALE.to(device)
        self.LONG_SCALE = self.LONG_SCALE.to(device)
        self.HEIGHT_SCALE = self.HEIGHT_SCALE.to(device)

        self.LNUM = self.LNUM.to(device)
        self.LDEM = self.LDEM.to(device)
        self.SNUM = self.SNUM.to(device)
        self.SDEM = self.SDEM.to(device)

        self.LATNUM = self.LATNUM.to(device)
        self.LATDEM = self.LATDEM.to(device)
        self.LONNUM = self.LONNUM.to(device)
        self.LONDEM = self.LONDEM.to(device)

        self.adjust_params = self.adjust_params.to(device)
        self.adjust_params_inv = self.adjust_params_inv.to(device)

        self.device = self.LINE_OFF.device

    def save_rpc_to_file(self, filepath):
        """
        Save the direct and inverse rpc to a file
        :param filepath: where to store the file
        :return:
        """
        addition0 = ['LINE_OFF:', 'SAMP_OFF:', 'LAT_OFF:', 'LONG_OFF:', 'HEIGHT_OFF:', 'LINE_SCALE:', 'SAMP_SCALE:',
                     'LAT_SCALE:', 'LONG_SCALE:', 'HEIGHT_SCALE:', 'LINE_NUM_COEFF_1:', 'LINE_NUM_COEFF_2:',
                     'LINE_NUM_COEFF_3:', 'LINE_NUM_COEFF_4:', 'LINE_NUM_COEFF_5:', 'LINE_NUM_COEFF_6:',
                     'LINE_NUM_COEFF_7:', 'LINE_NUM_COEFF_8:', 'LINE_NUM_COEFF_9:', 'LINE_NUM_COEFF_10:',
                     'LINE_NUM_COEFF_11:', 'LINE_NUM_COEFF_12:', 'LINE_NUM_COEFF_13:', 'LINE_NUM_COEFF_14:',
                     'LINE_NUM_COEFF_15:', 'LINE_NUM_COEFF_16:', 'LINE_NUM_COEFF_17:', 'LINE_NUM_COEFF_18:',
                     'LINE_NUM_COEFF_19:', 'LINE_NUM_COEFF_20:', 'LINE_DEN_COEFF_1:', 'LINE_DEN_COEFF_2:',
                     'LINE_DEN_COEFF_3:', 'LINE_DEN_COEFF_4:', 'LINE_DEN_COEFF_5:', 'LINE_DEN_COEFF_6:',
                     'LINE_DEN_COEFF_7:', 'LINE_DEN_COEFF_8:', 'LINE_DEN_COEFF_9:', 'LINE_DEN_COEFF_10:',
                     'LINE_DEN_COEFF_11:', 'LINE_DEN_COEFF_12:', 'LINE_DEN_COEFF_13:', 'LINE_DEN_COEFF_14:',
                     'LINE_DEN_COEFF_15:', 'LINE_DEN_COEFF_16:', 'LINE_DEN_COEFF_17:', 'LINE_DEN_COEFF_18:',
                     'LINE_DEN_COEFF_19:', 'LINE_DEN_COEFF_20:', 'SAMP_NUM_COEFF_1:', 'SAMP_NUM_COEFF_2:',
                     'SAMP_NUM_COEFF_3:', 'SAMP_NUM_COEFF_4:', 'SAMP_NUM_COEFF_5:', 'SAMP_NUM_COEFF_6:',
                     'SAMP_NUM_COEFF_7:', 'SAMP_NUM_COEFF_8:', 'SAMP_NUM_COEFF_9:', 'SAMP_NUM_COEFF_10:',
                     'SAMP_NUM_COEFF_11:', 'SAMP_NUM_COEFF_12:', 'SAMP_NUM_COEFF_13:', 'SAMP_NUM_COEFF_14:',
                     'SAMP_NUM_COEFF_15:', 'SAMP_NUM_COEFF_16:', 'SAMP_NUM_COEFF_17:', 'SAMP_NUM_COEFF_18:',
                     'SAMP_NUM_COEFF_19:', 'SAMP_NUM_COEFF_20:', 'SAMP_DEN_COEFF_1:', 'SAMP_DEN_COEFF_2:',
                     'SAMP_DEN_COEFF_3:', 'SAMP_DEN_COEFF_4:', 'SAMP_DEN_COEFF_5:', 'SAMP_DEN_COEFF_6:',
                     'SAMP_DEN_COEFF_7:', 'SAMP_DEN_COEFF_8:', 'SAMP_DEN_COEFF_9:', 'SAMP_DEN_COEFF_10:',
                     'SAMP_DEN_COEFF_11:', 'SAMP_DEN_COEFF_12:', 'SAMP_DEN_COEFF_13:', 'SAMP_DEN_COEFF_14:',
                     'SAMP_DEN_COEFF_15:', 'SAMP_DEN_COEFF_16:', 'SAMP_DEN_COEFF_17:', 'SAMP_DEN_COEFF_18:',
                     'SAMP_DEN_COEFF_19:', 'SAMP_DEN_COEFF_20:', 'LAT_NUM_COEFF_1:', 'LAT_NUM_COEFF_2:',
                     'LAT_NUM_COEFF_3:', 'LAT_NUM_COEFF_4:', 'LAT_NUM_COEFF_5:', 'LAT_NUM_COEFF_6:',
                     'LAT_NUM_COEFF_7:', 'LAT_NUM_COEFF_8:', 'LAT_NUM_COEFF_9:', 'LAT_NUM_COEFF_10:',
                     'LAT_NUM_COEFF_11:', 'LAT_NUM_COEFF_12:', 'LAT_NUM_COEFF_13:', 'LAT_NUM_COEFF_14:',
                     'LAT_NUM_COEFF_15:', 'LAT_NUM_COEFF_16:', 'LAT_NUM_COEFF_17:', 'LAT_NUM_COEFF_18:',
                     'LAT_NUM_COEFF_19:', 'LAT_NUM_COEFF_20:', 'LAT_DEN_COEFF_1:', 'LAT_DEN_COEFF_2:',
                     'LAT_DEN_COEFF_3:', 'LAT_DEN_COEFF_4:', 'LAT_DEN_COEFF_5:', 'LAT_DEN_COEFF_6:',
                     'LAT_DEN_COEFF_7:', 'LAT_DEN_COEFF_8:', 'LAT_DEN_COEFF_9:', 'LAT_DEN_COEFF_10:',
                     'LAT_DEN_COEFF_11:', 'LAT_DEN_COEFF_12:', 'LAT_DEN_COEFF_13:', 'LAT_DEN_COEFF_14:',
                     'LAT_DEN_COEFF_15:', 'LAT_DEN_COEFF_16:', 'LAT_DEN_COEFF_17:', 'LAT_DEN_COEFF_18:',
                     'LAT_DEN_COEFF_19:', 'LAT_DEN_COEFF_20:', 'LONG_NUM_COEFF_1:', 'LONG_NUM_COEFF_2:',
                     'LONG_NUM_COEFF_3:', 'LONG_NUM_COEFF_4:', 'LONG_NUM_COEFF_5:', 'LONG_NUM_COEFF_6:',
                     'LONG_NUM_COEFF_7:', 'LONG_NUM_COEFF_8:', 'LONG_NUM_COEFF_9:', 'LONG_NUM_COEFF_10:',
                     'LONG_NUM_COEFF_11:', 'LONG_NUM_COEFF_12:', 'LONG_NUM_COEFF_13:', 'LONG_NUM_COEFF_14:',
                     'LONG_NUM_COEFF_15:', 'LONG_NUM_COEFF_16:', 'LONG_NUM_COEFF_17:', 'LONG_NUM_COEFF_18:',
                     'LONG_NUM_COEFF_19:', 'LONG_NUM_COEFF_20:', 'LONG_DEN_COEFF_1:', 'LONG_DEN_COEFF_2:',
                     'LONG_DEN_COEFF_3:', 'LONG_DEN_COEFF_4:', 'LONG_DEN_COEFF_5:', 'LONG_DEN_COEFF_6:',
                     'LONG_DEN_COEFF_7:', 'LONG_DEN_COEFF_8:', 'LONG_DEN_COEFF_9:', 'LONG_DEN_COEFF_10:',
                     'LONG_DEN_COEFF_11:', 'LONG_DEN_COEFF_12:', 'LONG_DEN_COEFF_13:', 'LONG_DEN_COEFF_14:',
                     'LONG_DEN_COEFF_15:', 'LONG_DEN_COEFF_16:', 'LONG_DEN_COEFF_17:', 'LONG_DEN_COEFF_18:',
                     'LONG_DEN_COEFF_19:', 'LONG_DEN_COEFF_20:']
        addition1 = ['pixels', 'pixels', 'degrees', 'degrees', 'meters', 'pixels', 'pixels', 'degrees', 'degrees',
                     'meters']
        
        addition2 = ['CL0:','CLS:','CLL:','CS0:','CSS:','CSL:']

        corection_params = [self.adjust_params[0,2],self.adjust_params[0,1],self.adjust_params[0,0] - 1.,self.adjust_params[1,2],self.adjust_params[1,1] - 1.,self.adjust_params[1,0]]

        text = ""

        text += addition0[0] + " " + str(self.LINE_OFF.item()) + " " + addition1[0] + "\n"
        text += addition0[1] + " " + str(self.SAMP_OFF.item()) + " " + addition1[1] + "\n"
        text += addition0[2] + " " + str(self.LAT_OFF.item()) + " " + addition1[2] + "\n"
        text += addition0[3] + " " + str(self.LONG_OFF.item()) + " " + addition1[3] + "\n"
        text += addition0[4] + " " + str(self.HEIGHT_OFF.item()) + " " + addition1[4] + "\n"
        text += addition0[5] + " " + str(self.LINE_SCALE.item()) + " " + addition1[5] + "\n"
        text += addition0[6] + " " + str(self.SAMP_SCALE.item()) + " " + addition1[6] + "\n"
        text += addition0[7] + " " + str(self.LAT_SCALE.item()) + " " + addition1[7] + "\n"
        text += addition0[8] + " " + str(self.LONG_SCALE.item()) + " " + addition1[8] + "\n"
        text += addition0[9] + " " + str(self.HEIGHT_SCALE.item()) + " " + addition1[9] + "\n"

        for i in range(10, 30):
            text += addition0[i] + " " + str(self.LNUM[i - 10].item()) + "\n"
        for i in range(30, 50):
            text += addition0[i] + " " + str(self.LDEM[i - 30].item()) + "\n"
        for i in range(50, 70):
            text += addition0[i] + " " + str(self.SNUM[i - 50].item()) + "\n"
        for i in range(70, 90):
            text += addition0[i] + " " + str(self.SDEM[i - 70].item()) + "\n"
        for i in range(90, 110):
            text += addition0[i] + " " + str(self.LATNUM[i - 90].item()) + "\n"
        for i in range(110, 130):
            text += addition0[i] + " " + str(self.LATDEM[i - 110].item()) + "\n"
        for i in range(130, 150):
            text += addition0[i] + " " + str(self.LONNUM[i - 130].item()) + "\n"
        for i in range(150, 170):
            text += addition0[i] + " " + str(self.LONDEM[i - 150].item()) + "\n"
        
        if corection_params != [0.,0.,1.,0.,1.,0.]:
            text += "\nRFM CORRECTION PARAMETERS:\n"

            for i in range(6):
                text += addition2[i] + " " + str(corection_params[i].item()) + "\n"

        f = open(filepath, "w")
        f.write(text)
        f.close()

def load_rpc(rpc_path:str) -> RPCModelParameterTorch:
    rpc = RPCModelParameterTorch()
    rpc.load_from_file(rpc_path)
    return rpc