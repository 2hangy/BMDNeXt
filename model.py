import torch
import torch.nn as nn
from torchdiffeq import odeint
import torch.nn.functional as F
from fds import *
import loguru

logger = loguru.logger

class nmODE(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(nmODE, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.W = nn.Parameter(torch.randn(input_dim, output_dim))
        self.b = nn.Parameter(torch.randn(output_dim))

        nn.init.xavier_uniform_(self.W)
        nn.init.zeros_(self.b)

    def neuronODE(self, t, y, X):
        gamma = torch.matmul(X, self.W) # + self.b
        dydt = -y + torch.sin(y + gamma)**2
        return dydt

    def forward(self, t, X, method='dopri5'):
        y0 = torch.ones(X.size(0), X.size(1), self.output_dim, device=X.device) * 0.5
        ode_func = lambda t, y: self.neuronODE(t, y, X)
        y = odeint(ode_func, y0, t, method=method)
        return y[-1]
    
class NMA(nn.Module):
    def __init__(self, num_channels):
        super(NMA, self).__init__()
        self.num_channels = num_channels

        self.avg_pool_d = nn.AdaptiveAvgPool3d((1, 1, None))
        self.avg_pool_w = nn.AdaptiveAvgPool3d((1, None, 1))
        self.avg_pool_h = nn.AdaptiveAvgPool3d((None, 1, 1))
        self.avg_pool_c = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.nmODE_c = nmODE(num_channels, num_channels)

        self.alpha = nn.Parameter(torch.ones(3))

        self.nmODE_w = None
        self.nmODE_h = None
        self.nmODE_d = None
        self.initialized = False
        self.infer = False

    def initialize_spatial_nmODEs(self, H, W, D):
        self.nmODE_w = nmODE(W, W).to(self.alpha.device)
        self.nmODE_h = nmODE(H, H).to(self.alpha.device)
        self.nmODE_d = nmODE(D, D).to(self.alpha.device)
        self.initialized = True

    def forward(self, input_tensor):
        if self.infer:
            method = 'euler'
            t = torch.linspace(0, 3, 10, device=input_tensor.device)
        else:
            method = 'dopri5'
            t = torch.linspace(0, 3, 1, device=input_tensor.device)

        batch_size, num_channels, H, W, D = input_tensor.size()
        
        if not self.initialized:
            self.initialize_spatial_nmODEs(H, W, D)
        
        # Spatial embedding
        z_w = self.avg_pool_w(input_tensor)  # (B, C, W)
        z_h = self.avg_pool_h(input_tensor)  # (B, C, H)
        z_d = self.avg_pool_d(input_tensor)  # (B, C, D)

        # Channel embedding
        z_c = self.avg_pool_c(input_tensor).squeeze(-1).squeeze(-1).squeeze(-1)  # (B, C)

        # Spatial attention weights
        a_w = self.nmODE_w(t, z_w.view(batch_size, num_channels, W),method) # (B, C, W)
        a_h = self.nmODE_h(t, z_h.view(batch_size, num_channels, H),method) # (B, C, H)
        a_d = self.nmODE_d(t, z_d.view(batch_size, num_channels, D),method) # (B, C, D)

        a_w = a_w.view(batch_size, num_channels, W, 1, 1).expand(batch_size, num_channels, W, H, D)
        a_h = a_h.view(batch_size, num_channels, 1, H, 1).expand(batch_size, num_channels, W, H, D)
        a_d = a_d.view(batch_size, num_channels, 1, 1, D).expand(batch_size, num_channels, W, H, D)

        alpha_softmax = F.softmax(self.alpha, dim=0)
        A_spatial = alpha_softmax[0] * a_w + alpha_softmax[1] * a_h + alpha_softmax[2] * a_d
        
        A_channel = self.nmODE_c(t, z_c.unsqueeze(0),method)  # (1, B, C)
        A_channel = A_channel.squeeze(0).view(batch_size, num_channels, 1, 1, 1).expand(batch_size, num_channels, W, H, D)

        A = A_spatial * A_channel

        output_tensor = input_tensor * A

        return output_tensor
    
class Block3D(nn.Module):
    def __init__(self, dim, k_size=3):
        super().__init__()
        self.dwconv = nn.Conv3d(dim, dim, kernel_size=k_size, padding=(k_size-1) // 2, groups=dim)
        self.pwconv1 = nn.Conv3d(dim, 4 * dim, kernel_size=1)
        self.act = nn.GELU()
        self.pwconv2 = nn.Conv3d(4 * dim, dim, kernel_size=1)

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        
        x = x.permute(0, 2, 3, 4, 1)
        norm = nn.LayerNorm(x.shape[-1], elementwise_affine=False).to(x.device)
        x = norm(x)
        x = x.permute(0, 4, 1, 2, 3)
        
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = input + x
        return x


class Stem(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_chans, out_chans, kernel_size=1, stride=1 ,padding=0),
        )
 
    def forward(self, x):
        return self.conv(x) + x


class BMDNeXt(nn.Module):
    def __init__(self, in_chans=1, depths=[3, 3, 18, 3], dims=[96, 192, 384, 768], start_fds_epoch=3):
        super().__init__()
        self.downsample_layers = nn.ModuleList()  # stem and 3 intermediate downsampling conv layers
        stem = Stem(in_chans, dims[0])

        self.downsample_layers.append(stem)
        for i in range(3):   
            downsample_layer = nn.Sequential(
                nn.Conv3d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList()  # 4 feature resolution stages
        for i in range(4):
            stage = nn.Sequential(
                *[Block3D(dim=dims[i],k_size=5) for j in range(depths[i])],
                NMA(dims[i]),
            )
            self.stages.append(stage)

        self.AFDS = MS_AFDS(feature_dim=dims[-1], bucket_num=20, start_update=start_fds_epoch, start_smooth= start_fds_epoch+1)
        self.start_fds_epoch = start_fds_epoch
        self.is_train = True
        self.dropout = nn.Dropout(p=0.15)
        self.linear = nn.Linear(dims[-1], 1)
        self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

    def set_train(self,state):
        self.is_train = state

    def forward_features(self, x):
        outs = []
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
            outs.append(x)
        return tuple(outs)

    def forward(self, x, labels = None, epoch = 10):
        x = self.forward_features(x)
        stage_features = x
        out = x[-1]
        #out = self.dropout(out)

        feature = self.avg_pool(out).flatten(1)
        
        if self.is_train and epoch >= self.start_fds_epoch+1:
            feature = self.AFDS.smooth(feature, labels, epoch)

        pred = self.linear(feature)

        return pred , feature, stage_features
    