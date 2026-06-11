import math

import torch
import torch.nn as nn
from timm.models.layers import DropPath
from einops import rearrange
import torch.nn.functional as F
from functools import partial
from typing import Callable

from lib.fs_simply import FrequencySelection
from lib.vmamba import SaliencyMB_CATA1D_K1
from lib.util import LayerNorm, Mlp


class up_conv(nn.Module):
    def __init__(self, ch_in, ch_out):
        super(up_conv, self).__init__()
        self.up = nn.Sequential(
            nn.Upsample(scale_factor=2),
            nn.Conv2d(ch_in, ch_out, kernel_size=1),
            nn.BatchNorm2d(ch_out),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.up(x)
        return x


class LayerNorm2d(nn.Module):

    def __init__(self, normalized_shape, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.norm = nn.LayerNorm(normalized_shape, eps, elementwise_affine)

    def forward(self, x):
        x = rearrange(x, 'b c h w -> b h w c').contiguous()
        x = self.norm(x)
        x = rearrange(x, 'b h w c -> b c h w').contiguous()
        return x


class ConvNormAct(nn.Module):

    def __init__(self, dim_in, dim_out, kernel_size, stride=1, dilation=1, groups=1, bias=False, skip=False,
                 inplace=True, drop_path_rate=0.):
        super(ConvNormAct, self).__init__()
        self.has_skip = skip and dim_in == dim_out
        padding = math.ceil((kernel_size - stride) / 2)
        self.conv = nn.Conv2d(dim_in, dim_out, kernel_size, stride, padding, dilation, groups, bias)
        self.norm = nn.BatchNorm2d(dim_out)
        self.act = nn.ReLU(inplace=inplace)
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate else nn.Identity()

    def forward(self, x):
        shortcut = x
        x = self.conv(x)
        x = self.norm(x)
        x = self.act(x)
        if self.has_skip:
            x = self.drop_path(x) + shortcut
        return x




class BSR(nn.Module):
    def __init__(self, in_channel, out_channel, bands=(2,4,8), fs_freeze_iters=182):
        super(BSR, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel),
        )

        self.fs = FrequencySelection(
            in_channels=out_channel,
            k_list=list(bands),
            lowfreq_att=True,
            fs_feat='feat', lp_type='freq',
            act='softmax',
            spatial='conv', spatial_group=1, spatial_kernel=3,
            init='zero', global_selection=False,
        )

        self.gamma_fs  = nn.Parameter(torch.zeros(1))
        self.gamma_exp = nn.Parameter(torch.zeros(1))


        self.expert_high = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.expert_mid  = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=3, dilation=3, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.expert_low  = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=5, dilation=5, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )


        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(out_channel, out_channel * 3, 1, bias=True),
        )

        self.post_bn = nn.BatchNorm2d(out_channel)


        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel * 2, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel * 2),
            nn.Conv2d(out_channel * 2, out_channel, 3, padding=1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(True),
        )
        self.reduce = nn.Sequential(
            nn.Conv2d(out_channel * 2, out_channel, 3, 1, 1, bias=False),
            nn.BatchNorm2d(out_channel), nn.ReLU(True)
        )


        self.register_buffer("fs_counter", torch.zeros(1, dtype=torch.long))
        self.fs_freeze_iters = int(fs_freeze_iters)

    def forward(self, x):
        x0 = self.conv1(x)
        x1 = self.conv3(x0)


        use_fs = (self.fs_counter.item() >= self.fs_freeze_iters) or (not self.training)
        if self.training: self.fs_counter += 1
        if use_fs:
            fs_y = self.fs(x1)
            x_fs = x1 + self.gamma_fs.tanh() * (fs_y - x1)
        else:
            x_fs = x1


        yh = self.expert_high(x1); ym = self.expert_mid(x1); yl = self.expert_low(x1)


        gate = self.gate(x1).view(x1.size(0), 3, -1, 1, 1).softmax(dim=1)
        g_low, g_mid, g_high = gate[:,0], gate[:,1], gate[:,2]
        y_mix = (g_low * yl + g_mid * ym + g_high * yh)

        self.debug_gamma_fs = float(self.gamma_fs.tanh().detach().cpu())
        self.debug_gamma_exp = float(self.gamma_exp.tanh().detach().cpu())
        self.debug_gate_mean = {
            'low': g_low.mean().item(),
            'mid': g_mid.mean().item(),
            'high': g_high.mean().item(),
        }
        self.debug_fs_on = int(use_fs)
        self.debug_fs_counter = int(self.fs_counter.item())

        y = x_fs + self.gamma_exp.tanh() * y_mix
        y = self.post_bn(y)


        x_res = self.res(x)
        out = self.reduce(torch.cat((y, x_res), 1)) + x0
        return out



class MFE_module(nn.Module):
    def __init__(self, in_channel, out_channel, exp_ratio=1.0):
        super(MFE_module, self).__init__()

        mid_channel = in_channel * exp_ratio

        self.DWConv = ConvNormAct(mid_channel, mid_channel, kernel_size=3, groups=out_channel // 2)
        self.DWConv3x3 = ConvNormAct(in_channel // 4, in_channel // 4, kernel_size=3, groups=in_channel // 4)
        self.DWConv5x5 = ConvNormAct(in_channel // 4, in_channel // 4, kernel_size=5, groups=in_channel // 4)
        self.DWConv7x7 = ConvNormAct(in_channel // 4, in_channel // 4, kernel_size=7, groups=in_channel // 4)
        self.PWConv1 = ConvNormAct(in_channel, mid_channel, kernel_size=1)
        self.PWConv2 = ConvNormAct(mid_channel, out_channel, kernel_size=1)
        self.norm = nn.BatchNorm2d(in_channel)
        # MaxPool2d Capture high-frequency information
        self.Maxpool = nn.MaxPool2d(3, stride=1, padding=1)
        self.gelu = nn.GELU()

    def forward(self, x):
        shortcut = x
        x = self.norm(x)
        channels = x.size(1)
        channels_per_part = channels // 4
        x1 = x[:, :channels_per_part, :, :]
        x2 = x[:, channels_per_part:2*channels_per_part, :, :]
        x3 = x[:, 2*channels_per_part:3*channels_per_part, :, :]
        x4 = x[:, 3*channels_per_part:, :, :]
        x1 = self.Maxpool(x1)
        x2 = self.DWConv3x3(x2)
        x3 = self.DWConv5x5(x3)
        x4 = self.DWConv7x7(x4)

        x2 = x1 * x2
        x3 = x2 * x3
        x4 = x3 * x4
        x = torch.cat((x1, x2, x3, x4), dim=1)
        x = self.PWConv1(x)
        x = x + self.DWConv(x)
        x = self.PWConv2(x)
        x = x + shortcut

        return x




class SaliencyMambaBlock(nn.Module):
    '''
    Saliency Mamba (SaMB), with 2d SSM
    '''

    def __init__(
            self,
            dim: int = 0,
            drop_path: float = 0,
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),
            attn_drop_rate: float = 0,
            d_state: int = 4,
            dt_rank="auto",
            d_conv=3,
            use_sal_prompt=True,
            ssm_ratio=2.0,
            softmax_version=False,
            mlp_ratio=0.0,
            act_layer=nn.GELU,
            drop: float = 0.0,
            LayerNorm_type='WithBias',
            cata_num_centers=None,
            **kwargs,
    ):
        super().__init__()

        self.dim = dim
        # self.norm = norm_layer(dim)

        self.op = SaliencyMB_CATA1D_K1(
            d_model=dim,
            dropout=attn_drop_rate,
            d_state=d_state,
            ssm_ratio=ssm_ratio,
            dt_rank=dt_rank,
            d_conv=d_conv,
            softmax_version=softmax_version,
            # CATA 参数
            cata_num_centers=cata_num_centers,
            cata_iters=3,
            cata_ema=0.999,
            use_sal_prompt=use_sal_prompt,
        )

        self.drop_path = DropPath(drop_path)
        self.norm2 = LayerNorm(dim, LayerNorm_type)
        self.mlp = Mlp(in_features=dim, hidden_features=int(dim * mlp_ratio), act_layer=act_layer, drop=drop)

    def forward(self, x: torch.Tensor, gt: torch.Tensor):

        B, C, H, W = x.shape
        x_n = x.permute(0, 2, 3, 1).contiguous()  # B,H,W,C

        gt_r = F.interpolate(gt, size=(H, W), mode='bilinear', align_corners=False)
        gt_r = gt_r.to(device=x.device, dtype=torch.float32)

        with torch.cuda.amp.autocast(enabled=False):
            y_n = self.op(x_n.float(), gt_r)

        y_n = y_n.to(dtype=x_n.dtype)
        y = y_n.permute(0, 3, 1, 2).contiguous()
        x1 = x + self.drop_path(y)

        # FFN
        z = self.norm2(x1)
        z = self.mlp(z.permute(0, 2, 3, 1).contiguous().reshape(B * H * W, C)) \
            .reshape(B, H, W, C).permute(0, 3, 1, 2).contiguous()
        out = x1 + self.drop_path(z)
        return out

class SAM(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(SAM, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1), nn.BatchNorm2d(out_channel),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(out_channel, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel),
        )

        self.res = nn.Sequential(
            nn.Conv2d(in_channel, out_channel*2, 3, 1, 1), nn.BatchNorm2d(out_channel*2),
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel),
        )

        self.reduce = nn.Sequential(
            nn.Conv2d(out_channel*2, out_channel, 3, padding=1, dilation=1), nn.BatchNorm2d(out_channel), nn.ReLU(True)
        )
        self.relu = nn.ReLU(True)
        self.Global = SaliencyMambaBlock(dim=out_channel)


    def forward(self, x, gt):
        x0 = self.conv1(x)
        x1 = self.conv3(x0)
        x_global = self.Global(x1, gt)
        x_res = self.res(x)
        x = self.reduce(torch.cat((x_res, x_global), 1)) + x0
        return x





class SEBlock(nn.Module):
    def __init__(self, in_ch, r=16):
        super().__init__()
        hid = max(1, in_ch // r)
        self.avg = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(in_ch, hid, 1, bias=True), nn.ReLU(inplace=True),
            nn.Conv2d(hid, in_ch, 1, bias=True), nn.Sigmoid()
        )
    def forward(self, x):
        w = self.fc(self.avg(x))
        return x * w

class UCD_Fuse(nn.Module):
    """Uncertainty-aware Boundary-Guided fusion"""
    def __init__(self, in_channels, mid_channels):
        super().__init__()
        # 融合前的降维 + 轻量卷积
        self.pre_fuse = nn.Sequential(
            ConvNormAct(in_channels * 3, in_channels, kernel_size=1),
            ConvNormAct(in_channels, in_channels, kernel_size=3)
        )
        self.se = SEBlock(in_channels, r=16)
        self.head = nn.Sequential(
            ConvNormAct(in_channels, mid_channels, kernel_size=3),
            nn.Conv2d(mid_channels, mid_channels, kernel_size=3, padding=1),
        )

    def forward(self, ra_out, conv_out, edge_feat, weight_ra, weight_conv, weight_edge):

        ra_out_w   = ra_out   * weight_ra
        conv_out_w = conv_out * weight_conv
        edge_feat_w= edge_feat* weight_edge
        x = torch.cat([ra_out_w, conv_out_w, edge_feat_w], dim=1)
        x = self.pre_fuse(x)
        x = self.se(x)
        y = self.head(x)
        return y

class UCD(nn.Module):
    def __init__(self, in_channels, mid_channels, bias=False):
        super(UCD, self).__init__()
        # 正向增强分支
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels * 2, in_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=bias), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=bias), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, stride=1, bias=bias), nn.BatchNorm2d(in_channels), nn.ReLU(inplace=True),
        )
        # 边界编码
        self.edge_logits = nn.Conv2d(in_channels, 1, 1)

        # 三分支不确定性门控融合
        self.fuse = UCD_Fuse(in_channels=in_channels, mid_channels=mid_channels)

    def forward(self, GL, origin_x, prior_cam):
        B, C, H, W = GL.shape
        prior_cam = F.interpolate(prior_cam, size=(H, W), mode='bilinear', align_corners=False)

        # 正向语义
        yt = self.conv(torch.cat([GL, prior_cam.expand(-1, C, -1, -1)], dim=1))
        conv_out = self.conv3(yt)

        # 反向注意力（背景）
        r_prior = 1.0 - torch.sigmoid(prior_cam.detach())           # 背景概率
        ra_out  = r_prior.expand(-1, C, -1, -1) * GL

        # 边界：浅层 + 不确定性
        e_logits   = self.edge_logits(origin_x)
        e_prob     = torch.sigmoid(e_logits)

        # 不确定性
        p = torch.sigmoid(prior_cam)
        u = 1.0 - torch.abs(2.0 * p - 1.0)
        edge_weight = torch.clamp(0.5 * e_prob + 0.5 * u, 0, 1)
        conv_weight = 1.0 - edge_weight
        ra_weight   = r_prior

        # GL 细化边界特征
        edge_feat = GL * edge_weight + origin_x

        # 三分支融合 + 预测
        y = self.fuse(ra_out, conv_out, edge_feat, ra_weight, conv_weight, edge_weight)
        y = y + prior_cam

        return y



class SmoCD(nn.Module):
    def __init__(self, channels=[512, 320, 128, 64], stages=4):
        super(SmoCD, self).__init__()

        self.stages = stages
        self.Conv_1x1 = nn.ModuleList([
            nn.Conv2d(2 * channels[i], channels[i], kernel_size=1, stride=1, padding=0) for i in range(1, self.stages)
        ])
        self.LocalBlock_x = nn.Sequential(
            ConvNormAct(dim_in=3, dim_out=channels[3], kernel_size=3, stride=2),
            ConvNormAct(dim_in=channels[3], dim_out=channels[3], kernel_size=3, stride=2),
            MFE_module(in_channel=channels[3], out_channel=channels[3], exp_ratio=2),
            MFE_module(in_channel=channels[3], out_channel=channels[3], exp_ratio=2),
        )
        self.LocalBlocks = nn.ModuleList([
            MFE_module(in_channel=channels[i], out_channel=channels[i], exp_ratio=2) for i in range(self.stages)
        ])

        self.MixBlock0 = SAM(in_channel=2 * channels[0], out_channel=channels[0])  # MixBlock0：512
        self.MixBlock1 = SAM(in_channel=2 * channels[1], out_channel=channels[1])  # MixBlock1：192
        self.MixBlock2 = SAM(in_channel=2 * channels[2], out_channel=channels[2])  # MixBlock2：88
        self.MixBlock3 = SAM(in_channel=2 * channels[3], out_channel=channels[3])  # MixBlock2：88

        self.HardParts = nn.ModuleList(
            UCD(in_channels=channels[i], mid_channels=channels[i]) for i in range(1, self.stages)
        )

        self.Ups = nn.ModuleList([
            up_conv(ch_in=channels[i], ch_out=channels[i+1]) for i in range(self.stages-1)
        ])
        self.down = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.Semantic_fusion1 = BSR(in_channel=channels[3], out_channel=channels[3])
        self.Semantic_fusion2 = BSR(in_channel=channels[3], out_channel=channels[2])
        self.Semantic_fusion3 = BSR(in_channel=channels[3] + channels[2], out_channel=channels[1])
        self.Semantic_fusion4 = BSR(in_channel=channels[3] + channels[2] + channels[1], out_channel=channels[0])

        self.origin_down1 = nn.Sequential(
            ConvNormAct(dim_in=64, dim_out=64, kernel_size=3, stride=2),
            ConvNormAct(dim_in=64, dim_out=128, kernel_size=1)
        )
        self.origin_down2 = nn.Sequential(
            ConvNormAct(dim_in=128, dim_out=128, kernel_size=3, stride=2),
            ConvNormAct(dim_in=128, dim_out=320, kernel_size=1)
        )

        # Prediction heads initialization
        n_class = 1
        self.out_head1 = nn.Conv2d(512, n_class, 1)
        self.out_head2 = nn.Conv2d(320, n_class, 1)
        self.out_head3 = nn.Conv2d(128, n_class, 1)
        self.out_head4 = nn.Conv2d(64, n_class, 1)
        self.pred_saliency = nn.Conv2d(in_channels=512, out_channels=1, kernel_size=1, bias=True)

    def forward(self, x, x4, x3, x2, x1):
        origin_x1 = self.LocalBlock_x(x)
        origin_x2 = self.origin_down1(origin_x1)
        origin_x3 = self.origin_down2(origin_x2)

        origin_x1_ = self.Semantic_fusion1(origin_x1)
        guide = self.pred_saliency(x4).detach()
        x1 = self.MixBlock3(torch.cat((origin_x1_, x1), 1), guide)

        x1_down = self.down(x1)
        x1_ = self.Semantic_fusion2(x1_down)
        x2 = self.MixBlock2(torch.cat((x1_, x2), 1), guide)

        x1_down = self.down(x1_down)
        x2_down = self.down(x2)
        x1_x2 = torch.cat((x1_down, x2_down), dim=1)
        x1_x2 = self.Semantic_fusion3(x1_x2)
        x3 = self.MixBlock1(torch.cat((x1_x2, x3), 1), guide)

        x1_down = self.down(x1_down)
        x2_down = self.down(x2_down)
        x3_down = self.down(x3)
        x1_x2_x3 = torch.cat((x1_down, x2_down, x3_down), dim=1)
        x1_x2_x3 = self.Semantic_fusion4(x1_x2_x3)
        x4 = self.MixBlock0(torch.cat((x1_x2_x3, x4), 1), guide)

        pred4 = self.out_head1(x4)

        u4 = self.Ups[0](x4)

        d3 = torch.mul(x3, u4) + x3
        d3 = self.HardParts[0](d3, origin_x3, pred4)
        pred3 = self.out_head2(d3)

        u3 = self.Ups[1](d3)

        d2 = torch.mul(x2, u3) + x2
        d2 = self.HardParts[1](d2, origin_x2, pred3)
        pred2 = self.out_head3(d2)

        u2 = self.Ups[2](d2)

        d1 = torch.mul(x1, u2) + x1
        d1 = self.HardParts[2](d1, origin_x1, pred2)
        pred1 = self.out_head4(d1)

        return pred4, pred3, pred2, pred1


