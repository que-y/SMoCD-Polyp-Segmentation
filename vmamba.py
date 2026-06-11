import math
import copy
from functools import partial
from typing import Optional, Callable, Any
from collections import OrderedDict
from torch.cuda.amp import autocast
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath, trunc_normal_
DropPath.__repr__ = lambda self: f"timm.DropPath({self.drop_prob})"


# cross selective scan ===============================

if True:
    import selective_scan_cuda_core as selective_scan_cuda


    class SelectiveScan(torch.autograd.Function):
        @staticmethod
        @torch.cuda.amp.custom_fwd(cast_inputs=torch.float32)
        def forward(ctx, u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False, nrows=1):
            assert nrows in [1, 2, 3, 4], f"{nrows}"
            assert u.shape[1] % (B.shape[1] * nrows) == 0, f"{nrows}, {u.shape}, {B.shape}"
            ctx.delta_softplus = delta_softplus
            ctx.nrows = nrows

            # all in float
            if u.stride(-1) != 1:
                u = u.contiguous()
            if delta.stride(-1) != 1:
                delta = delta.contiguous()
            if D is not None:
                D = D.contiguous()
            if B.stride(-1) != 1:
                B = B.contiguous()
            if C.stride(-1) != 1:
                C = C.contiguous()
            if B.dim() == 3:
                B = B.unsqueeze(dim=1)
                ctx.squeeze_B = True
            if C.dim() == 3:
                C = C.unsqueeze(dim=1)
                ctx.squeeze_C = True

            out, x, *rest = selective_scan_cuda.fwd(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

            ctx.save_for_backward(u, delta, A, B, C, D, delta_bias, x)
            return out

        @staticmethod
        @torch.cuda.amp.custom_bwd
        def backward(ctx, dout, *args):
            u, delta, A, B, C, D, delta_bias, x = ctx.saved_tensors
            if dout.stride(-1) != 1:
                dout = dout.contiguous()
            du, ddelta, dA, dB, dC, dD, ddelta_bias, *rest = selective_scan_cuda.bwd(
                u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, 1
                # u, delta, A, B, C, D, delta_bias, dout, x, ctx.delta_softplus, ctx.nrows,
            )
            dB = dB.squeeze(1) if getattr(ctx, "squeeze_B", False) else dB
            dC = dC.squeeze(1) if getattr(ctx, "squeeze_C", False) else dC
            return (du, ddelta, dA, dB, dC, dD, ddelta_bias, None, None)


    class CrossScan(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x: torch.Tensor):
            B, C, H, W = x.shape
            ctx.shape = (B, C, H, W)
            xs = x.new_empty((B, 4, C, H * W))
            xs[:, 0] = x.flatten(2, 3)
            xs[:, 1] = x.transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            return xs

        @staticmethod
        def backward(ctx, ys: torch.Tensor):
            # out: (b, k, d, l)
            B, C, H, W = ctx.shape
            L = H * W
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, -1, L)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, -1, L)
            return y.view(B, -1, H, W)


    class CrossMerge(torch.autograd.Function):
        @staticmethod
        def forward(ctx, ys: torch.Tensor):
            B, K, D, H, W = ys.shape
            ctx.shape = (H, W)
            ys = ys.view(B, K, D, -1)
            ys = ys[:, 0:2] + ys[:, 2:4].flip(dims=[-1]).view(B, 2, D, -1)
            y = ys[:, 0] + ys[:, 1].view(B, -1, W, H).transpose(dim0=2, dim1=3).contiguous().view(B, D, -1)
            return y

        @staticmethod
        def backward(ctx, x: torch.Tensor):
            # B, D, L = x.shape
            # out: (b, k, d, l)
            H, W = ctx.shape
            B, C, L = x.shape
            xs = x.new_empty((B, 4, C, L))
            xs[:, 0] = x
            xs[:, 1] = x.view(B, C, H, W).transpose(dim0=2, dim1=3).flatten(2, 3)
            xs[:, 2:4] = torch.flip(xs[:, 0:2], dims=[-1])
            xs = xs.view(B, 4, C, H, W)
            return xs, None, None

    def cross_selective_scan(
            x: torch.Tensor = None,
            x_proj_weight: torch.Tensor = None,
            x_proj_bias: torch.Tensor = None,
            dt_projs_weight: torch.Tensor = None,
            dt_projs_bias: torch.Tensor = None,
            A_logs: torch.Tensor = None,
            Ds: torch.Tensor = None,
            out_norm: torch.nn.Module = None,
            softmax_version=False,
            nrows=-1,
            delta_softplus=True,
    ):
        B, D, H, W = x.shape
        D, N = A_logs.shape
        K, D, R = dt_projs_weight.shape
        L = H * W
        if nrows < 1:
            if D % 4 == 0:
                nrows = 4
            elif D % 3 == 0:
                nrows = 3
            elif D % 2 == 0:
                nrows = 2
            else:
                nrows = 1

        xs = CrossScan.apply(x)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs, x_proj_weight)
        if x_proj_bias is not None:
            x_dbl = x_dbl + x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts, dt_projs_weight)

        xs = xs.view(B, -1, L).to(torch.float)
        dts = dts.contiguous().view(B, -1, L).to(torch.float)
        As = -torch.exp(A_logs.to(torch.float))  # (k * c, d_state)
        Bs = Bs.contiguous().to(torch.float)
        Cs = Cs.contiguous().to(torch.float)
        Ds = Ds.to(torch.float)  # (K * c)
        delta_bias = dt_projs_bias.view(-1).to(torch.float)

        def selective_scan(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=True, nrows=1):
            return SelectiveScan.apply(u, delta, A, B, C, D, delta_bias, delta_softplus, nrows)

        ys: torch.Tensor = selective_scan(
            xs, dts, As, Bs, Cs, Ds, delta_bias, delta_softplus, nrows,
        ).view(B, K, -1, H, W)

        y = CrossMerge.apply(ys)

        if softmax_version:
            y = y.softmax(y, dim=-1).to(x.dtype)
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        else:
            y = y.transpose(dim0=1, dim1=2).contiguous().view(B, H, W, -1)
            y = out_norm(y).to(x.dtype)

        return y


DEV = False
class SS2D(nn.Module):
    def __init__(
            self,
            d_model=96,
            d_state=16,
            ssm_ratio=2,
            dt_rank="auto",
            d_conv=3,
            conv_bias=True,
            dropout=0.,
            bias=False,
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            softmax_version=False,

            **kwargs,
    ):
        if DEV:
            d_conv = -1

        factory_kwargs = {"device": None, "dtype": None}
        super().__init__()
        self.softmax_version = softmax_version
        self.d_model = d_model
        self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_state  # 20240109
        self.d_conv = d_conv
        self.expand = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)

        if self.d_conv > 1:
            self.conv2d = nn.Conv2d(
                in_channels=self.d_inner,
                out_channels=self.d_inner,
                groups=self.d_inner,
                bias=conv_bias,
                kernel_size=d_conv,
                padding=(d_conv - 1) // 2,
                **factory_kwargs,
            )
            self.act = nn.SiLU()

        self.K = 4

        self.x_proj = [
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K, N, inner)
        del self.x_proj

        self.dt_projs = [
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs)
            for _ in range(self.K)
        ]
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K, inner)
        del self.dt_projs

        self.K2 = self.K


        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=self.K2, merge=True)  # (K * D, N)
        self.Ds = self.D_init(self.d_inner, copies=self.K2, merge=True)  # (K * D)

        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else nn.Identity()

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)

        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)


        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=-1, device=None, merge=True):

        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)
        if copies > 0:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=-1, device=None, merge=True):
        D = torch.ones(d_inner, device=device)
        if copies > 0:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)
        D._no_weight_decay = True
        return D

    def forward_corev2(self, x: torch.Tensor, nrows=-1):
        return cross_selective_scan(
            x, self.x_proj_weight, None, self.dt_projs_weight, self.dt_projs_bias,
            self.A_logs, self.Ds, getattr(self, "out_norm", None), self.softmax_version,
            nrows=nrows,
        )

    forward_core = forward_corev2  # vmamba

    def forward(self, x: torch.Tensor, **kwargs):
        xz = self.in_proj(x)
        if self.d_conv > 1:
            x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            x = self.act(self.conv2d(x))  # (b, d, h, w)
            y = self.forward_core(x)
            if self.softmax_version:
                y = y * z
            else:
                y = y * F.silu(z)
        else:
            if self.softmax_version:
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
                x = F.silu(x)
            else:
                xz = F.silu(xz)
                x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)
            x = x.permute(0, 3, 1, 2).contiguous()
            y = self.forward_core(x)
            y = y * z
        out = self.dropout(self.out_proj(y))
        return out


class SaliencyMB_CATA1D_K1(nn.Module):
    def __init__(self,
                 d_model=96, d_state=4, ssm_ratio=2, dt_rank="auto",
                 d_conv=3, conv_bias=True, dropout=0., bias=False,
                 dt_min=0.001, dt_max=0.1, dt_init="random", dt_scale=1.0, dt_init_floor=1e-4,
                 softmax_version=False,
                 # CATA
                 cata_num_centers=None, cata_iters=3, cata_ema=0.999,
                 use_sal_prompt=True):
        super().__init__()
        self.softmax_version = softmax_version
        self.use_sal_prompt = use_sal_prompt

        # dims
        self.d_model = d_model
        self.d_state = math.ceil(d_model/6) if d_state == "auto" else d_state
        self.expand  = ssm_ratio
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(d_model/16) if dt_rank == "auto" else dt_rank

        # in-proj + depthwise conv
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias)
        self.d_conv  = d_conv
        if d_conv >= 2:
            self.conv2d = nn.Conv2d(self.d_inner, self.d_inner, groups=self.d_inner,
                                    kernel_size=d_conv, padding=(d_conv-1)//2, bias=conv_bias)
        self.act = nn.SiLU()

        # x -> [R + 2N]
        self.x_proj = nn.Linear(self.d_inner, (self.dt_rank + 2*self.d_state), bias=False)

        # dt 投影：R -> D（带 bias）
        self.dt_proj = SS2D.dt_init(self.dt_rank, self.d_inner,
                                    dt_scale, dt_init, dt_min, dt_max, dt_init_floor)

        # A_log, D（复用 SS2D 的构造）
        self.A_logs = SS2D.A_log_init(self.d_state, self.d_inner, copies=-1, merge=True) # (D,N)
        self.Ds     = SS2D.D_init(self.d_inner, copies=-1, merge=True)                    # (D,)

        if not self.softmax_version:
            self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout  = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # 显著性 prompt
        self.prompt_head = nn.Sequential(nn.LayerNorm(1), nn.Linear(1, self.d_state, bias=False))

        # CATA buffers
        self.cata_num_centers = cata_num_centers
        self.cata_iters = cata_iters
        self.cata_ema = cata_ema
        M = (cata_num_centers or 64)
        self.register_buffer('cata_means', torch.randn(M, self.d_inner))
        self.register_buffer('cata_initted', torch.tensor(False))

    def _seq_reorder(self, tokens, sal_vec):

        B, L, D = tokens.shape
        cnt = self.cata_num_centers or min(64, L)

        # 首次/形状变更时初始化中心
        if (not bool(self.cata_initted.item())) or (self.cata_means.shape[0] != cnt):
            self.cata_means = self.cata_means.new_empty(cnt, D)
            xn = F.normalize(tokens, dim=-1)
            pad = (cnt - (L % cnt)) % cnt
            xpad = torch.cat([xn, xn[:, L-pad:L, :].flip(1)], dim=1) if pad>0 else xn
            means0 = rearrange(xpad, 'b (cnt m) d -> cnt (b m) d', cnt=cnt).mean(1)
            self.cata_means.copy_(means0)
            self.cata_initted.fill_(True)

        # 训练期细化 + EMA
        xn = F.normalize(tokens, dim=-1)
        if self.training and self.cata_iters > 1:
            with torch.no_grad():
                m = F.normalize(self.cata_means, dim=-1)
                for _ in range(self.cata_iters-1):
                    sim = torch.einsum('bld,md->blm', xn, m)
                    bucket = sim.argmax(-1)  # (B,L)

                    sums = tokens.new_zeros(m.shape[0], D)
                    cnts = tokens.new_zeros(m.shape[0], 1)
                    for b in range(B):
                        sums.index_add_(0, bucket[b], xn[b])

                        cnts.index_add_(0, bucket[b], torch.ones(L,1, device=tokens.device, dtype=tokens.dtype))
                    m = F.normalize(torch.where(cnts>0, sums/(cnts+1e-6), m), dim=-1)
            self.cata_means.mul_(self.cata_ema).add_(m, alpha=1 - self.cata_ema)

        # 簇分配 → 稳定排序
        buckets = torch.einsum('bld,md->blm', tokens, F.normalize(self.cata_means, dim=-1)).argmax(-1)  # (B,L)
        perm = torch.argsort(buckets, dim=-1, stable=True)  # (B,L)
        # 逆序 index
        rev = torch.zeros_like(perm)
        arange = torch.arange(L, device=perm.device).unsqueeze(0).expand(B, L)
        rev.scatter_(1, perm, arange)


        seq   = torch.gather(tokens, 1, perm.unsqueeze(-1).expand(-1, -1, D))
        gtseq = torch.gather(sal_vec, 1, perm.unsqueeze(-1))


        return seq, gtseq, perm, rev

    def forward(self, x: torch.Tensor, gt: torch.Tensor):
        """
        x: (B,H,W,C=d_model), gt: (B,1,H,W) or (B,H,W[,1])
        """
        B, H, W, C = x.shape
        L = H * W

        # 显著性：仅作 prompt
        if gt.dim() == 3: gt = gt.unsqueeze(1)
        if gt.shape[1] != 1: gt = gt[:, :1]
        gt  = F.interpolate(gt, size=(H,W), mode='bilinear', align_corners=False)
        sal = torch.sigmoid(gt).clamp(0,1)
        sal_vec = sal.flatten(2).transpose(1,2)  # (B,L,1)

        # in-proj + 可选 DWConv
        x_in = self.in_proj(x)                   # (B,H,W,D)
        if self.d_conv >= 2:
            x_dw = self.act(self.conv2d(x_in.permute(0,3,1,2))).permute(0,2,3,1).contiguous()
        else:
            x_dw = self.act(x_in)
        tokens = x_dw.view(B, L, self.d_inner)   # (B,L,D)

        # CATA：得到单向序列
        seq, gtseq, perm, rev = self._seq_reorder(tokens, sal_vec)
        xs = seq.transpose(1,2).unsqueeze(1)     # (B,1,D,L)

        # x -> [R+2N]
        x_dbl = torch.einsum('b k d l, c d -> b k c l', xs, self.x_proj.weight)  # (B,1,R+2N,L)
        R, N = self.dt_rank, self.d_state
        dts, Bs, Cs = torch.split(x_dbl, [R, N, N], dim=2)                       # (B,1,R,L)/(B,1,N,L)/(B,1,N,L)

        # dt: R -> D（带 bias）
        dts = torch.einsum('b k r l, d r -> b k d l', dts, self.dt_proj.weight)  # (B,1,D,L)
        dtb = self.dt_proj.bias                                                  # (D,)

        # ASE：显著性 prompt 加到 C
        if self.use_sal_prompt:
            P = self.prompt_head(gtseq).transpose(1,2).unsqueeze(1)              # (B,1,N,L)
            Cs = Cs + P

        # pack 成 kernel 需要的形状
        u     = xs.view(B, -1, L).to(torch.float32)                               # (B,D,L)
        delta = dts.contiguous().view(B, -1, L).to(torch.float32)                 # (B,D,L)
        Bm    = Bs.contiguous().view(B, -1, L).to(torch.float32)                  # (B,N,L)
        Cm    = Cs.contiguous().view(B, -1, L).to(torch.float32)                  # (B,N,L)
        A     = (-torch.exp(self.A_logs)).to(torch.float32)                       # (D,N)
        Dskip = self.Ds.view(-1).to(torch.float32)                                # (D,)
        dbias = dtb.view(-1).to(torch.float32)                                    # (D,)

        y = SelectiveScan.apply(u, delta, A, Bm, Cm, Dskip, dbias, True, 1)       # (B,D,L)

        # 映射回输出 + 逆置
        y = y.transpose(1,2).contiguous()                                         # (B,L,D)
        y = self.out_proj(self.out_norm(y) if not self.softmax_version else y)
        y = torch.gather(y, 1, rev.unsqueeze(-1).expand(-1, -1, self.d_model)).view(B, H, W, C)
        return self.dropout(y)








