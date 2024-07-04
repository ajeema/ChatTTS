import math
from typing import List, Optional

import numpy as np
import pybase16384 as b14
import torch
import torch.nn as nn
import torch.nn.functional as F
from vector_quantize_pytorch import GroupedResidualFSQ


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim: int, intermediate_dim: int, kernel: int, dilation: int, layer_scale_init_value: float = 1e-6):
        super().__init__()
        self.dwconv = nn.Conv1d(dim, dim, kernel_size=kernel, padding=dilation * (kernel // 2), dilation=dilation, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, intermediate_dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(intermediate_dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True) if layer_scale_init_value > 0 else None

    def forward(self, x: torch.Tensor, cond=None) -> torch.Tensor:
        residual = x

        y = self.dwconv(x)
        y = y.transpose(1, 2)  # (B, C, T) -> (B, T, C)
        y = self.norm(y)
        y = self.pwconv1(y)
        y = self.act(y)
        y = self.pwconv2(y)
        if self.gamma is not None:
            y *= self.gamma
        y = y.transpose(1, 2)  # (B, T, C) -> (B, C, T)

        return y + residual


class GFSQ(nn.Module):
    def __init__(self, dim: int, levels: List[int], G: int, R: int, eps=1e-5, transpose=True):
        super().__init__()
        self.quantizer = GroupedResidualFSQ(dim=dim, levels=levels, num_quantizers=R, groups=G)
        self.n_ind = math.prod(levels)
        self.eps = eps
        self.transpose = transpose
        self.G = G
        self.R = R

    def _embed(self, x: torch.Tensor):
        if self.transpose:
            x = x.transpose(1, 2)
        x = x.view(x.size(0), x.size(1), self.G, self.R).permute(2, 0, 1, 3)
        feat = self.quantizer.get_output_from_indices(x)
        return feat.transpose(1, 2) if self.transpose else feat

    def forward(self, x):
        if self.transpose:
            x = x.transpose(1, 2)
        feat, ind = self.quantizer(x)
        ind = ind.permute(1, 2, 0, 3).contiguous().view(ind.size(1), ind.size(2), -1)
        embed_onehot = F.one_hot(ind.long(), self.n_ind).to(x.dtype)
        e_mean = embed_onehot.mean(dim=[0, 1])
        e_mean /= (e_mean.sum(dim=1) + self.eps).unsqueeze(1)
        perplexity = torch.exp(-torch.sum(e_mean * torch.log(e_mean + self.eps), dim=1))

        return torch.zeros(perplexity.shape, dtype=x.dtype, device=x.device), feat.transpose(1, 2) if self.transpose else feat, perplexity, None, ind.transpose(1, 2) if self.transpose else ind


class DVAEDecoder(nn.Module):
    def __init__(self, idim: int, odim: int, n_layer=12, bn_dim=64, hidden=256, kernel=7, dilation=2, up=False):
        super().__init__()
        self.up = up
        self.conv_in = nn.Sequential(nn.Conv1d(idim, bn_dim, 3, 1, 1), nn.GELU(), nn.Conv1d(bn_dim, hidden, 3, 1, 1))
        self.decoder_block = nn.ModuleList([ConvNeXtBlock(hidden, hidden * 4, kernel, dilation) for _ in range(n_layer)])
        self.conv_out = nn.Conv1d(hidden, odim, kernel_size=1, bias=False)

    def forward(self, x: torch.Tensor, conditioning=None) -> torch.Tensor:
        y = self.conv_in(x)
        for block in self.decoder_block:
            y = block(y, conditioning)
        return self.conv_out(y)


class DVAE(nn.Module):
    def __init__(self, decoder_config, vq_config, dim=512, coef: Optional[str] = None):
        super().__init__()
        coef = torch.rand(100) if coef is None else torch.from_numpy(np.frombuffer(b14.decode_from_string(coef), dtype=np.float32).copy())
        self.register_buffer("coef", coef.unsqueeze(0).unsqueeze(2))

        self.decoder = DVAEDecoder(**decoder_config)
        self.out_conv = nn.Conv1d(dim, 100, 3, 1, 1, bias=False)
        self.vq_layer = GFSQ(**vq_config) if vq_config else None

    def __repr__(self) -> str:
        return b14.encode_to_string(self.coef.cpu().numpy().astype(np.float32).tobytes())

    def forward(self, inp: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            vq_feats = self.vq_layer._embed(inp) if self.vq_layer else inp
            vq_feats = vq_feats.view(vq_feats.size(0), 2, vq_feats.size(1) // 2, vq_feats.size(2)).permute(0, 2, 3, 1).flatten(2)
            dec_out = self.out_conv(self.decoder(vq_feats))
            return dec_out * self.coef
