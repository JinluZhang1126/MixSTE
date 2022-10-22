# Following the implementation of RELA(https://github.com/rishikksh20/rectified-linear-attention/blob/master/attention.py)
import torch
import math
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class RectifiedLinearAttention(nn.Module):
    def __init__(self, dim, num_heads = 8, attn_drop = 0., proj_drop=0., qk_scale=None, qkv_bias=False, comb=False, vis=False):
        super().__init__()
        dim_head = dim // num_heads
        inner_dim = dim_head *  num_heads
        project_out = not (num_heads == 1 and dim_head == dim)

        self.heads = num_heads
        self.scale = qk_scale or dim_head ** -0.5

        self.to_qkv = nn.Linear(dim, inner_dim * 3, qkv_bias)

        self.norm = nn.LayerNorm(inner_dim)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(proj_drop)
        ) if project_out else nn.Identity()

    def forward(self, x, vis=False):
        b, n, _, h = *x.shape, self.heads
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = h), qkv)

        dots = einsum('b h i d, b h j d -> b h i j', q, k) * self.scale

        attn = F.relu(dots)

        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')

        out =  self.to_out(self.norm(out))
        return out