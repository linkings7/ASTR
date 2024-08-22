import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
from lib.models.layers.rpe import generate_2d_concatenated_self_attention_relative_positional_encoding_index




class Cross_Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        # self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, z, x, return_attention=False):
        B, N, C = x.shape
        N_z = z.shape[1]
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3) # B N 3C -> B N 1 H E -> 1 B H N E (B, 12, 256, 64)
        kv = self.kv(z).reshape(B, N_z, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # B N 2C -> B N 2 H E -> 2 B H N E
        #qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4) # B N 3C -> B N 3 H E -> 3 B H N E
        k, v = kv[0], kv[1] # make torchscript happy (cannot use tensor as tuple)

        # x: (B, 256, 768)
        # q: (B, 12, 256, 64)
        # k: (B, 12, 320, 64)
        # v: (B, 12, 320, 64)

        attn = (q @ k.transpose(-2, -1)) * self.scale # (B, 12, 256, 320)

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # # (B, 12, 256, 320) @ (B, 12, 320->(P3->192) , 64) -> (B, 12, 256, 64) -> (B, 256, 12, 64)


        x = self.proj(x)
        x = self.proj_drop(x)

        if return_attention:
            return x, attn
        return x
