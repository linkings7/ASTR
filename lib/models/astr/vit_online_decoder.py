import logging
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import to_2tuple

from lib.models.layers.patch_embed import PatchEmbed
from lib.models.layers.attn_blocks import Block
from lib.models.layers.module.cross_blocks import Cross_Block
from .utils import combine_tokens
from .vit import VisionTransformer

_logger = logging.getLogger(__name__)

class VisionTransformerCE(VisionTransformer):
    """ Vision Transformer with candidate elimination (CE) module

    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`
        - https://arxiv.org/abs/2010.11929

    Includes distillation token & head support for `DeiT: Data-efficient Image Transformers`
        - https://arxiv.org/abs/2012.12877
    """

    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=True, representation_size=None, distilled=False,
                 drop_rate=0., attn_drop_rate=0., drop_path_rate=0., embed_layer=PatchEmbed, norm_layer=None,
                 act_layer=None, weight_init='', search_size=None, template_size=None,
                 new_patch_size=None, num_decoder=1, prompt_size=5):
        """
        Args:
            img_size (int, tuple): input image size
            patch_size (int, tuple): patch size
            in_chans (int): number of input channels
            num_classes (int): number of classes for classification head
            embed_dim (int): embedding dimension
            depth (int): depth of transformer
            num_heads (int): number of attention heads
            mlp_ratio (int): ratio of mlp hidden dim to embedding dim
            qkv_bias (bool): enable bias for qkv if True
            representation_size (Optional[int]): enable and set representation layer (pre-logits) to this value if set
            distilled (bool): model includes a distillation token and head as in DeiT models
            drop_rate (float): dropout rate
            attn_drop_rate (float): attention dropout rate
            drop_path_rate (float): stochastic depth rate
            embed_layer (nn.Module): patch embedding layer
            norm_layer: (nn.Module): normalization layer
            weight_init: (str): weight init scheme
            new_patch_size: backbone stride
        """
        super().__init__()
        if isinstance(img_size, tuple):
            self.img_size = img_size
        else:
            self.img_size = to_2tuple(img_size)
        self.patch_size = patch_size
        self.in_chans = in_chans

        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.num_tokens = 2 if distilled else 1
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)
        act_layer = act_layer or nn.GELU

        self.patch_embed = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        self.patch_embed_prompt = embed_layer(
            img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)

        # num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dist_token = nn.Parameter(torch.zeros(1, 1, embed_dim)) if distilled else None
        # self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + self.num_tokens, embed_dim)) # it's redundant
        self.pos_drop = nn.Dropout(p=drop_rate)

        '''
        prompt parameters
        '''
        H, W = search_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_search=new_P_H * new_P_W
        H, W = template_size
        new_P_H, new_P_W = H // new_patch_size, W // new_patch_size
        self.num_patches_template=new_P_H * new_P_W
        """add here, no need use backbone.finetune_track """
        self.pos_embed_z = nn.Parameter(torch.zeros(1, self.num_patches_template, embed_dim))
        self.pos_embed_x = nn.Parameter(torch.zeros(1, self.num_patches_search, embed_dim))

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule

        self.blocks = nn.Sequential(*[
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(depth)])

        cross_depth = num_decoder
        self.cross_blocks = nn.Sequential(*[
            Cross_Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, drop=drop_rate,
                attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, act_layer=act_layer)
            for i in range(cross_depth)])

        self.num_mix = prompt_size
        self.mixup_ratio = nn.Parameter(torch.zeros(self.num_mix, 1, 2))

        nn.init.xavier_uniform_(self.mixup_ratio)

        self.norm = norm_layer(embed_dim)
        self.init_weights(weight_init)

    def forward_features(self, z, x, cz):
        B, H, W = x.shape[0], x.shape[2], x.shape[3]

        x_rgb = x[:, :3, :, :]
        z_rgb = z[:, :3, :, :]
        z_rgb_online = cz[:, :3, :, :]
        # depth thermal event images
        x_dte = x[:, 3:, :, :]
        z_dte = z[:, 3:, :, :]
        z_dte_online = cz[:, 3:, :, :]

        x_rgb = self.patch_embed(x_rgb)
        z_rgb = self.patch_embed(z_rgb)
        z_rgb_online = self.patch_embed(z_rgb_online)

        x_dte = self.patch_embed_prompt(x_dte)
        z_dte = self.patch_embed_prompt(z_dte)
        z_dte_online = self.patch_embed_prompt(z_dte_online)

        p_z_rgb = z_rgb_online
        p_z_dte_t = z_dte_online

        x = x_rgb + x_dte
        z = z_rgb + z_dte
        z_online = z_rgb_online + z_dte_online

        x += self.pos_embed_x
        z += self.pos_embed_z
        z_online += self.pos_embed_z

        z = combine_tokens(z, z_online, mode=self.cat_mode)
        x = combine_tokens(z, x, mode=self.cat_mode)

        x = self.pos_drop(x)

        for i, blk in enumerate(self.blocks):
            x = blk(x)

        x = self.norm(x)

        lens_z = self.pos_embed_z.shape[1]
        x = x[:, 2 * lens_z:, :]

        _, N_t, C_t = z_rgb.shape
        mixup_ratio = F.softmax(self.mixup_ratio, dim=2)

        r1, r2 = torch.chunk(mixup_ratio, 2, dim=2)
        z_mix = p_z_rgb.unsqueeze(1) * r1 + p_z_dte_t.unsqueeze(1) * r2
        z_mix = z_mix.reshape(B, self.num_mix * N_t, C_t)

        z_mix = self.pos_drop(z_mix)

        for i, cross_blk in enumerate(self.cross_blocks):
            x = cross_blk(z_mix, x)

        aux_dict = {"attn": None}
        return x, aux_dict

    def forward(self, z, x, cz):

        x, aux_dict = self.forward_features(z, x, cz)

        return x, aux_dict


def _create_vision_transformer(pretrained=False, **kwargs):
    model = VisionTransformerCE(**kwargs)

    if pretrained:
        if 'npz' in pretrained:
            model.load_pretrained(pretrained, prefix='')
        else:
            checkpoint = torch.load(pretrained, map_location="cpu")
            missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)
            print('Load pretrained OSTrack from: ' + pretrained)
            print(f"missing_keys: {missing_keys}")
            print(f"unexpected_keys: {unexpected_keys}")

    return model


def vit_online_decoder(pretrained=False, **kwargs):
    """ ViT-Base model (ViT-B/16) from original paper (https://arxiv.org/abs/2010.11929).
    """
    model_kwargs = dict(
        patch_size=16, embed_dim=768, depth=12, num_heads=12, **kwargs)
    model = _create_vision_transformer(pretrained=pretrained, **model_kwargs)
    return model
