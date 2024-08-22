"""
Basic ViPT model.
"""
import math
import os
from typing import List
from timm.models.layers import to_2tuple
import torch
from torch import nn
from torch.nn.modules.transformer import _get_clones
from lib.models.layers.head import build_box_head
from lib.models.astr.vit import vit
from lib.models.astr.vit_online_decoder import vit_online_decoder
from lib.utils.box_ops import box_xyxy_to_cxcywh

class ASTR(nn.Module):
    """ This is the base class for ViPTrack """

    def __init__(self, transformer, box_head, aux_loss=False, head_type="CORNER"):
        """ Initializes the model.
        Parameters:
            transformer: torch module of the transformer architecture.
            aux_loss: True if auxiliary decoding losses (loss at each decoder layer) are to be used.
        """
        super().__init__()

        self.backbone = transformer
        self.box_head = box_head
        self.aux_loss = aux_loss
        self.head_type = head_type

        if head_type == "CORNER" or head_type == "CENTER":
            self.feat_sz_s = int(box_head.feat_sz)
            self.feat_len_s = int(box_head.feat_sz ** 2)

        if self.aux_loss:
            self.box_head = _get_clones(self.box_head, 6)


    def forward(self, template: torch.Tensor,
                search: torch.Tensor,
                cz=None
                ):

        x, aux_dict = self.backbone(z=template, x=search, cz=cz)

        # Forward head
        feat_last = x
        if isinstance(x, list):
            feat_last = x[-1]

        out = self.forward_head(feat_last, None)

        out.update(aux_dict)
        out['backbone_feat'] = x
        return out

    def forward_head(self, cat_feature, gt_score_map=None):
        """
        cat_feature: output embeddings of the backbone, it can be (HW1+HW2, B, C) or (HW2, B, C)
        """
        enc_opt = cat_feature[:, -self.feat_len_s:]  # encoder output for the search region (B, HW, C)
        opt = (enc_opt.unsqueeze(-1)).permute((0, 3, 2, 1)).contiguous()
        bs, Nq, C, HW = opt.size()
        opt_feat = opt.view(-1, C, self.feat_sz_s, self.feat_sz_s)

        if self.head_type == "CORNER":
            # run the corner head
            pred_box, score_map = self.box_head(opt_feat, True)
            outputs_coord = box_xyxy_to_cxcywh(pred_box)
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map,
                   }
            return out

        elif self.head_type == "CENTER":
            # run the center head
            score_map_ctr, bbox, size_map, offset_map = self.box_head(opt_feat, gt_score_map)
            # outputs_coord = box_xyxy_to_cxcywh(bbox)
            outputs_coord = bbox
            outputs_coord_new = outputs_coord.view(bs, Nq, 4)
            out = {'pred_boxes': outputs_coord_new,
                   'score_map': score_map_ctr,
                   'size_map': size_map,
                   'offset_map': offset_map}
            return out
        else:
            raise NotImplementedError


def build_astr(cfg, training=True):
    current_dir = os.path.dirname(os.path.abspath(__file__))  # This is your Project Root
    pretrained_path = os.path.join(current_dir, '../../../pretrained_models')  # use pretrained OSTrack as initialization
    if cfg.MODEL.PRETRAIN_FILE and ('OSTrack' not in cfg.MODEL.PRETRAIN_FILE) and training:
        pretrained = os.path.join(pretrained_path, cfg.MODEL.PRETRAIN_FILE)
    else:
        pretrained = ''

    backbone = create_backbone(pretrained, cfg)
    backbone.finetune_track(cfg=cfg, patch_start_index=1)
    box_head = build_box_head(cfg, backbone.embed_dim)

    model = ASTR(
        backbone,
        box_head,
        aux_loss=False,
        head_type=cfg.MODEL.HEAD.TYPE,
    )

    if 'OSTrack' in cfg.MODEL.PRETRAIN_FILE and training:
        checkpoint = torch.load(cfg.MODEL.PRETRAIN_FILE, map_location="cpu")
        missing_keys, unexpected_keys = model.load_state_dict(checkpoint["net"], strict=False)

    return model


def create_backbone(pretrained, cfg):
    backbone_type = cfg.MODEL.BACKBONE.TYPE
    common_params = {
        'pretrained': pretrained,
        'drop_path_rate': cfg.TRAIN.DROP_PATH_RATE,
        'search_size': to_2tuple(cfg.DATA.SEARCH.SIZE),
        'template_size': to_2tuple(cfg.DATA.TEMPLATE.SIZE),
        'new_patch_size': cfg.MODEL.BACKBONE.STRIDE,
    }

    backbone_map = {
        'vit': vit,
        'vit_online_decoder': vit_online_decoder,
    }

    if backbone_type not in backbone_map:
        raise NotImplementedError(f"Backbone type '{backbone_type}' is not implemented")

    backbone = backbone_map[backbone_type](**common_params)

    return backbone

