# code reference : https://github.com/Megvii-BaseDetection/YOLOX/
from copy import deepcopy
from thop import profile
import torch


def get_model_info(model, template_size):

    template_img = torch.zeros((1, 6, 128, 128), device=next(model.parameters()).device)

    if template_size == 1:
        online_template = None
    elif template_size == 2:
        online_template = template_img
    else:
        online_template = [template_img]*template_size
    search_img = torch.zeros((1, 6, 256, 256), device=next(model.parameters()).device)
    flops, _ = profile(deepcopy(model), inputs=(template_img, search_img, online_template), verbose=False)

    params = sum(p.numel() for p in model.parameters())
    learnable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    params /= 1e6
    learnable_params /= 1e6
    flops /= 1e9
    info = "Total Params: {:.2f}M, Learnable Params: {:.2f}M, Gflops: {:.2f}".format(params, learnable_params, flops)
    return info
