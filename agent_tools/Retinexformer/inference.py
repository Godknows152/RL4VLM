# Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement
# Yuanhao Cai, Hao Bian, Jing Lin, Haoqian Wang, Radu Timofte, Yulun Zhang
# International Conference on Computer Vision (ICCV), 2023
# https://arxiv.org/abs/2303.06705
# https://github.com/caiyuanhao1998/Retinexformer

import numpy as np
import os
import torch
import torch.nn.functional as F
from .Enhancement import utils as utils

from skimage import img_as_ubyte

from basicsr_retinexformer.models import create_model
from basicsr_retinexformer.utils.options import parse

def self_ensemble(x, model):
    def forward_transformed(x, hflip, vflip, rotate, model):
        if hflip:
            x = torch.flip(x, (-2,))
        if vflip:
            x = torch.flip(x, (-1,))
        if rotate:
            x = torch.rot90(x, dims=(-2, -1))
        x = model(x)
        if rotate:
            x = torch.rot90(x, dims=(-2, -1), k=3)
        if vflip:
            x = torch.flip(x, (-1,))
        if hflip:
            x = torch.flip(x, (-2,))
        return x
    t = []
    for hflip in [False, True]:
        for vflip in [False, True]:
            for rot in [False, True]:
                t.append(forward_transformed(x, hflip, vflip, rot, model))
    t = torch.stack(t)
    return torch.mean(t, dim=0)



def load_retinexformer_model(model_path=None, device=torch.device('cuda:0')):

    opt = os.path.join(os.path.dirname(os.path.abspath(__file__)), f'Options/RetinexFormer_FiveK.yml')
    weights = f'{model_path}/FiveK.pth'
    opt = parse(opt, is_train=False)
    opt['dist'] = False
    
    model_restoration = create_model(opt).net_g
    # 加载模型
    checkpoint = torch.load(weights, weights_only=False)

    try:
        model_restoration.load_state_dict(checkpoint['params'])
    except:
        new_checkpoint = {}
        for k in checkpoint['params']:
            new_checkpoint['module.' + k] = checkpoint['params'][k]
        model_restoration.load_state_dict(new_checkpoint)

    print("===>Testing using weights: ", weights)
    model_restoration.to(device)
    #model_restoration = nn.DataParallel(model_restoration)
    model_restoration.eval()
    return model_restoration


def retinexformer_predict(model_restoration, input_img, output_dir, is_self_ensemble=False, device=torch.device('cuda:0')):
    factor = 4

    torch.cuda.ipc_collect()
    torch.cuda.empty_cache()

    img = np.float32(utils.load_img(input_img)) / 255.

    img = torch.from_numpy(img).permute(2, 0, 1)
    input_ = img.unsqueeze(0).to(device)

    # Padding in case images are not multiples of 4
    b, c, h, w = input_.shape
    H, W = ((h + factor) // factor) * \
        factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input_ = F.pad(input_, (0, padw, 0, padh), 'reflect').to(device)
    if h < 3000 and w < 3000:
        if is_self_ensemble:
            restored = self_ensemble(input_, model_restoration)
        else:
            restored = model_restoration(input_)
    else:
        # split and test
        input_1 = input_[:, :, :, 1::2]
        input_2 = input_[:, :, :, 0::2]
        if is_self_ensemble:
            restored_1 = self_ensemble(input_1, model_restoration)
            restored_2 = self_ensemble(input_2, model_restoration)
        else:
            restored_1 = model_restoration(input_1)
            restored_2 = model_restoration(input_2)
        restored = torch.zeros_like(input_)
        restored[:, :, :, 1::2] = restored_1
        restored[:, :, :, 0::2] = restored_2

    # Unpad images to original dimensions
    restored = restored[:, :, :h, :w]

    restored = torch.clamp(restored, 0, 1).cpu(
    ).detach().permute(0, 2, 3, 1).squeeze(0).numpy()

    save_path = os.path.join(output_dir, os.path.splitext(
            os.path.split(input_img)[-1])[0] + '.png')
    utils.save_img(save_path, img_as_ubyte(restored))
    return save_path
        
