import torch
import torch.nn as nn
from .LD_model1 import Dehaze
import math
import os
from PIL import Image
from torchvision.transforms import Compose, ToTensor, Normalize
os.environ['KMP_DUPLICATE_LIB_OK']='True'

import os
import numpy as np
from torchvision.utils import save_image


def PSNR(img1, img2):
    b,_,_,_=img1.shape
    img1=np.clip(img1,0,255)
    img2=np.clip(img2,0,255)
    mse = np.mean((img1/ 255. - img2/ 255.) ** 2)
    if mse == 0:
        return 100
    PIXEL_MAX = 1
    return 20 * math.log10(PIXEL_MAX / math.sqrt(mse))
def ensure_dir(dir_path):
    if not os.path.isdir(dir_path):
        os.makedirs(dir_path)

def load_kanet_model(model_path, device='cuda'):
    """
    Load KANet model from a given checkpoint.
    
    Args:
        model_path (str): Path to the model checkpoint
        device (str, optional): Device to load the model on. Defaults to 'cuda'.
    
    Returns:
        nn.Module: Loaded and prepared KANet model
    """
    path = os.path.join(model_path, 'trained_model_epoch1.pk')
    # Ensure device is set correctly
    if isinstance(device, torch.device):
        device_str = str(device)
    else:
        device_str = device if torch.cuda.is_available() else 'cpu'
    
    # Load checkpoint
    ckp = torch.load(path, map_location=device_str, weights_only=False)
    
    # Initialize model
    net = Dehaze(3, 3).to(device_str)
    
    # 加载权重：处理 DataParallel 保存的模型（移除 'module.' 前缀）
    # 避免在多 GPU DeepSpeed 训练中使用 DataParallel，因为它会导致设备冲突
    state_dict = ckp['model']
    new_state_dict = {}
    for k, v in state_dict.items():
        # 移除 'module.' 前缀
        name = k[7:] if k.startswith('module.') else k
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    net.eval()
    
    return net

def kanet_predict(model, input_img_path, output_dir, device='cuda'):
    """
    Perform single image prediction using KANet model.
    
    Args:
        model (nn.Module): Loaded KANet model
        input_img (path): Input image path
        output_dir (str): Directory to save output images
        device (str, optional): Device to run inference on. Defaults to 'cuda'.
    
    Returns:
        torch.Tensor: Predicted output image
    """
    
    img_name, ext = os.path.splitext(os.path.basename(input_img_path))
    # Ensure device is set correctly
    device = device if torch.cuda.is_available() else 'cpu'
    
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Ensure input is on the correct device
    input_img = Image.open(input_img_path).convert('RGB')
    transform = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    input_img = transform(input_img)
    input_img = input_img.to(device)
    input_img = input_img.unsqueeze(0)
    
    # Adjust image dimensions if not divisible by 16
    b, c, h, w = input_img.size()
    
    # Height adjustment
    if h % 16 != 0:
        h0 = h % 16
        if h0 % 2 == 0:
            h1 = int(h0 / 2)
            input_img = input_img[:, :, h1:(h-h1), :]
        else:
            h1 = h0 % 2
            input_img = input_img[:, :, h1:(h+h1-h0), :]
    
    # Width adjustment
    if w % 16 != 0:
        w0 = w % 16
        if w0 % 2 == 0:
            w1 = int(w0 / 2)
            input_img = input_img[:, :, :, w1:(w-w1)]
        else:
            w1 = w0 % 2
            input_img = input_img[:, :, :, w1:(w-w0+w1)]
    
    # Perform inference
    with torch.no_grad():
        predict_y, _ = model(input_img)
    
    save_path = os.path.join(output_dir, img_name+'.png')
    save_image(predict_y[0], save_path)
    
    return save_path



