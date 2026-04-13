import os
import numpy as np
import torch
from PIL import Image
from .net.CIDNet import CIDNet
import torchvision.transforms as transforms
import torch.nn.functional as F
import platform
import argparse

def load_hvicidnet_model(path, device):
    model = CIDNet().to(device)
    model.load_state_dict(torch.load(path, map_location=lambda storage, loc: storage, weights_only=False))
    model.eval()
    return model

def hvicidnet_predict(model, input_img, output_dir, device):
    # Load image if path is provided as string
    if isinstance(input_img, str):
        img_name = os.path.basename(input_img).split('.')[0]
        input_img = Image.open(input_img)
    else:
        img_name = "output" # default name if input is not a path

    torch.set_grad_enabled(False)
    pil2tensor = transforms.Compose([transforms.ToTensor()])
    input = pil2tensor(input_img)
    factor = 8
    h, w = input.shape[1], input.shape[2]
    H, W = ((h + factor) // factor) * factor, ((w + factor) // factor) * factor
    padh = H - h if h % factor != 0 else 0
    padw = W - w if w % factor != 0 else 0
    input = F.pad(input.unsqueeze(0), (0,padw,0,padh), 'reflect')
    gamma = 1
    alpha_s = 1.0
    alpha_i = 1.0
    with torch.no_grad():
        model.trans.alpha_s = alpha_s
        model.trans.alpha = alpha_i
        output = model(input.to(device)**gamma)

    output = torch.clamp(output, 0, 1).cpu()
    output = output[:, :, :h, :w]
    enhanced_img = transforms.ToPILImage()(output.squeeze(0))
    if isinstance(input_img, str):
        original_img = Image.open(input_img)
        enhanced_img = enhanced_img.resize(original_img.size, Image.LANCZOS)

    # Save the output
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, img_name+'.png')
    enhanced_img.save(save_path)
    return save_path


if __name__ == '__main__':
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_path = "checkpoints/HVICIDNet/generalization.pth"
    img_path = "./Test_Input/108.png"
    output_folder = "./output"
    model = load_hvicidnet_model(model_path, device)
    save_path = hvicidnet_predict(model, img_path, output_folder, device)
    print(f"processed image saved to: {save_path}")
    

