import argparse
import os
import yaml
import torch
import numpy as np
import torchvision
from . import utils
from .models import DenoisingDiffusion, DiffusiveRestoration
from PIL import Image
import torch.nn.functional as F
def load_img(filepath):
    img = Image.open(filepath).convert('RGB')
    return img

def parse_args_and_config(model_path):
    class Config:
        def __init__(self):
            self.config = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/unsupervised.yml')
            self.mode = 'evaluation'
            self.resume = os.path.join(model_path, 'stage2_weight.pth.tar')
            self.image_folder = os.path.join(model_path, 'results/')

    args = Config()

    with open(os.path.join(model_path, args.config), "r") as f:
        config = yaml.safe_load(f)
    new_config = dict2namespace(config)

    return args, new_config


def dict2namespace(config):
    namespace = argparse.Namespace()
    for key, value in config.items():
        if isinstance(value, dict):
            new_value = dict2namespace(value)
        else:
            new_value = value
        setattr(namespace, key, new_value)
    return namespace


def load_lightdiff_model(model_path, device):

    args, config = parse_args_and_config(model_path)
    config.device = device

    # set up the model
    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)



    return model

def lightdiff_predict(model, input_img, output_dir, device):
    with torch.no_grad():
        os.makedirs(output_dir, exist_ok=True)  # Replaced util.mkdir with os.makedirs

        img_name, ext = os.path.splitext(os.path.basename(input_img))

        img = load_img(input_img)
        transform = torchvision.transforms.ToTensor()
        img_tensor = transform(img).to(device)
        x = img_tensor.unsqueeze(0)
        with torch.no_grad():
            x_cond = x[:, :3, :, :].to(model.diffusion.device)
            b, c, h, w = x_cond.shape
            img_h_64 = int(64 * np.ceil(h / 64.0))
            img_w_64 = int(64 * np.ceil(w / 64.0))
            x_cond = F.pad(x_cond, (0, img_w_64 - w, 0, img_h_64 - h), 'reflect')

            pred_x = model.diffusion.model(torch.cat((x_cond, x_cond),
                                                    dim=1))["pred_x"][:, :, :h, :w]
        # ------------------------------------
        # save results
        # ------------------------------------
        save_path = os.path.join(output_dir, img_name+'.png')
        utils.logging.save_image(pred_x, save_path)
        return save_path