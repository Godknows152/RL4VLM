import argparse
import os
import yaml
import torch
import numpy as np
from . import utils
import cv2
from .models import DenoisingDiffusion, DiffusiveRestoration
from torchvision.transforms.functional import crop
from PIL import Image  # Add this import
# import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "4"

class Config:
    # def __init__(self, config_path='IDT/configs/daytime_128.yml', resume_path='IDT/ckpt/IDT/epoch100.pth.tar', grid_r=16, sampling_timesteps=25, test_set='IDT', image_folder='results/', seed=61, sid='1'):
    def __init__(self, prefix = '', model_type='day', config_path='configs/daytime_128.yml', resume_path='ckpt/Night/IDT/epoch100.pth.tar', grid_r=16, sampling_timesteps=25, test_set='IDT', image_folder='results/', seed=61, sid='1'):
        if model_type == 'day':
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/daytime_128.yml')
            resume_path = 'epoch100.pth.tar'
        elif model_type == 'night':
            config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'configs/nighttime_128.yml')
            resume_path = 'epoch100.pth.tar'

        config_path = os.path.join(prefix, config_path)
        resume_path = os.path.join(prefix, resume_path)

        self.config_path = config_path
        self.resume = resume_path
        self.grid_r = grid_r
        self.sampling_timesteps = sampling_timesteps
        self.test_set = test_set
        self.image_folder = image_folder
        self.seed = seed
        self.sid = sid

    def load_config(self):
        with open(os.path.join(self.config_path), "r") as f:
            config = yaml.safe_load(f)
        return dict2namespace(config)

def parse_args_and_config(prefix, model_type):
    
    args = Config(prefix=prefix, model_type=model_type)
    new_config = args.load_config()

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

def load_img(filepath):
    return cv2.cvtColor(cv2.imread(filepath), cv2.COLOR_BGR2RGB)

def data_transform(X):
    return 2 * X - 1.0

def inverse_data_transform(X):
    return torch.clamp((X + 1.0) / 2.0, 0.0, 1.0)

def load_idt_model(model_type, model_path, device):
        
    args, config = parse_args_and_config(model_path, model_type)
    config.device = device

    diffusion = DenoisingDiffusion(args, config)
    model = DiffusiveRestoration(diffusion, args, config)
    
    return model

def idt_predict(model, input_img, output_dir, device):
    
    os.makedirs(output_dir, exist_ok=True) 
    img_name, ext = os.path.splitext(os.path.basename(input_img))

    img_L = Image.open(input_img)
    img_L = img_L.resize((512, 512))
    img_L = np.array(img_L)

    if img_L.shape[2] > 3:
        img_L = img_L[:, :, :3]

    img_L = torch.from_numpy(img_L).float().div(255.).permute(2,0,1).unsqueeze(0).to(device)

    with torch.no_grad():

        input_res = 128
        stride = 48
        h_list = [i for i in range(0, img_L.shape[2] - input_res + 1, stride)]
        w_list = [i for i in range(0, img_L.shape[3] - input_res + 1, stride)]
        h_list = h_list + [img_L.shape[2]-input_res]
        w_list = w_list + [img_L.shape[3]-input_res]

        corners = [(i, j) for i in h_list for j in w_list]

        p_size = input_res
        x_grid_mask = torch.zeros_like(img_L).to(device)
        for (hi, wi) in corners:
            x_grid_mask[:, :, hi:hi + p_size, wi:wi + p_size] += 1

        et_output = torch.zeros_like(img_L).to(device)
        manual_batching_size = 256
        x_cond_patch = torch.cat([crop(img_L, hi, wi, p_size, p_size) for (hi, wi) in corners], dim=0)
        
        for i in range(0, len(corners), manual_batching_size):
            
            Output = model.diffusion.model( data_transform(x_cond_patch[i:i+manual_batching_size]).float() )

            for didx, (hi, wi) in enumerate(corners[i:i+manual_batching_size]):
                et_output[0, :, hi:hi + p_size, wi:wi + p_size] += Output[didx]

        x_output = torch.div(et_output, x_grid_mask)

        x_output = inverse_data_transform(x_output)
        save_path = os.path.join(output_dir, img_name+'.png')
        
        utils.logging.save_image(x_output, save_path)
    return save_path