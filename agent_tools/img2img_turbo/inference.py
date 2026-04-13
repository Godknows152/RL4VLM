import os
from PIL import Image
import torch
from torchvision import transforms
from .src.cyclegan_turbo import CycleGAN_Turbo
from .src.my_utils.training_utils import build_transform



def load_turbo_model(name, model_path, device):
    if name == 'rain':
        model = CycleGAN_Turbo(pretrained_path=os.path.join(model_path, 'rainy2day.pkl'), device=device)
    elif name == 'snow':
        model = CycleGAN_Turbo(pretrained_path=os.path.join(model_path, 'snow2day.pkl'), device=device)
    model.direction = 'b2a'
    model.caption = 'driving in the day'
    model.eval()
    model.unet.enable_xformers_memory_efficient_attention()
    return model



def turbo_predict(model, input_image, output_dir, device):
    bname = os.path.splitext(
            os.path.split(input_image)[-1])[0] + '.png'
    
    T_val = build_transform('resize_512x512')
    input_image = Image.open(input_image).convert('RGB')
    # translate the image
    with torch.no_grad():
        input_img = T_val(input_image)
        x_t = transforms.ToTensor()(input_img)
        x_t = transforms.Normalize([0.5], [0.5])(x_t).unsqueeze(0).to(device)
        output = model(x_t)

    output_pil = transforms.ToPILImage()(output[0].cpu() * 0.5 + 0.5)
    output_pil = output_pil.resize((input_image.width, input_image.height), Image.LANCZOS)
    # output_pil = output_pil

    # save the output image
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, bname)
    output_pil.save(save_path)
    return save_path
