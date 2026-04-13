import os.path
import torch
from .utils import utils_image as util
from .models.network_scunet import SCUNet as net

def load_scu_model(model_path, device):
    n_channels = 3

    model = net(in_nc=n_channels,config=[4,4,4,4,4,4,4],dim=64)
    model.load_state_dict(torch.load(model_path, weights_only=False), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)
    return model

def scu_predict(model, input_img, output_dir, device):
    with torch.no_grad():
        n_channels = 3
        util.mkdir(output_dir)

        img_name, ext = os.path.splitext(os.path.basename(input_img))

        img_L = util.imread_uint(input_img, n_channels=n_channels)

        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)
        img_E = model(img_L)
        img_E = util.tensor2uint(img_E)

        # ------------------------------------
        # save results
        # ------------------------------------
        save_path = os.path.join(output_dir, img_name+'.png')
        util.imsave(img_E, save_path)
        return save_path

def scu_predict_tensor(model, input_img, output_dir, device):
    with torch.no_grad():
        n_channels = 3
        util.mkdir(output_dir)

        img_name, ext = os.path.splitext(os.path.basename(input_img))

        img_L = util.imread_uint(input_img, n_channels=n_channels)

        img_L = util.uint2tensor4(img_L)
        img_L = img_L.to(device)
        img_E = model(img_L)
        return img_E