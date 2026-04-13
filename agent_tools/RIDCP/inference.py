import os.path
import torch
from basicsr_ridcp.archs.dehaze_vq_weight_arch import VQWeightDehazeNet
from .utils import utils_image as util
from basicsr_ridcp.utils import img2tensor, tensor2img, imwrite
import cv2
def load_ridcp_model(model_path, device):


    # set up the model
    model = VQWeightDehazeNet(codebook_params=[[64, 1024, 512]], LQ_stage=True, use_weight=True, weight_alpha=-21.25, weight_path=os.path.join(model_path, 'weight_for_matching_dehazing_Flickr.pth'))
    model.load_state_dict(torch.load(os.path.join(model_path, 'pretrained_RIDCP.pth'), weights_only=False)['params'], strict=False)
    model.eval()

    model = model.to(device)
    return model

def ridcp_predict(model, input_img, output_dir, device):
    with torch.no_grad():
        util.mkdir(output_dir)

        img_name, ext = os.path.splitext(os.path.basename(input_img))

        img = cv2.imread(input_img, cv2.IMREAD_UNCHANGED)
        if img.max() > 255.0:
            img = img / 255.0
        if img.shape[-1] > 3:
            img = img[:, :, :3]
        img_tensor = img2tensor(img).to(device) / 255.
        img_tensor = img_tensor.unsqueeze(0)

        max_size = 1500 ** 2 
        h, w = img_tensor.shape[2:]
        if h * w < max_size: 
            output, _ = model.test(img_tensor)
        else:
            down_img = torch.nn.UpsamplingBilinear2d((h//2, w//2))(img_tensor)
            output, _ = model.test(down_img)
            output = torch.nn.UpsamplingBilinear2d((h, w))(output)
        output_img = tensor2img(output)


        # ------------------------------------
        # save results
        # ------------------------------------
        save_path = os.path.join(output_dir, img_name+'.png')
        imwrite(output_img, save_path)
        return save_path