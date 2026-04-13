import cv2
import os
from basicsr.archs.rrdbnet_arch import RRDBNet

from .realesrgan import RealESRGANer


def load_esrgan_model(model_path, device):
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    netscale = 4

    
    # use dni to control the denoise strength
    dni_weight = None


    # restorer
    upsampler = RealESRGANer(
        scale=netscale,
        model_path=model_path,
        dni_weight=dni_weight,
        model=model,
        tile=0,
        tile_pad=10,
        pre_pad=0,
        half=True,
        device=device)
    return upsampler

def esrgan_predict(upsampler, input_image, output_dir, device,):
    
    # determine models according to model names
    outscale = 4 # the final upsampling scale
    imgname, extension = os.path.splitext(os.path.basename(input_image))

    img = cv2.imread(input_image, cv2.IMREAD_COLOR)
    h, w, _ = img.shape
    try:
        output, _ = upsampler.enhance(img, outscale=outscale)
    except RuntimeError as error:
        print('Error', error)
        print('If you encounter CUDA out of memory, try to set --tile with a smaller number.')
    # resize back to the original resolution
    output = cv2.resize(output, (w, h), interpolation=cv2.INTER_CUBIC)
    
    save_path = os.path.join(output_dir, f'{imgname}.png')
    cv2.imwrite(save_path, output)

    return save_path