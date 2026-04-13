import torch
from torchvision.utils import save_image
import torchvision.transforms as tfs
from PIL import Image
import yaml
import os
from tqdm import tqdm
from .UDR_S2Former import Transformer


def load_s2former_model(model_path, device):
    """
    Load S2Former model weights
    
    Args:
        model_path (str): Path to model weights
        device (torch.device): Device to load model on
    
    Returns:
        torch.nn.Module: Loaded S2Former model
    """
    path = os.path.join(model_path, 'udrs2former_demo.pth')
    # Assuming the model input size is fixed at 320x320
    model = Transformer((320, 320)).to(device)
    model.load_state_dict(torch.load(path, map_location=device, weights_only=False))
    model.eval()  # Set model to evaluation mode
    return model

def s2former_predict(model, input_img, output_dir, device):
    """
    Perform S2Former prediction on a single image
    
    Args:
        model (torch.nn.Module): Loaded S2Former model
        input_img (str or PIL.Image.Image): Input image path or PIL Image
        output_dir (str): Directory to save output image
        device (torch.device): Device to run inference on
    
    Returns:
        torch.Tensor: Processed output image
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Get image name for saving
    img_name, ext = os.path.splitext(os.path.basename(input_img))
    
    # Load image if path is provided
    if isinstance(input_img, str):
        input_img = Image.open(input_img).convert('RGB')
    
    # Convert to tensor and add batch dimension
    img_tensor = tfs.ToTensor()(input_img).unsqueeze(0).to(device)
    
    # Get image dimensions
    b, c, h, w = img_tensor.shape
    
    # Inference parameters
    tile = min(320, h, w)
    tile_overlap = 64
    sf = 1  # scale factor
    
    # Prepare output tensors
    stride = tile - tile_overlap
    h_idx_list = list(range(0, h-tile, stride)) + [h-tile]
    w_idx_list = list(range(0, w-tile, stride)) + [w-tile]
    
    E1 = torch.zeros(b, c, h*sf, w*sf).type_as(img_tensor)
    W1 = torch.zeros_like(E1)
    
    # Tile-based inference
    with torch.no_grad():
        for h_idx in h_idx_list:
            for w_idx in w_idx_list:
                in_patch = img_tensor[..., h_idx:h_idx+tile, w_idx:w_idx+tile]
                out_patch1, _ = model(in_patch)
                out_patch1 = out_patch1[0]
                out_patch_mask1 = torch.ones_like(out_patch1)
                
                E1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch1)
                W1[..., h_idx*sf:(h_idx+tile)*sf, w_idx*sf:(w_idx+tile)*sf].add_(out_patch_mask1)
        
        # Normalize output
        output = E1.div_(W1)
    
    # Save output image
    save_path = os.path.join(output_dir, f'{img_name}.png')
    save_image(output, save_path, normalize=False)
    
    return save_path

# Example usage
if __name__ == "__main__":
    # Set up device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load model
    model_path = 'weights/udrs2former_demo.pth'
    model = load_s2former_model(model_path, device)
    
    # Predict on an image
    input_img = 'assests/1.jpg'
    output_dir = 'newoutput'
    s2former_predict(model, input_img, output_dir, device)