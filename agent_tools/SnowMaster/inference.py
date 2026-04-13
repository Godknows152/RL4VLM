import torch
import torchvision.transforms as tfs
from PIL import Image
import os
from torch.utils.data import Dataset, DataLoader
import sys

from .nafnet import NAFNetLocal

# Add project root to Python path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)


class SnowDataset(Dataset):
    """Simple dataset class for snow images"""
    def __init__(self, input_path, desnow_input_size=None, model_name=None):
        self.input_path = input_path
        self.image_list = [f for f in os.listdir(input_path) if f.endswith(('.png', '.jpg', '.jpeg'))]
        
        self.transform = tfs.Compose([
            tfs.ToTensor()
        ])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        image_name = self.image_list[idx]
        image_path = os.path.join(self.input_path, image_name)
        image = Image.open(image_path).convert('RGB')
        
        # Check and resize image if too large
        max_size = 1500
        if image.size[0] > max_size or image.size[1] > max_size:
            ratio = max_size / max(image.size)
            new_size = tuple(int(dim * ratio) for dim in image.size)
            image = image.resize(new_size, Image.LANCZOS)
        
        snow_input = self.transform(image)
        
        image_id = os.path.splitext(image_name)[0]
        return snow_input, torch.tensor([]), torch.tensor([]), image_id


def load_snowmaster_model(model_path, device='cuda'):
    """
    Load the SnowMaster model
    
    Args:
        model_path (str): Path to model checkpoint
        device (str): Device for inference, default is 'cuda'
    
    Returns:
        torch.nn.Module: Loaded model moved to specified device
    """
    path = os.path.join(model_path, 'checkpoint_0318.pth')
    # Create model architecture
    model = NAFNetLocal(img_channel=3, width=32, middle_blk_num=1, enc_blk_nums=[1, 1, 1, 28], dec_blk_nums=[1, 1, 1, 1])
    
    # Load checkpoint
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    # Get actual model weights
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    elif 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'params' in checkpoint:
        state_dict = checkpoint['params']
    else:
        state_dict = checkpoint

    # Process weight keys, remove module prefix (module.)
    new_state_dict = {}
    for k, v in state_dict.items():
        if k.startswith('module.'):
            k = k[7:]
        new_state_dict[k] = v

    try:
        model.load_state_dict(new_state_dict)
        print("Model loaded successfully")
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        raise

    # Set to evaluation mode and move to specified device
    model.eval()
    return model.to(device)


def snowmaster_predict(model, input_img, output_dir, device='cuda'):
    """
    Use SnowMaster model to perform snow removal on a single image
    
    Args:
        model (torch.nn.Module): Loaded model
        input_img (str or PIL.Image.Image): Input image path or PIL image
        output_dir (str): Output directory
        device (str): Device for inference, default is 'cuda'
    
    Returns:
        str: Path to the saved output image
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # If input is an image path, open the image
    if isinstance(input_img, str):
        img = Image.open(input_img).convert('RGB')
        img_name = os.path.splitext(os.path.basename(input_img))[0]
    else:
        img = input_img
        img_name = "predicted_image"
    
    # Resize image if too large
    max_size = 1500
    if img.size[0] > max_size or img.size[1] > max_size:
        ratio = max_size / max(img.size)
        new_size = tuple(int(dim * ratio) for dim in img.size)
        img = img.resize(new_size, Image.LANCZOS)
    
    # Convert to tensor
    transform = tfs.Compose([tfs.ToTensor()])
    input_tensor = transform(img).unsqueeze(0)
    
    # Inference
    with torch.no_grad():
        # Check image size, use patch-based processing if too large
        if input_tensor.shape[-1] * input_tensor.shape[-2] > 2**24:
            patch_size = 1024
            stride = 1000
            b, c, h, w = input_tensor.shape
            output_tensor = torch.zeros_like(input_tensor)
            for i in range(0, h - patch_size + 1, stride):
                for j in range(0, w - patch_size + 1, stride):
                    patch = input_tensor[:, :, i:i+patch_size, j:j+patch_size]
                    patch_output = model(patch.to(device))
                    output_tensor[:, :, i:i+patch_size, j:j+patch_size] = patch_output.cpu()
        else:
            input_tensor = input_tensor.to(device)
            output_tensor = model(input_tensor)
        
        # Clamp output values to valid range
        output_tensor = output_tensor.clamp_(0, 1)
        
        # Convert to PIL image
        output_image = tfs.ToPILImage()(output_tensor.squeeze().cpu())
    
    # Save image
    save_path = os.path.join(output_dir, f"{img_name}.png")
    output_image.save(save_path)
    
    return save_path


def inference(model_path, input_path, output_path, model_name, device='cuda'):
    """
    Batch inference function:
    Args:
        model_path: Path to model checkpoint (string)
        input_path: Path to input image folder (string)
        output_path: Path to output image save folder (string)
        model_name: Model name
        device: Device for inference
    """
    # Check if model file and input directory exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file does not exist: {model_path}")
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Input path does not exist: {input_path}")
        
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    print(f"\nUsing model: {model_path}")
    print(f"Input directory: {input_path}")
    print(f"Output directory: {output_path}")
    
    # Load current model
    try:
        model = load_snowmaster_model(model_path, device)
    except Exception as e:
        print(f"Error loading model: {str(e)}")
        sys.exit(1)
        
    dataset = SnowDataset(
        input_path=input_path,
        desnow_input_size=None,
        model_name=model_name
    )
    
    dataloader = DataLoader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0
    )
    
    with torch.no_grad():
        for snow_input, _, _, image_id in dataloader:
            # Check image size, use patch-based processing if too large
            if snow_input.shape[-1] * snow_input.shape[-2] > 2**24:
                patch_size = 1024  # Can be adjusted as needed
                stride = 1000      # Can be adjusted as needed
                b, c, h, w = snow_input.shape
                output_tensor = torch.zeros_like(snow_input)
                for i in range(0, h - patch_size + 1, stride):
                    for j in range(0, w - patch_size + 1, stride):
                        patch = snow_input[:, :, i:i+patch_size, j:j+patch_size]
                        patch_output = model(patch.to(device))
                        output_tensor[:, :, i:i+patch_size, j:j+patch_size] = patch_output.cpu()
            else:
                snow_input = snow_input.to(device)
                output_tensor = model(snow_input)
            
            output_tensor = output_tensor.clamp_(0, 1)
            
            output_image = tfs.ToPILImage()(output_tensor.squeeze().cpu())
            save_path = os.path.join(output_path, f"{image_id[0]}.png")
            output_image.save(save_path)
    return save_path


if __name__ == "__main__":
    ckpt_path = "weights/checkpoint_0318.pth"
    input_img = "assests/1.jpg"
    output_path = "output"
    device = "cuda"

    # Load model
    model = load_snowmaster_model(ckpt_path, device)
    
    # Single image inference
    snowmaster_predict(model, input_img, output_path, device)