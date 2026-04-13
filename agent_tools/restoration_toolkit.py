import os
import sys
import logging
from contextlib import contextmanager

# Setup logger for this module
logger = logging.getLogger(__name__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))

@contextmanager
def suppress_stdout():
    """Context manager to suppress stdout output (e.g., model initialization prints)."""
    with open(os.devnull, 'w') as devnull:
        old_stdout = sys.stdout
        sys.stdout = devnull
        try:
            yield
        finally:
            sys.stdout = old_stdout

# Force CUDA visibility before any CUDA operations
# Ray workers without GPU allocation may have CUDA_VISIBLE_DEVICES set to empty
if not os.environ.get("CUDA_VISIBLE_DEVICES"):
    # Try to detect available GPUs and use GPU 0 by default
    try:
        import subprocess
        result = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True)
        if result.returncode == 0 and 'GPU' in result.stdout:
            num_gpus = result.stdout.count('GPU')
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))
    except Exception:
        pass

os.environ["BASICSR_JIT"] = "True"
from PIL import Image
import torch
from .SCUNet.inference import load_scu_model, scu_predict
from .Retinexformer.inference import load_retinexformer_model, retinexformer_predict
from .img2img_turbo.inference import load_turbo_model, turbo_predict
from .ESRGAN.inference import load_esrgan_model, esrgan_predict
from .RIDCP.inference import load_ridcp_model, ridcp_predict
from .LightenDiffusion.inference import load_lightdiff_model, lightdiff_predict
from .IDT.inference import load_idt_model, idt_predict
from .iqa_reward import IQAScore
from .SnowMaster.inference import load_snowmaster_model, snowmaster_predict
from .S2Former.inference import load_s2former_model, s2former_predict
from .KANet.inference import load_kanet_model, kanet_predict
from .HVICIDNet.inference import load_hvicidnet_model, hvicidnet_predict

class RestorationToolkit():
    """
    A toolkit for image restoration, providing access to various models and evaluation capabilities.
    
    Supports two modes:
    - preload=True: Load all models at initialization (faster inference, more memory)
    - preload=False (lazy mode): Load models on-demand, unload after use (slower, less memory)
    """
    def __init__(self, models=None, device='cuda', score_weight=None, load_iqa=True, 
                 preload=True, auto_unload=False):
        """
        Initialize the toolkit engine.
        
        Args:
            models (list, optional): A list of models to load. Defaults to all available models.
            device (str, optional): The computation device ('cuda' or 'cpu'). Defaults to 'cuda'.
            score_weight (dict, optional): Weights for IQA score calculation. Defaults to None.
            load_iqa (bool, optional): Whether to load IQA models. Defaults to True.
                Set to False to save GPU memory when IQA is handled separately.
            preload (bool, optional): Whether to preload all models at initialization. Defaults to True.
                Set to False for lazy loading mode (load on demand).
            auto_unload (bool, optional): Whether to automatically unload models after use. Defaults to False.
                Only effective when preload=False. Helps avoid GPU memory conflicts.
        """
        logger.info(f"Initializing RestorationToolkit (preload={preload}, auto_unload={auto_unload})")
        self.all_model_paths = [
            'scunet', 
            'retinexformer_fivek', 'hvicidnet', 'lightdiff',
            'turbo_rain', 'idt', 's2former',
            'ridcp', 'kanet', 
            'turbo_snow', 'snowmaster',
            'real_esrgan',
        ]
        
        if models is not None:
            self.all_model_paths = models
            
        # Model file paths configuration
        self.model_paths = {
            'scunet': os.path.join(os.path.dirname(os.path.abspath(__file__)), '../checkpoints/agent_tools/checkpoints/SCUNet/scunet_color_real_gan.pth'),
            'retinexformer': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/Retinexformer'),
            'real_esrgan': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/ESRGAN/RealESRGAN_x4plus.pth'),
            'ridcp': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/RIDCP'),
            'idt': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/IDT'),
            'img2img_turbo': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/img2img_turbo'),
            'lightdiff': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/LightenDiffusion'),
            'hvicidnet': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/HVICIDNet/generalization.pth'),
            's2former': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/S2Former'),
            'kanet': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/KANet'),
            'snowmaster': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/SnowMaster'),
            'turbo': os.path.join(os.path.dirname(os.path.abspath(__file__)),'../checkpoints/agent_tools/checkpoints/Img2img_turbo')
        }
        
        self.device = device
        self.models = {}
        self.preload = preload
        self.auto_unload = auto_unload
        
        # Only preload models if requested
        if preload:
            self.load_models()
        
        # Optionally load IQA models (can save ~8GB GPU memory)
        self.iqa = None
        if load_iqa:
            self.iqa = IQAScore(self.device, score_weight)
        
        logger.info(f"RestorationToolkit initialized. Loaded models: {list(self.models.keys())}")

    def load_models(self):
        """
        Load all configured models to device.
        """
        for model_name in self.all_model_paths:
            self.load_single_model(model_name)
    
    def load_single_model(self, model_name):
        """
        Load a single model to the device.
        
        Args:
            model_name: Name of the model to load
            
        Returns:
            The loaded model, or None if loading failed
        """
        if model_name in self.models:
            return self.models[model_name]
            
        try:
            # Suppress stdout to avoid model initialization prints (e.g., "Block Initial Type: W")
            with suppress_stdout():
                if model_name == 'scunet':
                    self.models['scunet'] = load_scu_model(self.model_paths[model_name], self.device)
                elif model_name == 'retinexformer_fivek':
                    self.models['retinexformer_fivek'] = load_retinexformer_model(self.model_paths['retinexformer'], self.device)
                elif model_name == 'turbo_rain':
                    self.models['turbo_rain'] = load_turbo_model('rain', self.model_paths['turbo'], self.device)
                elif model_name == 'turbo_snow':
                    self.models['turbo_snow'] = load_turbo_model('snow', self.model_paths['turbo'], self.device)
                elif model_name == 'real_esrgan':
                    self.models['real_esrgan'] = load_esrgan_model(self.model_paths[model_name], self.device)
                elif model_name == 'ridcp':
                    self.models['ridcp'] = load_ridcp_model(self.model_paths[model_name], self.device)
                elif model_name == 'idt':
                    self.models['idt'] = load_idt_model('day', self.model_paths['idt'], self.device)
                elif model_name == 'lightdiff':
                    self.models['lightdiff'] = load_lightdiff_model(self.model_paths[model_name], self.device)
                elif model_name == 'snowmaster':
                    self.models['snowmaster'] = load_snowmaster_model(self.model_paths[model_name], self.device)
                elif model_name == 's2former':
                    self.models['s2former'] = load_s2former_model(self.model_paths[model_name], self.device)
                elif model_name == 'kanet':
                    self.models['kanet'] = load_kanet_model(self.model_paths[model_name], self.device)
                elif model_name == 'hvicidnet':
                    self.models['hvicidnet'] = load_hvicidnet_model(self.model_paths[model_name], self.device)
                else:
                    logger.warning(f"Unknown model: {model_name}")
                    return None
            
            logger.debug(f"Loaded model: {model_name}")
            return self.models.get(model_name)
        except Exception as e:
            logger.error(f"Error loading model {model_name}: {e}")
            return None
    
    def unload_single_model(self, model_name):
        """
        Unload a single model from GPU memory.
        
        Args:
            model_name: Name of the model to unload
        """
        if model_name not in self.models:
            return
            
        try:
            model = self.models.pop(model_name)
            del model
            # Clear CUDA cache
            if 'cuda' in str(self.device):
                torch.cuda.empty_cache()
            logger.debug(f"Unloaded model: {model_name}")
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
    
    def unload_all_models(self):
        """
        Unload all models from GPU memory.
        """
        model_names = list(self.models.keys())
        for model_name in model_names:
            self.unload_single_model(model_name)
        
        # Force garbage collection
        import gc
        gc.collect()
        if 'cuda' in str(self.device):
            torch.cuda.empty_cache()
        logger.debug("All models unloaded")

    def resize_image(self, img_path, output_dir):
        """
        Resize image to 512x512.
        
        Args:
            img_path (str): Path to the input image.
            output_dir (str): Directory to save the resized image.
            
        Returns:
            str: Path to the resized image.
        """
        with Image.open(img_path) as img:
            img = img.convert('RGB')  # Ensure consistent color mode
            img = img.resize((512, 512), Image.LANCZOS)  # Use high-quality resampling
            img_name = os.path.splitext(os.path.basename(img_path))[0]
            save_path = os.path.join(output_dir, f"{img_name}.png")
            img.save(save_path, format='PNG')
        return save_path

    def process_image_with_models(self, model_list, img_path, output_dir):
        """
        Process an image with a specified sequence of models.
        
        Supports dynamic loading/unloading when preload=False:
        - Loads model on-demand if not already loaded
        - Unloads model after use if auto_unload=True
        
        Args:
            model_list (list): A list of model names to use for processing.
            img_path (str): Path to the input image.
            output_dir (str): Directory to save the output images.
            
        Returns:
            str: The absolute path to the final processed image.
        """
        os.makedirs(output_dir, exist_ok=True)
        with torch.no_grad():
            img_path = self.resize_image(img_path, output_dir)
            for model_name in model_list:
                if model_name not in self.all_model_paths:
                    logger.warning(f"Model {model_name} not found in available models")
                    continue
                
                # Dynamic loading: load model if not in memory
                should_unload = False
                if model_name not in self.models:
                    if not self.preload:
                        # Lazy mode: load the model now
                        self.load_single_model(model_name)
                        should_unload = self.auto_unload  # Mark for unload after use
                    else:
                        logger.warning(f"Model {model_name} not loaded and preload=True, skipping")
                        continue
                
                # Process image with the model
                if model_name == 'scunet':
                    img_path = scu_predict(self.models['scunet'], img_path, output_dir, device=self.device)
                elif model_name == 'retinexformer_fivek':
                    img_path = retinexformer_predict(self.models['retinexformer_fivek'], img_path, output_dir, device=self.device)
                elif model_name == 'turbo_rain':
                    img_path = turbo_predict(self.models['turbo_rain'], img_path, output_dir, device=self.device)
                elif model_name == 'turbo_snow':
                    img_path = turbo_predict(self.models['turbo_snow'], img_path, output_dir, device=self.device)
                elif model_name == 'real_esrgan':
                    img_path = esrgan_predict(self.models['real_esrgan'], img_path, output_dir, device=self.device)
                elif model_name == 'ridcp':
                    img_path = ridcp_predict(self.models['ridcp'], img_path, output_dir, device=self.device)
                elif model_name == 'idt':
                    img_path = idt_predict(self.models['idt'], img_path, output_dir, device=self.device)
                elif model_name == 'lightdiff':
                    img_path = lightdiff_predict(self.models['lightdiff'], img_path, output_dir, device=self.device)
                elif model_name == 'snowmaster':
                    img_path = snowmaster_predict(self.models['snowmaster'], img_path, output_dir, device=self.device)
                elif model_name == 's2former':
                    img_path = s2former_predict(self.models['s2former'], img_path, output_dir, device=self.device)
                elif model_name == 'kanet':
                    img_path = kanet_predict(self.models['kanet'], img_path, output_dir, device=self.device)
                elif model_name == 'hvicidnet':
                    img_path = hvicidnet_predict(self.models['hvicidnet'], img_path, output_dir, device=self.device)
                
                # Dynamic unloading: unload model after use if auto_unload=True
                if should_unload:
                    self.unload_single_model(model_name)
                    
        return os.path.abspath(img_path)  # Return the absolute path to the final image
    
    def process_image(self, tools, img_path, output_dir, is_identify=True):
        # 调用处理函数
        tool_dict = {
            "night": ["retinexformer_fivek", "hvicidnet", "lightdiff"],
            "rain_drop": ["idt", "turbo_rain", "s2former"], 
            "rain_drive": ["idt", "turbo_rain", "s2former"], 
            "rain_streak": ["idt", "turbo_rain", "s2former"], 
            "fog":["ridcp", "kanet"], 
            "snow":["turbo_snow", "snowmaster"], 
        }
        if not is_identify:
            for tool_name, tool_list in tool_dict.items():
                if tool_name in img_path:
                    con_tools = list(set(tools) & set(tool_list))
                    if len(con_tools) == 0:
                        logger.warning(f"tool {tools} not in {tool_list}!")
                        return {"output_path": "error!", "score": [-2, -2, -2, -2, -2]}

        output_path = self.process_image_with_models(tools, img_path, output_dir)
        
        # Get IQA score if available, otherwise return None
        score = None
        if self.iqa is not None:
            score = self.iqa.get_iqa_score(output_path)
        
        return {"output_path": output_path, "score": score}