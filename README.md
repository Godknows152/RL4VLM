# Image Restoration Expert Toolkit

JarvisIR integrates diverse expert tools for image restoration, targeting problems like low-light conditions, rain/snow, haze, resolution enhancement, and noise. We've optimized our toolkit for both efficiency and performance, with some variations from the original paper.

## Key Features

- Unified framework for multiple expert models.
- Simple API for testing each expert model.
- Quality assessment capabilities through reward functions
- Modular design for easy extension with new models


## Expert Model List
| Task | Tools | Model Description |
|---------|---------|------|
| **Super-resolution & Deblurring & Artifact removal** | Real-ESRGAN | Fast GAN for super-resolution, deblurring, and artifact removal. |
| **Denoising** | SCUNet | Hybrid UNet-based model combining convolution and transformer blocks, designed for robust denoising. |
| **Deraining** | UDR-S2Former | An uncertainty-aware transformer model for rain streak removal. |
| | Img2img-turbo-rain | Efficient model based on SD-turbo, designed for fast and effective rain removal in real-world images. |
| **Raindrop removal** | IDT | Transformer-based model for de-raining and raindrop removal. |
| **Dehazing** | RIDCP | Efficient dehazing model utilizing high-quality codebook priors |
| | KANet | Efficient dehazing network using a localization-and-removal pipeline. |
| **Desnowing** | Img2img-turbo-snow | Efficient model for removing snow artifacts while preserving natural scene details. |
| | Snowmaster | Real-world Image Desnowing via MLLM with Multi-Model Feedback Optimization |
| **Low-light enhancement** | Retinexformer | One-stage Retinex-based Transformer for Low-light Image Enhancement |
| | HVICIDNet | Lightweight transformer for low-light and exposure correction |
| | LightenDiff | Diffusion-based framework for low-light enhancement |

## Usage

### Basic Usage

```python
from agent_tools import RestorationToolkit

# Initialize the toolkit
toolkit = RestorationToolkit(device='cuda')

# Process an image with a sequence of models
result = toolkit.process_image(
    tools=['scunet', 'real_esrgan'],  # Models to apply in sequence
    img_path='path/to/input.jpg',
    output_dir='path/to/output'
)

# Access the result
output_path = result['output_path']
quality_score = result['score']
```

### Running the Test API Server

```python
from agent_tools import start_server

# Start the server with default models
start_server(host='0.0.0.0', port=5010)
```

Or use the API with curl:

```bash
curl -X POST http://localhost:5010/process_image \
  -H "Content-Type: application/json" \
  -d '{"img_path": "path/to/image.jpg", "models": ["scunet", "real_esrgan"]}'
```

## Cite

```
# Real-esrgan
@inproceedings{wang2021real,
  title={Real-esrgan: Training real-world blind super-resolution with pure synthetic data},
  author={Wang, Xintao and Xie, Liangbin and Dong, Chao and Shan, Ying},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={1905--1914},
  year={2021}
}
# img2img-turbo
@article{parmar2024one,
  title={One-step image translation with text-to-image models},
  author={Parmar, Gaurav and Park, Taesung and Narasimhan, Srinivasa and Zhu, Jun-Yan},
  journal={arXiv preprint arXiv:2403.12036},
  year={2024}
}
# Lightendiffusion
@article{jiang2024lightendiffusion,
  title={Lightendiffusion: Unsupervised low-light image enhancement with latent-retinex diffusion models},
  author={Jiang, Hai and Luo, Ao and Liu, Xiaohong and Han, Songchen and Liu, Shuaicheng},
  journal={arXiv preprint arXiv:2407.08939},
  year={2024}
}
# KANet
@article{feng2024advancing,
  title={Advancing real-world image dehazing: perspective, modules, and training},
  author={Feng, Yuxin and Ma, Long and Meng, Xiaozhe and Zhou, Fan and Liu, Risheng and Su, Zhuo},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2024},
  publisher={IEEE}
}
# IDT
@article{xiao2022image,
  title={Image de-raining transformer},
  author={Xiao, Jie and Fu, Xueyang and Liu, Aiping and Wu, Feng and Zha, Zheng-Jun},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  volume={45},
  number={11},
  pages={12978--12995},
  year={2022},
  publisher={IEEE}
}
# SCUNet
@article{zhang2023practical,
   author = {Zhang, Kai and Li, Yawei and Liang, Jingyun and Cao, Jiezhang and Zhang, Yulun and Tang, Hao and Fan, Deng-Ping and Timofte, Radu and Gool, Luc Van},
   title = {Practical Blind Image Denoising via Swin-Conv-UNet and Data Synthesis},
   journal = {Machine Intelligence Research},
   DOI = {10.1007/s11633-023-1466-0},
   url = {https://doi.org/10.1007/s11633-023-1466-0},
   volume={20},
   number={6},
   pages={822--836},
   year={2023},
   publisher={Springer}
}
# S2Former
@inproceedings{chen2023sparse,
  title={Sparse sampling transformer with uncertainty-driven ranking for unified removal of raindrops and rain streaks},
  author={Chen, Sixiang and Ye, Tian and Bai, Jinbin and Chen, Erkang and Shi, Jun and Zhu, Lei},
  booktitle={Proceedings of the IEEE/CVF international conference on computer vision},
  pages={13106--13117},
  year={2023}
}
# RIDCP
@inproceedings{wu2023ridcp,
  title={Ridcp: Revitalizing real image dehazing via high-quality codebook priors},
  author={Wu, Rui-Qi and Duan, Zheng-Peng and Guo, Chun-Le and Chai, Zhi and Li, Chongyi},
  booktitle={Proceedings of the IEEE/CVF conference on computer vision and pattern recognition},
  pages={22282--22291},
  year={2023}
}
# HVI-CIDNet 
@article{yan2024you,
  title={You only need one color space: An efficient network for low-light image enhancement},
  author={Yan, Qingsen and Feng, Yixu and Zhang, Cheng and Wang, Pei and Wu, Peng and Dong, Wei and Sun, Jinqiu and Zhang, Yanning},
  journal={arXiv preprint arXiv:2402.05809},
  year={2024}
}

# Retinexformer
@inproceedings{retinexformer,
  title={Retinexformer: One-stage Retinex-based Transformer for Low-light Image Enhancement},
  author={Yuanhao Cai and Hao Bian and Jing Lin and Haoqian Wang and Radu Timofte and Yulun Zhang},
  booktitle={ICCV},
  year={2023}
}