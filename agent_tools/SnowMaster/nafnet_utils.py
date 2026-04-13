# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Source: https://github.com/megvii-research/NAFNet

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LayerNormFunction(torch.autograd.Function):
    """Custom autograd function for Layer Normalization for 2D data."""

    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        # Calculate mean and variance along the channel dimension
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        # Normalize the input
        y = (x - mu) / (var + eps).sqrt()
        # Save tensors for backward pass
        ctx.save_for_backward(y, var, weight)
        # Apply scale and shift
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps
        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        
        # Chain rule for layer norm backward pass
        # Gradient w.r.t. scaled output
        g = grad_output * weight.view(1, C, 1, 1)
        
        # Intermediate gradients
        mean_g = g.mean(dim=1, keepdim=True)
        mean_gy = (g * y).mean(dim=1, keepdim=True)
        
        # Gradient w.r.t. input x
        gx = (1. / torch.sqrt(var + eps)) * (g - y * mean_gy - mean_g)
        
        # Gradient w.r.t. weight (sum over all dimensions except channel)
        grad_weight = (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0)
        
        # Gradient w.r.t. bias (sum over all dimensions except channel)
        grad_bias = grad_output.sum(dim=3).sum(dim=2).sum(dim=0)
        
        return gx, grad_weight, grad_bias, None

class LayerNorm2d(nn.Module):
    """
    Layer Normalization for 2D data (e.g., images).

    Applies Layer Normalization over a mini-batch of 2D inputs. The mean and
    standard-deviation are calculated over the channel dimension.
    """
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    
class AvgPool2d(nn.Module):
    """
    Custom 2D Average Pooling that adapts to different input resolutions.

    This module replaces standard AdaptiveAvgPool2d to allow the model to handle
    arbitrary input sizes during inference. It computes the pooling kernel size
    dynamically based on the ratio of the current input size to a base training size.
    It uses the integral image (summed-area table) method for efficient computation.
    """
    def __init__(self, kernel_size=None, base_size=None, auto_pad=True, fast_imp=False, train_size=None):
        super().__init__()
        self.kernel_size = kernel_size
        self.base_size = base_size
        self.auto_pad = auto_pad
        self.fast_imp = fast_imp
        self.train_size = train_size
        
        # Factors for fast implementation, used for downsampling before integral image calculation.
        self.rs = [5, 4, 3, 2, 1]
        self.max_r1 = self.rs[0]
        self.max_r2 = self.rs[0]

    def extra_repr(self) -> str:
        return 'kernel_size={}, base_size={}, stride={}, fast_imp={}'.format(
            self.kernel_size, self.base_size, self.kernel_size, self.fast_imp
        )

    def forward(self, x):
        # Dynamically calculate kernel size if not specified
        if self.kernel_size is None and self.base_size:
            if isinstance(self.base_size, int):
                self.base_size = (self.base_size, self.base_size)
            
            # Scale kernel size based on the ratio of current input size to training size
            self.kernel_size = list(self.base_size)
            self.kernel_size[0] = x.shape[2] * self.base_size[0] // self.train_size[-2]
            self.kernel_size[1] = x.shape[3] * self.base_size[1] // self.train_size[-1]

            # Update max reduction factors for fast implementation
            self.max_r1 = max(1, self.rs[0] * x.shape[2] // self.train_size[-2])
            self.max_r2 = max(1, self.rs[0] * x.shape[3] // self.train_size[-1])

        # If kernel size is larger than or equal to input size, use adaptive average pooling
        if self.kernel_size[0] >= x.size(-2) and self.kernel_size[1] >= x.size(-1):
            return F.adaptive_avg_pool2d(x, 1)

        if self.fast_imp:  # A faster but non-equivalent implementation
            h, w = x.shape[2:]
            if self.kernel_size[0] >= h and self.kernel_size[1] >= w:
                out = F.adaptive_avg_pool2d(x, 1)
            else:
                # Find a suitable downsampling ratio
                r1 = next((r for r in self.rs if h % r == 0), 1)
                r2 = next((r for r in self.rs if w % r == 0), 1)
                # Constrain the reduction ratio
                r1 = min(self.max_r1, r1)
                r2 = min(self.max_r2, r2)
                
                # Downsample and compute integral image
                s = x[:, :, ::r1, ::r2].cumsum(dim=-1).cumsum(dim=-2)
                n, c, h_s, w_s = s.shape
                k1, k2 = min(h_s - 1, self.kernel_size[0] // r1), min(w_s - 1, self.kernel_size[1] // r2)
                
                # Compute average using the integral image
                out = (s[:, :, :-k1, :-k2] - s[:, :, :-k1, k2:] - s[:, :, k1:, :-k2] + s[:, :, k1:, k2:]) / (k1 * k2)
                # Upsample back to original resolution
                out = torch.nn.functional.interpolate(out, scale_factor=(r1, r2))
        else: # Exact implementation using integral image
            # Compute integral image (summed-area table)
            s = x.cumsum(dim=-1).cumsum_(dim=-2)
            s = torch.nn.functional.pad(s, (1, 0, 1, 0))  # Pad for easier boundary handling
            
            k1, k2 = min(x.shape[-2], self.kernel_size[0]), min(x.shape[-1], self.kernel_size[1])
            
            # Get the four corners of the sum rectangle from the integral image
            s1 = s[:, :, :-k1, :-k2]
            s2 = s[:, :, :-k1, k2:]
            s3 = s[:, :, k1:, :-k2]
            s4 = s[:, :, k1:, k2:]
            
            # Calculate the sum over the kernel window and then the average
            out = s4 + s1 - s2 - s3
            out = out / (k1 * k2)

        # Pad the output to match the input spatial dimensions
        if self.auto_pad:
            n, c, h, w = x.shape
            _h, _w = out.shape[2:]
            pad2d = ((w - _w) // 2, (w - _w + 1) // 2, (h - _h) // 2, (h - _h + 1) // 2)
            out = torch.nn.functional.pad(out, pad2d, mode='replicate')

        return out

def replace_layers(model, base_size, train_size, fast_imp, **kwargs):
    """
    Recursively traverses a model and replaces all `nn.AdaptiveAvgPool2d`
    layers with the custom `AvgPool2d` layer.
    """
    for n, m in model.named_children():
        if len(list(m.children())) > 0:
            # If the module has children, recurse into it
            replace_layers(m, base_size, train_size, fast_imp, **kwargs)

        if isinstance(m, nn.AdaptiveAvgPool2d):
            # Create and set the new pooling layer
            pool = AvgPool2d(base_size=base_size, fast_imp=fast_imp, train_size=train_size)
            assert m.output_size == 1, "Only AdaptiveAvgPool2d with output_size=1 is supported"
            setattr(model, n, pool)


class Local_Base():
    """
    A base class (mixin) to make a neural network model resolution-agnostic.
    It achieves this by replacing adaptive pooling layers with a custom version
    that can handle variable input sizes.
    """
    def convert(self, *args, train_size, **kwargs):
        """
        Converts the model to be resolution-agnostic.

        This method replaces standard adaptive pooling layers and performs a
        dry run with a dummy tensor to initialize dynamic parameters within
        the new layers.

        Args:
            train_size (tuple): The input size (N, C, H, W) used during training.
            *args, **kwargs: Arguments passed to `replace_layers`.
        """
        replace_layers(self, *args, train_size=train_size, **kwargs)
        # Create a dummy input tensor with the training size
        imgs = torch.rand(train_size)
        # Perform a forward pass to initialize the dynamic kernel sizes in AvgPool2d
        with torch.no_grad():
            self.forward(imgs)