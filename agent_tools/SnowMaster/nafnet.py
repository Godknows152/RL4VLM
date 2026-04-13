# ------------------------------------------------------------------------
# Copyright (c) 2022 megvii-model. All Rights Reserved.
# ------------------------------------------------------------------------
# Source: https://github.com/megvii-research/NAFNet

import torch
import torch.nn as nn
import torch.nn.functional as F
from .nafnet_utils import Local_Base, LayerNorm2d


class SimpleGate(nn.Module):
    """
    A simple gating mechanism that splits the input tensor into two halves
    along the channel dimension and performs element-wise multiplication.
    """
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2

class NAFBlock(nn.Module):
    """
    NAFNet Block. This block is the main building component of NAFNet.
    It consists of a main branch with LayerNorm, Depth-wise convolution,
    Simplified Channel Attention (SCA), and a skip connection. It also
    has a Feed-Forward Network (FFN) branch.

    Args:
        c (int): Number of input and output channels.
        DW_Expand (int): Expansion factor for the depth-wise convolution.
        FFN_Expand (int): Expansion factor for the FFN.
        drop_out_rate (float): Dropout rate.
    """
    def __init__(self, c, DW_Expand=2, FFN_Expand=2, drop_out_rate=0.):
        super().__init__()
        dw_channel = c * DW_Expand
        # Main Branch: Depth-wise convolution and Simplified Channel Attention
        self.conv1 = nn.Conv2d(in_channels=c, out_channels=dw_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv2 = nn.Conv2d(in_channels=dw_channel, out_channels=dw_channel, kernel_size=3, padding=1, stride=1, groups=dw_channel,
                               bias=True)
        self.conv3 = nn.Conv2d(in_channels=dw_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        
        # Simplified Channel Attention (SCA)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(in_channels=dw_channel // 2, out_channels=dw_channel // 2, kernel_size=1, padding=0, stride=1,
                      groups=1, bias=True),
        )

        # SimpleGating
        self.sg = SimpleGate()

        # FFN Branch
        ffn_channel = FFN_Expand * c
        self.conv4 = nn.Conv2d(in_channels=c, out_channels=ffn_channel, kernel_size=1, padding=0, stride=1, groups=1, bias=True)
        self.conv5 = nn.Conv2d(in_channels=ffn_channel // 2, out_channels=c, kernel_size=1, padding=0, stride=1, groups=1, bias=True)

        # Layer Normalization
        self.norm1 = LayerNorm2d(c)
        self.norm2 = LayerNorm2d(c)

        # Dropout
        self.dropout1 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(drop_out_rate) if drop_out_rate > 0. else nn.Identity()

        # Learnable parameters for scaling skip connections
        self.beta = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros((1, c, 1, 1)), requires_grad=True)

    def forward(self, inp):
        x = inp

        # Main Branch
        x = self.norm1(x)
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.sg(x)      # Gating
        x = x * self.sca(x) # Apply Simplified Channel Attention
        x = self.conv3(x)
        x = self.dropout1(x)
        
        # First skip connection
        y = inp + x * self.beta

        # FFN Branch
        x = self.conv4(self.norm2(y))
        x = self.sg(x)      # Gating
        x = self.conv5(x)
        x = self.dropout2(x)

        # Second skip connection
        return y + x * self.gamma


class NAFNet(nn.Module):
    """
    NAFNet (Non-linear Activation Free Network) for image restoration.
    This model uses a U-Net like architecture with NAFBlocks.

    Args:
        img_channel (int): Number of input image channels.
        width (int): Base width of the network.
        middle_blk_num (int): Number of NAFBlocks in the bottleneck.
        enc_blk_nums (list[int]): Number of NAFBlocks in each encoder stage.
        dec_blk_nums (list[int]): Number of NAFBlocks in each decoder stage.
    """
    def __init__(self, img_channel=3, width=16, middle_blk_num=1, enc_blk_nums=[], dec_blk_nums=[]):
        super().__init__()

        # Initial and final convolutions
        self.intro = nn.Conv2d(in_channels=img_channel, out_channels=width, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)
        self.ending = nn.Conv2d(in_channels=width, out_channels=img_channel, kernel_size=3, padding=1, stride=1, groups=1,
                              bias=True)

        # U-Net components
        self.encoders = nn.ModuleList()
        self.decoders = nn.ModuleList()
        self.middle_blks = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()

        chan = width
        # Encoder stages
        for num in enc_blk_nums:
            self.encoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )
            self.downs.append(
                nn.Conv2d(chan, 2*chan, 2, 2)
            )
            chan = chan * 2

        # Bottleneck
        self.middle_blks = \
            nn.Sequential(
                *[NAFBlock(chan) for _ in range(middle_blk_num)]
            )

        # Decoder stages
        for num in dec_blk_nums:
            self.ups.append(
                nn.Sequential(
                    nn.Conv2d(chan, chan * 2, 1, bias=False),
                    nn.PixelShuffle(2)
                )
            )
            chan = chan // 2
            self.decoders.append(
                nn.Sequential(
                    *[NAFBlock(chan) for _ in range(num)]
                )
            )

        self.padder_size = 2 ** len(self.encoders)

    def forward(self, inp):
        B, C, H, W = inp.shape
        # Pad input to be divisible by the downsampling factor
        inp = self.check_image_size(inp)

        # Initial convolution
        x = self.intro(inp)

        # Encoder path
        encs = []
        for encoder, down in zip(self.encoders, self.downs):
            x = encoder(x)
            encs.append(x)
            x = down(x)

        # Bottleneck
        x = self.middle_blks(x)

        # Decoder path with skip connections
        for decoder, up, enc_skip in zip(self.decoders, self.ups, encs[::-1]):
            x = up(x)
            x = x + enc_skip # Skip connection
            x = decoder(x)

        # Final convolution and residual connection
        x = self.ending(x)
        x = x + inp

        # Crop to original size
        return x[:, :, :H, :W]

    def check_image_size(self, x):
        """
        Pads the input image so its height and width are divisible by the
        total downsampling factor.
        """
        _, _, h, w = x.size()
        mod_pad_h = (self.padder_size - h % self.padder_size) % self.padder_size
        mod_pad_w = (self.padder_size - w % self.padder_size) % self.padder_size
        # Pad the right and bottom sides of the image
        x = F.pad(x, (0, mod_pad_w, 0, mod_pad_h))
        return x

class NAFNetLocal(Local_Base, NAFNet):
    """
    A local version of NAFNet that adapts the model for inference on
    arbitrary-sized images by replacing pooling layers. This is useful
    when the training and testing resolutions differ.

    Args:
        train_size (tuple): The size of the training images (N, C, H, W).
        fast_imp (bool): Whether to use a faster, non-equivalent implementation
                         for the replaced pooling layers.
    """
    def __init__(self, *args, train_size=(1, 3, 256, 256), fast_imp=False, **kwargs):
        Local_Base.__init__(self)
        NAFNet.__init__(self, *args, **kwargs)

        N, C, H, W = train_size
        # Base size for adaptive pooling layers, typically larger than train size
        base_size = (int(H * 1.5), int(W * 1.5))

        self.eval()
        with torch.no_grad():
            # Convert the model to use custom pooling layers
            self.convert(base_size=base_size, train_size=train_size, fast_imp=fast_imp)


def create_nafnet(input_channels = 3, width = 32, enc_blks = [2, 2, 4, 8], middle_blk_num = 12, dec_blks = [2, 2, 2, 2]):
    """
    Creates a NAFNet model with specific architecture parameters.
    Default parameters are based on the NAFNet-width32 configuration for SIDD.
    Reference: https://github.com/megvii-research/NAFNet/blob/main/options/test/SIDD/NAFNet-width32.yml

    Args:
        input_channels (int): Number of input image channels.
        width (int): Base width of the network.
        enc_blks (list[int]): Number of NAFBlocks in each encoder stage.
        middle_blk_num (int): Number of NAFBlocks in the bottleneck.
        dec_blks (list[int]): Number of NAFBlocks in each decoder stage.

    Returns:
        nn.Module: The created NAFNet model.
    """
    
    net = NAFNet(img_channel=input_channels, width=width, middle_blk_num=middle_blk_num,
                      enc_blk_nums=enc_blks, dec_blk_nums=dec_blks)
    
    return net