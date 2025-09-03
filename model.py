import torch
import torch.nn as nn
import torch.nn.functional as F

class DeformConvBlock(nn.Module):
    """Deformable Convolution Block with R1=2 repetitions"""
    def __init__(self, in_channels, out_channels):
        super(DeformConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x

class DepthConvBlock(nn.Module):
    """Depth-wise Convolution Block"""
    def __init__(self, in_channels, out_channels):
        super(DepthConvBlock, self).__init__()
        self.depth_conv = nn.Conv2d(in_channels, in_channels, kernel_size=3, stride=1, padding=1, groups=in_channels)
        self.point_conv = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        x = self.depth_conv(x)
        x = self.point_conv(x)
        x = self.activation(x)
        return x

class ConvBlock(nn.Module):
    """Standard Convolution Block with R2=2 repetitions"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.Tanh()
        
    def forward(self, x):
        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return x

class DDSRNet(nn.Module):
    """
    DDSRNet: A Deep Model for Denoising and Super-Resolution
    Two-stage learning architecture for simultaneous image denoising and super-resolution
    """
    def __init__(self, channel_in=3, channel_out=3, factor=4, R1=2, R2=2):
        super(DDSRNet, self).__init__()
        self.factor = factor
        self.R1 = R1
        self.R2 = R2
        
        # Stage 1: Initial feature extraction
        self.conv1 = nn.Conv2d(channel_in, 64, kernel_size=3, stride=1, padding=1)
        
        # DeformConv Blocks (R1=2 repetitions)
        self.deform_conv_blocks = nn.ModuleList([
            DeformConvBlock(64 if i == 0 else 128, 128) 
            for i in range(R1)
        ])
        
        # DepthConv Block
        self.depth_conv = DepthConvBlock(128, 128)
        
        # Downscale using PixelUnshuffle
        self.downscale = nn.PixelUnshuffle(factor)
        downscale_channels = 128 * (factor * factor)  # 128 * 16 = 2048 for factor=4
        
        # ConvBlocks (R2=2 repetitions) at downscaled resolution
        # Need to output 128 * factor * factor = 128 * 16 = 2048 channels for the last ConvBlock
        # so that after PixelShuffle we get 128 channels
        self.conv_blocks = nn.ModuleList([
            ConvBlock(downscale_channels if i == 0 else 128, 128 if i < R2-1 else 128 * factor * factor)
            for i in range(R2)
        ])
        
        # Upscale using PixelShuffle
        self.upscale = nn.PixelShuffle(factor)
        
        # Denoising path (1x1 kernel for final output)
        self.conv_denoised = nn.Conv2d(128, channel_out, kernel_size=1, stride=1, padding=0)
        
        # Super-resolution path concatenation and processing
        # Concatenate: Out_cat0 (64), Out_cat1 (128), Out_cat2 (128), img_denoised (channel_out)
        concat_channels = 64 + 128 + 128 + channel_out  # 64 + 128 + 128 + 1 = 321 for grayscale
        self.conv_super_intermediate = nn.Conv2d(concat_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv_super = nn.Conv2d(64, channel_out, kernel_size=1, stride=1, padding=0)
        
        # Upscaling for super-resolution output
        self.upscale_sr = nn.Upsample(scale_factor=factor, mode='bilinear', align_corners=False)
        
        # Activation functions
        self.tanh = nn.Tanh()
        self.sigmoid = nn.Sigmoid()

    def forward(self, input_img):
        # Ensure proper input shape
        if len(input_img.shape) == 3:
            input_img = input_img.unsqueeze(0)
        
        # Check if input dimensions are divisible by factor, pad if necessary
        B, C, H, W = input_img.shape
        pad_h = (self.factor - H % self.factor) % self.factor
        pad_w = (self.factor - W % self.factor) % self.factor
        
        if pad_h > 0 or pad_w > 0:
            # Pad to make dimensions divisible by factor
            input_img = F.pad(input_img, (0, pad_w, 0, pad_h), mode='reflect')
            B, C, H, W = input_img.shape
        
        # Stage 1: Initial feature extraction
        Out_cat0 = self.tanh(self.conv1(input_img))  # [B, 64, H, W]
        
        # DeformConv Blocks (R1=2 repetitions)
        Out = Out_cat0
        for deform_block in self.deform_conv_blocks:
            Out = deform_block(Out)
        
        # DepthConv Block
        Out = self.depth_conv(Out)  # [B, 128, H, W]
        Out_cat1 = Out
        
        # Downscale using PixelUnshuffle
        Out = self.downscale(Out)  # [B, 2048, H/4, W/4]
        
        # ConvBlocks (R2=2 repetitions) at downscaled resolution
        for conv_block in self.conv_blocks:
            Out = conv_block(Out)
        
        # Upscale using PixelShuffle
        # Out should now have 128 * factor * factor = 2048 channels
        Out_cat2 = self.upscale(Out)  # [B, 128, H, W] after PixelShuffle
        
        # Stage 2a: Denoising path
        img_denoised = self.sigmoid(self.conv_denoised(Out_cat2))  # [B, channel_out, H, W]
        
        # Stage 2b: Super-resolution path with concatenation
        # Concatenate features from different stages
        concat = torch.cat((Out_cat0, Out_cat1, Out_cat2, img_denoised), dim=1)  # [B, concat_channels, H, W]
        
        # Process concatenated features
        Out = self.tanh(self.conv_super_intermediate(concat))
        
        # Final super-resolution output
        img_super = self.sigmoid(self.conv_super(Out))  # [B, channel_out, H, W]
        
        # Upscale super-resolution output to HR size
        img_super = self.upscale_sr(img_super)  # [B, channel_out, H*factor, W*factor]
        
        return img_super, img_denoised

# Legacy class name for backward compatibility
class my_net(DDSRNet):
    def __init__(self, channel_in=3, channel_out=3, factor=4):
        super(my_net, self).__init__(channel_in, channel_out, factor)