from pytorch_msssim import MS_SSIM, ms_ssim, SSIM, ssim
from torchmetrics.image import PeakSignalNoiseRatio
import torch
import torch.nn as nn

class MS_SSIM_Loss(MS_SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(MS_SSIM_Loss, self).forward(img1, img2) )

class SSIM_Loss(SSIM):
    def forward(self, img1, img2):
        return 100*( 1 - super(SSIM_Loss, self).forward(img1, img2) )

class CompositeLoss(nn.Module):
    """
    Composite loss function implementing:
    D = L1(I_ref - I_out) + 10 * [1 - SSIM(I_ref, I_out)] + 100/PSNR(I_ref, I_out)
    """
    def __init__(self, data_range=1.0, size_average=True, channel=1):
        super(CompositeLoss, self).__init__()
        self.l1_loss = nn.L1Loss()
        self.ssim_loss = SSIM_Loss(data_range=data_range, size_average=size_average, channel=channel)
        self.psnr_metric = PeakSignalNoiseRatio(data_range=data_range)
        
    def forward(self, image_ref, image_out):
        # L1 loss
        l1_term = self.l1_loss(image_out, image_ref)
        
        # SSIM term: 10 * [1 - SSIM(I_ref, I_out)]
        ssim_term = 10 * self.ssim_loss(image_out, image_ref) / 100  # Divide by 100 since SSIM_Loss already multiplies by 100
        
        # PSNR term: 100 / PSNR(I_ref, I_out)
        # Add small epsilon to avoid division by zero
        psnr_value = self.psnr_metric(image_out, image_ref)
        psnr_term = 100 / (psnr_value + 1e-8)
        
        return l1_term + ssim_term + psnr_term

class CombinedLoss(nn.Module):
    """
    Combined loss function implementing:
    Loss = λ * D_denoising + β * D_super_resolution
    """
    def __init__(self, lambda_weight=1.0, beta_weight=1.0, data_range=1.0, size_average=True, channel=1):
        super(CombinedLoss, self).__init__()
        self.lambda_weight = lambda_weight
        self.beta_weight = beta_weight
        self.denoising_loss = CompositeLoss(data_range=data_range, size_average=size_average, channel=channel)
        self.super_resolution_loss = CompositeLoss(data_range=data_range, size_average=size_average, channel=channel)
        
    def forward(self, denoising_ref, denoising_out, sr_ref, sr_out):
        # D_denoising
        d_denoising = self.denoising_loss(denoising_ref, denoising_out)
        
        # D_super_resolution
        d_super_resolution = self.super_resolution_loss(sr_ref, sr_out)
        
        # Combined loss: λ * D_denoising + β * D_super_resolution
        total_loss = self.lambda_weight * d_denoising + self.beta_weight * d_super_resolution
        
        return total_loss, d_denoising, d_super_resolution