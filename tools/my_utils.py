import numpy as np
from PIL import Image
from skimage.util import random_noise
import skimage
from skimage.filters import gaussian
from torchvision.transforms.functional import to_tensor
import random
import torch
import cv2
import os
from typing import Tuple, List, Union, Optional
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def ensure_dir(path: Union[str, Path]) -> None:
    """Ensure directory exists, create if it doesn't."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)

def validate_patch_config(patch_size: int, factor: int) -> int:
    """
    Validate and adjust patch size to be divisible by the downscale factor.
    
    Args:
        patch_size: Original patch size
        factor: Downscale factor
    
    Returns:
        Adjusted patch size that is divisible by factor
    """
    if patch_size % factor != 0:
        # Find the largest patch size smaller than or equal to the original that's divisible by factor
        adjusted_patch_size = (patch_size // factor) * factor
        logger.warning(f"Patch size {patch_size} not divisible by factor {factor}. "
                      f"Adjusting to {adjusted_patch_size}")
        return adjusted_patch_size
    return patch_size
        
def NormalizeData(data: np.ndarray) -> np.ndarray:
    """Normalize data to [0, 1] range."""
    data_min, data_max = data.min(), data.max()
    if data_max == data_min:
        return np.zeros_like(data)
    return (data - data_min) / (data_max - data_min)

def img_loader(path: Union[str, Path], type_out: str) -> Image.Image:
    """Load image with error handling."""
    try:
        return Image.open(path).convert(type_out)
    except Exception as e:
        logger.error(f"Failed to load image {path}: {e}")
        raise

def CropPatches_hr_lr(image: Image.Image, patch_size: int, stride: int, 
                      factor: int, type_in: str, noise: Union[str, bool], noise_type: str) -> Tuple[Tuple[torch.Tensor, ...], 
                                                                                 Tuple[torch.Tensor, ...], 
                                                                                 Tuple[torch.Tensor, ...]]:
    """
    Crop high-resolution image into patches and generate corresponding low-resolution patches.
    
    Args:
        image: Input high-resolution image
        patch_size: Size of patches to extract
        stride: Stride for patch extraction
        factor: Downscaling factor
        type_in: Input image type for LR patches
        noise: Noise type to add to LR patches
        noise_type: Noise type to add to LR patches
    Returns:
        Tuple of (HR patches, LR patches with noise, LR patches without noise)
    """
    w, h = image.size
    
    # Ensure patch_size is divisible by factor to avoid dimension issues
    if patch_size % factor != 0:
        # Adjust patch_size to be divisible by factor
        adjusted_patch_size = ((patch_size // factor) * factor)
        logger.warning(f"Patch size {patch_size} not divisible by factor {factor}. "
                      f"Adjusting to {adjusted_patch_size}")
        patch_size = adjusted_patch_size
    
    # Pre-allocate lists for better memory efficiency
    patches_hr: List[torch.Tensor] = []
    patches_lr: List[torch.Tensor] = []
    patches_lr_no_noise: List[torch.Tensor] = []
    
    # Calculate new dimensions for LR patches (should be divisible by factor)
    new_w, new_h = patch_size // factor, patch_size // factor
    
    # Use range with step for better performance
    for i in range(0, h - patch_size + 1, stride):
        for j in range(0, w - patch_size + 1, stride):
            # Extract HR patch
            patch_hr = image.crop((j, i, j + patch_size, i + patch_size))
            
            # Create LR patch by downscaling
            patch_lr = patch_hr.resize((new_w, new_h), Image.BICUBIC)
            patch_lr = patch_lr.convert(type_in)
            
            # Convert to tensor for no-noise version
            patch_lr_tensor = to_tensor(patch_lr)
            patches_lr_no_noise.append(patch_lr_tensor)
            
            # Add noise if specified
            if noise_type:
                try:
                    patch_lr = add_noise(patch_lr, noise_type, degree=0.01)
                except Exception as e:
                    logger.warning(f"Failed to add noise to patch: {e}")
                    # Use original patch if noise addition fails
            
            # Convert to tensor and store
            patches_hr.append(to_tensor(patch_hr))
            patches_lr.append(to_tensor(patch_lr))
    
    return tuple(patches_hr), tuple(patches_lr), tuple(patches_lr_no_noise)

def add_noise(image: Image.Image, mode: str, degree: float) -> Image.Image:
    """
    Add noise to image with optimized processing.
    
    Args:
        image: Input image
        mode: Noise type ('gaussian', 'speckle', 's&p', 'blur')
        degree: Noise intensity
    
    Returns:
        Image with added noise
    """
    try:
        # Convert to numpy array once
        img_array = np.array(image, dtype=np.float64) / 255.0
        
        if mode in ["gaussian", "speckle"]:
            output = random_noise(img_array, mode=mode, var=degree)
        elif mode == "s&p":
            output = random_noise(img_array, mode=mode, amount=degree)
        elif mode == "blur":
            output = gaussian(img_array, sigma=1)
        else:
            logger.warning(f"Unknown noise mode: {mode}, returning original image")
            return image
        
        # Convert back to uint8
        output = np.clip(output * 255, 0, 255).astype(np.uint8)
        return Image.fromarray(output)
        
    except Exception as e:
        logger.error(f"Failed to add noise: {e}")
        return image

def add_salt_and_pepper(image: Image.Image, amount: float) -> Image.Image:
    """
    Add salt and pepper noise to image with optimized implementation.
    
    Args:
        image: Input image
        amount: Amount of noise to add
    
    Returns:
        Image with salt and pepper noise
    """
    try:
        output = np.array(image, copy=True)
        total_pixels = output.size
        
        # Calculate number of salt and pepper pixels
        nb_salt = int(np.ceil(amount * total_pixels * 0.5))
        nb_pepper = int(np.ceil(amount * total_pixels * 0.5))
        
        # Add salt (white pixels)
        if nb_salt > 0:
            salt_coords = tuple(np.random.randint(0, i, nb_salt) for i in output.shape)
            output[salt_coords] = 255
        
        # Add pepper (black pixels)
        if nb_pepper > 0:
            pepper_coords = tuple(np.random.randint(0, i, nb_pepper) for i in output.shape)
            output[pepper_coords] = 0
        
        return Image.fromarray(output)
        
    except Exception as e:
        logger.error(f"Failed to add salt and pepper noise: {e}")
        return image

def add_snow(image: Image.Image) -> Image.Image:
    """
    Add snow effect to image with optimized color space conversion.
    
    Args:
        image: Input RGB image
    
    Returns:
        Image with snow effect
    """
    try:
        image_array = np.asarray(image)
        
        # Convert to HLS color space
        image_HLS = cv2.cvtColor(image_array, cv2.COLOR_RGB2HLS)
        image_HLS = image_HLS.astype(np.float64)
        
        # Snow parameters
        brightness_coefficient = 1.0
        snow_point = 1400
        
        # Apply snow effect to lightness channel
        lightness_mask = image_HLS[:, :, 1] < snow_point
        image_HLS[:, :, 1][lightness_mask] *= brightness_coefficient
        
        # Clip values to valid range
        image_HLS[:, :, 1] = np.clip(image_HLS[:, :, 1], 0, 255)
        
        # Convert back to RGB
        image_HLS = image_HLS.astype(np.uint8)
        image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
        
        return Image.fromarray(image_RGB)
        
    except Exception as e:
        logger.error(f"Failed to add snow effect: {e}")
        return image

def add_rain(image: Image.Image) -> Image.Image:
    """
    Add rain effect to image with optimized line generation.
    
    Args:
        image: Input RGB image
    
    Returns:
        Image with rain effect
    """
    try:
        image_array = np.asarray(image)
        imshape = image_array.shape
        
        # Rain parameters
        slant_extreme = 1
        slant = np.random.randint(-slant_extreme, slant_extreme)
        drop_length = 5
        drop_width = 2
        drop_color = (200, 200, 200)
        
        # Generate rain drops
        rain_drops = generate_random_lines(imshape, slant, drop_length)
        
        # Draw rain drops
        for rain_drop in rain_drops:
            x, y = rain_drop
            end_x = x + slant
            end_y = y + drop_length
            
            # Ensure coordinates are within image bounds
            if (0 <= x < imshape[1] and 0 <= y < imshape[0] and 
                0 <= end_x < imshape[1] and 0 <= end_y < imshape[0]):
                cv2.line(image_array, (x, y), (end_x, end_y), drop_color, drop_width)
        
        # Apply blur and brightness adjustment
        image_array = cv2.blur(image_array, (2, 2))
        
        # Convert to HLS for brightness adjustment
        image_HLS = cv2.cvtColor(image_array, cv2.COLOR_RGB2HLS)
        image_HLS = image_HLS.astype(np.float64)
        
        # Reduce brightness for rainy effect
        brightness_coefficient = 0.7
        image_HLS[:, :, 1] *= brightness_coefficient
        
        # Convert back to RGB
        image_HLS = np.clip(image_HLS, 0, 255).astype(np.uint8)
        image_RGB = cv2.cvtColor(image_HLS, cv2.COLOR_HLS2RGB)
        
        return Image.fromarray(image_RGB)
        
    except Exception as e:
        logger.error(f"Failed to add rain effect: {e}")
        return image

def generate_random_lines(imshape: Tuple[int, ...], slant: int, drop_length: int) -> List[Tuple[int, int]]:
    """
    Generate random line coordinates for rain effect.
    
    Args:
        imshape: Image shape (height, width, channels)
        slant: Slant angle for rain drops
        drop_length: Length of rain drops
    
    Returns:
        List of (x, y) coordinates for rain drops
    """
    drops = []
    height, width = imshape[0], imshape[1]
    
    # Generate random rain drop positions
    for _ in range(1500):  # Number of rain drops
        if slant < 0:
            x = np.random.randint(slant, width)
        else:
            x = np.random.randint(0, width - slant)
        
        y = np.random.randint(0, height - drop_length)
        drops.append((x, y))
    
    return drops

def apply_super_resolution(image_lr_in: Image.Image, model: torch.nn.Module, 
                         channel_out: int, in_patch_size: int, img_output_size: Tuple[int, int], 
                         factor: int, device: torch.device) -> torch.Tensor:
    """
    Apply super-resolution to image using patch-based processing.
    
    Args:
        image_lr_in: Input low-resolution image
        model: Super-resolution model
        channel_out: Number of output channels
        in_patch_size: Input patch size
        img_output_size: Output image size (width, height)
        factor: Upscaling factor
        device: Device to run model on
    
    Returns:
        High-resolution output image
    """
    model = model.to(device)
    in_w, in_h = image_lr_in.size
    out_w, out_h = img_output_size
    out_patch_size = in_patch_size * factor
    
    # Initialize output tensor
    image_hr_out = torch.zeros(channel_out, out_h, out_w, device=device)
    
    # Process patches
    for i, m in zip(range(0, in_h, in_patch_size), range(0, out_h, out_patch_size)):
        for j, p in zip(range(0, in_w, in_patch_size), range(0, out_w, out_patch_size)):
            
            # Check if patch is within bounds
            if (i + in_patch_size) <= in_h and (j + in_patch_size) <= in_w:
                patch_lr = image_lr_in.crop((j, i, j + in_patch_size, i + in_patch_size))
                patch_lr = to_tensor(patch_lr).to(device)
                
                # Get model output (assuming first output is super-resolution)
                with torch.no_grad():
                    model_output = model(patch_lr)
                    patch_hr = model_output[0] if isinstance(model_output, tuple) else model_output
                
                # Place patch in output image
                image_hr_out[:, m:m + out_patch_size, p:p + out_patch_size] = patch_hr.squeeze(0)
    
    # Handle edge cases for non-divisible patch sizes
    _handle_edge_patches(image_lr_in, image_hr_out, model, in_patch_size, 
                        out_patch_size, channel_out, device, is_super_res=True)
    
    return image_hr_out

def apply_denoising(image_lr_in: Image.Image, model: torch.nn.Module, 
                   channel_out: int, in_patch_size: int, factor: int, 
                   device: torch.device) -> torch.Tensor:
    """
    Apply denoising to image using patch-based processing.
    
    Args:
        image_lr_in: Input noisy image
        model: Denoising model
        channel_out: Number of output channels
        in_patch_size: Input patch size
        factor: Upscaling factor (unused for denoising)
        device: Device to run model on
    
    Returns:
        Denoised output image
    """
    model = model.to(device)
    in_w, in_h = image_lr_in.size
    out_w, out_h = in_w, in_h  # Same size for denoising
    out_patch_size = in_patch_size
    
    # Initialize output tensor
    image_hr_out = torch.zeros(channel_out, out_h, out_w, device=device)
    
    # Process patches
    for i, m in zip(range(0, in_h, in_patch_size), range(0, out_h, in_patch_size)):
        for j, p in zip(range(0, in_w, in_patch_size), range(0, out_w, in_patch_size)):
            
            # Check if patch is within bounds
            if (i + in_patch_size) <= in_h and (j + in_patch_size) <= in_w:
                patch_lr = image_lr_in.crop((j, i, j + in_patch_size, i + in_patch_size))
                patch_lr = to_tensor(patch_lr).to(device)
                
                # Get model output (assuming second output is denoising)
                with torch.no_grad():
                    model_output = model(patch_lr)
                    patch_hr = model_output[1] if isinstance(model_output, tuple) else model_output
                
                # Place patch in output image
                image_hr_out[:, m:m + out_patch_size, p:p + out_patch_size] = patch_hr.squeeze(0)
    
    # Handle edge cases for non-divisible patch sizes
    _handle_edge_patches(image_lr_in, image_hr_out, model, in_patch_size, 
                        out_patch_size, channel_out, device, is_super_res=False)
    
    return image_hr_out

def _handle_edge_patches(image_lr_in: Image.Image, image_hr_out: torch.Tensor, 
                        model: torch.nn.Module, in_patch_size: int, out_patch_size: int,
                        channel_out: int, device: torch.device, is_super_res: bool) -> None:
    """
    Handle edge patches for non-divisible image sizes.
    
    Args:
        image_lr_in: Input image
        image_hr_out: Output tensor
        model: Model to use for inference
        in_patch_size: Input patch size
        out_patch_size: Output patch size
        channel_out: Number of output channels
        device: Device to run model on
        is_super_res: Whether this is super-resolution (affects output indexing)
    """
    in_w, in_h = image_lr_in.size
    out_w, out_h = image_hr_out.shape[-2:] if len(image_hr_out.shape) >= 3 else (in_w, in_h)
    
    # Handle right edge
    if in_w % in_patch_size != 0:
        for i in range(0, in_h, in_patch_size):
            if i + in_patch_size <= in_h:
                patch_lr = image_lr_in.crop((in_w - in_patch_size, i, in_w, i + in_patch_size))
                patch_lr = to_tensor(patch_lr).to(device)
                
                with torch.no_grad():
                    model_output = model(patch_lr)
                    patch_hr = model_output[0 if is_super_res else 1] if isinstance(model_output, tuple) else model_output
                
                if is_super_res:
                    image_hr_out[:, i:i + out_patch_size, out_w - out_patch_size:out_w] = patch_hr.squeeze(0)
                else:
                    image_hr_out[:, i:i + out_patch_size, out_w - out_patch_size:out_w] = patch_hr.squeeze(0)
    
    # Handle bottom edge
    if in_h % in_patch_size != 0:
        for j in range(0, in_w, in_patch_size):
            if j + in_patch_size <= in_w:
                patch_lr = image_lr_in.crop((j, in_h - in_patch_size, j + in_patch_size, in_h))
                patch_lr = to_tensor(patch_lr).to(device)
                
                with torch.no_grad():
                    model_output = model(patch_lr)
                    patch_hr = model_output[0 if is_super_res else 1] if isinstance(model_output, tuple) else model_output
                
                if is_super_res:
                    image_hr_out[:, out_h - out_patch_size:out_h, j:j + out_patch_size] = patch_hr.squeeze(0)
                else:
                    image_hr_out[:, out_h - out_patch_size:out_h, j:j + out_patch_size] = patch_hr.squeeze(0)
    
    # Handle bottom-right corner
    if in_w % in_patch_size != 0 and in_h % in_patch_size != 0:
        patch_lr = image_lr_in.crop((in_w - in_patch_size, in_h - in_patch_size, in_w, in_h))
        patch_lr = to_tensor(patch_lr).to(device)
        
        with torch.no_grad():
            model_output = model(patch_lr)
            patch_hr = model_output[0 if is_super_res else 1] if isinstance(model_output, tuple) else model_output
        
        if is_super_res:
            image_hr_out[:, out_h - out_patch_size:out_h, out_w - out_patch_size:out_w] = patch_hr.squeeze(0)
        else:
            image_hr_out[:, out_h - out_patch_size:out_h, out_w - out_patch_size:out_w] = patch_hr.squeeze(0)
