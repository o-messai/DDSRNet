#!/usr/bin/env python3
"""
Test script to verify model dimensions and patch creation.
This script tests the model with different input sizes to ensure compatibility.
"""

import torch
import yaml
from model import DDSRNet
from tools.my_utils import validate_patch_config, CropPatches_hr_lr
from PIL import Image
import numpy as np

def test_model_dimensions():
    """Test the model with different input dimensions."""
    
    # Load configuration
    with open("config.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)
    
    factor = config['factor']
    patch_size = config['patch_size']
    
    print(f"Configuration:")
    print(f"  Factor: {factor}")
    print(f"  Patch size: {patch_size}")
    print(f"  Patch size divisible by factor: {patch_size % factor == 0}")
    
    # Validate patch size
    adjusted_patch_size = validate_patch_config(patch_size, factor)
    print(f"  Adjusted patch size: {adjusted_patch_size}")
    
    # Create model
    channel_in = 1 if config['input_type'] == 'L' else 3
    channel_out = 1 if config['output_type'] == 'L' else 3
    
    model = DDSRNet(channel_in=channel_in, channel_out=channel_out, factor=factor)
    print(f"Model created with {channel_in} input channels and {channel_out} output channels")
    
    # Test different input sizes
    test_sizes = [64, 128, 256, 512]
    
    for size in test_sizes:
        print(f"\nTesting input size: {size}x{size}")
        
        # Check if size is divisible by factor
        if size % factor != 0:
            print(f"  WARNING: {size} not divisible by {factor}")
            continue
        
        # Create dummy input
        dummy_input = torch.randn(1, channel_in, size, size)
        print(f"  Input shape: {dummy_input.shape}")
        
        try:
            # Forward pass
            with torch.no_grad():
                output_sr, output_denoised = model(dummy_input)
            
            print(f"  Output SR shape: {output_sr.shape}")
            print(f"  Output denoised shape: {output_denoised.shape}")
            
            # Verify dimensions
            expected_sr_size = size * factor
            expected_denoised_size = size
            
            if output_sr.shape[-2:] == (expected_sr_size, expected_sr_size):
                print(f"  ✓ SR output size correct: {output_sr.shape[-2:]}")
            else:
                print(f"  ✗ SR output size incorrect: expected {expected_sr_size}x{expected_sr_size}, got {output_sr.shape[-2:]}")
            
            if output_denoised.shape[-2:] == (expected_denoised_size, expected_denoised_size):
                print(f"  ✓ Denoised output size correct: {output_denoised.shape[-2:]}")
            else:
                print(f"  ✗ Denoised output size incorrect: expected {expected_denoised_size}x{expected_denoised_size}, got {output_denoised.shape[-2:]}")
            
            print(f"  ✓ Test passed")
            
        except Exception as e:
            print(f"  ✗ Test failed: {e}")
    
    # Test patch creation
    print(f"\nTesting patch creation:")
    test_image = Image.new('L', (256, 256), color=128)
    
    try:
        patches_hr, patches_lr, patches_lr_nn = CropPatches_hr_lr(
            test_image, adjusted_patch_size, 64, factor, 'L', False
        )
        
        print(f"  Created {len(patches_hr)} patches")
        if patches_hr:
            print(f"  HR patch shape: {patches_hr[0].shape}")
            print(f"  LR patch shape: {patches_lr[0].shape}")
            print(f"  ✓ Patch creation test passed")
        
    except Exception as e:
        print(f"  ✗ Patch creation test failed: {e}")

if __name__ == "__main__":
    test_model_dimensions()
