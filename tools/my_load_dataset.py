from torchvision import transforms
import torch
from torch.utils.data import Dataset
from pathlib import Path
from tqdm import tqdm
from .my_utils import img_loader, CropPatches_hr_lr
import os
from typing import List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class load_hr_and_lr(Dataset):
    """Optimized dataset loader for HR and LR image pairs with caching and efficient memory management."""
    
    def __init__(self, dataset: str, config: dict, status: str):
        self.img_loader = img_loader
        self.status = status
        
        # Validate status
        if status not in ['train', 'test', 'valid']:
            raise ValueError(f"Invalid status: {status}. Must be 'train', 'test', or 'valid'.")
        
        # Get folder paths
        self.train_folder = Path(config[dataset]['train_dir'])
        self.test_folder = Path(config[dataset]['test_dir'])
        self.valid_folder = Path(config[dataset]['valid_dir'])
        
        # Configuration parameters
        self.factor = config['factor']
        self.type_out = config['output_type']
        self.type_in = config['input_type']
        self.patch_size = config['patch_size']
        self.stride = config['stride']
        self.noise = config['noise']
        self.noise_type = config['noise_type']
        
        # Validate and adjust patch size to be divisible by factor
        from .my_utils import validate_patch_config
        self.patch_size = validate_patch_config(self.patch_size, self.factor)
        
        # Use lists instead of tuples for better memory efficiency
        self.patchesHR: List[torch.Tensor] = []
        self.patchesLR: List[torch.Tensor] = []
        self.patchesLR_nn: List[torch.Tensor] = []
        
        # Load data based on status
        self._load_data()
        
        # Convert to tensors for faster access
        self._convert_to_tensors()
        
        logger.info(f"Loaded {len(self.patchesHR)} patches for {status} dataset")
    
    def _get_folder_path(self) -> Path:
        """Get the appropriate folder path based on status."""
        if self.status == 'train':
            return self.train_folder
        elif self.status == 'test':
            return self.test_folder
        else:  # valid
            return self.valid_folder
    
    def _load_data(self):
        """Load and process image data efficiently."""
        folder_path = self._get_folder_path()
        
        if not folder_path.exists():
            raise FileNotFoundError(f"Dataset folder not found: {folder_path}")
        
        # Get sorted list of image files
        image_files = sorted([f for f in folder_path.iterdir() 
                            if f.suffix.lower() in {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}])
        
        if not image_files:
            raise ValueError(f"No image files found in {folder_path}")
        
        # Process images with progress bar
        for img_path in tqdm(image_files, desc=f'Loading {self.status} images'):
            try:
                img_hr = self.img_loader(img_path, self.type_out)
                patchesHR, patchesLR, patchesLR_nn = CropPatches_hr_lr(
                    img_hr, self.patch_size, self.stride, self.factor, 
                    self.type_in, self.noise, self.noise_type
                )
                
                # Extend lists instead of concatenating tuples
                self.patchesHR.extend(patchesHR)
                self.patchesLR.extend(patchesLR)
                self.patchesLR_nn.extend(patchesLR_nn)
                
            except Exception as e:
                logger.warning(f"Failed to load image {img_path}: {e}")
                continue
    
    def _convert_to_tensors(self):
        """Convert all patches to tensors for faster access."""
        if self.patchesHR:
            # Validate patch dimensions before stacking
            self._validate_patch_dimensions()
            
            # Stack tensors for better memory efficiency
            try:
                self.patchesHR = torch.stack(self.patchesHR)
                self.patchesLR = torch.stack(self.patchesLR)
                self.patchesLR_nn = torch.stack(self.patchesLR_nn)
            except Exception as e:
                logger.error(f"Failed to convert patches to tensors: {e}")
                # Fallback to list if stacking fails
                pass
    
    def _validate_patch_dimensions(self):
        """Validate that all patches have correct dimensions for the model."""
        if not self.patchesHR:
            return
        
        # Check first patch for reference dimensions
        first_hr = self.patchesHR[0]
        first_lr = self.patchesLR[0]
        
        expected_hr_size = (self.patch_size, self.patch_size)
        expected_lr_size = (self.patch_size // self.factor, self.patch_size // self.factor)
        
        # Validate HR patches
        for i, patch in enumerate(self.patchesHR):
            if patch.shape[-2:] != expected_hr_size:
                logger.warning(f"HR patch {i} has incorrect size: {patch.shape[-2:]}, expected {expected_hr_size}")
        
        # Validate LR patches
        for i, patch in enumerate(self.patchesLR):
            if patch.shape[-2:] != expected_lr_size:
                logger.warning(f"LR patch {i} has incorrect size: {patch.shape[-2:]}, expected {expected_lr_size}")
        
        logger.info(f"Patch validation complete. HR size: {expected_hr_size}, LR size: {expected_lr_size}")
    
    def __len__(self) -> int:
        return len(self.patchesHR)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of range for dataset of size {len(self)}")
        
        # Return tensors directly if conversion was successful
        if isinstance(self.patchesHR, torch.Tensor):
            return self.patchesHR[idx], self.patchesLR[idx], self.patchesLR_nn[idx]
        else:
            # Fallback to list access
            return self.patchesHR[idx], self.patchesLR[idx], self.patchesLR_nn[idx]
    
    def get_patch_info(self) -> dict:
        """Get information about the dataset patches."""
        if not self.patchesHR:
            return {}
        
        sample_patch = self.patchesHR[0] if isinstance(self.patchesHR, list) else self.patchesHR[0]
        
        return {
            'num_patches': len(self),
            'patch_size': sample_patch.shape[-2:] if len(sample_patch.shape) >= 2 else None,
            'channels': sample_patch.shape[0] if len(sample_patch.shape) >= 3 else None,
            'factor': self.factor,
            'stride': self.stride,
            'noise_type': self.noise_type
        }