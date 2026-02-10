"""
Data Transforms and Augmentations
=================================

Augmentation strategies for pathology images.
"""

import torch
import numpy as np
from typing import Optional, Dict, Any


class FeatureAugmentation:
    """
    Augmentation for pre-extracted features.
    
    Args:
        dropout_rate: Rate for feature dropout (default: 0.1)
        noise_std: Standard deviation of Gaussian noise (default: 0.01)
        mixup_alpha: Alpha for mixup augmentation (default: 0.0)
    """
    
    def __init__(
        self,
        dropout_rate: float = 0.1,
        noise_std: float = 0.01,
        mixup_alpha: float = 0.0,
    ):
        self.dropout_rate = dropout_rate
        self.noise_std = noise_std
        self.mixup_alpha = mixup_alpha
    
    def __call__(self, features: np.ndarray) -> np.ndarray:
        """Apply augmentations to features."""
        # Feature dropout
        if self.dropout_rate > 0:
            mask = np.random.binomial(1, 1 - self.dropout_rate, features.shape)
            features = features * mask / (1 - self.dropout_rate)
        
        # Gaussian noise
        if self.noise_std > 0:
            noise = np.random.normal(0, self.noise_std, features.shape)
            features = features + noise
        
        return features.astype(np.float32)


class PatchAugmentation:
    """
    Augmentation for raw patch images.
    
    Includes standard pathology augmentations:
    - Color jittering
    - Random rotation
    - Random flipping
    """
    
    def __init__(
        self,
        color_jitter: bool = True,
        random_rotation: bool = True,
        random_flip: bool = True,
    ):
        self.color_jitter = color_jitter
        self.random_rotation = random_rotation
        self.random_flip = random_flip
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Apply augmentations to image."""
        if self.random_flip:
            if np.random.random() > 0.5:
                image = np.fliplr(image)
            if np.random.random() > 0.5:
                image = np.flipud(image)
        
        if self.random_rotation:
            k = np.random.randint(4)
            image = np.rot90(image, k)
        
        if self.color_jitter:
            image = self._color_jitter(image)
        
        return image
    
    def _color_jitter(
        self,
        image: np.ndarray,
        brightness: float = 0.1,
        contrast: float = 0.1,
        saturation: float = 0.1,
    ) -> np.ndarray:
        """Apply color jittering."""
        image = image.astype(np.float32) / 255.0
        
        # Brightness
        image = image + np.random.uniform(-brightness, brightness)
        
        # Contrast
        mean = image.mean()
        image = (image - mean) * (1 + np.random.uniform(-contrast, contrast)) + mean
        
        # Clip and convert back
        image = np.clip(image * 255, 0, 255).astype(np.uint8)
        
        return image


def get_augmentations(
    mode: str = 'train',
    feature_level: bool = True,
) -> Optional[callable]:
    """
    Get augmentation transform based on mode.
    
    Args:
        mode: 'train', 'val', or 'test'
        feature_level: Whether to use feature-level augmentation
        
    Returns:
        Augmentation transform or None
    """
    if mode != 'train':
        return None
    
    if feature_level:
        return FeatureAugmentation(
            dropout_rate=0.1,
            noise_std=0.01,
        )
    else:
        return PatchAugmentation(
            color_jitter=True,
            random_rotation=True,
            random_flip=True,
        )
