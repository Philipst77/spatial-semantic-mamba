"""
WSI Preprocessing Module
========================

Patch extraction and tissue segmentation for Whole Slide Images.
"""

import os
import numpy as np
import cv2
from typing import List, Tuple, Optional
from pathlib import Path
from tqdm import tqdm

try:
    import openslide
except ImportError:
    openslide = None


class WSIPreprocessor:
    """
    Preprocessor for Whole Slide Images.
    
    Handles tissue segmentation and patch extraction.
    
    Args:
        patch_size: Size of extracted patches (default: 256)
        magnification: Target magnification level (default: 20)
        overlap: Overlap between patches (default: 0)
        tissue_threshold: Threshold for tissue detection (default: 0.5)
    """
    
    def __init__(
        self,
        patch_size: int = 256,
        magnification: int = 20,
        overlap: int = 0,
        tissue_threshold: float = 0.5,
    ):
        self.patch_size = patch_size
        self.magnification = magnification
        self.overlap = overlap
        self.tissue_threshold = tissue_threshold
        self.stride = patch_size - overlap
    
    def process_slide(
        self,
        slide_path: str,
        output_dir: str,
        save_patches: bool = True,
    ) -> Tuple[List[np.ndarray], List[Tuple[int, int]]]:
        """
        Process a single WSI slide.
        
        Args:
            slide_path: Path to WSI file
            output_dir: Output directory for patches
            save_patches: Whether to save patches to disk
            
        Returns:
            Tuple of (patches, coordinates)
        """
        if openslide is None:
            raise ImportError("openslide-python is required for WSI processing")
        
        slide = openslide.OpenSlide(slide_path)
        
        # Get target level
        level = self._get_level(slide)
        level_dims = slide.level_dimensions[level]
        level_downsample = slide.level_downsamples[level]
        
        # Get tissue mask
        tissue_mask = self._get_tissue_mask(slide)
        
        # Extract patches
        patches = []
        coords = []
        
        for y in range(0, level_dims[1] - self.patch_size, self.stride):
            for x in range(0, level_dims[0] - self.patch_size, self.stride):
                # Check tissue content
                mask_x = int(x / level_dims[0] * tissue_mask.shape[1])
                mask_y = int(y / level_dims[1] * tissue_mask.shape[0])
                
                if tissue_mask[mask_y, mask_x] < self.tissue_threshold * 255:
                    continue
                
                # Extract patch
                patch = slide.read_region(
                    (int(x * level_downsample), int(y * level_downsample)),
                    level,
                    (self.patch_size, self.patch_size)
                )
                patch = np.array(patch.convert('RGB'))
                
                patches.append(patch)
                coords.append((x, y))
        
        slide.close()
        
        # Save patches
        if save_patches:
            self._save_patches(patches, coords, output_dir, Path(slide_path).stem)
        
        return patches, coords
    
    def _get_level(self, slide) -> int:
        """Get the level closest to target magnification."""
        try:
            base_mag = float(slide.properties.get('openslide.objective-power', 40))
        except:
            base_mag = 40
        
        target_downsample = base_mag / self.magnification
        
        for level, downsample in enumerate(slide.level_downsamples):
            if downsample >= target_downsample:
                return level
        
        return len(slide.level_downsamples) - 1
    
    def _get_tissue_mask(self, slide, level: int = -1) -> np.ndarray:
        """Generate tissue mask using Otsu thresholding."""
        # Read thumbnail
        thumb_size = (512, 512)
        thumbnail = slide.get_thumbnail(thumb_size)
        thumbnail = np.array(thumbnail.convert('RGB'))
        
        # Convert to grayscale
        gray = cv2.cvtColor(thumbnail, cv2.COLOR_RGB2GRAY)
        
        # Otsu thresholding
        _, mask = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        
        return mask
    
    def _save_patches(
        self,
        patches: List[np.ndarray],
        coords: List[Tuple[int, int]],
        output_dir: str,
        slide_name: str,
    ):
        """Save patches to disk."""
        patch_dir = Path(output_dir) / slide_name
        patch_dir.mkdir(parents=True, exist_ok=True)
        
        for i, (patch, (x, y)) in enumerate(zip(patches, coords)):
            patch_path = patch_dir / f"patch_{x}_{y}.png"
            cv2.imwrite(str(patch_path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))


def extract_patches(
    slide_dir: str,
    output_dir: str,
    patch_size: int = 256,
    magnification: int = 20,
    num_workers: int = 4,
) -> None:
    """
    Extract patches from all slides in a directory.
    
    Args:
        slide_dir: Directory containing WSI files
        output_dir: Output directory for patches
        patch_size: Size of patches
        magnification: Target magnification
        num_workers: Number of parallel workers
    """
    slide_dir = Path(slide_dir)
    slide_files = list(slide_dir.glob("*.tif")) + \
                  list(slide_dir.glob("*.svs")) + \
                  list(slide_dir.glob("*.ndpi"))
    
    preprocessor = WSIPreprocessor(
        patch_size=patch_size,
        magnification=magnification,
    )
    
    for slide_path in tqdm(slide_files, desc="Processing slides"):
        try:
            preprocessor.process_slide(str(slide_path), output_dir)
        except Exception as e:
            print(f"Error processing {slide_path}: {e}")
