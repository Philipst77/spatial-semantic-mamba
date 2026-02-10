#!/usr/bin/env python
"""
Feature Extraction Script
=========================

Extract patch features from WSIs using pre-trained encoders.

Usage:
    python tools/extract_features.py --slide_dir /path/to/slides --output_dir /path/to/features
"""

import sys
import argparse
import torch
import numpy as np
import h5py
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ss_mamba.data.preprocessing import WSIPreprocessor


def parse_args():
    parser = argparse.ArgumentParser(description='Extract features from WSIs')
    parser.add_argument('--slide_dir', type=str, required=True,
                        help='Directory containing WSI files')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='Output directory for features')
    parser.add_argument('--encoder', type=str, default='resnet50',
                        choices=['resnet50', 'vit', 'dino'],
                        help='Feature encoder')
    parser.add_argument('--patch_size', type=int, default=256)
    parser.add_argument('--magnification', type=int, default=20)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--device', type=str, default='cuda')
    return parser.parse_args()


def get_encoder(encoder_name: str, device: str):
    """Load pre-trained encoder."""
    if encoder_name == 'resnet50':
        import torchvision.models as models
        model = models.resnet50(pretrained=True)
        model.fc = torch.nn.Identity()
        feature_dim = 2048
    elif encoder_name == 'vit':
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.head = torch.nn.Identity()
        feature_dim = 768
    elif encoder_name == 'dino':
        model = torch.hub.load('facebookresearch/dino:main', 'dino_vitb16')
        feature_dim = 768
    else:
        raise ValueError(f"Unknown encoder: {encoder_name}")
    
    model = model.to(device)
    model.eval()
    
    return model, feature_dim


def extract_features_from_patches(patches, model, batch_size, device):
    """Extract features from patches using encoder."""
    import torchvision.transforms as transforms
    
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    features = []
    
    for i in range(0, len(patches), batch_size):
        batch_patches = patches[i:i + batch_size]
        batch_tensors = torch.stack([transform(p) for p in batch_patches]).to(device)
        
        with torch.no_grad():
            batch_features = model(batch_tensors)
        
        features.append(batch_features.cpu().numpy())
    
    return np.concatenate(features, axis=0)


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading encoder: {args.encoder}")
    model, feature_dim = get_encoder(args.encoder, device)
    
    preprocessor = WSIPreprocessor(
        patch_size=args.patch_size,
        magnification=args.magnification,
    )
    
    slide_dir = Path(args.slide_dir)
    slide_files = list(slide_dir.glob("*.tif")) + \
                  list(slide_dir.glob("*.svs")) + \
                  list(slide_dir.glob("*.ndpi"))
    
    print(f"Found {len(slide_files)} slides")
    
    for slide_path in tqdm(slide_files, desc="Processing slides"):
        slide_name = slide_path.stem
        output_path = output_dir / f"{slide_name}.h5"
        
        if output_path.exists():
            print(f"Skipping {slide_name} (already exists)")
            continue
        
        try:
            patches, coords = preprocessor.process_slide(
                str(slide_path), 
                str(output_dir / 'patches'),
                save_patches=False
            )
            
            if len(patches) == 0:
                print(f"No patches extracted for {slide_name}")
                continue
            
            features = extract_features_from_patches(
                patches, model, args.batch_size, device
            )
            
            with h5py.File(output_path, 'w') as f:
                f.create_dataset('features', data=features)
                f.create_dataset('coords', data=np.array(coords))
            
            print(f"Saved {len(features)} features for {slide_name}")
            
        except Exception as e:
            print(f"Error processing {slide_name}: {e}")
    
    print("Feature extraction complete!")


if __name__ == '__main__':
    main()
