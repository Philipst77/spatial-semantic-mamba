"""
WSI Dataset Module
==================

Generic dataset class for Whole Slide Image classification.
Supports Camelyon16, TCGA-BRCA, BRACS, and other pathology datasets.
"""

import os
import torch
import numpy as np
import pandas as pd
import h5py
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Dict, Tuple
from pathlib import Path


class WSIDataset(Dataset):
    """
    Generic WSI dataset for Multiple Instance Learning.
    
    Args:
        data_path: Path to data directory
        split: 'train', 'val', or 'test'
        dataset_name: 'camelyon16', 'tcga_brca', or 'bracs'
        feature_dim: Feature dimension (default: 1024)
        max_patches: Maximum patches per slide (default: None)
    """
    
    def __init__(
        self,
        data_path: str,
        split: str = 'train',
        dataset_name: str = 'camelyon16',
        feature_dim: int = 1024,
        max_patches: Optional[int] = None,
    ):
        super().__init__()
        
        self.data_path = Path(data_path)
        self.split = split
        self.dataset_name = dataset_name.lower()
        self.feature_dim = feature_dim
        self.max_patches = max_patches
        
        self.slide_ids, self.labels = self._load_labels()
        self.num_classes = self._get_num_classes()
        self.class_names = self._get_class_names()
    
    def _load_labels(self) -> Tuple[List[str], List[int]]:
        csv_path = self.data_path / f"{self.split}_labels.csv"
        if not csv_path.exists():
            csv_path = self.data_path / "labels.csv"
        
        df = pd.read_csv(csv_path)
        if 'split' in df.columns:
            df = df[df['split'] == self.split]
        
        return df['slide_id'].tolist(), df['label'].tolist()
    
    def _get_num_classes(self) -> int:
        return {'camelyon16': 2, 'tcga_brca': 4, 'bracs': 7}.get(
            self.dataset_name, max(self.labels) + 1
        )
    
    def _get_class_names(self) -> List[str]:
        return {
            'camelyon16': ['Normal', 'Tumor'],
            'tcga_brca': ['Luminal_A', 'Luminal_B', 'HER2', 'Basal'],
            'bracs': ['Normal', 'Benign', 'UDH', 'ADH', 'FEA', 'DCIS', 'Invasive'],
        }.get(self.dataset_name, [f'Class_{i}' for i in range(self.num_classes)])
    
    def __len__(self) -> int:
        return len(self.slide_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        slide_id = self.slide_ids[idx]
        label = self.labels[idx]
        features, coords = self._load_features(slide_id)
        
        if self.max_patches and len(features) > self.max_patches:
            indices = np.random.choice(len(features), self.max_patches, replace=False)
            features, coords = features[indices], coords[indices]
        
        return {
            'features': torch.FloatTensor(features),
            'coords': torch.FloatTensor(coords),
            'label': torch.LongTensor([label])[0],
            'slide_id': slide_id,
        }
    
    def _load_features(self, slide_id: str) -> Tuple[np.ndarray, np.ndarray]:
        h5_path = self.data_path / 'features' / f"{slide_id}.h5"
        if h5_path.exists():
            with h5py.File(h5_path, 'r') as f:
                features = f['features'][:]
                coords = f['coords'][:] if 'coords' in f else np.zeros((len(features), 2))
            return features, coords
        
        pt_path = self.data_path / 'features' / f"{slide_id}.pt"
        if pt_path.exists():
            data = torch.load(pt_path)
            features = data['features'].numpy() if isinstance(data['features'], torch.Tensor) else data['features']
            coords = data.get('coords', np.zeros((len(features), 2)))
            if isinstance(coords, torch.Tensor):
                coords = coords.numpy()
            return features, coords
        
        raise FileNotFoundError(f"Features not found for slide: {slide_id}")


class WSIBagCollator:
    """Custom collator for WSI bags with variable sizes."""
    
    def __init__(self, max_patches: Optional[int] = None):
        self.max_patches = max_patches
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        max_len = max(item['features'].shape[0] for item in batch)
        if self.max_patches:
            max_len = min(max_len, self.max_patches)
        
        batch_size = len(batch)
        feature_dim = batch[0]['features'].shape[1]
        
        features = torch.zeros(batch_size, max_len, feature_dim)
        coords = torch.zeros(batch_size, max_len, 2)
        masks = torch.zeros(batch_size, max_len, dtype=torch.bool)
        labels = torch.zeros(batch_size, dtype=torch.long)
        slide_ids = []
        
        for i, item in enumerate(batch):
            n = min(item['features'].shape[0], max_len)
            features[i, :n] = item['features'][:n]
            coords[i, :n] = item['coords'][:n]
            masks[i, :n] = True
            labels[i] = item['label']
            slide_ids.append(item['slide_id'])
        
        return {
            'features': features,
            'coords': coords,
            'masks': masks,
            'labels': labels,
            'slide_ids': slide_ids,
        }


def get_dataloader(
    data_path: str,
    split: str,
    dataset_name: str,
    batch_size: int = 1,
    num_workers: int = 4,
    shuffle: bool = True,
    max_patches: Optional[int] = None,
) -> DataLoader:
    """Create dataloader for WSI dataset."""
    dataset = WSIDataset(
        data_path=data_path,
        split=split,
        dataset_name=dataset_name,
        max_patches=max_patches,
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=WSIBagCollator(max_patches),
        pin_memory=True,
    )
