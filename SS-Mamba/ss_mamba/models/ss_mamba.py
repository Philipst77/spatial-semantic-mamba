"""
SS-Mamba: Spatial-Semantic Mamba Model
======================================

Full model combining 2D-SSM backbone, semantic ordering, and BiMamba aggregator.
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple, Dict

from .backbone.mamba_2d import Mamba2D
from .ordering.semantic_ordering import SemanticOrdering
from .aggregator.bimamba_mil import BiMambaAggregator


class SSMamba(nn.Module):
    """
    Spatial-Semantic Mamba for Whole Slide Image Classification.
    
    This model addresses two key limitations in computational pathology:
    1. Spatial discrepancy from 2D-to-1D flattening
    2. Random patch ordering in MIL
    
    Args:
        feature_dim: Input feature dimension (default: 1024)
        hidden_dim: Hidden dimension for Mamba layers (default: 512)
        num_classes: Number of output classes
        num_layers: Number of Mamba layers (default: 2)
        dropout: Dropout rate (default: 0.25)
        lambda_spatial: Balance factor for semantic ordering (default: 0.5)
        sigma: Spatial decay parameter (default: 100)
        use_2d_ssm: Whether to use 2D-SSM backbone (default: True)
        use_semantic_ordering: Whether to use semantic ordering (default: True)
    """
    
    def __init__(
        self,
        feature_dim: int = 1024,
        hidden_dim: int = 512,
        num_classes: int = 2,
        num_layers: int = 2,
        dropout: float = 0.25,
        lambda_spatial: float = 0.5,
        sigma: float = 100.0,
        use_2d_ssm: bool = True,
        use_semantic_ordering: bool = True,
    ):
        super().__init__()
        
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_2d_ssm = use_2d_ssm
        self.use_semantic_ordering = use_semantic_ordering
        
        # Feature projection
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # 2D-SSM Backbone for spatial-aware feature extraction
        if use_2d_ssm:
            self.backbone = Mamba2D(
                d_model=hidden_dim,
                d_state=16,
                d_conv=4,
                expand=2,
            )
        
        # Semantic ordering module
        if use_semantic_ordering:
            self.ordering = SemanticOrdering(
                lambda_spatial=lambda_spatial,
                sigma=sigma,
            )
        
        # BiMamba MIL Aggregator
        self.aggregator = BiMambaAggregator(
            input_dim=hidden_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        )
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),  # *2 for bidirectional
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        features: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass of SS-Mamba.
        
        Args:
            features: Patch features [B, N, D] or [N, D]
            coords: Patch coordinates [B, N, 2] or [N, 2] (optional)
            return_attention: Whether to return attention weights
            
        Returns:
            Dictionary containing:
                - logits: Classification logits [B, num_classes]
                - attention: Attention weights (if return_attention=True)
                - order: Semantic ordering indices (if use_semantic_ordering=True)
        """
        # Handle single sample (no batch dimension)
        if features.dim() == 2:
            features = features.unsqueeze(0)
            if coords is not None:
                coords = coords.unsqueeze(0)
        
        B, N, D = features.shape
        output = {}
        
        # Project features to hidden dimension
        h = self.feature_proj(features)  # [B, N, hidden_dim]
        
        # Apply 2D-SSM backbone for spatial-aware features
        if self.use_2d_ssm:
            h = self.backbone(h)  # [B, N, hidden_dim]
        
        # Apply semantic ordering
        if self.use_semantic_ordering and coords is not None:
            h, order = self.ordering(h, coords)
            output['order'] = order
        
        # Aggregate with BiMamba
        z, attention = self.aggregator(h, return_attention=True)
        
        if return_attention:
            output['attention'] = attention
        
        # Classification
        logits = self.classifier(z)
        output['logits'] = logits
        
        return output
    
    def get_attention_scores(
        self,
        features: torch.Tensor,
        coords: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Get attention scores for visualization."""
        with torch.no_grad():
            output = self.forward(features, coords, return_attention=True)
        return output.get('attention', None)


def build_ss_mamba(config: Dict) -> SSMamba:
    """
    Build SS-Mamba model from configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        SSMamba model instance
    """
    model_config = config.get('model', {})
    ordering_config = config.get('ordering', {})
    
    return SSMamba(
        feature_dim=model_config.get('feature_dim', 1024),
        hidden_dim=model_config.get('hidden_dim', 512),
        num_classes=model_config.get('num_classes', 2),
        num_layers=model_config.get('num_layers', 2),
        dropout=model_config.get('dropout', 0.25),
        lambda_spatial=ordering_config.get('lambda_spatial', 0.5),
        sigma=ordering_config.get('sigma', 100.0),
        use_2d_ssm=True,
        use_semantic_ordering=ordering_config.get('enabled', True),
    )
