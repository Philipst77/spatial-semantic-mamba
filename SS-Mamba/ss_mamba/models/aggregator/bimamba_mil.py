"""
BiMamba MIL Aggregator
======================

Bidirectional Mamba aggregator for Multiple Instance Learning.
Processes semantically ordered sequences in both forward and backward
directions to capture dependencies from all parts of the sequence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional

from ..backbone.mamba_2d import SelectiveSSM


class MambaLayer(nn.Module):
    """Single Mamba layer with FFN."""
    
    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(d_model)
        self.mamba = SelectiveSSM(d_model, d_state, d_conv, expand)
        
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.mamba(self.norm1(x))
        x = x + self.ffn(self.norm2(x))
        return x


class BiMambaAggregator(nn.Module):
    """
    Bidirectional Mamba aggregator for MIL.
    
    Processes the semantically ordered patch sequence in both forward
    and backward directions, enabling each position to attend to the
    entire sequence context.
    
    Architecture:
        z_forward = Mamba_→(h_π1, h_π2, ..., h_πN)
        z_backward = Mamba_←(h_πN, ..., h_π2, h_π1)
        z = MLP([z_forward; z_backward])
    
    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for Mamba
        num_layers: Number of Mamba layers (default: 2)
        d_state: State dimension for SSM (default: 16)
        dropout: Dropout rate (default: 0.1)
        pooling: Pooling method - 'attention', 'mean', or 'max'
    """
    
    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 512,
        num_layers: int = 2,
        d_state: int = 16,
        dropout: float = 0.1,
        pooling: str = 'attention',
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.pooling = pooling
        
        # Input projection
        self.input_proj = nn.Identity()
        if input_dim != hidden_dim:
            self.input_proj = nn.Linear(input_dim, hidden_dim)
        
        # Forward Mamba layers
        self.forward_layers = nn.ModuleList([
            MambaLayer(hidden_dim, d_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Backward Mamba layers
        self.backward_layers = nn.ModuleList([
            MambaLayer(hidden_dim, d_state, dropout=dropout)
            for _ in range(num_layers)
        ])
        
        # Attention pooling
        if pooling == 'attention':
            self.attention = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.Tanh(),
                nn.Linear(hidden_dim // 2, 1),
            )
        
        # Final layer norm
        self.norm = nn.LayerNorm(hidden_dim * 2)
        
    def forward(
        self,
        x: torch.Tensor,
        return_attention: bool = False,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass of BiMamba aggregator.
        
        Args:
            x: Input sequence [B, N, D]
            return_attention: Whether to return attention weights
            
        Returns:
            Tuple of:
                - Aggregated representation [B, hidden_dim * 2]
                - Attention weights [B, N] (if return_attention=True)
        """
        B, N, D = x.shape
        
        # Project input
        x = self.input_proj(x)
        
        # Forward pass
        h_forward = x
        for layer in self.forward_layers:
            h_forward = layer(h_forward)
        
        # Backward pass
        h_backward = torch.flip(x, dims=[1])
        for layer in self.backward_layers:
            h_backward = layer(h_backward)
        h_backward = torch.flip(h_backward, dims=[1])
        
        # Pool sequences
        attention_weights = None
        
        if self.pooling == 'attention':
            attn_forward = self.attention(h_forward).squeeze(-1)
            attn_backward = self.attention(h_backward).squeeze(-1)
            
            attn_forward = F.softmax(attn_forward, dim=-1)
            attn_backward = F.softmax(attn_backward, dim=-1)
            
            z_forward = torch.einsum('bn,bnd->bd', attn_forward, h_forward)
            z_backward = torch.einsum('bn,bnd->bd', attn_backward, h_backward)
            
            attention_weights = (attn_forward + attn_backward) / 2
            
        elif self.pooling == 'mean':
            z_forward = h_forward.mean(dim=1)
            z_backward = h_backward.mean(dim=1)
            
        elif self.pooling == 'max':
            z_forward = h_forward.max(dim=1)[0]
            z_backward = h_backward.max(dim=1)[0]
        
        # Concatenate
        z = torch.cat([z_forward, z_backward], dim=-1)
        z = self.norm(z)
        
        return z, attention_weights
