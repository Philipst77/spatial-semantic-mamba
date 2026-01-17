"""
SS-Mamba: Spatial-Semantic Mamba for Computational Pathology
============================================================

A novel framework for WSI classification that preserves 2D spatial 
structure and ensures semantically meaningful patch sequences.

Main Components:
- 2D-SSM Backbone: Bidirectional scanning for spatial preservation
- Semantic Pre-ordering: Feature-based patch ordering
- BiMamba Aggregator: Bidirectional MIL aggregation
"""

from .models.ss_mamba import SSMamba
from .models.backbone.mamba_2d import Mamba2D
from .models.ordering.semantic_ordering import SemanticOrdering
from .models.aggregator.bimamba_mil import BiMambaAggregator

__version__ = "1.0.0"
__author__ = "Sina Mansouri, Neelesh Prakash Wadhwani, Philip Stavrev"

__all__ = [
    "SSMamba",
    "Mamba2D", 
    "SemanticOrdering",
    "BiMambaAggregator",
]
