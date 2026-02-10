"""
Semantic Pre-ordering Module
============================

Implements parameter-free semantic ordering of patches before MIL aggregation.
Groups similar patches together to create meaningful sequences.

Key Innovation:
- Standard MIL processes patches in arbitrary order, wasting model capacity
- Our semantic ordering arranges patches by feature similarity + spatial proximity
- Greedy nearest-neighbor traversal creates coherent tissue region sequences
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class SemanticOrdering(nn.Module):
    """
    Semantic Pre-ordering for meaningful patch sequences.
    
    Computes combined similarity score balancing semantic similarity
    with spatial proximity, then applies greedy nearest-neighbor ordering.
    
    Similarity Score:
        S_ij = cos(h_i, h_j) + λ * exp(-d_ij² / σ²)
        
    Where:
        - cos(h_i, h_j): Cosine similarity between patch features
        - d_ij: Euclidean distance between patch coordinates
        - λ: Balance factor (default: 0.5)
        - σ: Spatial decay parameter (default: 100)
    
    Args:
        lambda_spatial: Balance factor between semantic and spatial (default: 0.5)
        sigma: Spatial decay parameter (default: 100.0)
    """
    
    def __init__(
        self,
        lambda_spatial: float = 0.5,
        sigma: float = 100.0,
    ):
        super().__init__()
        
        self.lambda_spatial = lambda_spatial
        self.sigma = sigma
        
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Reorder patches based on semantic similarity and spatial proximity.
        
        Args:
            features: Patch features [B, N, D]
            coords: Patch coordinates [B, N, 2]
            
        Returns:
            Tuple of:
                - Reordered features [B, N, D]
                - Ordering indices [B, N]
        """
        B, N, D = features.shape
        
        reordered_features = []
        order_indices = []
        
        for b in range(B):
            feat = features[b]  # [N, D]
            coord = coords[b]   # [N, 2]
            
            # Compute similarity matrix
            similarity = self.compute_similarity(feat, coord)
            
            # Greedy nearest-neighbor ordering
            order = self.greedy_nearest_neighbor(similarity)
            
            # Reorder features
            reordered_feat = feat[order]
            
            reordered_features.append(reordered_feat)
            order_indices.append(order)
        
        reordered_features = torch.stack(reordered_features, dim=0)
        order_indices = torch.stack(order_indices, dim=0)
        
        return reordered_features, order_indices
    
    def compute_similarity(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute combined similarity matrix.
        
        Args:
            features: Patch features [N, D]
            coords: Patch coordinates [N, 2]
            
        Returns:
            Similarity matrix [N, N]
        """
        N = features.shape[0]
        
        # Semantic similarity (cosine)
        features_norm = F.normalize(features, p=2, dim=-1)
        semantic_sim = torch.mm(features_norm, features_norm.t())  # [N, N]
        
        # Spatial proximity (Gaussian decay)
        coords_diff = coords.unsqueeze(0) - coords.unsqueeze(1)  # [N, N, 2]
        spatial_dist = torch.norm(coords_diff, p=2, dim=-1)  # [N, N]
        spatial_sim = torch.exp(-spatial_dist.pow(2) / (self.sigma ** 2))
        
        # Combined similarity
        similarity = semantic_sim + self.lambda_spatial * spatial_sim
        
        return similarity
    
    def greedy_nearest_neighbor(
        self,
        similarity: torch.Tensor,
    ) -> torch.Tensor:
        """
        Greedy nearest-neighbor traversal for ordering.
        
        Algorithm:
            1. Start with most connected node (highest sum of similarities)
            2. Iteratively add nearest unvisited neighbor
            3. Continue until all nodes visited
        
        Args:
            similarity: Similarity matrix [N, N]
            
        Returns:
            Ordering indices [N]
        """
        N = similarity.shape[0]
        device = similarity.device
        
        # Start with most connected node
        connectivity = similarity.sum(dim=1)
        start_idx = connectivity.argmax().item()
        
        # Track visited nodes
        visited = torch.zeros(N, dtype=torch.bool, device=device)
        order = torch.zeros(N, dtype=torch.long, device=device)
        
        # Initialize
        current = start_idx
        order[0] = current
        visited[current] = True
        
        # Greedy traversal
        for i in range(1, N):
            # Get similarities to unvisited nodes
            sim_to_unvisited = similarity[current].clone()
            sim_to_unvisited[visited] = float('-inf')
            
            # Select nearest unvisited neighbor
            next_idx = sim_to_unvisited.argmax().item()
            
            order[i] = next_idx
            visited[next_idx] = True
            current = next_idx
        
        return order
    
    def compute_coherence(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
        order: torch.Tensor,
    ) -> float:
        """
        Compute sequence coherence metric.
        
        Coherence = (1/N-1) * Σ S[π_i, π_{i+1}]
        
        Higher coherence indicates more structured sequences.
        
        Args:
            features: Original features [N, D]
            coords: Patch coordinates [N, 2]
            order: Ordering indices [N]
            
        Returns:
            Coherence score (float)
        """
        similarity = self.compute_similarity(features, coords)
        
        coherence = 0.0
        N = len(order)
        
        for i in range(N - 1):
            coherence += similarity[order[i], order[i + 1]].item()
        
        coherence /= (N - 1)
        
        return coherence


class HierarchicalOrdering(nn.Module):
    """
    Hierarchical semantic ordering using clustering.
    
    First clusters patches into groups, then orders within groups.
    Provides coarse-to-fine semantic structure.
    """
    
    def __init__(
        self,
        n_clusters: int = 8,
        lambda_spatial: float = 0.5,
        sigma: float = 100.0,
    ):
        super().__init__()
        
        self.n_clusters = n_clusters
        self.base_ordering = SemanticOrdering(lambda_spatial, sigma)
    
    def forward(
        self,
        features: torch.Tensor,
        coords: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Hierarchical ordering: cluster then order within clusters.
        """
        B, N, D = features.shape
        
        reordered_features = []
        order_indices = []
        
        for b in range(B):
            feat = features[b]  # [N, D]
            coord = coords[b]   # [N, 2]
            
            # Cluster patches (simplified k-means)
            cluster_ids = self._cluster(feat)
            
            # Order within each cluster, then concatenate
            final_order = []
            for c in range(self.n_clusters):
                mask = cluster_ids == c
                if mask.sum() == 0:
                    continue
                    
                cluster_feat = feat[mask]
                cluster_coord = coord[mask]
                cluster_indices = torch.where(mask)[0]
                
                if len(cluster_feat) > 1:
                    # Order within cluster
                    sim = self.base_ordering.compute_similarity(
                        cluster_feat, cluster_coord
                    )
                    local_order = self.base_ordering.greedy_nearest_neighbor(sim)
                    final_order.extend(cluster_indices[local_order].tolist())
                else:
                    final_order.extend(cluster_indices.tolist())
            
            order = torch.tensor(final_order, device=features.device)
            reordered_features.append(feat[order])
            order_indices.append(order)
        
        return torch.stack(reordered_features), torch.stack(order_indices)
    
    def _cluster(self, features: torch.Tensor) -> torch.Tensor:
        """Simple k-means clustering."""
        N, D = features.shape
        
        # Random initialization
        indices = torch.randperm(N)[:self.n_clusters]
        centroids = features[indices].clone()
        
        # Iterate
        for _ in range(10):
            # Assign clusters
            dists = torch.cdist(features, centroids)
            cluster_ids = dists.argmin(dim=1)
            
            # Update centroids
            for c in range(self.n_clusters):
                mask = cluster_ids == c
                if mask.sum() > 0:
                    centroids[c] = features[mask].mean(dim=0)
        
        return cluster_ids
