"""
Evaluation Metrics
==================

Metrics for evaluating WSI classification performance.
"""

import numpy as np
import torch
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    confusion_matrix,
    roc_curve,
    precision_recall_curve,
)


class AUCMeter:
    """
    Meter for computing AUC during training/evaluation.
    
    Supports both binary and multi-class classification.
    """
    
    def __init__(self, num_classes: int = 2):
        self.num_classes = num_classes
        self.reset()
    
    def reset(self):
        """Reset meter."""
        self.predictions = []
        self.targets = []
        self.probabilities = []
    
    def update(
        self,
        preds: torch.Tensor,
        targets: torch.Tensor,
        probs: Optional[torch.Tensor] = None,
    ):
        """
        Update meter with new predictions.
        
        Args:
            preds: Predicted class indices [B]
            targets: Ground truth labels [B]
            probs: Class probabilities [B, C] (optional)
        """
        self.predictions.extend(preds.cpu().numpy().tolist())
        self.targets.extend(targets.cpu().numpy().tolist())
        
        if probs is not None:
            self.probabilities.extend(probs.cpu().numpy().tolist())
    
    def compute(self) -> Dict[str, float]:
        """
        Compute metrics.
        
        Returns:
            Dictionary with AUC, accuracy, F1, etc.
        """
        predictions = np.array(self.predictions)
        targets = np.array(self.targets)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(targets, predictions)
        
        # F1 Score
        average = 'binary' if self.num_classes == 2 else 'macro'
        metrics['f1'] = f1_score(targets, predictions, average=average)
        
        # AUC
        if len(self.probabilities) > 0:
            probs = np.array(self.probabilities)
            
            if self.num_classes == 2:
                metrics['auc'] = roc_auc_score(targets, probs[:, 1])
            else:
                try:
                    metrics['auc'] = roc_auc_score(
                        targets, probs, multi_class='ovr', average='macro'
                    )
                except ValueError:
                    metrics['auc'] = 0.0
        
        return metrics


def compute_metrics(
    predictions: np.ndarray,
    targets: np.ndarray,
    probabilities: Optional[np.ndarray] = None,
    num_classes: int = 2,
) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: Predicted class indices
        targets: Ground truth labels
        probabilities: Class probabilities (optional)
        num_classes: Number of classes
        
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    # Accuracy
    metrics['accuracy'] = accuracy_score(targets, predictions)
    
    # F1 Score
    average = 'binary' if num_classes == 2 else 'macro'
    metrics['f1'] = f1_score(targets, predictions, average=average)
    metrics['f1_weighted'] = f1_score(targets, predictions, average='weighted')
    
    # Per-class F1
    f1_per_class = f1_score(targets, predictions, average=None)
    for i, f1 in enumerate(f1_per_class):
        metrics[f'f1_class_{i}'] = f1
    
    # Confusion Matrix
    cm = confusion_matrix(targets, predictions)
    metrics['confusion_matrix'] = cm
    
    # Sensitivity and Specificity (for binary)
    if num_classes == 2:
        tn, fp, fn, tp = cm.ravel()
        metrics['sensitivity'] = tp / (tp + fn) if (tp + fn) > 0 else 0
        metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    # AUC
    if probabilities is not None:
        if num_classes == 2:
            metrics['auc'] = roc_auc_score(targets, probabilities[:, 1])
            
            # ROC curve points
            fpr, tpr, thresholds = roc_curve(targets, probabilities[:, 1])
            metrics['roc_curve'] = {'fpr': fpr, 'tpr': tpr, 'thresholds': thresholds}
        else:
            try:
                metrics['auc'] = roc_auc_score(
                    targets, probabilities, multi_class='ovr', average='macro'
                )
                
                # Per-class AUC
                for i in range(num_classes):
                    binary_targets = (targets == i).astype(int)
                    metrics[f'auc_class_{i}'] = roc_auc_score(
                        binary_targets, probabilities[:, i]
                    )
            except ValueError:
                metrics['auc'] = 0.0
    
    return metrics


def compute_ci(
    values: List[float],
    confidence: float = 0.95,
) -> Tuple[float, float, float]:
    """
    Compute confidence interval.
    
    Args:
        values: List of values (e.g., from cross-validation)
        confidence: Confidence level
        
    Returns:
        Tuple of (mean, lower_bound, upper_bound)
    """
    values = np.array(values)
    mean = np.mean(values)
    std = np.std(values)
    n = len(values)
    
    # t-value for confidence interval
    from scipy import stats
    t_value = stats.t.ppf((1 + confidence) / 2, n - 1)
    
    margin = t_value * std / np.sqrt(n)
    
    return mean, mean - margin, mean + margin
