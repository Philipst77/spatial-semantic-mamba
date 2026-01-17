"""
Visualization Utilities
=======================

Visualization tools for attention maps and results.
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Optional, List, Tuple
from pathlib import Path


def visualize_attention(
    attention_weights: np.ndarray,
    coords: np.ndarray,
    slide_size: Tuple[int, int] = (1000, 1000),
    patch_size: int = 256,
    save_path: Optional[str] = None,
    title: str = "Attention Heatmap",
) -> np.ndarray:
    """
    Visualize attention weights as a heatmap.
    
    Args:
        attention_weights: Attention weights [N]
        coords: Patch coordinates [N, 2]
        slide_size: Size of output heatmap
        patch_size: Size of patches
        save_path: Path to save figure
        title: Plot title
        
    Returns:
        Heatmap as numpy array
    """
    heatmap = np.zeros(slide_size)
    count_map = np.zeros(slide_size)
    
    for (x, y), weight in zip(coords, attention_weights):
        x, y = int(x), int(y)
        x_end = min(x + patch_size, slide_size[0])
        y_end = min(y + patch_size, slide_size[1])
        
        heatmap[y:y_end, x:x_end] += weight
        count_map[y:y_end, x:x_end] += 1
    
    # Average overlapping regions
    count_map[count_map == 0] = 1
    heatmap = heatmap / count_map
    
    # Normalize
    heatmap = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min() + 1e-8)
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 10))
    im = ax.imshow(heatmap, cmap='jet', interpolation='bilinear')
    ax.set_title(title)
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return heatmap


def plot_roc_curve(
    fpr: np.ndarray,
    tpr: np.ndarray,
    auc_score: float,
    save_path: Optional[str] = None,
    title: str = "ROC Curve",
) -> None:
    """
    Plot ROC curve.
    
    Args:
        fpr: False positive rates
        tpr: True positive rates
        auc_score: AUC score
        save_path: Path to save figure
        title: Plot title
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    ax.plot(fpr, tpr, color='darkorange', lw=2,
            label=f'ROC curve (AUC = {auc_score:.3f})')
    ax.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel('False Positive Rate', fontsize=12)
    ax.set_ylabel('True Positive Rate', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='lower right', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_confusion_matrix(
    cm: np.ndarray,
    class_names: List[str],
    save_path: Optional[str] = None,
    title: str = "Confusion Matrix",
    normalize: bool = True,
) -> None:
    """
    Plot confusion matrix.
    
    Args:
        cm: Confusion matrix
        class_names: List of class names
        save_path: Path to save figure
        title: Plot title
        normalize: Whether to normalize
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    
    fig, ax = plt.subplots(figsize=(8, 8))
    
    im = ax.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        xticklabels=class_names,
        yticklabels=class_names,
        title=title,
        ylabel='True label',
        xlabel='Predicted label'
    )
    
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    
    # Add text annotations
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                   ha="center", va="center",
                   color="white" if cm[i, j] > thresh else "black")
    
    fig.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def plot_training_curves(
    train_losses: List[float],
    val_losses: List[float],
    train_aucs: List[float],
    val_aucs: List[float],
    save_path: Optional[str] = None,
) -> None:
    """
    Plot training curves.
    
    Args:
        train_losses: Training losses per epoch
        val_losses: Validation losses per epoch
        train_aucs: Training AUCs per epoch
        val_aucs: Validation AUCs per epoch
        save_path: Path to save figure
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    # Loss plot
    ax1.plot(epochs, train_losses, 'b-', label='Train Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # AUC plot
    ax2.plot(epochs, train_aucs, 'b-', label='Train AUC')
    ax2.plot(epochs, val_aucs, 'r-', label='Val AUC')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('AUC')
    ax2.set_title('Training and Validation AUC')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
