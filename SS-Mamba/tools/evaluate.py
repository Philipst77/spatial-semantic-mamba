#!/usr/bin/env python
"""
SS-Mamba Evaluation Script
==========================

Evaluate trained SS-Mamba model.

Usage:
    python tools/evaluate.py --dataset camelyon16 --checkpoint checkpoints/best_model.pth --data_path /path/to/data
"""

import sys
import argparse
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from ss_mamba.models import SSMamba
from ss_mamba.data import get_dataloader
from ss_mamba.utils import compute_metrics, get_logger
from ss_mamba.utils.visualization import plot_roc_curve, plot_confusion_matrix


def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate SS-Mamba')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['camelyon16', 'tcga_brca', 'bracs'])
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to model checkpoint')
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results')
    return parser.parse_args()


@torch.no_grad()
def evaluate(model, dataloader, device):
    model.eval()
    
    all_preds = []
    all_labels = []
    all_probs = []
    all_slide_ids = []
    
    for batch in tqdm(dataloader, desc='Evaluating'):
        features = batch['features'].to(device)
        coords = batch['coords'].to(device)
        labels = batch['labels']
        slide_ids = batch['slide_ids']
        
        output = model(features, coords)
        logits = output['logits']
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        
        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.numpy().tolist())
        all_probs.extend(probs.cpu().numpy().tolist())
        all_slide_ids.extend(slide_ids)
    
    return {
        'predictions': np.array(all_preds),
        'labels': np.array(all_labels),
        'probabilities': np.array(all_probs),
        'slide_ids': all_slide_ids,
    }


def main():
    args = parse_args()
    device = args.device if torch.cuda.is_available() else 'cpu'
    
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger('evaluate', str(output_dir))
    logger.info(f"Evaluating on {args.dataset}")
    
    # Load checkpoint
    checkpoint = torch.load(args.checkpoint, map_location=device)
    config = checkpoint['config']
    
    # Load model
    model = SSMamba(
        feature_dim=config['feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        lambda_spatial=config['lambda_spatial'],
        sigma=config['sigma'],
    ).to(device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    logger.info(f"Loaded checkpoint from epoch {checkpoint['epoch']}")
    
    # Load data
    dataloader = get_dataloader(
        args.data_path, args.split, args.dataset,
        args.batch_size, args.num_workers, shuffle=False
    )
    
    # Evaluate
    results = evaluate(model, dataloader, device)
    
    # Compute metrics
    metrics = compute_metrics(
        results['predictions'],
        results['labels'],
        results['probabilities'],
        num_classes=config['num_classes'],
    )
    
    # Log results
    logger.info(f"Results on {args.split} set:")
    logger.info(f"  AUC: {metrics.get('auc', 0):.4f}")
    logger.info(f"  Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  F1: {metrics['f1']:.4f}")
    
    if config['num_classes'] == 2:
        logger.info(f"  Sensitivity: {metrics.get('sensitivity', 0):.4f}")
        logger.info(f"  Specificity: {metrics.get('specificity', 0):.4f}")
    
    # Plot ROC curve
    if 'roc_curve' in metrics:
        plot_roc_curve(
            metrics['roc_curve']['fpr'],
            metrics['roc_curve']['tpr'],
            metrics['auc'],
            save_path=str(output_dir / 'roc_curve.png'),
        )
        logger.info(f"ROC curve saved to {output_dir / 'roc_curve.png'}")
    
    # Plot confusion matrix
    class_names = {
        'camelyon16': ['Normal', 'Tumor'],
        'tcga_brca': ['Luminal_A', 'Luminal_B', 'HER2', 'Basal'],
        'bracs': ['Normal', 'Benign', 'UDH', 'ADH', 'FEA', 'DCIS', 'Invasive'],
    }[args.dataset]
    
    plot_confusion_matrix(
        metrics['confusion_matrix'],
        class_names,
        save_path=str(output_dir / 'confusion_matrix.png'),
    )
    logger.info(f"Confusion matrix saved to {output_dir / 'confusion_matrix.png'}")
    
    # Save predictions
    import pandas as pd
    df = pd.DataFrame({
        'slide_id': results['slide_ids'],
        'label': results['labels'],
        'prediction': results['predictions'],
    })
    df.to_csv(output_dir / 'predictions.csv', index=False)
    logger.info(f"Predictions saved to {output_dir / 'predictions.csv'}")
    
    logger.info("Evaluation complete!")


if __name__ == '__main__':
    main()
