#!/usr/bin/env python
"""
SS-Mamba Training Script
========================

Usage:
    python tools/train.py --dataset camelyon16 --data_path /path/to/data --num_classes 2
    python tools/train.py --dataset tcga_brca --data_path /path/to/data --num_classes 4
    python tools/train.py --dataset bracs --data_path /path/to/data --num_classes 7
"""

import os
import sys
import argparse
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from pathlib import Path
from tqdm import tqdm
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

from ss_mamba.models import SSMamba
from ss_mamba.data import get_dataloader
from ss_mamba.utils import AUCMeter, get_logger, TensorboardLogger


def parse_args():
    parser = argparse.ArgumentParser(description='Train SS-Mamba')
    parser.add_argument('--dataset', type=str, required=True,
                        choices=['camelyon16', 'tcga_brca', 'bracs'])
    parser.add_argument('--data_path', type=str, required=True)
    parser.add_argument('--num_classes', type=int, required=True)
    parser.add_argument('--feature_dim', type=int, default=1024)
    parser.add_argument('--hidden_dim', type=int, default=512)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.25)
    parser.add_argument('--lambda_spatial', type=float, default=0.5)
    parser.add_argument('--sigma', type=float, default=100.0)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--weight_decay', type=float, default=0.05)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--max_patches', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--save_dir', type=str, default='checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs')
    parser.add_argument('--config', type=str, default=None)
    return parser.parse_args()


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_epoch(model, dataloader, criterion, optimizer, device, scaler, epoch):
    model.train()
    meter = AUCMeter(num_classes=model.num_classes)
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Train]')
    for batch in pbar:
        features = batch['features'].to(device)
        coords = batch['coords'].to(device)
        labels = batch['labels'].to(device)
        
        optimizer.zero_grad()
        
        with autocast():
            output = model(features, coords)
            logits = output['logits']
            loss = criterion(logits, labels)
        
        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()
        
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        meter.update(preds, labels, probs)
        total_loss += loss.item()
        pbar.set_postfix({'loss': loss.item()})
    
    metrics = meter.compute()
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


@torch.no_grad()
def validate(model, dataloader, criterion, device, epoch):
    model.eval()
    meter = AUCMeter(num_classes=model.num_classes)
    total_loss = 0.0
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch} [Val]')
    for batch in pbar:
        features = batch['features'].to(device)
        coords = batch['coords'].to(device)
        labels = batch['labels'].to(device)
        
        output = model(features, coords)
        logits = output['logits']
        loss = criterion(logits, labels)
        
        probs = torch.softmax(logits, dim=-1)
        preds = logits.argmax(dim=-1)
        meter.update(preds, labels, probs)
        total_loss += loss.item()
    
    metrics = meter.compute()
    metrics['loss'] = total_loss / len(dataloader)
    return metrics


def main():
    args = parse_args()
    
    if args.config:
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
        for key, value in vars(args).items():
            if value is not None:
                config[key] = value
    else:
        config = vars(args)
    
    set_seed(config['seed'])
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    
    save_dir = Path(config['save_dir']) / config['dataset']
    save_dir.mkdir(parents=True, exist_ok=True)
    log_dir = Path(config['log_dir']) / config['dataset']
    log_dir.mkdir(parents=True, exist_ok=True)
    
    logger = get_logger('train', str(log_dir))
    tb_logger = TensorboardLogger(str(log_dir), config['dataset'])
    
    logger.info(f"Training SS-Mamba on {config['dataset']}")
    
    train_loader = get_dataloader(
        config['data_path'], 'train', config['dataset'],
        config['batch_size'], config['num_workers'], True, config.get('max_patches')
    )
    val_loader = get_dataloader(
        config['data_path'], 'val', config['dataset'],
        config['batch_size'], config['num_workers'], False, config.get('max_patches')
    )
    
    model = SSMamba(
        feature_dim=config['feature_dim'],
        hidden_dim=config['hidden_dim'],
        num_classes=config['num_classes'],
        num_layers=config['num_layers'],
        dropout=config['dropout'],
        lambda_spatial=config['lambda_spatial'],
        sigma=config['sigma'],
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=1e-6)
    scaler = GradScaler()
    
    best_auc = 0.0
    
    for epoch in range(1, config['epochs'] + 1):
        train_metrics = train_epoch(model, train_loader, criterion, optimizer, device, scaler, epoch)
        val_metrics = validate(model, val_loader, criterion, device, epoch)
        scheduler.step()
        
        logger.info(f"Epoch {epoch}: Train AUC={train_metrics.get('auc', 0):.4f}, Val AUC={val_metrics.get('auc', 0):.4f}")
        tb_logger.log_metrics(train_metrics, epoch, prefix='train')
        tb_logger.log_metrics(val_metrics, epoch, prefix='val')
        
        val_auc = val_metrics.get('auc', 0)
        if val_auc > best_auc:
            best_auc = val_auc
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'best_auc': best_auc, 'config': config}, save_dir / 'best_model.pth')
            logger.info(f"Saved best model with AUC={best_auc:.4f}")
    
    logger.info(f"Training complete. Best AUC: {best_auc:.4f}")
    tb_logger.close()


if __name__ == '__main__':
    main()
