#!/bin/bash
# =============================================================================
# SS-Mamba Experiment Runner
# =============================================================================
# Usage: bash scripts/run_experiment.sh <dataset_name>
# Example: bash scripts/run_experiment.sh camelyon16

set -e

if [ $# -lt 1 ]; then
    echo "Usage: bash scripts/run_experiment.sh <dataset_name>"
    echo "Available datasets: camelyon16, tcga_brca, bracs"
    exit 1
fi

DATASET=$1
DATA_PATH=${2:-"./data/${DATASET}"}

case $DATASET in
    camelyon16)
        NUM_CLASSES=2
        EPOCHS=100
        ;;
    tcga_brca)
        NUM_CLASSES=4
        EPOCHS=100
        ;;
    bracs)
        NUM_CLASSES=7
        EPOCHS=150
        ;;
    *)
        echo "Unknown dataset: $DATASET"
        exit 1
        ;;
esac

echo "=============================================="
echo "SS-Mamba Training: $DATASET"
echo "=============================================="

python tools/train.py \
    --dataset $DATASET \
    --data_path $DATA_PATH \
    --num_classes $NUM_CLASSES \
    --epochs $EPOCHS \
    --feature_dim 1024 \
    --hidden_dim 512 \
    --num_layers 2 \
    --dropout 0.25 \
    --lambda_spatial 0.5 \
    --sigma 100 \
    --lr 0.0002 \
    --weight_decay 0.05 \
    --batch_size 1 \
    --seed 42

echo "Training Complete!"

python tools/evaluate.py \
    --dataset $DATASET \
    --checkpoint checkpoints/${DATASET}/best_model.pth \
    --data_path $DATA_PATH \
    --split test

echo "Evaluation Complete!"
