#!/bin/bash
# =============================================================================
# SS-Mamba: Run All Experiments
# =============================================================================
# Runs training and evaluation on all three datasets
# Usage: bash scripts/run_all_datasets.sh

set -e

echo "=============================================="
echo "SS-Mamba: Running All Experiments"
echo "=============================================="

# Camelyon16
echo ""
echo "[1/3] Running Camelyon16..."
bash scripts/run_experiment.sh camelyon16 ./data/camelyon16

# TCGA-BRCA
echo ""
echo "[2/3] Running TCGA-BRCA..."
bash scripts/run_experiment.sh tcga_brca ./data/tcga_brca

# BRACS
echo ""
echo "[3/3] Running BRACS..."
bash scripts/run_experiment.sh bracs ./data/bracs

echo ""
echo "=============================================="
echo "All Experiments Complete!"
echo "=============================================="
echo "Results:"
echo "  - Camelyon16: results/camelyon16/"
echo "  - TCGA-BRCA: results/tcga_brca/"
echo "  - BRACS: results/bracs/"
echo "=============================================="
