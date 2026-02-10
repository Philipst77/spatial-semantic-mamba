# Spatial-Semantic Mamba (SS-Mamba)

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch 2.0+](https://img.shields.io/badge/pytorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Official implementation of **"Spatial-Semantic Mamba: Preserving 2D Structure and Meaningful Sequencing for Enhanced Computational Pathology"**

## Paper
📄 **Spatial-Semantic Mamba: Preserving 2D Structure and Meaningful Sequencing for Enhanced Computational Pathology**

- PDF: [`Spatial_Semantic_Mamba.pdf`](./Spatial_Semantic_Mamba.pdf)

> This work was developed as a graduate-level research project and is currently an unpublished manuscript / preprint.



## Overview

SS-Mamba is a novel framework for Whole Slide Image (WSI) classification that addresses two critical limitations in current computational pathology methods:

1. **Spatial Discrepancy**: Standard methods flatten 2D patches into 1D sequences, destroying spatial relationships
2. **Random Patch Ordering**: MIL frameworks process patches in arbitrary order, ignoring semantic relationships

Our approach introduces:
- **2D-SSM Backbone**: Bidirectional scanning along horizontal and vertical axes
- **Semantic Pre-ordering**: Parameter-free ordering based on feature similarity
- **BiMamba Aggregator**: Bidirectional Mamba for MIL aggregation


## Project Structure

```
SS-Mamba/
├── README.md
├── requirements.txt
├── setup.py
├── configs/
│   └── config.yaml              # Configuration file
├── ss_mamba/                    # Main package
│   ├── models/
│   │   ├── ss_mamba.py          # Full SS-Mamba model
│   │   ├── backbone/
│   │   │   └── mamba_2d.py      # 2D-SSM backbone
│   │   ├── ordering/
│   │   │   └── semantic_ordering.py
│   │   └── aggregator/
│   │       └── bimamba_mil.py   # BiMamba aggregator
│   ├── data/
│   │   ├── wsi_dataset.py       # WSI dataset class
│   │   ├── preprocessing.py     # Patch extraction
│   │   └── transforms.py        # Augmentations
│   └── utils/
│       ├── metrics.py           # Evaluation metrics
│       ├── logger.py            # Logging utilities
│       └── visualization.py     # Visualization tools
├── tools/
│   ├── train.py                 # Training script
│   ├── evaluate.py              # Evaluation script
│   └── extract_features.py      # Feature extraction
└── scripts/
    ├── run_experiment.sh        # Run single experiment
    └── run_all_datasets.sh      # Run all experiments
```


### 1. Data Preparation

Download datasets and organize as follows:

```
data/
├── camelyon16/
│   ├── train/
│   │   ├── slides/              # .tif files
│   │   └── labels.csv
│   └── test/
├── tcga_brca/
│   ├── slides/
│   └── labels.csv
└── bracs/
    ├── slides/
    └── labels.csv
```

### 2. Patch Extraction

```bash
python tools/extract_features.py \
    --data_path /path/to/slides \
    --output_path /path/to/patches \
    --patch_size 256 \
    --magnification 20
```

### 3. Training

```bash
# Camelyon16 (binary classification)
python tools/train.py \
    --dataset camelyon16 \
    --data_path /path/to/data \
    --num_classes 2 \
    --epochs 100

# TCGA-BRCA (4-class classification)
python tools/train.py \
    --dataset tcga_brca \
    --data_path /path/to/data \
    --num_classes 4 \
    --epochs 100

# BRACS (7-class classification)
python tools/train.py \
    --dataset bracs \
    --data_path /path/to/data \
    --num_classes 7 \
    --epochs 150
```

Or use shell scripts:

```bash
bash scripts/run_experiment.sh camelyon16
bash scripts/run_all_datasets.sh
```

### 4. Evaluation

```bash
python tools/evaluate.py \
    --dataset camelyon16 \
    --checkpoint /path/to/checkpoint.pth \
    --data_path /path/to/data
```

## Configuration

Edit `configs/config.yaml` for custom settings:

```yaml
model:
  hidden_dim: 512
  num_layers: 2
  dropout: 0.25

ordering:
  lambda_spatial: 0.5
  sigma: 100

training:
  lr: 0.0002
  weight_decay: 0.05
  epochs: 100
  batch_size: 1
```

## Datasets

| Dataset | WSIs | Classes | Task | Link |
|---------|------|---------|------|------|
| Camelyon16 | 399 | 2 | Metastasis Detection | [Link](https://camelyon16.grand-challenge.org/) |
| TCGA-BRCA | 1,098 | 4 | Molecular Subtyping | [Link](https://portal.gdc.cancer.gov/) |
| BRACS | 547 | 7 | Tumor Typing | [Link](https://www.bracs.icar.cnr.it/) |



## Acknowledgments

- [Mamba](https://github.com/state-spaces/mamba) for the State Space Model implementation
- [VMamba](https://github.com/MzeroMiko/VMamba) for 2D scanning patterns
- [CLAM](https://github.com/mahmoodlab/CLAM) for WSI preprocessing pipeline

## Contact

- Sina Mansouri - Smansou3@gmu.edu
- Neelesh Prakash Wadhwani - nwadhwan@gmu.edu
- Philip Stavrev - pstavrev@gmu.edu
