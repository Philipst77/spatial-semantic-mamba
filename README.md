# Spatial-Semantic Mamba (SS-Mamba)

This repository contains the research paper and supporting materials for **Spatial-Semantic Mamba (SS-Mamba)**, a novel Mamba-based framework for whole-slide image (WSI) analysis in computational pathology.

## Overview
Current Multiple Instance Learning (MIL) approaches for WSI analysis suffer from:
1. Destruction of spatial structure due to 2D → 1D flattening
2. Arbitrary ordering of patches that ignores semantic relationships

SS-Mamba addresses these limitations through:
- **2D State Space Models (2D-SSM)** with bidirectional scanning to preserve spatial continuity
- **Semantic pre-ordering** of patches to construct meaningful sequences before MIL aggregation

The method maintains **linear O(N) complexity** while achieving state-of-the-art performance.

## Paper
📄 **Spatial-Semantic Mamba: Preserving 2D Structure and Meaningful Sequencing for Enhanced Computational Pathology**

- PDF: [`Spatial_Semantic_Mamba.pdf`](./Spatial_Semantic_Mamba.pdf)

> This work was developed as a graduate-level research project and is currently an unpublished manuscript / preprint.

## Results
SS-Mamba was evaluated on three benchmark datasets:
- **Camelyon16**: 95.1% AUC
- **TCGA-BRCA**: 89.1% AUC
- **BRACS**: 85.8% AUC

Ablation studies demonstrate a +3.3% improvement over 1D Mamba baselines with only ~8% computational overhead.

## Research Contribution (Summary)
My primary contributions to this project include:
- Implementing data preprocessing and patch extraction pipelines
- Setting up and evaluating baseline methods
- Conducting literature review and comparative analysis
- Creating visualizations and figures
- Writing and editing sections of the manuscript

## Authors
- Sina Mansouri
- Neelesh Prakash Wadhwani
- **Philip Stavrev**

## Status
This repository is intended to showcase the research contribution for academic review and PhD applications.

## License
Academic use only.
