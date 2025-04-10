# KNIME Nodes for Vision Transformers
[![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.html)

This repository provides a KNIME extension for fine-tuning and predicting with Vision Transformer (ViT) models. The nodes are fully developed in Python using PyTorch and HuggingFace Transformers, and can be integrated into your KNIME workflows via the KNIME Analytics Platform.

## Features

- **ViT Classification Learner Node**
  - Train transformer models on image classification tasks.
  - Supports **ViT**, **Swin Transformer**, and **Pyramid Transformer** architectures.
  - Accepts training and validation image sets in PNG format.
  - Configurable epochs, batch size, learning rate, and model type.

- **ViT Classification Predictor Node**
  - Predict labels and class probabilities on new image data.
  - Auto-decodes predictions to original label strings.
  - Customizable output column names and probability formatting.

## Extension Info

- **Group ID**: `edu.luiss`
- **Name**: `vision_transformers`
- **Author**: Alberto de Leo
- **Version**: 1.0.0
- **Vendor**: KNIME AG, Zurich, Switzerland
- **License**: [GPL v3](https://www.gnu.org/licenses/gpl-3.0.html)
