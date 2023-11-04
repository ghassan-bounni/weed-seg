# Weed-Seg

a Python-based PyTorch training and inference pipeline for Weed Segmentation and Stem Detection.

## Table of Contents

- [Introduction](#introduction)
- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Project-Structure](#project-structure)
- [Data](#data)
- [Configuration](#configuration)
- [Usage](#usage)
  - [Training](#training)
  - [Inference](#inference)

## Introduction

This README provides an overview and information on how to set up and use these pipelines.

## Features

- Train deep learning models with PyTorch.
- Perform inference using pre-trained models.
- Customize hyperparameters to fit your specific needs.
- Handle data preprocessing and augmentation.
- Evaluate model performance using various metrics.
- **Checkpointing**: allows users to resume training in case of a sudden stop or continue training from a specific epoch.

## Requirements

Before you can use this project, make sure you have the following requirements in place:

- Python (3.8+)
- PyTorch
- GPU support (if applicable)

## Installation

Clone the repo using this command:

```bash
git clone https://github.com/lux-rob/weed-seg.git
```

To install the necessary dependencies, use the following command:

```bash
pip install -r requirements.txt
```

## Project Structure

To help you navigate and understand the organization of this project, here's an overview of the directory structure:

```plaintext
project-root/
├── config.yaml
├── configs/
├── data/
├── eval.py
├── eval.sh
├── guidelines.py
├── logger/
├── main.py
├── models/
├── output/
├── README.md
├── requirements.txt
├── scripts/
├── train.py
├── train.sh
├── utils/
```

- config.yaml: Configuration file for the project, described in the "Configuration" section of the README.
- configs/: Directory containing config loading script.
- data/: The directory where you should organize your training, validation, and test datasets as described in the "Data" section of the README.
- eval.py: Script for evaluation.
- eval.sh: Shell script related to evaluation, if applicable.
- guidelines.py: Python script containing guidelines or helper functions.
- logger/: Directory related to logging functionality.
- main.py: The entry point of the project, responsible for handling command-line arguments and orchestrating the training and evaluation processes.
- models/: Directory containing model-related code and components.
- output/: Directory where the output of the pipelines may be stored.
- scripts/: Directory for shell scripts or additional scripts used in the project.
- train.py: Script for training.
- train.sh: Shell script related to training, if applicable.
- utils/: Directory containing utility functions and modules used throughout the project.

## Data

To use your PyTorch training and inference pipelines effectively, it's essential to organize your data as follows:

- Prepare your data in a directory structure that includes three main subdirectories: `train`, `val`, and `test`.
- Inside each of these subdirectories, you should have two subdirectories: `images` and `masks`. The `images` directory contains the input images, and the `masks` directory contains corresponding segmentation masks.
- Ensure that the image files have the same name as their corresponding masks. This is necessary for pairing images and masks during the data loading process.
- In the case of multiple class semantic segmentation, you can organize your data as follows:

  - Inside the `masks` directory, create subdirectories, each with a name corresponding to an image.
  - Place separate masks per class for each image within the respective subdirectories.
  - Name the mask files the same as the corresponding class name.

Here's an example of the directory structure:

```plaintext
data/
├── train/
│   ├── images/
│   │   ├── image1.jpg
│   │   ├── image2.jpg
│   │   └── ...
│   └── masks/
│       ├── image1.jpg
│       ├── image2.jpg
│       └── ...
├── val/
│   ├── images/
│   ├── masks/
└── test/
    ├── images/
    ├── masks/
```

## Configuration

Your PyTorch pipelines are highly configurable through a `config.yaml` file. Here's an example of the structure and some of the key configuration options:

```yaml
model:
  in_channels: 3
  out_channels: 1
  num_init_features: 32
  num_layers: 3
  block_depth: 3
  growth_rate: 4
  exp_factor: 2
  drop_rate: 0.2
  kernel_size: 3
  conv_mode: "full"

train:
  path: "/data/train"
  epochs: 200
  batch_size: 3
  warmup_epochs: 3
  weight_decay: 0.01
  lr: 0.001
  lr_decay: 0.1
  lr_step: 1
  lr_scheduler: "poly"
  clip_grad: null
  loss_fn: "JaccardLoss"
  transforms:
    image_transforms:
    mask_transforms:
    # Define image and mask transforms here.

val:
  path: "/data/val"
  batch_size: 3
  transforms:
    image_transforms:
    mask_transforms:
    # Define image and mask transforms here.

test:
  path: "/data/test"
  batch_size: 3
  transforms:
    image_transforms:
    mask_transforms:
    # Define image and mask transforms here.
```

## Usage

### Training

To train your model, use the following command:

```bash
python main.py --config config.yaml --mode train --output output_directory
```

You can customize the training process by modifying the parameters in the command, such as `--config`, `--output`, `--seed`, and `--save-interval`.

### Inference

To perform inference with a pre-trained model, use the following command:

```bash
python main.py --config config.yaml --mode eval --checkpoint path/to/pretrained_model.pth --output output_directory
```

Replace `path/to/pretrained_model.pth` with the path to the checkpoint file you want to use for inference.
