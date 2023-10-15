# weed-seg

Implementation of weed segmentation and stem detection...

## Configuration File (`config.yaml`) Guide

This guide explains how to fill out the `config.yaml` file to customize the behavior of your custom pipeline.

### 1. Configuration Template

Below is the template for the `config.yaml` file. Fill in the empty placeholders with your custom settings:

```yaml
model:
  name: "" # Name of the model class
  nc: # Number of channels
  input_size: # Input size of the model
  summary: # Set to true to print a detailed summary of the model

  layers:
    - name: ""
      type: ""
      in_features:
      out_features:
      # Add other layer configurations here...

  hyperparameters:
    learning_rate: 
    batch_size: 
    num_epochs: 
    optimizer: ""
    loss_fn: ""

data:
  train:
    path: ""
    transforms:
      # Define data transformations for training data

  valid:
    path: ""
    transforms:
      # Define data transformations for validation data

  test:
    path: ""
    transforms:
      # Define data transformations for test data
```

### 2. Instructions

Follow the instructions below to fill out the `config.yaml` file with your custom settings.

#### Model Settings

- `name`: Replace with the name of the model class you intend to use.
- `n_classes`: Define the number of classes your model should predict.
- `input_size`: Set the input size as a list of [channels, height, width].

#### Hyperparameters

- Adjust hyperparameters such as learning rate, batch size, number of epochs, optimizer, and loss function based on your project's requirements.

#### Data Settings

- Fill in the paths for your training, validation, and test datasets under the `train`, `valid`, and `test` sections.

#### Data Transformations

- Define data transformations under the `transforms` section for each dataset (train, valid, test). Follow the provided examples for data augmentation.


### 3. Example Configuration

For reference, here's an example of a filled-out `config.yaml` file:

```yaml
model:
  name: "CustomModel"
  n_classes: 10
  input_size: [3, 28, 28]
  summary: true

  layers:
    - name: "linear1"
      type: "Linear"
      in_features: 784
      out_features: 256

    - name: "relu1"
      type: "ReLU"

    - name: "dropout1"
      type: "Dropout"
      p: 0.5

    - name: "linear2"
      type: "Linear"
      in_features: 256
      out_features: 10

  hyperparameters:
    learning_rate: 0.001
    batch_size: 64
    num_epochs: 10
    optimizer: "Adam"
    loss_fn: "CrossEntropyLoss"

data:
  train:
    path: "data/train/"
    transforms:
      - RandomRotation:
          degrees: 10
      - RandomAffine:
          degrees: 10
          translate: [0.1, 0.1]
          scale: [0.9, 1.1]
          shear: 10
      - ToTensor: null
      - Normalize:
          mean: 0.5
          std: 0.5
```
