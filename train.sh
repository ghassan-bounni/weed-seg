#!/bin/bash

source /home/ubuntu/anaconda3/etc/profile.d/conda.sh # change this path to your anaconda3 path
conda activate your_env_name

export WANDB_API_KEY=your_wandb_api_key
export WANDB_PROJECT=your_wandb_project_name # optional, default is "weed-seg"
export WANDB_NAME=your_wandb_run_name # optional
export WANDB_NOTES=your_wandb_run_notes # optional

# Replace the following with your own training command.
# refer to README.md for more details, or try python main.py --help

python main.py \
  --mode train \
  --config config.yaml \
  --data data/ \
  --output output/ \
  --seed 42 \
  --save-interval 1