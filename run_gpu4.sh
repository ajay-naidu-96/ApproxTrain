#!/usr/bin/env bash
# run_gpu4.sh
export CUDA_VISIBLE_DEVICES=4
python3 train_transformer.py --multiplier pos5e1_mit --experiment_name pos5e1_mit_$(date +%Y%m%d_%H%M%S)
python3 train_transformer.py --multiplier pos5e2_mit --experiment_name pos5e2_mit_$(date +%Y%m%d_%H%M%S)
