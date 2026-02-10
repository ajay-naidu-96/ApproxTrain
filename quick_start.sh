#!/bin/bash
# Quick start script for transformer baseline training

set -e

echo "=========================================="
echo "Quick Start: Transformer Baseline Training"
echo "=========================================="
echo ""

# Test mode - fast iteration for debugging
echo "Running quick test (1 epoch, small batch)..."
python3 train_transformer.py \
    --multiplier fp32 \
    --epochs 1 \
    --batch_size 32 \
    --test_mode \
    --experiment_name test_run
