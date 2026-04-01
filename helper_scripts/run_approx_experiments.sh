#!/bin/bash
# Script to run approximate multiplier experiments with 4 layers

set -e

# Configuration
EPOCHS=50
BATCH_SIZE=64
NUM_LAYERS=4
D_MODEL=256
NUM_HEADS=4

# List of approximate multipliers to test
# Format: multiplier_name:bits
MULTIPLIERS=("mbm:7" "mbm:3" "mit:7")

echo "=========================================="
echo "Step 1: Compiling Custom Operations"
echo "=========================================="
echo "Compiling for Approximate Multiplier Simulator..."
make clean
make convam MULTIPLIER=AMSIMULATOR
make denseam_gpu.so MULTIPLIER=AMSIMULATOR
make matmulam_gpu.so MULTIPLIER=AMSIMULATOR
echo "✓ Compilation complete"
echo ""

echo "=========================================="
echo "Step 2: Running Experiments"
echo "=========================================="

for item in "${MULTIPLIERS[@]}"; do
    IFS=":" read -r type bits <<< "$item"
    experiment_name="${type}_${bits}_l${NUM_LAYERS}"
    multiplier_flag="${type}_${bits}"
    
    echo "Running experiment: $experiment_name"
    
    python3 train_transformer.py \
        --multiplier "$multiplier_flag" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --num_layers "$NUM_LAYERS" \
        --d_model "$D_MODEL" \
        --num_heads "$NUM_HEADS" \
        --experiment_name "$experiment_name"
    
    echo "✓ Finished $experiment_name"
    echo "--------------------------------------------------"
done

echo "All experiments completed!"
