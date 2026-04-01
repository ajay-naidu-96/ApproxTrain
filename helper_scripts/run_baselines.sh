#!/bin/bash
# Automated script to run both FP32 and approximate multiplier baselines

set -e  # Exit on error

echo "=========================================="
echo "Transformer Baseline Training Script"
echo "=========================================="
echo ""

# Configuration
EPOCHS=50
BATCH_SIZE=64
D_MODEL=256
NUM_LAYERS=4
NUM_HEADS=4

# Parse arguments
MULTIPLIERS=("fp32" "mbm_7" "mbm_5" "mbm_3")
RUN_COMPARISON=true

while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --batch_size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --multipliers)
            IFS=',' read -ra MULTIPLIERS <<< "$2"
            shift 2
            ;;
        --no-comparison)
            RUN_COMPARISON=false
            shift
            ;;
        *)
            echo "Unknown option: $1"
            exit 1
            ;;
    esac
done

echo "Configuration:"
echo "  Epochs: $EPOCHS"
echo "  Batch size: $BATCH_SIZE"
echo "  Model: ${D_MODEL}d, ${NUM_LAYERS} layers, ${NUM_HEADS} heads"
echo "  Multipliers: ${MULTIPLIERS[@]}"
echo ""

# Step 1: Compile custom ops
echo "=========================================="
echo "Step 1: Compiling custom operations"
echo "=========================================="
echo ""

echo "Compiling with FP32..."
make clean && make convam MULTIPLIER=FP32 && make denseam_gpu.so MULTIPLIER=FP32
echo "✓ FP32 compilation complete"
echo ""

echo "Compiling with AM Simulator..."
make clean && make convam MULTIPLIER=AMSIMULATOR && make denseam_gpu.so MULTIPLIER=AMSIMULATOR
echo "✓ AM Simulator compilation complete"
echo ""

# Step 2: Generate LUT files
echo "=========================================="
echo "Step 2: Generating LUT files"
echo "=========================================="
echo ""

cd lut
./lut_gen.sh
cd ..
echo "✓ LUT files generated"
echo ""

# Step 3: Train baselines
echo "=========================================="
echo "Step 3: Training baselines"
echo "=========================================="
echo ""

LOG_DIRS=()
NAMES=()

for multiplier in "${MULTIPLIERS[@]}"; do
    echo "Training with $multiplier..."
    echo "----------------------------------------"
    
    python3 train_transformer.py \
        --multiplier "$multiplier" \
        --epochs "$EPOCHS" \
        --batch_size "$BATCH_SIZE" \
        --d_model "$D_MODEL" \
        --num_layers "$NUM_LAYERS" \
        --num_heads "$NUM_HEADS" \
        --experiment_name "${multiplier}_baseline"
    
    LOG_DIRS+=("logs/${multiplier}_baseline")
    NAMES+=("$multiplier")
    
    echo "✓ $multiplier training complete"
    echo ""
done

# Step 4: Evaluate models
echo "=========================================="
echo "Step 4: Evaluating models"
echo "=========================================="
echo ""

for multiplier in "${MULTIPLIERS[@]}"; do
    echo "Evaluating $multiplier..."
    
    python3 evaluate_model.py \
        --checkpoint_dir "checkpoints/${multiplier}_baseline" \
        --generate_samples \
        --num_samples 3
    
    echo "✓ $multiplier evaluation complete"
    echo ""
done

# Step 5: Benchmark models
echo "=========================================="
echo "Step 5: Benchmarking models"
echo "=========================================="
echo ""

for multiplier in "${MULTIPLIERS[@]}"; do
    echo "Benchmarking $multiplier..."
    
    python3 benchmark.py \
        --checkpoint_dir "checkpoints/${multiplier}_baseline" \
        --output "benchmark_${multiplier}.json"
    
    echo "✓ $multiplier benchmark complete"
    echo ""
done

# Step 6: Compare baselines
if [ "$RUN_COMPARISON" = true ]; then
    echo "=========================================="
    echo "Step 6: Comparing baselines"
    echo "=========================================="
    echo ""
    
    python3 compare_baselines.py \
        --log_dirs "${LOG_DIRS[@]}" \
        --names "${NAMES[@]}" \
        --output_dir comparison
    
    echo "✓ Comparison complete"
    echo ""
fi

echo "=========================================="
echo "All baselines complete!"
echo "=========================================="
echo ""
echo "Results:"
for multiplier in "${MULTIPLIERS[@]}"; do
    echo "  - $multiplier: logs/${multiplier}_baseline/"
done
echo ""
if [ "$RUN_COMPARISON" = true ]; then
    echo "Comparison report: comparison/comparison_report.md"
fi
echo ""
