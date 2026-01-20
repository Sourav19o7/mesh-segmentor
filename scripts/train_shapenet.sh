#!/bin/bash
# Quick training script for ShapeNet pipeline validation
#
# Usage:
#   ./scripts/train_shapenet.sh
#
# This uses smaller settings for faster iteration

set -e

echo "=========================================="
echo "Training on ShapeNetPart Dataset"
echo "=========================================="

# Check if data exists
if [ ! -d "data/processed/train" ] || [ -z "$(ls -A data/processed/train 2>/dev/null)" ]; then
    echo "ERROR: Training data not found!"
    echo ""
    echo "Run these commands first:"
    echo "  ./scripts/download_shapenet.sh"
    echo "  python scripts/adapt_shapenet.py"
    exit 1
fi

# Count samples
TRAIN_COUNT=$(ls data/processed/train/*.npy 2>/dev/null | wc -l)
VAL_COUNT=$(ls data/processed/val/*.npy 2>/dev/null | wc -l)

echo "Training samples: $TRAIN_COUNT"
echo "Validation samples: $VAL_COUNT"
echo ""

# Run training with ShapeNet-appropriate settings
python -m training.train \
    --train-data data/processed/train \
    --train-labels data/labels/train \
    --val-data data/processed/val \
    --val-labels data/labels/val \
    --epochs 50 \
    --batch-size 16 \
    --num-points 2048 \
    --lr 0.001 \
    --model-size small \
    "$@"

echo ""
echo "=========================================="
echo "Training Complete!"
echo "=========================================="
echo ""
echo "Check results:"
echo "  - Best model: checkpoints/best_model.pt"
echo "  - Test inference: python scripts/test_inference.py"
