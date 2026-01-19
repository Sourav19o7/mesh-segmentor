#!/bin/bash
# Training launcher script
#
# Usage:
#   ./scripts/train.sh                    # Default training
#   ./scripts/train.sh --epochs 50        # Custom epochs
#   ./scripts/train.sh --resume checkpoint.pt  # Resume training

set -e

# Configuration
S3_BUCKET="${S3_BUCKET:-mesh-segmentor}"
CHECKPOINT_DIR="${CHECKPOINT_DIR:-./checkpoints}"
LOG_DIR="${LOG_DIR:-./logs}"
NUM_GPUS="${NUM_GPUS:-1}"

# Create directories
mkdir -p "$CHECKPOINT_DIR" "$LOG_DIR"

# Get timestamp for log file
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="$LOG_DIR/training_${TIMESTAMP}.log"

echo "=========================================="
echo "Starting Training"
echo "=========================================="
echo "S3 Bucket: $S3_BUCKET"
echo "Checkpoint Dir: $CHECKPOINT_DIR"
echo "Log File: $LOG_FILE"
echo "=========================================="

# Check GPU availability
python -c "import torch; print(f'GPUs available: {torch.cuda.device_count()}')"

# Sync data from S3 if not present
if [ ! -d "./data/processed/train" ] || [ -z "$(ls -A ./data/processed/train 2>/dev/null)" ]; then
    echo "Syncing training data from S3..."
    aws s3 sync "s3://${S3_BUCKET}/datasets/train" ./data/processed/train
    aws s3 sync "s3://${S3_BUCKET}/datasets/labels/train" ./data/labels/train
fi

if [ ! -d "./data/processed/val" ] || [ -z "$(ls -A ./data/processed/val 2>/dev/null)" ]; then
    echo "Syncing validation data from S3..."
    aws s3 sync "s3://${S3_BUCKET}/datasets/val" ./data/processed/val
    aws s3 sync "s3://${S3_BUCKET}/datasets/labels/val" ./data/labels/val
fi

# Run training
echo "Starting training..."

if [ "$NUM_GPUS" -gt 1 ]; then
    # Multi-GPU training with distributed data parallel
    python -m torch.distributed.launch \
        --nproc_per_node=$NUM_GPUS \
        -m training.train \
        --train-data ./data/processed/train \
        --train-labels ./data/labels/train \
        --val-data ./data/processed/val \
        --val-labels ./data/labels/val \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --s3-bucket "$S3_BUCKET" \
        --epochs 100 \
        --batch-size 8 \
        --num-points 20000 \
        --model-size base \
        --amp \
        "$@" 2>&1 | tee "$LOG_FILE"
else
    # Single GPU training
    python -m training.train \
        --train-data ./data/processed/train \
        --train-labels ./data/labels/train \
        --val-data ./data/processed/val \
        --val-labels ./data/labels/val \
        --checkpoint-dir "$CHECKPOINT_DIR" \
        --s3-bucket "$S3_BUCKET" \
        --epochs 100 \
        --batch-size 8 \
        --num-points 20000 \
        --model-size base \
        --amp \
        "$@" 2>&1 | tee "$LOG_FILE"
fi

# Upload final log to S3
echo "Uploading training log to S3..."
aws s3 cp "$LOG_FILE" "s3://${S3_BUCKET}/logs/training_${TIMESTAMP}.log"

echo "=========================================="
echo "Training complete!"
echo "=========================================="
