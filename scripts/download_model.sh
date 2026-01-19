#!/bin/bash
# Download trained model from S3
#
# Usage:
#   ./scripts/download_model.sh [model_name]
#
# Default: best_model.pt

set -e

MODEL_NAME="${1:-best_model.pt}"
S3_BUCKET="${S3_BUCKET:-mesh-segmentor}"
LOCAL_DIR="${LOCAL_DIR:-./checkpoints}"

echo "Downloading model from S3..."
echo "  Bucket: $S3_BUCKET"
echo "  Model: $MODEL_NAME"
echo "  Local: $LOCAL_DIR/$MODEL_NAME"

mkdir -p "$LOCAL_DIR"

aws s3 cp "s3://${S3_BUCKET}/models/${MODEL_NAME}" "${LOCAL_DIR}/${MODEL_NAME}"

echo "Download complete!"
ls -lh "${LOCAL_DIR}/${MODEL_NAME}"
