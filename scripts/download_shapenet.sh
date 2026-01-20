#!/bin/bash
# Download ShapeNetPart dataset for pipeline validation
#
# OPTIONS:
#   1. Kaggle (recommended) - requires kaggle CLI
#   2. Manual download from Kaggle website
#
# Usage:
#   ./scripts/download_shapenet.sh

set -e

DATA_DIR="${DATA_DIR:-data/shapenet}"

echo "=========================================="
echo "Downloading ShapeNetPart Dataset"
echo "=========================================="

mkdir -p "$DATA_DIR"
cd "$DATA_DIR"

# Check if already downloaded
if [ -d "hdf5_data" ] && [ "$(ls -A hdf5_data 2>/dev/null)" ]; then
    echo "Dataset already exists at $DATA_DIR/hdf5_data/"
    ls -la hdf5_data/
    exit 0
fi

# Try Kaggle CLI first
if command -v kaggle &> /dev/null; then
    echo "Using Kaggle CLI to download..."
    kaggle datasets download -d majdouline20/shapenetpart-dataset -p .

    # Extract
    echo "Extracting..."
    unzip -q shapenetpart-dataset.zip

    # The Kaggle dataset may have different structure, check and reorganize
    if [ -d "shapenetcore_partanno_segmentation_benchmark_v0_normal" ]; then
        mv shapenetcore_partanno_segmentation_benchmark_v0_normal hdf5_data
    fi

    echo "Download complete!"
else
    echo ""
    echo "Kaggle CLI not found. Please download manually:"
    echo ""
    echo "OPTION 1 - Install Kaggle CLI:"
    echo "  pip install kaggle"
    echo "  # Set up ~/.kaggle/kaggle.json with your API key"
    echo "  # Get key from: https://www.kaggle.com/settings -> API -> Create New Token"
    echo "  ./scripts/download_shapenet.sh"
    echo ""
    echo "OPTION 2 - Manual Download:"
    echo "  1. Go to: https://www.kaggle.com/datasets/majdouline20/shapenetpart-dataset"
    echo "  2. Click 'Download' (requires free Kaggle account)"
    echo "  3. Extract to: $DATA_DIR/"
    echo "  4. Rename extracted folder to 'hdf5_data' if needed"
    echo ""
    echo "OPTION 3 - Use HuggingFace synthetic data instead:"
    echo "  python scripts/generate_synthetic_data.py"
    echo ""
    exit 1
fi

echo ""
echo "=========================================="
echo "Download Complete!"
echo "=========================================="
echo "Location: $DATA_DIR/"
ls -la
echo ""
echo "Next step:"
echo "  python scripts/adapt_shapenet.py"
echo "=========================================="
