#!/bin/bash
# EC2 g5.xlarge Setup Script for Training
#
# This script sets up an EC2 g5.xlarge instance for training
# the Point Transformer model.
#
# Prerequisites:
# - Launch EC2 g5.xlarge with Deep Learning AMI (Ubuntu 20.04)
# - At least 100GB EBS storage
# - IAM role with S3 access attached
#
# Usage:
#   chmod +x setup_ec2.sh
#   ./setup_ec2.sh

set -e

echo "=========================================="
echo "Setting up EC2 g5.xlarge for training"
echo "=========================================="

# Update system
echo "Updating system packages..."
sudo apt-get update -y
sudo apt-get upgrade -y

# Install system dependencies
echo "Installing system dependencies..."
sudo apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    curl \
    htop \
    tmux \
    vim \
    unzip \
    awscli

# Activate conda environment (Deep Learning AMI has conda pre-installed)
echo "Setting up conda environment..."
source /opt/conda/etc/profile.d/conda.sh

# Create new environment for mesh-segmentor
conda create -n mesh-segmentor python=3.10 -y
conda activate mesh-segmentor

# Install PyTorch with CUDA 11.8
echo "Installing PyTorch with CUDA support..."
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118

# Install pytorch3d (requires compilation)
echo "Installing pytorch3d..."
pip install fvcore iopath
pip install --no-index --no-cache-dir pytorch3d -f https://dl.fbaipublicfiles.com/pytorch3d/packaging/wheels/py310_cu118_pyt210/download.html

# Install other dependencies
echo "Installing project dependencies..."
pip install \
    rhino3dm==8.4.0 \
    trimesh==4.0.5 \
    numpy==1.24.3 \
    scipy==1.11.3 \
    pyyaml==6.0.1 \
    python-json-logger==2.0.7 \
    boto3==1.33.6

# Verify CUDA installation
echo "Verifying CUDA installation..."
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Verify pytorch3d
echo "Verifying pytorch3d installation..."
python -c "from pytorch3d.ops import sample_points_from_meshes; print('pytorch3d OK')"

# Clone repository (if not already present)
echo "Setting up project..."
cd ~
if [ ! -d "mesh-segmentor" ]; then
    echo "Cloning repository..."
    # Replace with your actual repository URL
    git clone https://github.com/YOUR_ORG/mesh-segmentor.git
fi
cd mesh-segmentor

# Create data directories
mkdir -p data/processed/train data/processed/val data/labels/train data/labels/val
mkdir -p checkpoints logs

# Configure AWS CLI
echo "Configuring AWS region..."
aws configure set region us-east-1

echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Sync data from S3:"
echo "   aws s3 sync s3://mesh-segmentor/datasets/train ./data/processed/train"
echo "   aws s3 sync s3://mesh-segmentor/datasets/labels/train ./data/labels/train"
echo "   aws s3 sync s3://mesh-segmentor/datasets/val ./data/processed/val"
echo "   aws s3 sync s3://mesh-segmentor/datasets/labels/val ./data/labels/val"
echo ""
echo "2. Start training:"
echo "   conda activate mesh-segmentor"
echo "   ./scripts/train.sh"
echo ""
echo "3. Or run training directly:"
echo "   python -m training.train --epochs 100 --batch-size 8 --s3-bucket mesh-segmentor"
