#!/bin/bash
# ECR Setup Script
# Creates ECR repository and pushes Docker image
#
# Prerequisites:
# - AWS CLI configured
# - Docker installed and running
# - Logged in to Docker
#
# Usage:
#   ./ecr-setup.sh

set -e

# Configuration
AWS_REGION="${AWS_REGION:-us-east-1}"
AWS_ACCOUNT_ID="${AWS_ACCOUNT_ID:-$(aws sts get-caller-identity --query Account --output text)}"
ECR_REPOSITORY="mesh-segmentor"
IMAGE_TAG="${IMAGE_TAG:-latest}"

echo "=========================================="
echo "ECR Setup for Mesh Segmentor"
echo "=========================================="
echo "Region: $AWS_REGION"
echo "Account: $AWS_ACCOUNT_ID"
echo "Repository: $ECR_REPOSITORY"
echo "Tag: $IMAGE_TAG"
echo "=========================================="

# Create ECR repository if it doesn't exist
echo "Creating ECR repository..."
aws ecr describe-repositories --repository-names $ECR_REPOSITORY --region $AWS_REGION 2>/dev/null || \
aws ecr create-repository \
    --repository-name $ECR_REPOSITORY \
    --region $AWS_REGION \
    --image-scanning-configuration scanOnPush=true \
    --encryption-configuration encryptionType=AES256

# Get ECR login token
echo "Logging in to ECR..."
aws ecr get-login-password --region $AWS_REGION | \
    docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com

# Build Docker image
echo "Building Docker image..."
cd "$(dirname "$0")/.."
docker build \
    -f docker/Dockerfile.inference \
    -t $ECR_REPOSITORY:$IMAGE_TAG \
    .

# Tag for ECR
ECR_URI="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/$ECR_REPOSITORY:$IMAGE_TAG"
docker tag $ECR_REPOSITORY:$IMAGE_TAG $ECR_URI

# Push to ECR
echo "Pushing image to ECR..."
docker push $ECR_URI

echo "=========================================="
echo "ECR Setup Complete!"
echo "=========================================="
echo "Image URI: $ECR_URI"
echo ""
echo "Next steps:"
echo "1. Update ECS task definition with image URI"
echo "2. Deploy to ECS cluster"
