#!/bin/bash
# Deployment script for Mesh Segmentor
#
# Usage:
#   ./scripts/deploy.sh [environment]
#
# Environments: dev, staging, prod

set -e

ENVIRONMENT="${1:-prod}"
AWS_REGION="${AWS_REGION:-us-east-1}"
STACK_NAME="mesh-segmentor-${ENVIRONMENT}"
S3_BUCKET="mesh-segmentor-${ENVIRONMENT}"

echo "=========================================="
echo "Deploying Mesh Segmentor"
echo "=========================================="
echo "Environment: $ENVIRONMENT"
echo "Region: $AWS_REGION"
echo "Stack: $STACK_NAME"
echo "=========================================="

# Get AWS Account ID
AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)

# Step 1: Build and push Docker image
echo "Step 1: Building and pushing Docker image..."
./aws/ecr-setup.sh

# Step 2: Deploy CloudFormation stack
echo "Step 2: Deploying infrastructure..."
aws cloudformation deploy \
    --template-file aws/cloudformation.yaml \
    --stack-name $STACK_NAME \
    --parameter-overrides \
        EnvironmentName=$STACK_NAME \
        S3BucketName=$S3_BUCKET \
        MinCapacity=1 \
        MaxCapacity=4 \
        DesiredCapacity=1 \
    --capabilities CAPABILITY_NAMED_IAM \
    --region $AWS_REGION

# Step 3: Get stack outputs
echo "Step 3: Retrieving stack outputs..."
ALB_DNS=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='LoadBalancerDNS'].OutputValue" \
    --output text \
    --region $AWS_REGION)

CLUSTER_NAME=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='ClusterName'].OutputValue" \
    --output text \
    --region $AWS_REGION)

TASK_EXEC_ROLE=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='TaskExecutionRoleArn'].OutputValue" \
    --output text \
    --region $AWS_REGION)

TASK_ROLE=$(aws cloudformation describe-stacks \
    --stack-name $STACK_NAME \
    --query "Stacks[0].Outputs[?OutputKey=='TaskRoleArn'].OutputValue" \
    --output text \
    --region $AWS_REGION)

# Step 4: Update and register ECS task definition
echo "Step 4: Registering ECS task definition..."
ECR_IMAGE="$AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com/mesh-segmentor:latest"

# Create task definition from template
cat aws/ecs-task-definition.json | \
    sed "s|ACCOUNT_ID|$AWS_ACCOUNT_ID|g" | \
    sed "s|arn:aws:iam::ACCOUNT_ID:role/ecsTaskExecutionRole|$TASK_EXEC_ROLE|g" | \
    sed "s|arn:aws:iam::ACCOUNT_ID:role/mesh-segmentor-task-role|$TASK_ROLE|g" | \
    sed "s|mesh-segmentor|$S3_BUCKET|g" > /tmp/task-definition.json

aws ecs register-task-definition \
    --cli-input-json file:///tmp/task-definition.json \
    --region $AWS_REGION

# Step 5: Create/Update ECS service
echo "Step 5: Deploying ECS service..."

# Check if service exists
SERVICE_EXISTS=$(aws ecs describe-services \
    --cluster $CLUSTER_NAME \
    --services mesh-segmentor-service \
    --query "services[?status=='ACTIVE'].serviceName" \
    --output text \
    --region $AWS_REGION 2>/dev/null || echo "")

if [ -z "$SERVICE_EXISTS" ]; then
    echo "Creating new ECS service..."

    # Get target group ARN
    TG_ARN=$(aws elbv2 describe-target-groups \
        --names "${STACK_NAME}-tg" \
        --query "TargetGroups[0].TargetGroupArn" \
        --output text \
        --region $AWS_REGION)

    # Get subnet IDs
    SUBNET_IDS=$(aws ec2 describe-subnets \
        --filters "Name=tag:Name,Values=${STACK_NAME}-public-*" \
        --query "Subnets[].SubnetId" \
        --output text \
        --region $AWS_REGION | tr '\t' ',')

    # Get security group ID
    SG_ID=$(aws ec2 describe-security-groups \
        --filters "Name=tag:Name,Values=${STACK_NAME}-ecs-sg" \
        --query "SecurityGroups[0].GroupId" \
        --output text \
        --region $AWS_REGION)

    aws ecs create-service \
        --cluster $CLUSTER_NAME \
        --service-name mesh-segmentor-service \
        --task-definition mesh-segmentor-inference \
        --desired-count 1 \
        --launch-type EC2 \
        --network-configuration "awsvpcConfiguration={subnets=[$SUBNET_IDS],securityGroups=[$SG_ID],assignPublicIp=ENABLED}" \
        --load-balancers "targetGroupArn=$TG_ARN,containerName=mesh-segmentor-inference,containerPort=8000" \
        --region $AWS_REGION
else
    echo "Updating existing ECS service..."
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service mesh-segmentor-service \
        --task-definition mesh-segmentor-inference \
        --force-new-deployment \
        --region $AWS_REGION
fi

# Step 6: Wait for deployment
echo "Step 6: Waiting for deployment to stabilize..."
aws ecs wait services-stable \
    --cluster $CLUSTER_NAME \
    --services mesh-segmentor-service \
    --region $AWS_REGION

echo "=========================================="
echo "Deployment Complete!"
echo "=========================================="
echo "API URL: http://$ALB_DNS"
echo "Health Check: http://$ALB_DNS/api/v1/health"
echo "API Docs: http://$ALB_DNS/docs"
echo "=========================================="
