#!/bin/bash

# Vision Object Classifier - AWS Deployment Script
set -e

# Configuration
PROJECT_NAME="vision-classifier"
AWS_REGION="${AWS_REGION:-us-west-2}"
ENVIRONMENT="${ENVIRONMENT:-dev}"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +'%Y-%m-%d %H:%M:%S')] $1${NC}"
}

warn() {
    echo -e "${YELLOW}[$(date +'%Y-%m-%d %H:%M:%S')] WARNING: $1${NC}"
}

error() {
    echo -e "${RED}[$(date +'%Y-%m-%d %H:%M:%S')] ERROR: $1${NC}"
}

# Check prerequisites
check_prerequisites() {
    log "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        error "AWS CLI is not installed"
        exit 1
    fi
    
    # Check Docker
    if ! command -v docker &> /dev/null; then
        error "Docker is not installed"
        exit 1
    fi
    
    # Check Terraform
    if ! command -v terraform &> /dev/null; then
        error "Terraform is not installed"
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        error "AWS credentials not configured"
        exit 1
    fi
    
    log "Prerequisites check passed"
}

# Build and push Docker image
build_and_push_image() {
    log "Building and pushing Docker image..."
    
    # Get AWS account ID
    AWS_ACCOUNT_ID=$(aws sts get-caller-identity --query Account --output text)
    ECR_URI="${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${PROJECT_NAME}"
    
    # Login to ECR
    aws ecr get-login-password --region $AWS_REGION | docker login --username AWS --password-stdin $ECR_URI
    
    # Build image
    cd ../../  # Go back to project root
    docker build -f docker/Dockerfile -t $PROJECT_NAME:latest .
    
    # Tag image
    docker tag $PROJECT_NAME:latest $ECR_URI:latest
    docker tag $PROJECT_NAME:latest $ECR_URI:$(date +%Y%m%d-%H%M%S)
    
    # Push image
    docker push $ECR_URI:latest
    docker push $ECR_URI:$(date +%Y%m%d-%H%M%S)
    
    log "Docker image pushed to ECR: $ECR_URI"
    cd infrastructure/aws/  # Return to infrastructure directory
}

# Deploy infrastructure
deploy_infrastructure() {
    log "Deploying infrastructure with Terraform..."
    
    cd terraform/
    
    # Initialize Terraform
    terraform init
    
    # Plan deployment
    terraform plan \
        -var="aws_region=$AWS_REGION" \
        -var="environment=$ENVIRONMENT" \
        -var="project_name=$PROJECT_NAME" \
        -out=tfplan
    
    # Apply deployment
    terraform apply tfplan
    
    # Get outputs
    ALB_HOSTNAME=$(terraform output -raw alb_hostname)
    ECR_REPOSITORY_URL=$(terraform output -raw ecr_repository_url)
    
    log "Infrastructure deployed successfully"
    log "Load Balancer: $ALB_HOSTNAME"
    log "ECR Repository: $ECR_REPOSITORY_URL"
    
    cd ..
}

# Upload models to S3
upload_models() {
    log "Uploading models to S3..."
    
    cd terraform/
    MODELS_BUCKET=$(terraform output -raw s3_models_bucket)
    cd ..
    
    # Upload model files
    aws s3 sync ../../models/ s3://$MODELS_BUCKET/models/ \
        --exclude "*.txt" \
        --exclude "*.md" \
        --include "*.pth" \
        --include "*.json"
    
    log "Models uploaded to S3: s3://$MODELS_BUCKET"
}

# Update ECS service
update_service() {
    log "Updating ECS service..."
    
    cd terraform/
    CLUSTER_NAME=$(terraform output -raw ecs_cluster_name)
    SERVICE_NAME=$(terraform output -raw ecs_service_name)
    cd ..
    
    # Force new deployment
    aws ecs update-service \
        --cluster $CLUSTER_NAME \
        --service $SERVICE_NAME \
        --force-new-deployment \
        --region $AWS_REGION
    
    log "ECS service update initiated"
}

# Wait for deployment
wait_for_deployment() {
    log "Waiting for deployment to complete..."
    
    cd terraform/
    CLUSTER_NAME=$(terraform output -raw ecs_cluster_name)
    SERVICE_NAME=$(terraform output -raw ecs_service_name)
    ALB_HOSTNAME=$(terraform output -raw alb_hostname)
    cd ..
    
    # Wait for service stability
    aws ecs wait services-stable \
        --cluster $CLUSTER_NAME \
        --services $SERVICE_NAME \
        --region $AWS_REGION
    
    # Test health endpoint
    log "Testing health endpoint..."
    sleep 30  # Allow load balancer to register targets
    
    if curl -f "http://$ALB_HOSTNAME/health/status" &> /dev/null; then
        log "Health check passed"
    else
        warn "Health check failed - service may still be starting"
    fi
    
    log "Deployment completed"
    log "API URL: http://$ALB_HOSTNAME"
    log "Health Check: http://$ALB_HOSTNAME/health/status"
    log "API Docs: http://$ALB_HOSTNAME/docs"
}

# Main deployment function
main() {
    log "Starting AWS deployment for $PROJECT_NAME ($ENVIRONMENT)"
    
    check_prerequisites
    
    # Create ECR repository if it doesn't exist
    aws ecr describe-repositories --repository-names $PROJECT_NAME --region $AWS_REGION 2>/dev/null || \
    aws ecr create-repository --repository-name $PROJECT_NAME --region $AWS_REGION
    
    build_and_push_image
    deploy_infrastructure
    upload_models
    update_service
    wait_for_deployment
    
    log "Deployment completed successfully!"
}

# Cleanup function
cleanup() {
    log "Cleaning up resources..."
    
    cd terraform/
    terraform destroy \
        -var="aws_region=$AWS_REGION" \
        -var="environment=$ENVIRONMENT" \
        -var="project_name=$PROJECT_NAME" \
        -auto-approve
    
    log "Resources cleaned up"
}

# Handle command line arguments
case "${1:-deploy}" in
    deploy)
        main
        ;;
    cleanup|destroy)
        cleanup
        ;;
    *)
        echo "Usage: $0 [deploy|cleanup]"
        exit 1
        ;;
esac