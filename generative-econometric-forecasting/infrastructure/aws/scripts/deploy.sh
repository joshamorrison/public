#!/bin/bash
#
# AWS Deployment Script for Generative Econometric Forecasting Platform
# Usage: ./deploy.sh [environment] [region]
#

set -e  # Exit on error

# Default values
ENVIRONMENT=${1:-dev}
AWS_REGION=${2:-us-east-1}
STACK_NAME="econometric-forecasting-${ENVIRONMENT}"
TEMPLATE_FILE="infrastructure/aws/cloudformation/main.yaml"
LAMBDA_CODE_DIR="infrastructure/aws/lambda"

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check prerequisites
check_prerequisites() {
    log_info "Checking prerequisites..."
    
    # Check AWS CLI
    if ! command -v aws &> /dev/null; then
        log_error "AWS CLI not found. Please install AWS CLI."
        exit 1
    fi
    
    # Check AWS credentials
    if ! aws sts get-caller-identity &> /dev/null; then
        log_error "AWS credentials not configured. Please run 'aws configure'."
        exit 1
    fi
    
    # Check if CloudFormation template exists
    if [ ! -f "$TEMPLATE_FILE" ]; then
        log_error "CloudFormation template not found: $TEMPLATE_FILE"
        exit 1
    fi
    
    log_success "Prerequisites check passed"
}

# Validate CloudFormation template
validate_template() {
    log_info "Validating CloudFormation template..."
    
    if aws cloudformation validate-template \
        --template-body file://$TEMPLATE_FILE \
        --region $AWS_REGION > /dev/null; then
        log_success "Template validation passed"
    else
        log_error "Template validation failed"
        exit 1
    fi
}

# Package Lambda functions
package_lambda() {
    log_info "Packaging Lambda functions..."
    
    # Create Lambda deployment packages
    mkdir -p build/lambda
    
    # Package forecasting handler
    cd $LAMBDA_CODE_DIR
    zip -r ../../build/lambda/forecasting-handler.zip . -x "*.pyc" "__pycache__/*"
    cd - > /dev/null
    
    log_success "Lambda functions packaged"
}

# Create S3 bucket for deployment artifacts
create_deployment_bucket() {
    local bucket_name="econometric-forecasting-deployment-$(aws sts get-caller-identity --query Account --output text)"
    
    log_info "Creating deployment bucket: $bucket_name"
    
    # Check if bucket exists
    if aws s3api head-bucket --bucket "$bucket_name" 2>/dev/null; then
        log_info "Bucket already exists: $bucket_name"
    else
        # Create bucket
        if [ "$AWS_REGION" = "us-east-1" ]; then
            aws s3api create-bucket --bucket "$bucket_name" --region $AWS_REGION
        else
            aws s3api create-bucket \
                --bucket "$bucket_name" \
                --region $AWS_REGION \
                --create-bucket-configuration LocationConstraint=$AWS_REGION
        fi
        
        # Enable versioning
        aws s3api put-bucket-versioning \
            --bucket "$bucket_name" \
            --versioning-configuration Status=Enabled
        
        log_success "Created deployment bucket: $bucket_name"
    fi
    
    echo $bucket_name
}

# Upload Lambda code to S3
upload_lambda_code() {
    local deployment_bucket=$1
    
    log_info "Uploading Lambda code to S3..."
    
    # Upload Lambda package
    aws s3 cp build/lambda/forecasting-handler.zip \
        s3://$deployment_bucket/lambda/forecasting-handler.zip
    
    log_success "Lambda code uploaded"
}

# Deploy CloudFormation stack
deploy_stack() {
    local deployment_bucket=$1
    
    log_info "Deploying CloudFormation stack: $STACK_NAME"
    
    # Get default key pair (if exists)
    local key_pairs=$(aws ec2 describe-key-pairs --query 'KeyPairs[0].KeyName' --output text 2>/dev/null || echo "NONE")
    
    if [ "$key_pairs" = "NONE" ]; then
        log_warning "No EC2 key pairs found. Creating default key pair..."
        
        # Create key pair
        aws ec2 create-key-pair \
            --key-name "econometric-forecasting-${ENVIRONMENT}" \
            --query 'KeyMaterial' \
            --output text > ~/.ssh/econometric-forecasting-${ENVIRONMENT}.pem
        
        chmod 600 ~/.ssh/econometric-forecasting-${ENVIRONMENT}.pem
        key_pairs="econometric-forecasting-${ENVIRONMENT}"
        
        log_success "Created key pair: $key_pairs"
    fi
    
    # Deploy stack
    aws cloudformation deploy \
        --template-file $TEMPLATE_FILE \
        --stack-name $STACK_NAME \
        --parameter-overrides \
            Environment=$ENVIRONMENT \
            KeyPairName=$key_pairs \
        --capabilities CAPABILITY_IAM CAPABILITY_NAMED_IAM \
        --region $AWS_REGION \
        --tags \
            Environment=$ENVIRONMENT \
            Project=econometric-forecasting \
            DeployedBy=$(whoami) \
            DeployedAt=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    
    if [ $? -eq 0 ]; then
        log_success "Stack deployment completed successfully"
    else
        log_error "Stack deployment failed"
        exit 1
    fi
}

# Update Lambda function code
update_lambda_code() {
    local deployment_bucket=$1
    
    log_info "Updating Lambda function code..."
    
    # Get Lambda function names from stack outputs
    local forecasting_function=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query "Stacks[0].Outputs[?OutputKey=='ForecastingLambdaArn'].OutputValue" \
        --output text | cut -d':' -f7)
    
    local data_processing_function=$(aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query "Stacks[0].Outputs[?OutputKey=='DataProcessingLambdaArn'].OutputValue" \
        --output text | cut -d':' -f7)
    
    # Update function code
    if [ ! -z "$forecasting_function" ]; then
        aws lambda update-function-code \
            --function-name $forecasting_function \
            --s3-bucket $deployment_bucket \
            --s3-key lambda/forecasting-handler.zip
        
        log_success "Updated Lambda function: $forecasting_function"
    fi
}

# Display stack outputs
display_outputs() {
    log_info "Retrieving stack outputs..."
    
    # Get stack outputs
    aws cloudformation describe-stacks \
        --stack-name $STACK_NAME \
        --query 'Stacks[0].Outputs[*].[OutputKey,OutputValue]' \
        --output table
    
    log_success "Deployment completed successfully!"
    log_info "Access your resources using the outputs above"
}

# Cleanup function
cleanup() {
    log_info "Cleaning up temporary files..."
    rm -rf build/
}

# Main deployment flow
main() {
    log_info "Starting deployment for environment: $ENVIRONMENT in region: $AWS_REGION"
    
    # Run deployment steps
    check_prerequisites
    validate_template
    package_lambda
    
    # Create deployment bucket
    DEPLOYMENT_BUCKET=$(create_deployment_bucket)
    
    # Upload artifacts
    upload_lambda_code $DEPLOYMENT_BUCKET
    
    # Deploy infrastructure
    deploy_stack $DEPLOYMENT_BUCKET
    
    # Update Lambda code
    update_lambda_code $DEPLOYMENT_BUCKET
    
    # Display results
    display_outputs
    
    # Cleanup
    cleanup
    
    log_success "Deployment completed successfully!"
    
    # Show next steps
    echo
    log_info "Next steps:"
    echo "1. Update API keys in AWS Secrets Manager"
    echo "2. Configure Airflow on the EC2 instance"
    echo "3. Set up monitoring and alerting"
    echo "4. Test the forecasting endpoints"
}

# Handle script interruption
trap cleanup EXIT

# Check if script is being sourced or executed
if [[ "${BASH_SOURCE[0]}" == "${0}" ]]; then
    main "$@"
fi