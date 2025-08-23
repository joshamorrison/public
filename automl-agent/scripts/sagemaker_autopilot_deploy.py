#!/usr/bin/env python3
"""
SageMaker Autopilot Integration for AutoML Agent Platform

This script sets up SageMaker Autopilot integration for the AutoML platform:
- Creates Autopilot jobs for automatic model training
- Sets up model registry for trained models
- Deploys inference endpoints
- Integrates with the agent platform for intelligent routing
"""

import os
import sys
import json
import boto3
import time
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class SageMakerAutopilotIntegration:
    """SageMaker Autopilot integration for AutoML Agent Platform"""
    
    def __init__(self, region_name: str = 'us-east-1', environment: str = 'production'):
        self.region_name = region_name
        self.environment = environment
        self.session = boto3.Session(region_name=region_name)
        
        # AWS service clients
        self.sagemaker = self.session.client('sagemaker')
        self.s3 = self.session.client('s3')
        self.iam = self.session.client('iam')
        self.lambda_client = self.session.client('lambda')
        
        # Configuration
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.config = {
            'project_name': 'automl-agent-autopilot',
            'environment': environment,
            'timestamp': timestamp,
            's3_bucket': f'automl-autopilot-{environment}-{timestamp}',
            'execution_role_name': f'AutoML-Autopilot-ExecutionRole-{environment}',
            'model_registry_name': f'AutoML-Models-{environment}',
            'autopilot_job_prefix': f'automl-autopilot-{environment}',
            'inference_endpoint_name': f'automl-inference-{environment}',
            'lambda_orchestrator_name': f'automl-autopilot-orchestrator-{environment}'
        }
        
        print(f"🤖 SageMaker Autopilot Integration initialized")
        print(f"   Region: {region_name}")
        print(f"   Environment: {environment}")
        print(f"   S3 Bucket: {self.config['s3_bucket']}")
    
    def setup_iam_roles(self) -> str:
        """Create IAM execution role for SageMaker Autopilot"""
        role_name = self.config['execution_role_name']
        
        # Trust policy for SageMaker
        trust_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Principal": {"Service": "sagemaker.amazonaws.com"},
                    "Action": "sts:AssumeRole"
                }
            ]
        }
        
        # Execution policy
        execution_policy = {
            "Version": "2012-10-17",
            "Statement": [
                {
                    "Effect": "Allow",
                    "Action": [
                        "s3:GetObject",
                        "s3:PutObject",
                        "s3:DeleteObject",
                        "s3:ListBucket"
                    ],
                    "Resource": [
                        f"arn:aws:s3:::{self.config['s3_bucket']}",
                        f"arn:aws:s3:::{self.config['s3_bucket']}/*"
                    ]
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "sagemaker:CreateModel",
                        "sagemaker:CreateEndpointConfig",
                        "sagemaker:CreateEndpoint",
                        "sagemaker:DescribeModel",
                        "sagemaker:DescribeEndpointConfig",
                        "sagemaker:DescribeEndpoint",
                        "sagemaker:CreateModelPackage",
                        "sagemaker:DescribeModelPackage",
                        "sagemaker:UpdateModelPackage"
                    ],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "ecr:BatchCheckLayerAvailability",
                        "ecr:GetDownloadUrlForLayer",
                        "ecr:BatchGetImage"
                    ],
                    "Resource": "*"
                },
                {
                    "Effect": "Allow",
                    "Action": [
                        "cloudwatch:PutMetricData",
                        "logs:CreateLogGroup",
                        "logs:CreateLogStream",
                        "logs:DescribeLogStreams",
                        "logs:PutLogEvents"
                    ],
                    "Resource": "*"
                }
            ]
        }
        
        try:
            # Create IAM role
            role_response = self.iam.create_role(
                RoleName=role_name,
                AssumeRolePolicyDocument=json.dumps(trust_policy),
                Description="Execution role for AutoML SageMaker Autopilot"
            )
            
            role_arn = role_response['Role']['Arn']
            
            # Attach custom execution policy
            self.iam.put_role_policy(
                RoleName=role_name,
                PolicyName=f"{role_name}-ExecutionPolicy",
                PolicyDocument=json.dumps(execution_policy)
            )
            
            # Attach AWS managed policy for SageMaker
            self.iam.attach_role_policy(
                RoleName=role_name,
                PolicyArn='arn:aws:iam::aws:policy/AmazonSageMakerFullAccess'
            )
            
            print(f"✅ IAM execution role created: {role_name}")
            return role_arn
            
        except self.iam.exceptions.EntityAlreadyExistsException:
            # Role already exists, get ARN
            role_response = self.iam.get_role(RoleName=role_name)
            role_arn = role_response['Role']['Arn']
            print(f"✅ IAM execution role already exists: {role_name}")
            return role_arn
            
        except Exception as e:
            print(f"❌ Failed to create IAM role: {e}")
            raise
    
    def setup_s3_infrastructure(self) -> bool:
        """Create S3 bucket structure for Autopilot"""
        try:
            bucket_name = self.config['s3_bucket']
            
            # Create bucket
            if self.region_name != 'us-east-1':
                self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region_name}
                )
            else:
                self.s3.create_bucket(Bucket=bucket_name)
            
            # Create folder structure for Autopilot
            folders = [
                'input-data/',
                'autopilot-jobs/',
                'model-artifacts/',
                'inference-data/',
                'batch-transform/',
                'logs/',
                'reports/'
            ]
            
            for folder in folders:
                self.s3.put_object(Bucket=bucket_name, Key=folder)
            
            print(f"✅ S3 infrastructure created: {bucket_name}")
            return True
            
        except Exception as e:
            print(f"❌ S3 setup failed: {e}")
            return False
    
    def create_model_registry(self) -> bool:
        """Create SageMaker Model Registry for trained models"""
        try:
            registry_name = self.config['model_registry_name']
            
            # Create model package group (registry)
            self.sagemaker.create_model_package_group(
                ModelPackageGroupName=registry_name,
                ModelPackageGroupDescription="AutoML Agent Platform trained models"
            )
            
            print(f"✅ Model registry created: {registry_name}")
            return True
            
        except self.sagemaker.exceptions.ValidationException as e:
            if "already exists" in str(e):
                print(f"✅ Model registry already exists: {registry_name}")
                return True
            else:
                print(f"❌ Model registry creation failed: {e}")
                return False
        except Exception as e:
            print(f"❌ Model registry creation failed: {e}")
            return False
    
    def create_autopilot_job_function(self, role_arn: str) -> bool:
        """Create Lambda function to manage Autopilot jobs"""
        
        lambda_code = f'''
import json
import boto3
from datetime import datetime

sagemaker = boto3.client('sagemaker')
s3 = boto3.client('s3')

def lambda_handler(event, context):
    """
    Lambda function to create and manage SageMaker Autopilot jobs
    """
    
    try:
        # Extract task information from event
        task_description = event.get('task_description', 'AutoML Task')
        s3_input_path = event.get('s3_input_path')
        target_column = event.get('target_column')
        task_type = event.get('task_type', 'BinaryClassification')  # or Regression, MulticlassClassification
        
        if not s3_input_path or not target_column:
            return {{
                'statusCode': 400,
                'body': json.dumps('Missing required parameters: s3_input_path, target_column')
            }}
        
        # Generate unique job name
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        job_name = f"{self.config['autopilot_job_prefix']}-{{timestamp}}"
        
        # Autopilot job configuration
        autopilot_config = {{
            'AutoMLJobName': job_name,
            'InputDataConfig': [{{
                'DataSource': {{
                    'S3DataSource': {{
                        'S3DataType': 'S3Prefix',
                        'S3Uri': s3_input_path,
                        'S3DataDistributionType': 'FullyReplicated'
                    }}
                }},
                'TargetAttributeName': target_column
            }}],
            'OutputDataConfig': {{
                'S3OutputPath': f's3://{self.config["s3_bucket"]}/autopilot-jobs/{{job_name}}'
            }},
            'RoleArn': '{role_arn}',
            'AutoMLJobObjective': {{
                'MetricName': 'Accuracy' if task_type != 'Regression' else 'MSE'
            }},
            'ProblemType': task_type,
            'AutoMLJobConfig': {{
                'CompletionCriteria': {{
                    'MaxCandidates': 10,
                    'MaxRuntimePerTrainingJobInSeconds': 3600,
                    'MaxAutoMLJobRuntimeInSeconds': 7200
                }}
            }}
        }}
        
        # Create Autopilot job
        response = sagemaker.create_auto_ml_job(**autopilot_config)
        
        return {{
            'statusCode': 200,
            'body': json.dumps({{
                'job_name': job_name,
                'job_arn': response['AutoMLJobArn'],
                'status': 'InProgress',
                'message': f'Autopilot job {{job_name}} created successfully'
            }})
        }}
        
    except Exception as e:
        return {{
            'statusCode': 500,
            'body': json.dumps(f'Error creating Autopilot job: {{str(e)}}')
        }}
'''
        
        try:
            # Create Lambda function
            function_name = self.config['lambda_orchestrator_name']
            
            response = self.lambda_client.create_function(
                FunctionName=function_name,
                Runtime='python3.9',
                Role=role_arn,
                Handler='index.lambda_handler',
                Code={'ZipFile': lambda_code.encode()},
                Description='AutoML Agent Platform - SageMaker Autopilot orchestrator',
                Timeout=300,
                Environment={
                    'Variables': {
                        'S3_BUCKET': self.config['s3_bucket'],
                        'MODEL_REGISTRY': self.config['model_registry_name']
                    }
                }
            )
            
            print(f"✅ Lambda orchestrator created: {function_name}")
            return True
            
        except self.lambda_client.exceptions.ResourceConflictException:
            print(f"✅ Lambda orchestrator already exists: {function_name}")
            return True
        except Exception as e:
            print(f"❌ Lambda orchestrator creation failed: {e}")
            return False
    
    def create_autopilot_job(self, 
                            input_s3_path: str, 
                            target_column: str, 
                            problem_type: str = 'BinaryClassification') -> str:
        """Create a SageMaker Autopilot job"""
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
            job_name = f"{self.config['autopilot_job_prefix']}-{timestamp}"
            role_arn = f"arn:aws:iam::{self.session.client('sts').get_caller_identity()['Account']}:role/{self.config['execution_role_name']}"
            
            # Autopilot job configuration
            job_config = {
                'AutoMLJobName': job_name,
                'InputDataConfig': [{
                    'DataSource': {
                        'S3DataSource': {
                            'S3DataType': 'S3Prefix',
                            'S3Uri': input_s3_path,
                            'S3DataDistributionType': 'FullyReplicated'
                        }
                    },
                    'TargetAttributeName': target_column
                }],
                'OutputDataConfig': {
                    'S3OutputPath': f's3://{self.config["s3_bucket"]}/autopilot-jobs/{job_name}'
                },
                'RoleArn': role_arn,
                'ProblemType': problem_type,
                'AutoMLJobObjective': {
                    'MetricName': 'Accuracy' if problem_type != 'Regression' else 'MSE'
                },
                'AutoMLJobConfig': {
                    'CompletionCriteria': {
                        'MaxCandidates': 20,
                        'MaxRuntimePerTrainingJobInSeconds': 3600,  # 1 hour per training job
                        'MaxAutoMLJobRuntimeInSeconds': 7200        # 2 hours total
                    }
                }
            }
            
            # Create the job
            response = self.sagemaker.create_auto_ml_job(**job_config)
            
            print(f"✅ Autopilot job created: {job_name}")
            print(f"   Job ARN: {response['AutoMLJobArn']}")
            print(f"   Problem Type: {problem_type}")
            print(f"   Target Column: {target_column}")
            
            return job_name
            
        except Exception as e:
            print(f"❌ Failed to create Autopilot job: {e}")
            raise
    
    def monitor_autopilot_job(self, job_name: str) -> Dict[str, Any]:
        """Monitor Autopilot job progress"""
        
        try:
            response = self.sagemaker.describe_auto_ml_job(AutoMLJobName=job_name)
            
            status = response['AutoMLJobStatus']
            
            result = {
                'job_name': job_name,
                'status': status,
                'created_at': response.get('CreationTime'),
                'last_modified': response.get('LastModifiedTime')
            }
            
            if status == 'Completed':
                result.update({
                    'best_candidate': response.get('BestCandidate', {}),
                    'end_time': response.get('EndTime'),
                    'output_location': response.get('OutputDataConfig', {}).get('S3OutputPath')
                })
                
            elif status == 'Failed':
                result['failure_reason'] = response.get('FailureReason')
                
            elif status == 'InProgress':
                result['progress'] = f"Generating candidates... ({response.get('AutoMLJobSecondaryStatus', 'Unknown')})"
            
            return result
            
        except Exception as e:
            return {
                'job_name': job_name,
                'status': 'Error',
                'error': str(e)
            }
    
    def deploy_best_model(self, job_name: str) -> Optional[str]:
        """Deploy the best model from Autopilot job to inference endpoint"""
        
        try:
            # Get job details
            job_response = self.sagemaker.describe_auto_ml_job(AutoMLJobName=job_name)
            
            if job_response['AutoMLJobStatus'] != 'Completed':
                print(f"❌ Job {job_name} not completed yet")
                return None
            
            best_candidate = job_response['BestCandidate']
            
            # Create model
            model_name = f"{job_name}-model"
            inference_containers = best_candidate['InferenceContainers']
            
            self.sagemaker.create_model(
                ModelName=model_name,
                PrimaryContainer=inference_containers[0],
                ExecutionRoleArn=f"arn:aws:iam::{self.session.client('sts').get_caller_identity()['Account']}:role/{self.config['execution_role_name']}"
            )
            
            # Create endpoint configuration
            endpoint_config_name = f"{job_name}-endpoint-config"
            self.sagemaker.create_endpoint_config(
                EndpointConfigName=endpoint_config_name,
                ProductionVariants=[{
                    'VariantName': 'primary',
                    'ModelName': model_name,
                    'InitialInstanceCount': 1,
                    'InstanceType': 'ml.t2.medium',
                    'InitialVariantWeight': 1
                }]
            )
            
            # Create endpoint
            endpoint_name = self.config['inference_endpoint_name']
            self.sagemaker.create_endpoint(
                EndpointName=endpoint_name,
                EndpointConfigName=endpoint_config_name
            )
            
            print(f"✅ Model deployment initiated: {endpoint_name}")
            print(f"   Model: {model_name}")
            print(f"   Best candidate score: {best_candidate.get('FinalAutoMLJobObjectiveMetric', {}).get('Value', 'N/A')}")
            
            return endpoint_name
            
        except Exception as e:
            print(f"❌ Model deployment failed: {e}")
            return None
    
    def run_full_deployment(self) -> bool:
        """Run complete SageMaker Autopilot deployment"""
        
        print(f"🚀 Starting SageMaker Autopilot deployment")
        print("=" * 60)
        
        try:
            # 1. Setup IAM roles
            print("📋 Setting up IAM execution roles...")
            role_arn = self.setup_iam_roles()
            
            # 2. Setup S3 infrastructure  
            print("📦 Setting up S3 infrastructure...")
            if not self.setup_s3_infrastructure():
                return False
            
            # 3. Create model registry
            print("📚 Creating model registry...")
            if not self.create_model_registry():
                return False
            
            # 4. Create Lambda orchestrator
            print("⚡ Creating Lambda orchestrator...")
            if not self.create_autopilot_job_function(role_arn):
                return False
            
            print(f"\n" + "=" * 60)
            print(f"✅ SageMaker Autopilot deployment completed!")
            print(f"\n🎯 Next Steps:")
            print(f"1. Upload training data to: s3://{self.config['s3_bucket']}/input-data/")
            print(f"2. Create Autopilot job using the Lambda function: {self.config['lambda_orchestrator_name']}")
            print(f"3. Monitor job progress in SageMaker console")
            print(f"4. Deploy best model to inference endpoint")
            
            return True
            
        except Exception as e:
            print(f"❌ Deployment failed: {e}")
            return False

def main():
    parser = argparse.ArgumentParser(description='Deploy SageMaker Autopilot for AutoML Platform')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--environment', default='production', choices=['dev', 'staging', 'production'])
    parser.add_argument('--create-job', help='Create Autopilot job with S3 input path')
    parser.add_argument('--target-column', help='Target column name for ML task')
    parser.add_argument('--problem-type', default='BinaryClassification', 
                       choices=['BinaryClassification', 'MulticlassClassification', 'Regression'])
    parser.add_argument('--monitor-job', help='Monitor existing Autopilot job')
    parser.add_argument('--deploy-model', help='Deploy best model from completed job')
    
    args = parser.parse_args()
    
    # Initialize Autopilot integration
    autopilot = SageMakerAutopilotIntegration(
        region_name=args.region,
        environment=args.environment
    )
    
    if args.create_job:
        if not args.target_column:
            print("❌ --target-column required when creating job")
            sys.exit(1)
        job_name = autopilot.create_autopilot_job(args.create_job, args.target_column, args.problem_type)
        print(f"🎯 Monitor job: python {__file__} --monitor-job {job_name}")
        
    elif args.monitor_job:
        status = autopilot.monitor_autopilot_job(args.monitor_job)
        print(f"📊 Job Status: {json.dumps(status, indent=2, default=str)}")
        
    elif args.deploy_model:
        endpoint = autopilot.deploy_best_model(args.deploy_model)
        if endpoint:
            print(f"🎯 Model deployed to endpoint: {endpoint}")
            
    else:
        # Run full deployment
        success = autopilot.run_full_deployment()
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()