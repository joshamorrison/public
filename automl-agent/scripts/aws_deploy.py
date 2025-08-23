#!/usr/bin/env python3
"""
AWS Deployment Script for AutoML Agent Platform

Deploys the AutoML platform to AWS with:
- ECS Fargate for API containers
- Lambda for serverless agent execution
- SageMaker for model training/hosting
- S3 for data storage and model artifacts
- CloudWatch for monitoring
"""

import os
import sys
import json
import boto3
import argparse
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

class AutoMLAWSDeployment:
    """Complete AWS deployment for AutoML Agent Platform"""
    
    def __init__(self, region_name: str = 'us-east-1', environment: str = 'production'):
        self.region_name = region_name
        self.environment = environment
        self.session = boto3.Session(region_name=region_name)
        
        # AWS service clients
        self.s3 = self.session.client('s3')
        self.ecs = self.session.client('ecs')
        self.lambda_client = self.session.client('lambda')
        self.sagemaker = self.session.client('sagemaker')
        self.cloudformation = self.session.client('cloudformation')
        self.iam = self.session.client('iam')
        self.ecr = self.session.client('ecr')
        self.cloudwatch = self.session.client('logs')
        
        # Deployment configuration
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        self.config = {
            'project_name': 'automl-agent',
            'environment': environment,
            'timestamp': timestamp,
            's3_bucket': f'automl-agent-{environment}-{timestamp}',
            'ecr_repository': f'automl-agent-{environment}',
            'ecs_cluster': f'automl-cluster-{environment}',
            'cloudwatch_log_group': f'/aws/automl/{environment}',
            'stack_name': f'automl-platform-{environment}',
            'sagemaker_autopilot': {
                'job_name_prefix': f'automl-autopilot-{environment}',
                'execution_role': f'AutoMLExecutionRole-{environment}',
                'model_registry': f'automl-models-{environment}'
            },
            'sagemaker_endpoints': [
                'automl-inference-endpoint'
            ],
            'lambda_orchestrator': 'automl-orchestrator-function'  # Single orchestrator using Autopilot
        }
        
        print(f"🚀 AutoML AWS Deployment initialized")
        print(f"   Region: {region_name}")
        print(f"   Environment: {environment}")
        print(f"   S3 Bucket: {self.config['s3_bucket']}")
    
    def check_aws_credentials(self) -> bool:
        """Verify AWS credentials and permissions"""
        try:
            sts = self.session.client('sts')
            identity = sts.get_caller_identity()
            print(f"✅ AWS credentials verified")
            print(f"   Account: {identity['Account']}")
            print(f"   User/Role: {identity['Arn'].split('/')[-1]}")
            return True
        except Exception as e:
            print(f"❌ AWS credential error: {e}")
            return False
    
    def create_s3_infrastructure(self) -> bool:
        """Create S3 buckets for data and model storage"""
        try:
            bucket_name = self.config['s3_bucket']
            
            # Create main bucket
            if self.region_name != 'us-east-1':
                self.s3.create_bucket(
                    Bucket=bucket_name,
                    CreateBucketConfiguration={'LocationConstraint': self.region_name}
                )
            else:
                self.s3.create_bucket(Bucket=bucket_name)
            
            # Enable versioning
            self.s3.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            # Create folder structure
            folders = [
                'data/raw/',
                'data/processed/',
                'models/artifacts/',
                'models/trained/',
                'outputs/reports/',
                'outputs/visualizations/',
                'logs/agents/',
                'configs/'
            ]
            
            for folder in folders:
                self.s3.put_object(Bucket=bucket_name, Key=folder)
            
            print(f"✅ S3 infrastructure created: {bucket_name}")
            return True
            
        except Exception as e:
            print(f"❌ S3 setup failed: {e}")
            return False
    
    def create_ecr_repository(self) -> bool:
        """Create ECR repository for container images"""
        try:
            repo_name = self.config['ecr_repository']
            
            # Create repository
            response = self.ecr.create_repository(
                repositoryName=repo_name,
                imageScanningConfiguration={'scanOnPush': True},
                encryptionConfiguration={'encryptionType': 'AES256'}
            )
            
            repository_uri = response['repository']['repositoryUri']
            print(f"✅ ECR repository created: {repository_uri}")
            
            # Build and push Docker images
            self._build_and_push_images(repository_uri)
            
            return True
            
        except self.ecr.exceptions.RepositoryAlreadyExistsException:
            print(f"✅ ECR repository already exists: {repo_name}")
            return True
        except Exception as e:
            print(f"❌ ECR setup failed: {e}")
            return False
    
    def _build_and_push_images(self, repository_uri: str):
        """Build and push Docker images to ECR"""
        import subprocess
        
        try:
            # Get ECR login token
            token_response = self.ecr.get_authorization_token()
            token = token_response['authorizationData'][0]['authorizationToken']
            endpoint = token_response['authorizationData'][0]['proxyEndpoint']
            
            # Docker login to ECR
            subprocess.run([
                'aws', 'ecr', 'get-login-password', '--region', self.region_name
            ], check=True, capture_output=True, text=True)
            
            # Build API container
            print("🔨 Building FastAPI container...")
            subprocess.run([
                'docker', 'build', 
                '-t', f"{repository_uri}:api-latest",
                '-f', 'docker/api.Dockerfile', '.'
            ], check=True)
            
            # Build Streamlit container
            print("🔨 Building Streamlit container...")
            subprocess.run([
                'docker', 'build',
                '-t', f"{repository_uri}:streamlit-latest", 
                '-f', 'docker/streamlit.Dockerfile', '.'
            ], check=True)
            
            # Push images
            subprocess.run(['docker', 'push', f"{repository_uri}:api-latest"], check=True)
            subprocess.run(['docker', 'push', f"{repository_uri}:streamlit-latest"], check=True)
            
            print("✅ Docker images built and pushed to ECR")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Docker build/push failed: {e}")
        except Exception as e:
            print(f"❌ Image deployment error: {e}")
    
    def deploy_ecs_services(self) -> bool:
        """Deploy ECS Fargate services for API and Streamlit"""
        try:
            cluster_name = self.config['ecs_cluster']
            
            # Create ECS cluster
            self.ecs.create_cluster(
                clusterName=cluster_name,
                capacityProviders=['FARGATE'],
                defaultCapacityProviderStrategy=[
                    {'capacityProvider': 'FARGATE', 'weight': 1}
                ]
            )
            
            # Deploy task definitions and services
            self._deploy_api_service(cluster_name)
            self._deploy_streamlit_service(cluster_name)
            
            print(f"✅ ECS services deployed to cluster: {cluster_name}")
            return True
            
        except Exception as e:
            print(f"❌ ECS deployment failed: {e}")
            return False
    
    def deploy_lambda_agents(self) -> bool:
        """Deploy individual agents as Lambda functions"""
        try:
            for function_name in self.config['lambda_functions']:
                self._deploy_lambda_function(function_name)
            
            print(f"✅ Lambda agents deployed: {len(self.config['lambda_functions'])} functions")
            return True
            
        except Exception as e:
            print(f"❌ Lambda deployment failed: {e}")
            return False
    
    def setup_sagemaker_endpoints(self) -> bool:
        """Setup SageMaker endpoints for model training and inference"""
        try:
            # Create SageMaker execution role
            role_arn = self._create_sagemaker_role()
            
            # Deploy model endpoints
            for endpoint_name in self.config['sagemaker_endpoints']:
                self._deploy_sagemaker_endpoint(endpoint_name, role_arn)
            
            print(f"✅ SageMaker endpoints deployed")
            return True
            
        except Exception as e:
            print(f"❌ SageMaker setup failed: {e}")
            return False
    
    def setup_monitoring(self) -> bool:
        """Setup CloudWatch monitoring and logging"""
        try:
            log_group = self.config['cloudwatch_log_group']
            
            # Create log group
            self.cloudwatch.create_log_group(
                logGroupName=log_group,
                retentionInDays=30
            )
            
            # Create CloudWatch dashboards
            self._create_monitoring_dashboard()
            
            print(f"✅ CloudWatch monitoring configured")
            return True
            
        except self.cloudwatch.exceptions.ResourceAlreadyExistsException:
            print(f"✅ CloudWatch log group already exists")
            return True
        except Exception as e:
            print(f"❌ Monitoring setup failed: {e}")
            return False
    
    def deploy_infrastructure(self) -> bool:
        """Deploy complete infrastructure using CloudFormation"""
        try:
            template_path = Path(__file__).parent / 'cloudformation_template.yml'
            
            if template_path.exists():
                with open(template_path, 'r') as f:
                    template_body = f.read()
                
                self.cloudformation.create_stack(
                    StackName=self.config['stack_name'],
                    TemplateBody=template_body,
                    Parameters=[
                        {'ParameterKey': 'Environment', 'ParameterValue': self.environment},
                        {'ParameterKey': 'S3Bucket', 'ParameterValue': self.config['s3_bucket']}
                    ],
                    Capabilities=['CAPABILITY_IAM']
                )
                
                print(f"✅ CloudFormation stack deployment initiated")
            else:
                print("⚠️  CloudFormation template not found, using individual deployments")
            
            return True
            
        except Exception as e:
            print(f"❌ Infrastructure deployment failed: {e}")
            return False
    
    def run_deployment(self, components: List[str] = None) -> bool:
        """Run complete deployment process"""
        
        if components is None:
            components = ['all']
        
        print(f"🚀 Starting AutoML Platform AWS Deployment")
        print(f"   Components: {components}")
        print("=" * 60)
        
        # Check prerequisites
        if not self.check_aws_credentials():
            return False
        
        deployment_steps = {
            's3': ('S3 Infrastructure', self.create_s3_infrastructure),
            'ecr': ('Container Registry', self.create_ecr_repository),
            'ecs': ('ECS Services', self.deploy_ecs_services),
            'lambda': ('Lambda Agents', self.deploy_lambda_agents),
            'sagemaker': ('SageMaker Endpoints', self.setup_sagemaker_endpoints),
            'monitoring': ('CloudWatch Monitoring', self.setup_monitoring),
            'infrastructure': ('CloudFormation Stack', self.deploy_infrastructure)
        }
        
        success_count = 0
        total_count = 0
        
        for component, (description, deploy_func) in deployment_steps.items():
            if 'all' in components or component in components:
                total_count += 1
                print(f"\n📦 Deploying {description}...")
                
                if deploy_func():
                    success_count += 1
                    print(f"✅ {description} - SUCCESS")
                else:
                    print(f"❌ {description} - FAILED")
        
        # Deployment summary
        print(f"\n" + "=" * 60)
        print(f"🎯 Deployment Summary:")
        print(f"   Successful: {success_count}/{total_count}")
        print(f"   Environment: {self.environment}")
        print(f"   Region: {self.region_name}")
        
        if success_count == total_count:
            print(f"🎉 AutoML Platform deployment COMPLETED successfully!")
            self._print_access_urls()
        else:
            print(f"⚠️  Partial deployment - check failed components")
        
        return success_count == total_count
    
    def _print_access_urls(self):
        """Print access URLs and endpoints"""
        print(f"\n🌐 Access URLs:")
        print(f"   API Endpoint: https://api.automl.{self.region_name}.aws.example.com")
        print(f"   Streamlit UI: https://app.automl.{self.region_name}.aws.example.com")
        print(f"   CloudWatch: https://console.aws.amazon.com/cloudwatch/")
        print(f"   S3 Bucket: s3://{self.config['s3_bucket']}")

def main():
    parser = argparse.ArgumentParser(description='Deploy AutoML Platform to AWS')
    parser.add_argument('--region', default='us-east-1', help='AWS region')
    parser.add_argument('--environment', default='production', choices=['dev', 'staging', 'production'])
    parser.add_argument('--components', nargs='+', 
                       choices=['s3', 'ecr', 'ecs', 'lambda', 'sagemaker', 'monitoring', 'infrastructure', 'all'],
                       default=['all'], help='Components to deploy')
    parser.add_argument('--dry-run', action='store_true', help='Validate configuration without deploying')
    
    args = parser.parse_args()
    
    # Initialize deployment
    deployment = AutoMLAWSDeployment(
        region_name=args.region,
        environment=args.environment
    )
    
    if args.dry_run:
        print("🔍 Dry run mode - validating configuration...")
        if deployment.check_aws_credentials():
            print("✅ Configuration valid - ready for deployment")
        else:
            print("❌ Configuration issues found")
            sys.exit(1)
    else:
        # Run deployment
        success = deployment.run_deployment(args.components)
        sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()