#!/usr/bin/env python3
"""
Complete AWS Deployment Script for MMM Platform
Automated deployment of MMM models, data pipelines, and monitoring
"""

import os
import sys
import boto3
import json
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent.parent))

from data.media_data_client import MediaDataClient
from models.mmm.econometric_mmm import EconometricMMM
from infrastructure.aws.sagemaker_deployment import MMMSageMakerDeployment

class AWSMMDeployment:
    """Complete AWS deployment orchestration for MMM platform"""
    
    def __init__(self, region_name: str = 'us-east-1'):
        self.region_name = region_name
        self.session = boto3.Session(region_name=region_name)
        
        # Initialize deployment services
        self.sagemaker_deployer = MMMSageMakerDeployment(region_name=region_name)
        
        # Deployment configuration
        self.config = {
            'project_name': 'mmm-platform',
            'environment': 'production',
            'models_to_deploy': ['econometric_mmm', 'attribution_model'],
            's3_bucket': f'mmm-platform-{datetime.now().strftime("%Y%m%d")}',
            'cloudwatch_log_group': '/aws/mmm/platform'
        }
    
    def setup_aws_infrastructure(self) -> dict:
        """Set up required AWS infrastructure"""
        print("[AWS] Setting up infrastructure...")
        
        results = {
            's3_bucket': self._create_s3_bucket(),
            'cloudwatch_logs': self._setup_cloudwatch_logs(),
            'iam_roles': self._verify_iam_roles()
        }
        
        print(f"[AWS] Infrastructure setup completed: {results}")
        return results
    
    def _create_s3_bucket(self) -> str:
        """Create S3 bucket for model artifacts and data"""
        try:
            s3_client = self.session.client('s3')
            bucket_name = self.config['s3_bucket']
            
            s3_client.create_bucket(Bucket=bucket_name)
            
            # Enable versioning
            s3_client.put_bucket_versioning(
                Bucket=bucket_name,
                VersioningConfiguration={'Status': 'Enabled'}
            )
            
            print(f"[S3] Created bucket: {bucket_name}")
            return bucket_name
            
        except Exception as e:
            print(f"[S3] Bucket setup warning: {e}")
            return self.config['s3_bucket']
    
    def _setup_cloudwatch_logs(self) -> str:
        """Set up CloudWatch log group for monitoring"""
        try:
            logs_client = self.session.client('logs')
            log_group = self.config['cloudwatch_log_group']
            
            logs_client.create_log_group(logGroupName=log_group)
            
            # Set retention policy (30 days)
            logs_client.put_retention_policy(
                logGroupName=log_group,
                retentionInDays=30
            )
            
            print(f"[CLOUDWATCH] Created log group: {log_group}")
            return log_group
            
        except Exception as e:
            print(f"[CLOUDWATCH] Log setup warning: {e}")
            return self.config['cloudwatch_log_group']
    
    def _verify_iam_roles(self) -> dict:
        """Verify required IAM roles exist"""
        try:
            iam_client = self.session.client('iam')
            
            roles_status = {}
            required_roles = ['SageMakerExecutionRole', 'LambdaExecutionRole']
            
            for role_name in required_roles:
                try:
                    iam_client.get_role(RoleName=role_name)
                    roles_status[role_name] = 'exists'
                except iam_client.exceptions.NoSuchEntityException:
                    roles_status[role_name] = 'missing'
            
            print(f"[IAM] Roles status: {roles_status}")
            return roles_status
            
        except Exception as e:
            print(f"[IAM] Role verification warning: {e}")
            return {'status': 'unknown'}
    
    def train_and_deploy_models(self) -> dict:
        """Train MMM models and deploy to SageMaker"""
        print("[MODELS] Training and deploying MMM models...")
        
        deployment_results = {}
        
        # Get training data
        data_client = MediaDataClient()
        training_data, data_info, source_type = data_client.get_best_available_data()
        
        print(f"[DATA] Using {source_type} data: {len(training_data)} records")
        
        # Train econometric MMM model
        print("[TRAIN] Training econometric MMM...")
        mmm_model = EconometricMMM(
            adstock_rate=0.5,
            saturation_param=0.6,
            regularization_alpha=0.1
        )
        
        spend_columns = [col for col in training_data.columns if col.endswith('_spend')]
        
        mmm_results = mmm_model.fit(
            data=training_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=True
        )
        
        print(f"[TRAIN] Model trained - R²: {mmm_results['performance']['r2_score']:.3f}")
        
        # Deploy to SageMaker
        print("[DEPLOY] Deploying to SageMaker...")
        deployment_info = self.sagemaker_deployer.deploy_model(
            self.sagemaker_deployer.prepare_model_artifacts(mmm_model, 'econometric-mmm'),
            'econometric-mmm'
        )
        
        deployment_results['econometric_mmm'] = {
            'model_performance': mmm_results['performance'],
            'deployment_info': deployment_info,
            'training_data_size': len(training_data),
            'features_used': spend_columns
        }
        
        return deployment_results
    
    def setup_monitoring(self, deployment_results: dict) -> dict:
        """Set up CloudWatch monitoring for deployed models"""
        print("[MONITORING] Setting up model monitoring...")
        
        try:
            cloudwatch = self.session.client('cloudwatch')
            
            monitoring_config = {}
            
            for model_name, results in deployment_results.items():
                if 'deployment_info' in results and results['deployment_info']['status'] == 'deployed':
                    endpoint_name = results['deployment_info']['endpoint_name']
                    
                    # Create custom metrics dashboard
                    dashboard_body = {
                        "widgets": [
                            {
                                "type": "metric",
                                "properties": {
                                    "metrics": [
                                        ["AWS/SageMaker", "Invocations", "EndpointName", endpoint_name],
                                        [".", "ModelLatency", ".", "."],
                                        [".", "Invocation4XXErrors", ".", "."],
                                        [".", "Invocation5XXErrors", ".", "."]
                                    ],
                                    "period": 300,
                                    "stat": "Average",
                                    "region": self.region_name,
                                    "title": f"MMM Model Performance - {model_name}"
                                }
                            }
                        ]
                    }
                    
                    dashboard_name = f"MMM-{model_name}-Dashboard"
                    cloudwatch.put_dashboard(
                        DashboardName=dashboard_name,
                        DashboardBody=json.dumps(dashboard_body)
                    )
                    
                    # Set up alarms
                    cloudwatch.put_metric_alarm(
                        AlarmName=f"MMM-{model_name}-HighLatency",
                        ComparisonOperator='GreaterThanThreshold',
                        EvaluationPeriods=2,
                        MetricName='ModelLatency',
                        Namespace='AWS/SageMaker',
                        Period=300,
                        Statistic='Average',
                        Threshold=1000.0,  # 1 second
                        ActionsEnabled=True,
                        AlarmDescription=f'High latency alert for {model_name}',
                        Dimensions=[
                            {'Name': 'EndpointName', 'Value': endpoint_name}
                        ]
                    )
                    
                    monitoring_config[model_name] = {
                        'dashboard': dashboard_name,
                        'alarms': [f"MMM-{model_name}-HighLatency"],
                        'endpoint': endpoint_name
                    }
            
            print(f"[MONITORING] Monitoring setup completed: {list(monitoring_config.keys())}")
            return monitoring_config
            
        except Exception as e:
            print(f"[MONITORING] Setup warning: {e}")
            return {}
    
    def create_deployment_summary(self, 
                                infrastructure: dict,
                                deployments: dict, 
                                monitoring: dict) -> dict:
        """Create comprehensive deployment summary"""
        
        summary = {
            'deployment_info': {
                'timestamp': datetime.now().isoformat(),
                'region': self.region_name,
                'project': self.config['project_name'],
                'environment': self.config['environment']
            },
            'infrastructure': infrastructure,
            'model_deployments': deployments,
            'monitoring': monitoring,
            'status': 'completed',
            'endpoints': {}
        }
        
        # Extract endpoint URLs for easy access
        for model_name, results in deployments.items():
            if 'deployment_info' in results and results['deployment_info']['status'] == 'deployed':
                summary['endpoints'][model_name] = results['deployment_info']['endpoint_url']
        
        # Save deployment summary
        summary_path = Path('./outputs/aws_deployment_summary.json')
        summary_path.parent.mkdir(exist_ok=True)
        
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        
        print(f"[SUMMARY] Deployment summary saved to {summary_path}")
        return summary

def main():
    """Execute complete AWS deployment"""
    print("=" * 60)
    print("AWS MEDIA MIX MODELING PLATFORM DEPLOYMENT")
    print("=" * 60)
    
    try:
        # Initialize deployment
        aws_deployment = AWSMMDeployment()
        
        # Step 1: Setup infrastructure
        print("\n[STEP 1] Setting up AWS infrastructure...")
        infrastructure = aws_deployment.setup_aws_infrastructure()
        
        # Step 2: Train and deploy models
        print("\n[STEP 2] Training and deploying models...")
        deployments = aws_deployment.train_and_deploy_models()
        
        # Step 3: Setup monitoring
        print("\n[STEP 3] Setting up monitoring...")
        monitoring = aws_deployment.setup_monitoring(deployments)
        
        # Step 4: Create summary
        print("\n[STEP 4] Creating deployment summary...")
        summary = aws_deployment.create_deployment_summary(infrastructure, deployments, monitoring)
        
        print("\n" + "=" * 60)
        print("🚀 AWS DEPLOYMENT COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print("\n📊 DEPLOYED MODELS:")
        for model_name, endpoint_url in summary['endpoints'].items():
            print(f"  • {model_name}: {endpoint_url}")
        
        print(f"\n📈 MONITORING: {len(monitoring)} dashboards created")
        print(f"🗄️  STORAGE: {infrastructure['s3_bucket']}")
        print(f"📝 LOGS: {infrastructure['cloudwatch_logs']}")
        
        print("\n✅ Your MMM platform is now running on AWS!")
        
        return summary
        
    except Exception as e:
        print(f"\n❌ Deployment failed: {e}")
        return None

if __name__ == "__main__":
    main()