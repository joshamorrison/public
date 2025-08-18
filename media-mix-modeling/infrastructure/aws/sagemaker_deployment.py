#!/usr/bin/env python3
"""
AWS SageMaker Deployment for Media Mix Modeling
Production-ready model serving with auto-scaling and monitoring
"""

import os
import json
import boto3
import sagemaker
import pickle
import joblib
from datetime import datetime
from typing import Dict, Any, Optional
from sagemaker.pytorch import PyTorchModel
from sagemaker.sklearn import SKLearnModel
from sagemaker import get_execution_role
from pathlib import Path

class MMMSageMakerDeployment:
    """Deploy MMM models to AWS SageMaker for production serving"""
    
    def __init__(self, 
                 region_name: str = 'us-east-1',
                 role_arn: Optional[str] = None,
                 bucket_name: Optional[str] = None):
        """
        Initialize SageMaker deployment client
        
        Args:
            region_name: AWS region for deployment
            role_arn: SageMaker execution role ARN
            bucket_name: S3 bucket for model artifacts
        """
        self.region_name = region_name
        self.session = boto3.Session(region_name=region_name)
        self.sagemaker_session = sagemaker.Session(boto_session=self.session)
        
        # Set up execution role
        try:
            self.role = role_arn or get_execution_role()
        except:
            # Fallback role for local testing
            self.role = f"arn:aws:iam::123456789012:role/SageMakerExecutionRole"
            
        # S3 bucket for model artifacts
        self.bucket_name = bucket_name or f"mmm-models-{datetime.now().strftime('%Y%m%d')}"
        
        # Model configuration
        self.model_config = {
            'instance_type': 'ml.m5.large',
            'instance_count': 1,
            'max_capacity': 5,
            'target_value': 70.0  # Target CPU utilization for auto-scaling
        }
        
    def prepare_model_artifacts(self, mmm_model, model_name: str) -> str:
        """
        Prepare model artifacts for SageMaker deployment
        
        Args:
            mmm_model: Trained MMM model instance
            model_name: Name for the model deployment
            
        Returns:
            S3 URI of uploaded model artifacts
        """
        print(f"[SAGEMAKER] Preparing model artifacts for {model_name}...")
        
        # Create local artifacts directory
        artifacts_dir = Path(f"./artifacts/{model_name}")
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model using joblib (compatible with SageMaker)
        model_path = artifacts_dir / "model.joblib"
        joblib.dump(mmm_model, model_path)
        
        # Create model metadata
        metadata = {
            'model_name': model_name,
            'model_type': 'econometric_mmm',
            'created_at': datetime.now().isoformat(),
            'version': '1.0.0',
            'parameters': {
                'adstock_rate': getattr(mmm_model, 'adstock_rate', 0.5),
                'saturation_param': getattr(mmm_model, 'saturation_param', 0.6),
                'regularization_alpha': getattr(mmm_model, 'regularization_alpha', 0.1)
            },
            'input_features': getattr(mmm_model, 'feature_names_', []),
            'target_column': getattr(mmm_model, 'target_column', 'revenue')
        }
        
        metadata_path = artifacts_dir / "metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
            
        # Create inference script
        inference_script = '''
import os
import json
import joblib
import pandas as pd
import numpy as np
from io import StringIO

def model_fn(model_dir):
    """Load model from model directory"""
    model_path = os.path.join(model_dir, "model.joblib")
    model = joblib.load(model_path)
    
    # Load metadata
    metadata_path = os.path.join(model_dir, "metadata.json")
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    
    return {'model': model, 'metadata': metadata}

def input_fn(request_body, request_content_type):
    """Parse input data for inference"""
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        
        # Convert to DataFrame
        if 'data' in input_data:
            df = pd.DataFrame(input_data['data'])
        else:
            df = pd.DataFrame([input_data])
            
        return df
    
    elif request_content_type == 'text/csv':
        return pd.read_csv(StringIO(request_body))
    
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")

def predict_fn(input_data, model_artifacts):
    """Make predictions using the loaded model"""
    model = model_artifacts['model']
    metadata = model_artifacts['metadata']
    
    try:
        # Make predictions
        if hasattr(model, 'predict'):
            predictions = model.predict(input_data)
        else:
            # Fallback for custom MMM models
            predictions = model.forecast(input_data)
        
        # Format results
        results = {
            'predictions': predictions.tolist() if hasattr(predictions, 'tolist') else predictions,
            'model_metadata': metadata,
            'timestamp': pd.Timestamp.now().isoformat()
        }
        
        return results
        
    except Exception as e:
        return {
            'error': str(e),
            'model_metadata': metadata,
            'timestamp': pd.Timestamp.now().isoformat()
        }

def output_fn(predictions, response_content_type):
    """Format model output"""
    if response_content_type == 'application/json':
        return json.dumps(predictions)
    else:
        raise ValueError(f"Unsupported response content type: {response_content_type}")
'''
        
        inference_path = artifacts_dir / "inference.py"
        with open(inference_path, 'w') as f:
            f.write(inference_script)
            
        # Create requirements.txt for SageMaker
        requirements = '''
pandas>=1.5.0
numpy>=1.21.0
scikit-learn>=1.1.0
joblib>=1.2.0
scipy>=1.9.0
'''
        
        requirements_path = artifacts_dir / "requirements.txt"
        with open(requirements_path, 'w') as f:
            f.write(requirements)
            
        print(f"[SAGEMAKER] Model artifacts prepared at {artifacts_dir}")
        
        # Upload to S3
        s3_uri = self._upload_artifacts_to_s3(artifacts_dir, model_name)
        return s3_uri
    
    def _upload_artifacts_to_s3(self, artifacts_dir: Path, model_name: str) -> str:
        """Upload model artifacts to S3"""
        print(f"[S3] Uploading artifacts to S3...")
        
        try:
            # Create S3 client
            s3_client = self.session.client('s3')
            
            # Create bucket if it doesn't exist
            try:
                s3_client.create_bucket(Bucket=self.bucket_name)
                print(f"[S3] Created bucket: {self.bucket_name}")
            except s3_client.exceptions.BucketAlreadyExists:
                print(f"[S3] Using existing bucket: {self.bucket_name}")
            except Exception as e:
                print(f"[S3] Bucket creation warning: {e}")
            
            # Upload all files
            s3_prefix = f"models/{model_name}"
            
            for file_path in artifacts_dir.rglob('*'):
                if file_path.is_file():
                    s3_key = f"{s3_prefix}/{file_path.name}"
                    s3_client.upload_file(str(file_path), self.bucket_name, s3_key)
                    print(f"[S3] Uploaded {file_path.name}")
            
            s3_uri = f"s3://{self.bucket_name}/{s3_prefix}/"
            print(f"[S3] All artifacts uploaded to {s3_uri}")
            return s3_uri
            
        except Exception as e:
            print(f"[S3] Upload failed: {e}")
            # Return local path as fallback
            return str(artifacts_dir)
    
    def deploy_model(self, 
                    s3_model_uri: str, 
                    model_name: str,
                    endpoint_name: Optional[str] = None) -> Dict[str, Any]:
        """
        Deploy model to SageMaker endpoint
        
        Args:
            s3_model_uri: S3 URI of model artifacts
            model_name: Name of the model
            endpoint_name: Custom endpoint name
            
        Returns:
            Deployment configuration and endpoint details
        """
        print(f"[SAGEMAKER] Deploying model {model_name} to SageMaker...")
        
        endpoint_name = endpoint_name or f"mmm-{model_name}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        
        try:
            # Create SageMaker model
            sklearn_model = SKLearnModel(
                model_data=s3_model_uri,
                role=self.role,
                entry_point='inference.py',
                framework_version='1.0-1',
                py_version='py3',
                sagemaker_session=self.sagemaker_session
            )
            
            # Deploy to endpoint
            predictor = sklearn_model.deploy(
                initial_instance_count=self.model_config['instance_count'],
                instance_type=self.model_config['instance_type'],
                endpoint_name=endpoint_name
            )
            
            # Set up auto-scaling
            self._setup_autoscaling(endpoint_name)
            
            deployment_info = {
                'endpoint_name': endpoint_name,
                'endpoint_url': f"https://runtime.sagemaker.{self.region_name}.amazonaws.com/endpoints/{endpoint_name}/invocations",
                'model_name': model_name,
                'instance_type': self.model_config['instance_type'],
                'status': 'deployed',
                'created_at': datetime.now().isoformat()
            }
            
            print(f"[SAGEMAKER] Model deployed successfully!")
            print(f"[ENDPOINT] {endpoint_name}")
            
            return deployment_info
            
        except Exception as e:
            print(f"[SAGEMAKER] Deployment failed: {e}")
            return {
                'endpoint_name': endpoint_name,
                'model_name': model_name,
                'status': 'failed',
                'error': str(e),
                'created_at': datetime.now().isoformat()
            }
    
    def _setup_autoscaling(self, endpoint_name: str):
        """Set up auto-scaling for the endpoint"""
        try:
            autoscaling_client = self.session.client('application-autoscaling')
            
            # Register scalable target
            autoscaling_client.register_scalable_target(
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                MinCapacity=1,
                MaxCapacity=self.model_config['max_capacity']
            )
            
            # Create scaling policy
            autoscaling_client.put_scaling_policy(
                PolicyName=f'{endpoint_name}-scaling-policy',
                ServiceNamespace='sagemaker',
                ResourceId=f'endpoint/{endpoint_name}/variant/AllTraffic',
                ScalableDimension='sagemaker:variant:DesiredInstanceCount',
                PolicyType='TargetTrackingScaling',
                TargetTrackingScalingPolicyConfiguration={
                    'TargetValue': self.model_config['target_value'],
                    'PredefinedMetricSpecification': {
                        'PredefinedMetricType': 'SageMakerVariantInvocationsPerInstance'
                    },
                    'ScaleOutCooldown': 300,
                    'ScaleInCooldown': 300
                }
            )
            
            print(f"[AUTOSCALING] Configured for endpoint {endpoint_name}")
            
        except Exception as e:
            print(f"[AUTOSCALING] Setup warning: {e}")
    
    def test_endpoint(self, endpoint_name: str, test_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Test deployed endpoint with sample data
        
        Args:
            endpoint_name: Name of the deployed endpoint
            test_data: Sample data for testing
            
        Returns:
            Test results and predictions
        """
        print(f"[TEST] Testing endpoint {endpoint_name}...")
        
        try:
            # Create SageMaker runtime client
            runtime_client = self.session.client('sagemaker-runtime')
            
            # Prepare test payload
            payload = json.dumps(test_data)
            
            # Invoke endpoint
            response = runtime_client.invoke_endpoint(
                EndpointName=endpoint_name,
                ContentType='application/json',
                Body=payload
            )
            
            # Parse response
            result = json.loads(response['Body'].read().decode())
            
            test_results = {
                'endpoint_name': endpoint_name,
                'test_status': 'success',
                'response_time_ms': response['ResponseMetadata'].get('HTTPHeaders', {}).get('x-amzn-requestid', 'unknown'),
                'predictions': result,
                'test_data': test_data,
                'timestamp': datetime.now().isoformat()
            }
            
            print(f"[TEST] Endpoint test successful!")
            return test_results
            
        except Exception as e:
            print(f"[TEST] Endpoint test failed: {e}")
            return {
                'endpoint_name': endpoint_name,
                'test_status': 'failed',
                'error': str(e),
                'timestamp': datetime.now().isoformat()
            }
    
    def cleanup_endpoint(self, endpoint_name: str) -> bool:
        """
        Clean up SageMaker endpoint and associated resources
        
        Args:
            endpoint_name: Name of endpoint to delete
            
        Returns:
            True if cleanup successful
        """
        print(f"[CLEANUP] Deleting endpoint {endpoint_name}...")
        
        try:
            sagemaker_client = self.session.client('sagemaker')
            
            # Delete endpoint
            sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
            
            # Delete endpoint configuration
            sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
            
            print(f"[CLEANUP] Endpoint {endpoint_name} deleted successfully")
            return True
            
        except Exception as e:
            print(f"[CLEANUP] Failed to delete endpoint: {e}")
            return False

def deploy_mmm_to_sagemaker(mmm_model, 
                           model_name: str = "econometric-mmm",
                           region_name: str = "us-east-1") -> Dict[str, Any]:
    """
    Convenience function to deploy MMM model to SageMaker
    
    Args:
        mmm_model: Trained MMM model
        model_name: Name for the deployment
        region_name: AWS region
        
    Returns:
        Deployment information
    """
    print(f"[DEPLOY] Starting SageMaker deployment for {model_name}...")
    
    # Initialize deployment client
    deployer = MMMSageMakerDeployment(region_name=region_name)
    
    # Prepare and upload model artifacts
    s3_uri = deployer.prepare_model_artifacts(mmm_model, model_name)
    
    # Deploy to SageMaker
    deployment_info = deployer.deploy_model(s3_uri, model_name)
    
    if deployment_info['status'] == 'deployed':
        # Test the endpoint
        test_data = {
            'tv_spend': [10000],
            'digital_spend': [15000],
            'radio_spend': [5000],
            'print_spend': [3000],
            'social_spend': [8000]
        }
        
        test_results = deployer.test_endpoint(deployment_info['endpoint_name'], test_data)
        deployment_info['test_results'] = test_results
    
    print(f"[DEPLOY] SageMaker deployment completed!")
    return deployment_info

if __name__ == "__main__":
    print("AWS SageMaker Deployment for Media Mix Modeling")
    print("Usage: from infrastructure.aws.sagemaker_deployment import deploy_mmm_to_sagemaker")