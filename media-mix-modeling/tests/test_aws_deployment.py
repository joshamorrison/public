"""
Tests for AWS deployment functionality
"""

import pytest
import boto3
import json
from unittest.mock import patch, MagicMock, mock_open
from moto import mock_s3, mock_sagemaker, mock_iam

from infrastructure.aws.sagemaker_deployment import MMMSageMakerDeployment, deploy_mmm_to_sagemaker

@pytest.mark.aws
class TestSageMakerDeployment:
    """Test SageMaker deployment functionality"""
    
    def test_init(self):
        """Test initialization"""
        deployment = MMMSageMakerDeployment(
            region_name='us-west-2',
            bucket_name='test-bucket'
        )
        
        assert deployment.region_name == 'us-west-2'
        assert deployment.bucket_name == 'test-bucket'
        assert deployment.model_config['instance_type'] == 'ml.m5.large'
    
    @patch('infrastructure.aws.sagemaker_deployment.joblib')
    @patch('infrastructure.aws.sagemaker_deployment.Path')
    def test_prepare_model_artifacts(self, mock_path, mock_joblib):
        """Test model artifact preparation"""
        # Mock the model
        mock_model = MagicMock()
        mock_model.adstock_rate = 0.5
        mock_model.saturation_param = 0.6
        mock_model.regularization_alpha = 0.1
        
        # Mock path operations
        mock_artifacts_dir = MagicMock()
        mock_path.return_value = mock_artifacts_dir
        mock_artifacts_dir.mkdir.return_value = None
        mock_artifacts_dir.__truediv__ = lambda self, x: MagicMock()
        
        deployment = MMMSageMakerDeployment()
        
        with patch.object(deployment, '_upload_artifacts_to_s3', return_value='s3://test-bucket/model/'):
            s3_uri = deployment.prepare_model_artifacts(mock_model, 'test-model')
        
        # Should call joblib.dump
        mock_joblib.dump.assert_called_once()
        
        # Should return S3 URI
        assert s3_uri == 's3://test-bucket/model/'
    
    @mock_s3
    def test_upload_artifacts_to_s3(self):
        """Test S3 artifact upload"""
        # Create mock S3 bucket
        s3_client = boto3.client('s3', region_name='us-east-1')
        bucket_name = 'test-mmm-bucket'
        s3_client.create_bucket(Bucket=bucket_name)
        
        deployment = MMMSageMakerDeployment(bucket_name=bucket_name)
        
        # Create temporary directory structure
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            
            # Create mock files
            (artifacts_dir / 'model.joblib').write_text('mock model data')
            (artifacts_dir / 'metadata.json').write_text('{"model": "test"}')
            (artifacts_dir / 'inference.py').write_text('# inference script')
            
            # Test upload
            with patch.object(deployment, 'session') as mock_session:
                mock_session.client.return_value = s3_client
                
                s3_uri = deployment._upload_artifacts_to_s3(artifacts_dir, 'test-model')
        
        # Should return S3 URI
        assert s3_uri.startswith('s3://')
        assert bucket_name in s3_uri
    
    @patch('infrastructure.aws.sagemaker_deployment.SKLearnModel')
    @patch('infrastructure.aws.sagemaker_deployment.sagemaker.Session')
    def test_deploy_model(self, mock_sagemaker_session, mock_sklearn_model):
        """Test model deployment to SageMaker"""
        # Mock the SKLearn model and predictor
        mock_predictor = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.deploy.return_value = mock_predictor
        mock_sklearn_model.return_value = mock_model_instance
        
        deployment = MMMSageMakerDeployment()
        
        with patch.object(deployment, '_setup_autoscaling'):
            deployment_info = deployment.deploy_model(
                s3_model_uri='s3://test-bucket/model/',
                model_name='test-model'
            )
        
        # Should create SKLearn model
        mock_sklearn_model.assert_called_once()
        
        # Should deploy model
        mock_model_instance.deploy.assert_called_once()
        
        # Should return deployment info
        assert isinstance(deployment_info, dict)
        assert 'endpoint_name' in deployment_info
        assert 'status' in deployment_info
    
    @patch('infrastructure.aws.sagemaker_deployment.boto3.Session')
    def test_test_endpoint(self, mock_session):
        """Test endpoint testing functionality"""
        # Mock SageMaker runtime client
        mock_runtime_client = MagicMock()
        mock_response = {
            'Body': MagicMock(),
            'ResponseMetadata': {'HTTPHeaders': {'x-amzn-requestid': 'test-request-id'}}
        }
        mock_response['Body'].read.return_value.decode.return_value = '{"predictions": [100, 200, 300]}'
        mock_runtime_client.invoke_endpoint.return_value = mock_response
        
        mock_session.return_value.client.return_value = mock_runtime_client
        
        deployment = MMMSageMakerDeployment()
        
        test_data = {'tv_spend': [10000], 'digital_spend': [15000]}
        
        results = deployment.test_endpoint('test-endpoint', test_data)
        
        # Should invoke endpoint
        mock_runtime_client.invoke_endpoint.assert_called_once()
        
        # Should return test results
        assert isinstance(results, dict)
        assert 'test_status' in results
        assert 'predictions' in results
    
    @patch('infrastructure.aws.sagemaker_deployment.boto3.Session')
    def test_cleanup_endpoint(self, mock_session):
        """Test endpoint cleanup"""
        mock_sagemaker_client = MagicMock()
        mock_session.return_value.client.return_value = mock_sagemaker_client
        
        deployment = MMMSageMakerDeployment()
        
        result = deployment.cleanup_endpoint('test-endpoint')
        
        # Should delete endpoint and config
        mock_sagemaker_client.delete_endpoint.assert_called_once_with(EndpointName='test-endpoint')
        mock_sagemaker_client.delete_endpoint_config.assert_called_once_with(EndpointConfigName='test-endpoint')
        
        assert result is True

@pytest.mark.aws
class TestDeploymentIntegration:
    """Test deployment integration functionality"""
    
    @patch('infrastructure.aws.sagemaker_deployment.MMMSageMakerDeployment')
    def test_deploy_mmm_to_sagemaker(self, mock_deployment_class):
        """Test convenience deployment function"""
        # Mock deployment instance
        mock_deployment = MagicMock()
        mock_deployment.prepare_model_artifacts.return_value = 's3://test-bucket/model/'
        mock_deployment.deploy_model.return_value = {
            'endpoint_name': 'test-endpoint',
            'status': 'deployed',
            'endpoint_url': 'https://runtime.sagemaker.us-east-1.amazonaws.com/endpoints/test-endpoint/invocations'
        }
        mock_deployment.test_endpoint.return_value = {
            'test_status': 'success',
            'predictions': {'predictions': [100, 200]}
        }
        mock_deployment_class.return_value = mock_deployment
        
        # Mock MMM model
        mock_model = MagicMock()
        
        # Test deployment
        result = deploy_mmm_to_sagemaker(mock_model, 'test-model', 'us-east-1')
        
        # Should prepare artifacts
        mock_deployment.prepare_model_artifacts.assert_called_once_with(mock_model, 'test-model')
        
        # Should deploy model
        mock_deployment.deploy_model.assert_called_once()
        
        # Should test endpoint
        mock_deployment.test_endpoint.assert_called_once()
        
        # Should return deployment info with test results
        assert isinstance(result, dict)
        assert 'status' in result
        assert 'test_results' in result

@pytest.mark.aws 
class TestErrorHandling:
    """Test error handling in AWS deployment"""
    
    @patch('infrastructure.aws.sagemaker_deployment.boto3.Session')
    def test_s3_upload_failure(self, mock_session):
        """Test handling of S3 upload failures"""
        # Mock S3 client that raises exception
        mock_s3_client = MagicMock()
        mock_s3_client.create_bucket.side_effect = Exception("S3 error")
        mock_session.return_value.client.return_value = mock_s3_client
        
        deployment = MMMSageMakerDeployment()
        
        import tempfile
        from pathlib import Path
        
        with tempfile.TemporaryDirectory() as temp_dir:
            artifacts_dir = Path(temp_dir)
            (artifacts_dir / 'test.txt').write_text('test')
            
            # Should handle S3 error gracefully and return local path
            result = deployment._upload_artifacts_to_s3(artifacts_dir, 'test-model')
            assert str(artifacts_dir) in result
    
    @patch('infrastructure.aws.sagemaker_deployment.SKLearnModel')
    def test_deployment_failure(self, mock_sklearn_model):
        """Test handling of deployment failures"""
        # Mock model that raises exception during deployment
        mock_model_instance = MagicMock()
        mock_model_instance.deploy.side_effect = Exception("Deployment failed")
        mock_sklearn_model.return_value = mock_model_instance
        
        deployment = MMMSageMakerDeployment()
        
        result = deployment.deploy_model('s3://test-bucket/model/', 'test-model')
        
        # Should return failure status
        assert result['status'] == 'failed'
        assert 'error' in result
    
    @patch('infrastructure.aws.sagemaker_deployment.boto3.Session')
    def test_endpoint_test_failure(self, mock_session):
        """Test handling of endpoint test failures"""
        # Mock runtime client that raises exception
        mock_runtime_client = MagicMock()
        mock_runtime_client.invoke_endpoint.side_effect = Exception("Endpoint error")
        mock_session.return_value.client.return_value = mock_runtime_client
        
        deployment = MMMSageMakerDeployment()
        
        result = deployment.test_endpoint('test-endpoint', {'test': 'data'})
        
        # Should return failure status
        assert result['test_status'] == 'failed'
        assert 'error' in result

@pytest.mark.aws
class TestAWSConfiguration:
    """Test AWS configuration and setup"""
    
    def test_role_configuration(self):
        """Test IAM role configuration"""
        # Test with custom role
        custom_role = "arn:aws:iam::123456789012:role/CustomSageMakerRole"
        deployment = MMMSageMakerDeployment(role_arn=custom_role)
        
        assert deployment.role == custom_role
    
    def test_region_configuration(self):
        """Test region configuration"""
        deployment = MMMSageMakerDeployment(region_name='eu-west-1')
        
        assert deployment.region_name == 'eu-west-1'
    
    def test_model_configuration(self):
        """Test model configuration settings"""
        deployment = MMMSageMakerDeployment()
        
        config = deployment.model_config
        
        assert 'instance_type' in config
        assert 'instance_count' in config
        assert 'max_capacity' in config
        assert 'target_value' in config
        
        # Values should be reasonable
        assert config['instance_count'] >= 1
        assert config['max_capacity'] >= config['instance_count']
        assert 0 < config['target_value'] <= 100

@pytest.mark.integration
@pytest.mark.aws
class TestDeploymentWorkflow:
    """Test complete deployment workflow"""
    
    @patch('infrastructure.aws.sagemaker_deployment.MMMSageMakerDeployment')
    @patch('infrastructure.aws.sagemaker_deployment.EconometricMMM')
    @patch('infrastructure.aws.sagemaker_deployment.MediaDataClient')
    def test_complete_deployment_workflow(self, mock_data_client, mock_mmm, mock_deployment_class):
        """Test complete workflow from training to deployment"""
        # Mock data client
        mock_data = MagicMock()
        mock_data_client.return_value.get_best_available_data.return_value = (mock_data, {}, 'SYNTHETIC')
        
        # Mock MMM model
        mock_model = MagicMock()
        mock_model.fit.return_value = {
            'performance': {'r2_score': 0.85, 'mape': 0.05},
            'coefficients': {'tv': 0.8, 'digital': 1.2}
        }
        mock_mmm.return_value = mock_model
        
        # Mock deployment
        mock_deployment = MagicMock()
        mock_deployment.prepare_model_artifacts.return_value = 's3://test-bucket/model/'
        mock_deployment.deploy_model.return_value = {
            'endpoint_name': 'mmm-endpoint',
            'status': 'deployed',
            'endpoint_url': 'https://test-endpoint-url'
        }
        mock_deployment.test_endpoint.return_value = {'test_status': 'success'}
        mock_deployment_class.return_value = mock_deployment
        
        # Test the workflow
        result = deploy_mmm_to_sagemaker(mock_model, 'test-mmm', 'us-east-1')
        
        # Verify the workflow executed
        assert mock_deployment.prepare_model_artifacts.called
        assert mock_deployment.deploy_model.called
        assert mock_deployment.test_endpoint.called
        
        # Verify result structure
        assert isinstance(result, dict)
        assert result['status'] == 'deployed'
        assert 'test_results' in result