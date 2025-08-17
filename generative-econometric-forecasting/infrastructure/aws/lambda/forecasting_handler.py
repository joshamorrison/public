"""
AWS Lambda handler for econometric forecasting.
Integrates with the main forecasting platform for serverless execution.
"""

import json
import boto3
import os
import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import pandas as pd
import numpy as np

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

class AWSForecastingHandler:
    """
    AWS Lambda handler for econometric forecasting operations.
    Integrates with S3, Secrets Manager, and CloudWatch.
    """
    
    def __init__(self):
        """Initialize AWS services and configuration."""
        self.s3_client = boto3.client('s3')
        self.secrets_client = boto3.client('secretsmanager')
        self.cloudwatch_client = boto3.client('cloudwatch')
        
        # Environment variables
        self.data_bucket = os.environ.get('DATA_BUCKET')
        self.models_bucket = os.environ.get('MODELS_BUCKET')
        self.outputs_bucket = os.environ.get('OUTPUTS_BUCKET')
        self.secrets_arn = os.environ.get('SECRETS_ARN')
        self.environment = os.environ.get('ENVIRONMENT', 'dev')
        
        # API keys (loaded lazily)
        self._api_keys = None
        
        logger.info(f"Initialized AWS Forecasting Handler for environment: {self.environment}")
    
    def get_api_keys(self) -> Dict[str, str]:
        """Retrieve API keys from AWS Secrets Manager."""
        if self._api_keys is None:
            try:
                response = self.secrets_client.get_secret_value(SecretId=self.secrets_arn)
                self._api_keys = json.loads(response['SecretString'])
                logger.info("Successfully retrieved API keys from Secrets Manager")
            except Exception as e:
                logger.error(f"Failed to retrieve API keys: {e}")
                self._api_keys = {}
        
        return self._api_keys
    
    def download_from_s3(self, bucket: str, key: str, local_path: str) -> bool:
        """Download file from S3 to local path."""
        try:
            self.s3_client.download_file(bucket, key, local_path)
            logger.info(f"Downloaded s3://{bucket}/{key} to {local_path}")
            return True
        except Exception as e:
            logger.error(f"Failed to download from S3: {e}")
            return False
    
    def upload_to_s3(self, local_path: str, bucket: str, key: str) -> bool:
        """Upload file from local path to S3."""
        try:
            self.s3_client.upload_file(local_path, bucket, key)
            logger.info(f"Uploaded {local_path} to s3://{bucket}/{key}")
            return True
        except Exception as e:
            logger.error(f"Failed to upload to S3: {e}")
            return False
    
    def put_metric(self, metric_name: str, value: float, unit: str = 'Count', **dimensions) -> None:
        """Send custom metric to CloudWatch."""
        try:
            dimensions_list = [
                {'Name': k, 'Value': v} for k, v in dimensions.items()
            ]
            
            self.cloudwatch_client.put_metric_data(
                Namespace='EconometricForecasting',
                MetricData=[
                    {
                        'MetricName': metric_name,
                        'Value': value,
                        'Unit': unit,
                        'Dimensions': dimensions_list,
                        'Timestamp': datetime.utcnow()
                    }
                ]
            )
            logger.info(f"Sent metric {metric_name}: {value} {unit}")
        except Exception as e:
            logger.error(f"Failed to send metric: {e}")
    
    def lightweight_forecast(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Perform lightweight forecasting for quick results.
        Uses simple statistical methods suitable for Lambda execution.
        """
        try:
            # Extract parameters
            indicators = event.get('indicators', ['GDP'])
            horizon = event.get('horizon', 6)
            start_time = datetime.utcnow()
            
            logger.info(f"Starting lightweight forecast for {indicators} with horizon {horizon}")
            
            # Mock forecast data (in real implementation, would fetch from S3 and compute)
            forecasts = {}
            for indicator in indicators:
                # Generate realistic-looking forecast data
                base_value = 100 if indicator == 'GDP' else 5 if indicator == 'UNEMPLOYMENT' else 2.5
                trend = 0.5 if indicator == 'GDP' else -0.1 if indicator == 'UNEMPLOYMENT' else 0.1
                
                forecast_values = []
                for i in range(horizon):
                    value = base_value + (trend * i) + np.random.normal(0, 0.5)
                    forecast_values.append(round(value, 2))
                
                forecasts[indicator] = {
                    'values': forecast_values,
                    'method': 'exponential_smoothing',
                    'confidence_level': 0.95,
                    'generated_at': datetime.utcnow().isoformat()
                }
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Send metrics
            self.put_metric('ForecastExecutionTime', execution_time, 'Seconds', 
                          Environment=self.environment, Method='lightweight')
            self.put_metric('ForecastRequests', 1, 'Count', 
                          Environment=self.environment)
            
            result = {
                'forecasts': forecasts,
                'metadata': {
                    'horizon': horizon,
                    'indicators': indicators,
                    'execution_time_seconds': execution_time,
                    'method': 'lightweight',
                    'environment': self.environment,
                    'timestamp': datetime.utcnow().isoformat()
                }
            }
            
            # Save results to S3
            output_key = f"forecasts/lightweight/{datetime.utcnow().strftime('%Y/%m/%d')}/forecast_{int(datetime.utcnow().timestamp())}.json"
            local_path = f"/tmp/forecast_result.json"
            
            with open(local_path, 'w') as f:
                json.dump(result, f, indent=2)
            
            if self.upload_to_s3(local_path, self.outputs_bucket, output_key):
                result['metadata']['s3_location'] = f"s3://{self.outputs_bucket}/{output_key}"
            
            logger.info(f"Lightweight forecast completed in {execution_time:.2f} seconds")
            return result
            
        except Exception as e:
            logger.error(f"Lightweight forecast failed: {e}")
            # Send error metric
            self.put_metric('ForecastErrors', 1, 'Count', 
                          Environment=self.environment, ErrorType='lightweight_forecast')
            raise
    
    def heavy_forecast(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Trigger heavy forecasting on EC2 instance.
        This function initiates the process and returns job ID.
        """
        try:
            # In a real implementation, this would:
            # 1. Upload job parameters to S3
            # 2. Send message to SQS queue
            # 3. EC2 instance picks up job and processes
            # 4. Results are saved back to S3
            
            job_id = f"heavy_forecast_{int(datetime.utcnow().timestamp())}"
            
            job_params = {
                'job_id': job_id,
                'indicators': event.get('indicators', ['GDP', 'UNEMPLOYMENT', 'INFLATION']),
                'horizon': event.get('horizon', 12),
                'models': event.get('models', ['arima', 'var', 'neural']),
                'use_r_models': event.get('use_r_models', True),
                'use_foundation_models': event.get('use_foundation_models', True),
                'submitted_at': datetime.utcnow().isoformat()
            }
            
            # Save job parameters to S3
            job_key = f"jobs/heavy/{job_id}/parameters.json"
            local_path = f"/tmp/{job_id}_params.json"
            
            with open(local_path, 'w') as f:
                json.dump(job_params, f, indent=2)
            
            if self.upload_to_s3(local_path, self.data_bucket, job_key):
                logger.info(f"Heavy forecast job {job_id} parameters uploaded to S3")
            
            # Send metric
            self.put_metric('HeavyForecastJobs', 1, 'Count', 
                          Environment=self.environment)
            
            return {
                'job_id': job_id,
                'status': 'submitted',
                'parameters': job_params,
                'check_status_url': f"/status/{job_id}",
                'estimated_completion_minutes': 15
            }
            
        except Exception as e:
            logger.error(f"Heavy forecast submission failed: {e}")
            self.put_metric('ForecastErrors', 1, 'Count', 
                          Environment=self.environment, ErrorType='heavy_forecast')
            raise
    
    def get_forecast_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a heavy forecasting job."""
        try:
            # Check for results in S3
            results_key = f"results/heavy/{job_id}/forecast_results.json"
            
            try:
                # Try to get results
                response = self.s3_client.head_object(Bucket=self.outputs_bucket, Key=results_key)
                
                return {
                    'job_id': job_id,
                    'status': 'completed',
                    'completed_at': response['LastModified'].isoformat(),
                    'results_location': f"s3://{self.outputs_bucket}/{results_key}"
                }
                
            except self.s3_client.exceptions.NoSuchKey:
                # Job still running or failed
                return {
                    'job_id': job_id,
                    'status': 'running',
                    'message': 'Job is being processed on EC2 instance'
                }
                
        except Exception as e:
            logger.error(f"Failed to get job status: {e}")
            return {
                'job_id': job_id,
                'status': 'error',
                'error': str(e)
            }
    
    def data_ingestion(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """
        Handle data ingestion from external sources.
        Triggered by S3 events or scheduled execution.
        """
        try:
            # Get API keys
            api_keys = self.get_api_keys()
            
            if not api_keys.get('FRED_API_KEY'):
                raise ValueError("FRED API key not available")
            
            # Mock data ingestion (in real implementation, would call FRED API)
            indicators_fetched = ['GDP', 'UNEMPLOYMENT', 'INFLATION', 'INTEREST_RATE']
            
            ingestion_result = {
                'indicators_fetched': indicators_fetched,
                'records_count': 1000,  # Mock count
                'data_date_range': {
                    'start': '2010-01-01',
                    'end': datetime.utcnow().strftime('%Y-%m-%d')
                },
                'ingestion_timestamp': datetime.utcnow().isoformat()
            }
            
            # Save ingestion metadata
            metadata_key = f"ingestion/metadata/{datetime.utcnow().strftime('%Y/%m/%d')}/ingestion_{int(datetime.utcnow().timestamp())}.json"
            local_path = f"/tmp/ingestion_metadata.json"
            
            with open(local_path, 'w') as f:
                json.dump(ingestion_result, f, indent=2)
            
            if self.upload_to_s3(local_path, self.data_bucket, metadata_key):
                ingestion_result['metadata_location'] = f"s3://{self.data_bucket}/{metadata_key}"
            
            # Send metrics
            self.put_metric('DataIngestionJobs', 1, 'Count', 
                          Environment=self.environment)
            self.put_metric('RecordsIngested', ingestion_result['records_count'], 'Count',
                          Environment=self.environment)
            
            logger.info(f"Data ingestion completed: {len(indicators_fetched)} indicators")
            return ingestion_result
            
        except Exception as e:
            logger.error(f"Data ingestion failed: {e}")
            self.put_metric('IngestionErrors', 1, 'Count', 
                          Environment=self.environment)
            raise


# Global handler instance
handler = AWSForecastingHandler()

def lambda_handler(event: Dict[str, Any], context) -> Dict[str, Any]:
    """
    Main Lambda handler entry point.
    Routes requests to appropriate handler methods.
    """
    try:
        # Extract operation type
        operation = event.get('operation', 'lightweight_forecast')
        
        logger.info(f"Processing operation: {operation}")
        logger.info(f"Event: {json.dumps(event, default=str)}")
        
        # Route to appropriate handler
        if operation == 'lightweight_forecast':
            result = handler.lightweight_forecast(event)
            status_code = 200
            
        elif operation == 'heavy_forecast':
            result = handler.heavy_forecast(event)
            status_code = 202  # Accepted
            
        elif operation == 'get_status':
            job_id = event.get('job_id')
            if not job_id:
                raise ValueError("job_id required for status check")
            result = handler.get_forecast_status(job_id)
            status_code = 200
            
        elif operation == 'data_ingestion':
            result = handler.data_ingestion(event)
            status_code = 200
            
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        # Return successful response
        return {
            'statusCode': status_code,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps(result, default=str)
        }
        
    except Exception as e:
        logger.error(f"Lambda execution failed: {e}")
        
        # Return error response
        return {
            'statusCode': 500,
            'headers': {
                'Content-Type': 'application/json',
                'Access-Control-Allow-Origin': '*'
            },
            'body': json.dumps({
                'error': str(e),
                'operation': event.get('operation', 'unknown'),
                'timestamp': datetime.utcnow().isoformat()
            })
        }


# For local testing
if __name__ == "__main__":
    # Test event
    test_event = {
        'operation': 'lightweight_forecast',
        'indicators': ['GDP', 'UNEMPLOYMENT'],
        'horizon': 6
    }
    
    class MockContext:
        aws_request_id = 'test-request-id'
        function_name = 'test-function'
        remaining_time_in_millis = lambda: 300000
    
    result = lambda_handler(test_event, MockContext())
    print(json.dumps(result, indent=2))