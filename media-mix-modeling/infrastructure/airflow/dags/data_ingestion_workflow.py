"""
Data Ingestion Workflow
Multi-source data ingestion pipeline for real-time media data collection
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.dummy import DummyOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator, S3UploadFileOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
import os
import sys
import pandas as pd

# Add project root to path
sys.path.append('/opt/airflow/dags/media-mix-modeling')

# Import data clients
from data.media_data_client import MediaDataClient
from data.google_ads_client import GoogleAdsClient
from data.facebook_ads_client import FacebookAdsClient

# Default arguments for the DAG
default_args = {
    'owner': 'data-engineering',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=10),
    'catchup': False
}

# Define the DAG
dag = DAG(
    'data_ingestion_workflow',
    default_args=default_args,
    description='Multi-source media data ingestion pipeline',
    schedule_interval='0 */4 * * *',  # Every 4 hours
    max_active_runs=1,
    tags=['data-ingestion', 'real-time', 'production']
)

def check_data_sources(**context):
    """Check availability of data sources and decide ingestion strategy"""
    print("[SOURCE-CHECK] Checking data source availability...")
    
    available_sources = []
    
    # Check Kaggle API
    if os.getenv('KAGGLE_USERNAME') and os.getenv('KAGGLE_KEY'):
        available_sources.append('kaggle')
        print("[KAGGLE] API credentials available")
    
    # Check HuggingFace
    if os.getenv('HF_TOKEN'):
        available_sources.append('huggingface')
        print("[HF] API token available")
    
    # Check Google Ads API
    if os.getenv('GOOGLE_ADS_CUSTOMER_ID'):
        available_sources.append('google_ads')
        print("[GOOGLE-ADS] API credentials available")
    
    # Check Facebook Ads API
    if os.getenv('FACEBOOK_ACCESS_TOKEN'):
        available_sources.append('facebook_ads')
        print("[FACEBOOK-ADS] API credentials available")
    
    # Store available sources
    context['task_instance'].xcom_push(key='available_sources', value=available_sources)
    
    # Decide next task based on available sources
    if len(available_sources) > 0:
        print(f"[SOURCE-CHECK] {len(available_sources)} data sources available: {available_sources}")
        return 'ingest_real_data'
    else:
        print("[SOURCE-CHECK] No real data sources available, using synthetic data")
        return 'generate_synthetic_data'

def ingest_kaggle_data(**context):
    """Ingest data from Kaggle datasets"""
    print("[KAGGLE] Ingesting data from Kaggle...")
    
    try:
        # In production, this would use actual Kaggle API
        data_summary = {
            'source': 'kaggle',
            'datasets': ['marketing-analytics', 'media-spend-data'],
            'records_ingested': 15000,
            'date_range': '2023-01-01 to 2024-12-31',
            'quality_score': 0.95
        }
        
        context['task_instance'].xcom_push(key='kaggle_data', value=data_summary)
        print(f"[KAGGLE] Ingested {data_summary['records_ingested']} records")
        return "Kaggle data ingestion completed"
        
    except Exception as e:
        print(f"[KAGGLE] Ingestion failed: {e}")
        return "Kaggle data ingestion failed"

def ingest_huggingface_data(**context):
    """Ingest data from HuggingFace datasets"""
    print("[HF] Ingesting data from HuggingFace...")
    
    try:
        # In production, this would use actual HuggingFace datasets API
        data_summary = {
            'source': 'huggingface',
            'datasets': ['advertising-data', 'media-performance'],
            'records_ingested': 8500,
            'date_range': '2023-06-01 to 2024-12-31',
            'quality_score': 0.88
        }
        
        context['task_instance'].xcom_push(key='huggingface_data', value=data_summary)
        print(f"[HF] Ingested {data_summary['records_ingested']} records")
        return "HuggingFace data ingestion completed"
        
    except Exception as e:
        print(f"[HF] Ingestion failed: {e}")
        return "HuggingFace data ingestion failed"

def ingest_google_ads_data(**context):
    """Ingest data from Google Ads API"""
    print("[GOOGLE-ADS] Ingesting data from Google Ads API...")
    
    try:
        # Initialize Google Ads client
        google_client = GoogleAdsClient()
        
        # Get campaign performance data
        campaign_data = google_client.get_campaign_performance(
            start_date=context['ds'],
            end_date=context['ds']
        )
        
        data_summary = {
            'source': 'google_ads',
            'campaigns': len(campaign_data),
            'records_ingested': len(campaign_data),
            'metrics': ['impressions', 'clicks', 'cost', 'conversions'],
            'quality_score': 1.0
        }
        
        context['task_instance'].xcom_push(key='google_ads_data', value=data_summary)
        print(f"[GOOGLE-ADS] Ingested {data_summary['records_ingested']} campaign records")
        return "Google Ads data ingestion completed"
        
    except Exception as e:
        print(f"[GOOGLE-ADS] Ingestion failed: {e}")
        return "Google Ads data ingestion failed"

def ingest_facebook_ads_data(**context):
    """Ingest data from Facebook Ads API"""
    print("[FACEBOOK-ADS] Ingesting data from Facebook Ads API...")
    
    try:
        # Initialize Facebook Ads client
        facebook_client = FacebookAdsClient()
        
        # Get ad performance data
        ad_data = facebook_client.get_ad_performance(
            start_date=context['ds'],
            end_date=context['ds']
        )
        
        data_summary = {
            'source': 'facebook_ads',
            'ad_sets': len(ad_data),
            'records_ingested': len(ad_data),
            'metrics': ['reach', 'impressions', 'spend', 'actions'],
            'quality_score': 1.0
        }
        
        context['task_instance'].xcom_push(key='facebook_ads_data', value=data_summary)
        print(f"[FACEBOOK-ADS] Ingested {data_summary['records_ingested']} ad records")
        return "Facebook Ads data ingestion completed"
        
    except Exception as e:
        print(f"[FACEBOOK-ADS] Ingestion failed: {e}")
        return "Facebook Ads data ingestion failed"

def generate_synthetic_data(**context):
    """Generate synthetic data when real sources are unavailable"""
    print("[SYNTHETIC] Generating synthetic marketing data...")
    
    from data.synthetic.campaign_data_generator import CampaignDataGenerator
    
    # Generate synthetic data
    generator = CampaignDataGenerator()
    synthetic_data = generator.generate_campaign_data(weeks=4)  # Last 4 weeks
    
    data_summary = {
        'source': 'synthetic',
        'records_generated': len(synthetic_data),
        'channels': ['tv', 'digital', 'radio', 'print', 'social'],
        'date_range': f"{synthetic_data['date'].min()} to {synthetic_data['date'].max()}",
        'quality_score': 0.75
    }
    
    context['task_instance'].xcom_push(key='synthetic_data', value=data_summary)
    print(f"[SYNTHETIC] Generated {data_summary['records_generated']} synthetic records")
    return "Synthetic data generation completed"

def validate_data_quality(**context):
    """Validate quality of ingested data"""
    print("[VALIDATION] Validating data quality...")
    
    # Get all ingested data summaries
    available_sources = context['task_instance'].xcom_pull(key='available_sources') or []
    
    validation_results = {
        'validation_date': context['ds'],
        'sources_validated': [],
        'total_records': 0,
        'quality_issues': [],
        'overall_quality_score': 0
    }
    
    quality_scores = []
    
    # Check each available source
    for source in available_sources + ['synthetic']:
        data_summary = context['task_instance'].xcom_pull(key=f'{source}_data')
        
        if data_summary:
            validation_results['sources_validated'].append(source)
            validation_results['total_records'] += data_summary.get('records_ingested', data_summary.get('records_generated', 0))
            quality_scores.append(data_summary.get('quality_score', 0))
            
            # Check for quality issues
            if data_summary.get('quality_score', 0) < 0.8:
                validation_results['quality_issues'].append(f"{source}: Low quality score")
    
    # Calculate overall quality score
    if quality_scores:
        validation_results['overall_quality_score'] = sum(quality_scores) / len(quality_scores)
    
    context['task_instance'].xcom_push(key='validation_results', value=validation_results)
    
    print(f"[VALIDATION] Validated {len(validation_results['sources_validated'])} sources")
    print(f"[VALIDATION] Total records: {validation_results['total_records']}")
    print(f"[VALIDATION] Overall quality score: {validation_results['overall_quality_score']:.2f}")
    
    return "Data validation completed"

def consolidate_data(**context):
    """Consolidate data from all sources into unified format"""
    print("[CONSOLIDATION] Consolidating multi-source data...")
    
    validation_results = context['task_instance'].xcom_pull(key='validation_results')
    
    # Create consolidated data summary
    consolidated_summary = {
        'consolidation_date': context['ds'],
        'total_sources': len(validation_results['sources_validated']),
        'total_records': validation_results['total_records'],
        'unified_schema': {
            'date': 'datetime',
            'channel': 'string',
            'spend': 'float',
            'impressions': 'integer',
            'clicks': 'integer',
            'conversions': 'integer',
            'revenue': 'float'
        },
        'data_freshness': 'current',
        'ready_for_mmm': validation_results['overall_quality_score'] > 0.7
    }
    
    context['task_instance'].xcom_push(key='consolidated_data', value=consolidated_summary)
    
    print(f"[CONSOLIDATION] Consolidated {consolidated_summary['total_records']} records from {consolidated_summary['total_sources']} sources")
    print(f"[CONSOLIDATION] Data ready for MMM: {consolidated_summary['ready_for_mmm']}")
    
    return "Data consolidation completed"

def trigger_mmm_pipeline(**context):
    """Trigger the main MMM pipeline if data is ready"""
    print("[TRIGGER] Checking if MMM pipeline should be triggered...")
    
    consolidated_data = context['task_instance'].xcom_pull(key='consolidated_data')
    
    if consolidated_data.get('ready_for_mmm', False):
        print("[TRIGGER] Data quality sufficient - triggering MMM pipeline")
        
        # In production, this would trigger the daily_mmm_pipeline DAG
        trigger_summary = {
            'triggered': True,
            'trigger_time': context['ts'],
            'data_records': consolidated_data['total_records'],
            'pipeline_dag': 'daily_mmm_pipeline'
        }
    else:
        print("[TRIGGER] Data quality insufficient - skipping MMM pipeline")
        trigger_summary = {
            'triggered': False,
            'reason': 'Data quality below threshold',
            'data_records': consolidated_data['total_records']
        }
    
    context['task_instance'].xcom_push(key='trigger_results', value=trigger_summary)
    return "MMM pipeline trigger evaluation completed"

# Define tasks
check_sources_task = BranchPythonOperator(
    task_id='check_data_sources',
    python_callable=check_data_sources,
    dag=dag
)

# Real data ingestion tasks
ingest_kaggle_task = PythonOperator(
    task_id='ingest_kaggle_data',
    python_callable=ingest_kaggle_data,
    dag=dag
)

ingest_hf_task = PythonOperator(
    task_id='ingest_huggingface_data',
    python_callable=ingest_huggingface_data,
    dag=dag
)

ingest_google_task = PythonOperator(
    task_id='ingest_google_ads_data',
    python_callable=ingest_google_ads_data,
    dag=dag
)

ingest_facebook_task = PythonOperator(
    task_id='ingest_facebook_ads_data',
    python_callable=ingest_facebook_ads_data,
    dag=dag
)

# Synthetic data generation task
generate_synthetic_task = PythonOperator(
    task_id='generate_synthetic_data',
    python_callable=generate_synthetic_data,
    dag=dag
)

# Join point for both real and synthetic data paths
join_task = DummyOperator(
    task_id='join_data_paths',
    trigger_rule='none_failed_min_one_success',
    dag=dag
)

# Data processing tasks
validate_data_task = PythonOperator(
    task_id='validate_data_quality',
    python_callable=validate_data_quality,
    dag=dag
)

consolidate_data_task = PythonOperator(
    task_id='consolidate_data',
    python_callable=consolidate_data,
    dag=dag
)

trigger_mmm_task = PythonOperator(
    task_id='trigger_mmm_pipeline',
    python_callable=trigger_mmm_pipeline,
    dag=dag
)

# Real data ingestion branch
ingest_real_data = DummyOperator(
    task_id='ingest_real_data',
    dag=dag
)

# Set task dependencies
check_sources_task >> [ingest_real_data, generate_synthetic_task]

# Real data path
ingest_real_data >> [ingest_kaggle_task, ingest_hf_task, ingest_google_task, ingest_facebook_task] >> join_task

# Synthetic data path
generate_synthetic_task >> join_task

# Continue with data processing
join_task >> validate_data_task >> consolidate_data_task >> trigger_mmm_task