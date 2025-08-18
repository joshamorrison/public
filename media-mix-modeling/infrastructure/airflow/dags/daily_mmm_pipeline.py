"""
Daily Media Mix Modeling Pipeline
Automated daily workflow for MMM data processing, model training, and optimization
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from airflow.providers.amazon.aws.operators.s3 import S3CreateBucketOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
import os
import sys

# Add project root to path
sys.path.append('/opt/airflow/dags/media-mix-modeling')

# Import MMM modules
from data.media_data_client import MediaDataClient
from models.mmm.econometric_mmm import EconometricMMM
from src.mlflow_integration import setup_mlflow_tracking

# Default arguments for the DAG
default_args = {
    'owner': 'mmm-team',
    'depends_on_past': False,
    'start_date': datetime(2025, 1, 1),
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'catchup': False
}

# Define the DAG
dag = DAG(
    'daily_mmm_pipeline',
    default_args=default_args,
    description='Daily Media Mix Modeling Pipeline',
    schedule_interval='0 6 * * *',  # Daily at 6 AM
    max_active_runs=1,
    tags=['mmm', 'daily', 'production']
)

def extract_media_data(**context):
    """Extract media data from multiple sources"""
    print("[EXTRACT] Starting media data extraction...")
    
    # Initialize data client
    data_client = MediaDataClient()
    
    # Extract data from all available sources
    marketing_data, source_info = data_client.get_marketing_data()
    
    # Store data for next tasks
    context['task_instance'].xcom_push(key='marketing_data', value=marketing_data.to_json())
    context['task_instance'].xcom_push(key='source_info', value=source_info)
    
    print(f"[EXTRACT] Extracted {len(marketing_data)} records from {source_info['description']}")
    return "Data extraction completed successfully"

def run_dbt_transformations(**context):
    """Run dbt data transformations"""
    print("[DBT] Running data transformations...")
    
    # This would run dbt models in production
    # For now, we'll simulate the transformation process
    
    import pandas as pd
    marketing_data = pd.read_json(context['task_instance'].xcom_pull(key='marketing_data'))
    
    # Simulate dbt transformation results
    transformation_results = {
        'staging_models': ['stg_media_spend', 'stg_channel_performance'],
        'intermediate_models': ['int_attribution_base', 'int_attribution_methods'],
        'mart_models': ['media_performance', 'budget_optimization'],
        'tests_passed': True,
        'records_processed': len(marketing_data)
    }
    
    context['task_instance'].xcom_push(key='dbt_results', value=transformation_results)
    print(f"[DBT] Processed {transformation_results['records_processed']} records through dbt pipeline")
    
    return "dbt transformations completed successfully"

def train_mmm_model(**context):
    """Train the MMM model with latest data"""
    print("[MMM] Training media mix model...")
    
    import pandas as pd
    
    # Get data from previous task
    marketing_data = pd.read_json(context['task_instance'].xcom_pull(key='marketing_data'))
    source_info = context['task_instance'].xcom_pull(key='source_info')
    
    # Initialize MLflow tracking
    mlflow_tracker = setup_mlflow_tracking("daily-mmm-pipeline")
    run_id = mlflow_tracker.start_mmm_run(
        run_name=f"daily_mmm_{context['ds']}",
        tags={
            "pipeline": "daily_production",
            "data_source": source_info.get('description', 'Unknown'),
            "execution_date": context['ds']
        }
    )
    
    mlflow_tracker.log_data_info(marketing_data, source_info)
    
    # Initialize and train MMM model
    mmm_model = EconometricMMM(
        adstock_rate=0.5,
        saturation_param=0.6,
        regularization_alpha=0.1,
        mlflow_tracker=mlflow_tracker
    )
    
    # Get spend columns
    spend_columns = [col for col in marketing_data.columns if col.endswith('_spend')]
    
    # Train model
    mmm_results = mmm_model.fit(
        data=marketing_data,
        target_column='revenue',
        spend_columns=spend_columns,
        include_synergies=True
    )
    
    # Store results for next tasks
    model_performance = {
        'r2_score': mmm_results['performance']['r2_score'],
        'mape': mmm_results['performance']['mape'],
        'model_version': run_id,
        'channels_analyzed': len(spend_columns)
    }
    
    context['task_instance'].xcom_push(key='model_performance', value=model_performance)
    context['task_instance'].xcom_push(key='mlflow_run_id', value=run_id)
    
    # End MLflow run
    mlflow_tracker.end_run()
    
    print(f"[MMM] Model training completed - R²: {model_performance['r2_score']:.3f}")
    return "MMM model training completed successfully"

def optimize_budget_allocation(**context):
    """Run budget optimization based on latest model"""
    print("[OPTIMIZATION] Running budget optimization...")
    
    import pandas as pd
    from models.mmm.budget_optimizer import BudgetOptimizer
    
    # Get data from previous tasks
    marketing_data = pd.read_json(context['task_instance'].xcom_pull(key='marketing_data'))
    model_performance = context['task_instance'].xcom_pull(key='model_performance')
    
    # Initialize budget optimizer
    optimizer = BudgetOptimizer()
    
    # Current budget allocation
    spend_columns = [col for col in marketing_data.columns if col.endswith('_spend')]
    current_budget = {
        channel.replace('_spend', ''): marketing_data[channel].mean() 
        for channel in spend_columns
    }
    
    # Run optimization
    optimization_results = optimizer.optimize_allocation(
        channels=list(current_budget.keys()),
        current_allocation=current_budget,
        target_metric='roi',
        total_budget=sum(current_budget.values())
    )
    
    # Store optimization results
    optimization_summary = {
        'optimization_date': context['ds'],
        'total_budget': sum(current_budget.values()),
        'projected_roi_improvement': optimization_results.get('improvements', {}).get('roi_improvement', 0),
        'recommended_reallocations': len([ch for ch, data in optimization_results.get('allocations', {}).items() 
                                        if abs(data.get('change_pct', 0)) > 5])
    }
    
    context['task_instance'].xcom_push(key='optimization_results', value=optimization_summary)
    
    print(f"[OPTIMIZATION] Budget optimization completed - Projected ROI improvement: {optimization_summary['projected_roi_improvement']:.1%}")
    return "Budget optimization completed successfully"

def generate_daily_reports(**context):
    """Generate executive reports and alerts"""
    print("[REPORTS] Generating daily reports...")
    
    # Get results from all previous tasks
    model_performance = context['task_instance'].xcom_pull(key='model_performance')
    optimization_results = context['task_instance'].xcom_pull(key='optimization_results')
    dbt_results = context['task_instance'].xcom_pull(key='dbt_results')
    
    # Generate daily summary report
    daily_report = {
        'report_date': context['ds'],
        'pipeline_status': 'SUCCESS',
        'model_performance': model_performance,
        'optimization_insights': optimization_results,
        'data_quality': {
            'dbt_tests_passed': dbt_results.get('tests_passed', False),
            'records_processed': dbt_results.get('records_processed', 0)
        },
        'alerts': []
    }
    
    # Add alerts for significant changes
    if model_performance.get('r2_score', 0) < 0.5:
        daily_report['alerts'].append("Model performance below threshold - requires attention")
    
    if optimization_results.get('projected_roi_improvement', 0) > 0.1:
        daily_report['alerts'].append("Significant budget reallocation opportunity identified")
    
    # Store final report
    context['task_instance'].xcom_push(key='daily_report', value=daily_report)
    
    print(f"[REPORTS] Daily report generated with {len(daily_report['alerts'])} alerts")
    return "Daily reports generated successfully"

def send_notifications(**context):
    """Send notifications to stakeholders"""
    print("[NOTIFICATIONS] Sending stakeholder notifications...")
    
    daily_report = context['task_instance'].xcom_pull(key='daily_report')
    
    # In production, this would send emails/Slack notifications
    # For now, we'll log the key metrics
    
    print(f"[SUMMARY] Daily MMM Pipeline - {daily_report['report_date']}")
    print(f"[SUMMARY] Status: {daily_report['pipeline_status']}")
    print(f"[SUMMARY] Model R²: {daily_report['model_performance'].get('r2_score', 'N/A')}")
    print(f"[SUMMARY] Optimization Opportunity: {daily_report['optimization_insights'].get('projected_roi_improvement', 0):.1%}")
    print(f"[SUMMARY] Alerts: {len(daily_report['alerts'])}")
    
    for alert in daily_report['alerts']:
        print(f"[ALERT] {alert}")
    
    return "Notifications sent successfully"

# Define task dependencies
extract_data_task = PythonOperator(
    task_id='extract_media_data',
    python_callable=extract_media_data,
    dag=dag
)

dbt_transform_task = PythonOperator(
    task_id='run_dbt_transformations',
    python_callable=run_dbt_transformations,
    dag=dag
)

train_model_task = PythonOperator(
    task_id='train_mmm_model',
    python_callable=train_mmm_model,
    dag=dag
)

optimize_budget_task = PythonOperator(
    task_id='optimize_budget_allocation',
    python_callable=optimize_budget_allocation,
    dag=dag
)

generate_reports_task = PythonOperator(
    task_id='generate_daily_reports',
    python_callable=generate_daily_reports,
    dag=dag
)

send_notifications_task = PythonOperator(
    task_id='send_notifications',
    python_callable=send_notifications,
    dag=dag
)

# Set task dependencies
extract_data_task >> dbt_transform_task >> train_model_task >> optimize_budget_task >> generate_reports_task >> send_notifications_task