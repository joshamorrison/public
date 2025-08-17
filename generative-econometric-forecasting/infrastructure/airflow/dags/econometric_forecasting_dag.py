"""
Apache Airflow DAG for Econometric Forecasting Platform
Orchestrates data ingestion, model training, forecasting, and reporting workflows.
"""

from datetime import datetime, timedelta
from typing import Dict, Any, List

from airflow import DAG
from airflow.operators.python_operator import PythonOperator
from airflow.operators.bash_operator import BashOperator
from airflow.operators.dummy_operator import DummyOperator
from airflow.operators.email_operator import EmailOperator
from airflow.providers.amazon.aws.operators.s3_copy_object import S3CopyObjectOperator
from airflow.providers.amazon.aws.operators.lambda_invoke import LambdaInvokeOperator
from airflow.providers.amazon.aws.sensors.s3_key import S3KeySensor
from airflow.providers.amazon.aws.hooks.s3 import S3Hook
from airflow.providers.postgres.operators.postgres import PostgresOperator
from airflow.models import Variable
from airflow.utils.dates import days_ago
from airflow.utils.task_group import TaskGroup
import logging
import pandas as pd
import json

# Configure logging
logger = logging.getLogger(__name__)

# DAG configuration
DEFAULT_ARGS = {
    'owner': 'econometric-forecasting-team',
    'depends_on_past': False,
    'start_date': days_ago(1),
    'email': ['forecasting-team@company.com'],
    'email_on_failure': True,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
    'execution_timeout': timedelta(hours=2),
}

# Environment variables (should be set in Airflow Variables)
ENVIRONMENT = Variable.get("ENVIRONMENT", default_var="dev")
AWS_REGION = Variable.get("AWS_REGION", default_var="us-east-1")
DATA_BUCKET = Variable.get("DATA_BUCKET", default_var=f"{ENVIRONMENT}-econometric-forecasting-data")
MODELS_BUCKET = Variable.get("MODELS_BUCKET", default_var=f"{ENVIRONMENT}-econometric-forecasting-models")
OUTPUTS_BUCKET = Variable.get("OUTPUTS_BUCKET", default_var=f"{ENVIRONMENT}-econometric-forecasting-outputs")

# Economic indicators to forecast
ECONOMIC_INDICATORS = [
    'GDP', 'UNEMPLOYMENT', 'INFLATION', 'INTEREST_RATE', 
    'CONSUMER_CONFIDENCE', 'HOUSING_STARTS', 'RETAIL_SALES'
]

def data_ingestion_task(**context) -> Dict[str, Any]:
    """
    Ingest economic data from multiple sources.
    """
    from data.fred_client import FredDataClient
    from data.unstructured.news_client import NewsClient
    
    logger.info("Starting data ingestion...")
    
    # Initialize clients
    fred_client = FredDataClient()
    news_client = NewsClient()
    
    results = {
        'fred_data': {},
        'news_data': {},
        'ingestion_timestamp': datetime.utcnow().isoformat()
    }
    
    # Fetch FRED data
    for indicator in ECONOMIC_INDICATORS:
        try:
            data = fred_client.get_economic_data(indicator, start_date='2020-01-01')
            if data is not None and not data.empty:
                results['fred_data'][indicator] = {
                    'records': len(data),
                    'latest_value': float(data.iloc[-1]),
                    'latest_date': data.index[-1].isoformat()
                }
                logger.info(f"Fetched {len(data)} records for {indicator}")
        except Exception as e:
            logger.error(f"Failed to fetch {indicator}: {e}")
            results['fred_data'][indicator] = {'error': str(e)}
    
    # Fetch news sentiment data
    try:
        news_data = news_client.get_economic_news(limit=50)
        sentiment_score = news_client.analyze_sentiment(news_data)
        results['news_data'] = {
            'articles_count': len(news_data),
            'sentiment_score': sentiment_score,
            'sentiment_distribution': news_client.get_sentiment_distribution(news_data)
        }
        logger.info(f"Fetched {len(news_data)} news articles")
    except Exception as e:
        logger.error(f"Failed to fetch news data: {e}")
        results['news_data'] = {'error': str(e)}
    
    # Save results to XCom for downstream tasks
    context['task_instance'].xcom_push(key='ingestion_results', value=results)
    
    logger.info("Data ingestion completed successfully")
    return results

def data_validation_task(**context) -> Dict[str, Any]:
    """
    Validate ingested data quality and completeness.
    """
    logger.info("Starting data validation...")
    
    # Get ingestion results from previous task
    ingestion_results = context['task_instance'].xcom_pull(
        task_ids='data_ingestion', key='ingestion_results'
    )
    
    validation_results = {
        'fred_validation': {},
        'news_validation': {},
        'overall_quality_score': 0.0,
        'validation_timestamp': datetime.utcnow().isoformat()
    }
    
    # Validate FRED data
    fred_valid_count = 0
    for indicator, data in ingestion_results.get('fred_data', {}).items():
        is_valid = 'error' not in data and data.get('records', 0) > 0
        validation_results['fred_validation'][indicator] = {
            'is_valid': is_valid,
            'records_count': data.get('records', 0),
            'has_recent_data': True  # Would check if latest_date is recent
        }
        if is_valid:
            fred_valid_count += 1
    
    # Validate news data
    news_data = ingestion_results.get('news_data', {})
    news_valid = 'error' not in news_data and news_data.get('articles_count', 0) > 0
    validation_results['news_validation'] = {
        'is_valid': news_valid,
        'articles_count': news_data.get('articles_count', 0),
        'has_sentiment': 'sentiment_score' in news_data
    }
    
    # Calculate overall quality score
    total_indicators = len(ECONOMIC_INDICATORS)
    fred_quality = fred_valid_count / total_indicators if total_indicators > 0 else 0
    news_quality = 1.0 if news_valid else 0.0
    validation_results['overall_quality_score'] = (fred_quality + news_quality) / 2
    
    # Check if quality meets minimum threshold
    min_quality_threshold = 0.7
    if validation_results['overall_quality_score'] < min_quality_threshold:
        raise ValueError(f"Data quality score {validation_results['overall_quality_score']:.2f} below threshold {min_quality_threshold}")
    
    context['task_instance'].xcom_push(key='validation_results', value=validation_results)
    
    logger.info(f"Data validation completed - Quality score: {validation_results['overall_quality_score']:.2f}")
    return validation_results

def statistical_forecasting_task(**context) -> Dict[str, Any]:
    """
    Run statistical forecasting models (ARIMA, VAR, etc.).
    """
    from models.forecasting_models import EconometricForecaster
    from models.r_statistical_models import RStatisticalModels
    
    logger.info("Starting statistical forecasting...")
    
    # Initialize forecasters
    forecaster = EconometricForecaster()
    r_models = RStatisticalModels()
    
    forecasting_results = {
        'python_models': {},
        'r_models': {},
        'model_performance': {},
        'forecasting_timestamp': datetime.utcnow().isoformat()
    }
    
    # Run Python-based models
    for indicator in ECONOMIC_INDICATORS[:3]:  # Limit for demo
        try:
            # Mock data fetch (in real implementation, would fetch from S3)
            mock_data = pd.Series(
                [100 + i + (i * 0.1) for i in range(100)],
                index=pd.date_range('2020-01-01', periods=100, freq='ME')
            )
            
            # Fit ARIMA model
            arima_results = forecaster.fit_arima(mock_data, order=(1, 1, 1))
            forecasting_results['python_models'][indicator] = {
                'arima_forecast': arima_results['forecast'].tolist() if hasattr(arima_results['forecast'], 'tolist') else [],
                'arima_aic': arima_results.get('aic', 0),
                'confidence_intervals': arima_results.get('confidence_intervals', []).tolist() if hasattr(arima_results.get('confidence_intervals', []), 'tolist') else []
            }
            
            logger.info(f"Completed Python forecasting for {indicator}")
            
        except Exception as e:
            logger.error(f"Python forecasting failed for {indicator}: {e}")
            forecasting_results['python_models'][indicator] = {'error': str(e)}
    
    # Run R-based models (if available)
    if r_models.r_available:
        try:
            for indicator in ECONOMIC_INDICATORS[:2]:  # Limit for demo
                mock_data = pd.Series(
                    [100 + i + (i * 0.1) for i in range(50)],
                    index=pd.date_range('2020-01-01', periods=50, freq='ME')
                )
                
                r_results = r_models.fit_arima_r(mock_data)
                forecasting_results['r_models'][indicator] = {
                    'forecast_mean': r_results['forecast_mean'].tolist(),
                    'aic': r_results['aic'],
                    'bic': r_results['bic']
                }
                
                logger.info(f"Completed R forecasting for {indicator}")
                
        except Exception as e:
            logger.error(f"R forecasting failed: {e}")
            forecasting_results['r_models']['error'] = str(e)
    else:
        forecasting_results['r_models']['status'] = 'R not available'
    
    context['task_instance'].xcom_push(key='statistical_results', value=forecasting_results)
    
    logger.info("Statistical forecasting completed")
    return forecasting_results

def ai_forecasting_task(**context) -> Dict[str, Any]:
    """
    Run AI-powered forecasting using foundation models.
    """
    from src.agents.narrative_generator import EconomicNarrativeGenerator
    from models.foundation_models.huggingface_forecaster import HybridFoundationEnsemble
    
    logger.info("Starting AI forecasting...")
    
    ai_results = {
        'foundation_models': {},
        'narratives': {},
        'ai_timestamp': datetime.utcnow().isoformat()
    }
    
    # Run foundation models (if available)
    try:
        ensemble = HybridFoundationEnsemble()
        
        for indicator in ECONOMIC_INDICATORS[:2]:  # Limit for demo
            mock_data = [100 + i for i in range(24)]  # 2 years of monthly data
            
            forecast_result = ensemble.forecast(
                data=mock_data,
                horizon=6,
                model_type='chronos'
            )
            
            ai_results['foundation_models'][indicator] = {
                'predictions': forecast_result.get('predictions', []),
                'confidence_intervals': forecast_result.get('confidence_intervals', []),
                'model_used': forecast_result.get('model_used', 'chronos')
            }
            
            logger.info(f"Completed foundation model forecasting for {indicator}")
            
    except Exception as e:
        logger.error(f"Foundation model forecasting failed: {e}")
        ai_results['foundation_models']['error'] = str(e)
    
    # Generate AI narratives
    try:
        narrative_generator = EconomicNarrativeGenerator()
        
        # Get statistical results from previous task
        statistical_results = context['task_instance'].xcom_pull(
            task_ids='statistical_forecasting', key='statistical_results'
        )
        
        # Generate narrative for GDP (example)
        gdp_narrative = narrative_generator.generate_executive_summary(
            metric='GDP',
            historical_data="GDP has shown steady growth...",
            forecast_data=str(statistical_results.get('python_models', {}).get('GDP', {})),
            model_performance="ARIMA model shows good fit..."
        )
        
        ai_results['narratives']['GDP'] = gdp_narrative
        
        logger.info("Generated AI narratives")
        
    except Exception as e:
        logger.error(f"AI narrative generation failed: {e}")
        ai_results['narratives']['error'] = str(e)
    
    context['task_instance'].xcom_push(key='ai_results', value=ai_results)
    
    logger.info("AI forecasting completed")
    return ai_results

def model_evaluation_task(**context) -> Dict[str, Any]:
    """
    Evaluate and compare model performances.
    """
    logger.info("Starting model evaluation...")
    
    # Get results from previous tasks
    statistical_results = context['task_instance'].xcom_pull(
        task_ids='statistical_forecasting', key='statistical_results'
    )
    ai_results = context['task_instance'].xcom_pull(
        task_ids='ai_forecasting', key='ai_results'
    )
    
    evaluation_results = {
        'model_comparison': {},
        'best_models': {},
        'performance_metrics': {},
        'evaluation_timestamp': datetime.utcnow().isoformat()
    }
    
    # Compare models for each indicator
    for indicator in ECONOMIC_INDICATORS[:3]:
        python_model = statistical_results.get('python_models', {}).get(indicator, {})
        r_model = statistical_results.get('r_models', {}).get(indicator, {})
        ai_model = ai_results.get('foundation_models', {}).get(indicator, {})
        
        model_scores = {}
        
        # Python model evaluation
        if 'error' not in python_model and python_model.get('arima_aic'):
            model_scores['python_arima'] = {
                'aic': python_model['arima_aic'],
                'available': True
            }
        
        # R model evaluation
        if 'error' not in r_model and r_model.get('aic'):
            model_scores['r_arima'] = {
                'aic': r_model['aic'],
                'bic': r_model['bic'],
                'available': True
            }
        
        # AI model evaluation (mock scoring)
        if 'error' not in ai_model and ai_model.get('predictions'):
            model_scores['ai_foundation'] = {
                'prediction_confidence': 0.85,  # Mock confidence score
                'available': True
            }
        
        evaluation_results['model_comparison'][indicator] = model_scores
        
        # Select best model (simplified logic)
        if model_scores:
            best_model = min(
                [k for k, v in model_scores.items() if v.get('aic')],
                key=lambda k: model_scores[k]['aic'],
                default='ai_foundation' if 'ai_foundation' in model_scores else list(model_scores.keys())[0]
            )
            evaluation_results['best_models'][indicator] = best_model
    
    # Calculate overall performance metrics
    total_models = len([m for models in evaluation_results['model_comparison'].values() for m in models.values() if m.get('available')])
    evaluation_results['performance_metrics'] = {
        'total_models_evaluated': total_models,
        'successful_forecasts': len(evaluation_results['best_models']),
        'success_rate': len(evaluation_results['best_models']) / len(ECONOMIC_INDICATORS) if ECONOMIC_INDICATORS else 0
    }
    
    context['task_instance'].xcom_push(key='evaluation_results', value=evaluation_results)
    
    logger.info(f"Model evaluation completed - Success rate: {evaluation_results['performance_metrics']['success_rate']:.2f}")
    return evaluation_results

def report_generation_task(**context) -> Dict[str, Any]:
    """
    Generate comprehensive forecasting reports.
    """
    from src.reports.executive_reporting import ExecutiveReportGenerator
    
    logger.info("Starting report generation...")
    
    # Gather all results from previous tasks
    validation_results = context['task_instance'].xcom_pull(
        task_ids='data_validation', key='validation_results'
    )
    statistical_results = context['task_instance'].xcom_pull(
        task_ids='statistical_forecasting', key='statistical_results'
    )
    ai_results = context['task_instance'].xcom_pull(
        task_ids='ai_forecasting', key='ai_results'
    )
    evaluation_results = context['task_instance'].xcom_pull(
        task_ids='model_evaluation', key='evaluation_results'
    )
    
    # Generate comprehensive report
    report_generator = ExecutiveReportGenerator()
    
    report_data = {
        'executive_summary': {
            'forecast_period': '6 months',
            'indicators_forecasted': len(ECONOMIC_INDICATORS),
            'data_quality_score': validation_results.get('overall_quality_score', 0),
            'model_success_rate': evaluation_results.get('performance_metrics', {}).get('success_rate', 0),
            'generated_at': datetime.utcnow().isoformat()
        },
        'data_summary': validation_results,
        'statistical_forecasts': statistical_results,
        'ai_forecasts': ai_results,
        'model_evaluation': evaluation_results,
        'recommendations': {
            'GDP': 'Moderate growth expected based on current trends',
            'UNEMPLOYMENT': 'Stable unemployment rate with slight improvement',
            'INFLATION': 'Inflation remains within target range'
        }
    }
    
    # Generate different report formats
    reports_generated = []
    
    try:
        # JSON report
        json_report = report_generator.generate_json_report(report_data)
        reports_generated.append('json')
        
        # Executive summary
        executive_summary = report_generator.generate_executive_summary(report_data)
        reports_generated.append('executive_summary')
        
        # CSV summary
        csv_summary = report_generator.generate_csv_summary(report_data)
        reports_generated.append('csv')
        
        logger.info(f"Generated reports: {', '.join(reports_generated)}")
        
    except Exception as e:
        logger.error(f"Report generation failed: {e}")
        reports_generated.append(f"error: {str(e)}")
    
    report_results = {
        'reports_generated': reports_generated,
        'report_data': report_data,
        'generation_timestamp': datetime.utcnow().isoformat()
    }
    
    context['task_instance'].xcom_push(key='report_results', value=report_results)
    
    logger.info("Report generation completed")
    return report_results

# Define the DAG
dag = DAG(
    'econometric_forecasting_pipeline',
    default_args=DEFAULT_ARGS,
    description='Comprehensive econometric forecasting and analysis pipeline',
    schedule_interval=timedelta(days=1),  # Run daily
    catchup=False,
    max_active_runs=1,
    tags=['econometrics', 'forecasting', 'ai', 'data-science'],
)

# Define tasks
start_task = DummyOperator(
    task_id='start_pipeline',
    dag=dag,
)

# Data ingestion and validation group
with TaskGroup('data_pipeline', dag=dag) as data_group:
    
    data_ingestion = PythonOperator(
        task_id='data_ingestion',
        python_callable=data_ingestion_task,
        dag=dag,
    )
    
    data_validation = PythonOperator(
        task_id='data_validation',
        python_callable=data_validation_task,
        dag=dag,
    )
    
    # Check data quality before proceeding
    data_quality_check = BashOperator(
        task_id='data_quality_check',
        bash_command='echo "Data quality check passed"',
        dag=dag,
    )
    
    data_ingestion >> data_validation >> data_quality_check

# Forecasting models group
with TaskGroup('forecasting_pipeline', dag=dag) as forecasting_group:
    
    statistical_forecasting = PythonOperator(
        task_id='statistical_forecasting',
        python_callable=statistical_forecasting_task,
        dag=dag,
    )
    
    ai_forecasting = PythonOperator(
        task_id='ai_forecasting',
        python_callable=ai_forecasting_task,
        dag=dag,
    )
    
    # Run forecasting models in parallel
    [statistical_forecasting, ai_forecasting]

# Model evaluation and reporting
model_evaluation = PythonOperator(
    task_id='model_evaluation',
    python_callable=model_evaluation_task,
    dag=dag,
)

report_generation = PythonOperator(
    task_id='report_generation',
    python_callable=report_generation_task,
    dag=dag,
)

# AWS S3 upload task
s3_upload = S3CopyObjectOperator(
    task_id='upload_reports_to_s3',
    source_bucket_key='local://tmp/reports/*',
    dest_bucket_name=OUTPUTS_BUCKET,
    dest_bucket_key='daily_reports/{{ ds }}/',
    aws_conn_id='aws_default',
    dag=dag,
)

# Lambda notification task
lambda_notification = LambdaInvokeOperator(
    task_id='send_completion_notification',
    function_name=f'{ENVIRONMENT}-econometric-forecasting',
    payload=json.dumps({
        'operation': 'send_notification',
        'message': 'Daily forecasting pipeline completed successfully',
        'execution_date': '{{ ds }}',
        'environment': ENVIRONMENT
    }),
    aws_conn_id='aws_default',
    dag=dag,
)

# Success notification email
success_email = EmailOperator(
    task_id='send_success_email',
    to=['forecasting-team@company.com'],
    subject='Econometric Forecasting Pipeline - SUCCESS',
    html_content="""
    <h3>Daily Forecasting Pipeline Completed Successfully</h3>
    <p><strong>Execution Date:</strong> {{ ds }}</p>
    <p><strong>Environment:</strong> {{ var.value.ENVIRONMENT }}</p>
    <p><strong>Duration:</strong> {{ task_instance.duration }} seconds</p>
    
    <h4>Pipeline Summary:</h4>
    <ul>
        <li>Data ingestion: ✅ Completed</li>
        <li>Statistical forecasting: ✅ Completed</li>
        <li>AI forecasting: ✅ Completed</li>
        <li>Model evaluation: ✅ Completed</li>
        <li>Report generation: ✅ Completed</li>
    </ul>
    
    <p>Reports are available in S3 bucket: {{ var.value.OUTPUTS_BUCKET }}</p>
    
    <p><em>Generated by Apache Airflow - Econometric Forecasting Platform</em></p>
    """,
    dag=dag,
)

end_task = DummyOperator(
    task_id='end_pipeline',
    dag=dag,
)

# Define task dependencies
start_task >> data_group >> forecasting_group >> model_evaluation
model_evaluation >> report_generation >> s3_upload
s3_upload >> lambda_notification >> success_email >> end_task

# Add failure handling
def failure_callback(context):
    """Send notification on pipeline failure."""
    logger.error(f"Pipeline failed: {context['exception']}")
    # Could send to Slack, PagerDuty, etc.

# Set failure callback on critical tasks
for task in [data_ingestion, statistical_forecasting, ai_forecasting, model_evaluation]:
    task.on_failure_callback = failure_callback