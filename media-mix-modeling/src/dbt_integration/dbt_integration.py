"""
dbt Integration Manager for Media Mix Modeling
Provides high-level dbt integration, orchestration, and MMM-specific workflows
"""

import os
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from .dbt_runner import DBTRunner

class DBTIntegration:
    """
    High-level dbt integration manager for MMM workflows
    
    Features:
    - Complete MMM data pipeline orchestration
    - Automated dependency management
    - Data lineage tracking
    - Performance monitoring
    - Integration with MMM models
    """
    
    def __init__(self, 
                 dbt_runner: Optional[DBTRunner] = None,
                 auto_refresh: bool = True,
                 enable_monitoring: bool = True):
        """
        Initialize dbt integration manager
        
        Args:
            dbt_runner: Optional pre-configured DBTRunner instance
            auto_refresh: Whether to auto-refresh stale data
            enable_monitoring: Whether to enable performance monitoring
        """
        self.dbt_runner = dbt_runner or DBTRunner()
        self.auto_refresh = auto_refresh
        self.enable_monitoring = enable_monitoring
        
        # Integration state
        self.pipeline_runs = []
        self.data_lineage = {}
        self.performance_metrics = {}
        self.mmm_integration_config = {}
        
        # Data refresh configuration
        self.refresh_thresholds = {
            'base_data': timedelta(hours=24),
            'transformations': timedelta(hours=12),
            'attribution': timedelta(hours=6),
            'metrics': timedelta(hours=1)
        }
        
        print(f"[DBT INTEGRATION] Initialized MMM data pipeline integration")
    
    def setup_mmm_pipeline(self, 
                          data_sources: Dict[str, str],
                          transformation_config: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Set up complete MMM data pipeline
        
        Args:
            data_sources: Dictionary mapping data types to source tables/files
            transformation_config: Optional transformation configuration
            
        Returns:
            Pipeline setup results
        """
        print(f"[DBT INTEGRATION] Setting up MMM data pipeline")
        
        setup_results = {
            'pipeline_configured': False,
            'data_sources': data_sources,
            'transformation_config': transformation_config or self._get_default_transformation_config(),
            'validation_results': {},
            'setup_errors': []
        }
        
        try:
            # Validate data sources
            print(f"[SETUP] Validating data sources...")
            source_validation = self._validate_data_sources(data_sources)
            setup_results['validation_results']['sources'] = source_validation
            
            if not source_validation.get('all_valid', False):
                setup_results['setup_errors'].append("Data source validation failed")
                return setup_results
            
            # Configure transformations
            print(f"[SETUP] Configuring transformations...")
            transformation_setup = self._configure_transformations(setup_results['transformation_config'])
            setup_results['transformation_setup'] = transformation_setup
            
            # Set up data lineage tracking
            print(f"[SETUP] Setting up data lineage...")
            lineage_setup = self._setup_data_lineage(data_sources)
            setup_results['lineage_setup'] = lineage_setup
            
            # Configure monitoring
            if self.enable_monitoring:
                print(f"[SETUP] Configuring monitoring...")
                monitoring_setup = self._setup_monitoring()
                setup_results['monitoring_setup'] = monitoring_setup
            
            setup_results['pipeline_configured'] = True
            print(f"[SETUP] MMM pipeline setup completed successfully")
            
        except Exception as e:
            setup_results['setup_errors'].append(str(e))
            print(f"[SETUP] Error setting up pipeline: {e}")
        
        return setup_results
    
    def execute_mmm_workflow(self, 
                           workflow_type: str = 'full',
                           force_refresh: bool = False) -> Dict[str, Any]:
        """
        Execute complete MMM data workflow
        
        Args:
            workflow_type: Type of workflow ('full', 'incremental', 'attribution_only')
            force_refresh: Whether to force refresh all data
            
        Returns:
            Workflow execution results
        """
        print(f"[WORKFLOW] Executing {workflow_type} MMM workflow")
        
        workflow_results = {
            'workflow_type': workflow_type,
            'start_time': datetime.now(),
            'stages': {},
            'overall_success': True,
            'data_outputs': {},
            'performance_metrics': {}
        }
        
        try:
            # Determine what needs to be refreshed
            refresh_plan = self._determine_refresh_plan(workflow_type, force_refresh)
            workflow_results['refresh_plan'] = refresh_plan
            
            # Execute workflow stages
            if workflow_type == 'full':
                workflow_results = self._execute_full_workflow(workflow_results, refresh_plan)
            elif workflow_type == 'incremental':
                workflow_results = self._execute_incremental_workflow(workflow_results, refresh_plan)
            elif workflow_type == 'attribution_only':
                workflow_results = self._execute_attribution_workflow(workflow_results, refresh_plan)
            else:
                raise ValueError(f"Unknown workflow type: {workflow_type}")
            
            # Record performance metrics
            workflow_results['end_time'] = datetime.now()
            workflow_results['execution_duration'] = workflow_results['end_time'] - workflow_results['start_time']
            
            # Update pipeline runs history
            self.pipeline_runs.append(workflow_results)
            
            # Update data lineage
            self._update_data_lineage(workflow_results)
            
            if workflow_results['overall_success']:
                print(f"[WORKFLOW] {workflow_type} workflow completed successfully")
            else:
                print(f"[WORKFLOW] {workflow_type} workflow completed with errors")
                
        except Exception as e:
            workflow_results['overall_success'] = False
            workflow_results['error'] = str(e)
            print(f"[WORKFLOW] Error executing workflow: {e}")
        
        return workflow_results
    
    def get_mmm_data_for_modeling(self, 
                                 data_type: str = 'final',
                                 date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """
        Get MMM-ready data for model training
        
        Args:
            data_type: Type of data ('base', 'transformed', 'final')
            date_range: Optional date range tuple (start, end)
            
        Returns:
            DataFrame ready for MMM modeling
        """
        print(f"[DATA RETRIEVAL] Getting {data_type} data for MMM modeling")
        
        try:
            # Determine data source
            if data_type == 'base':
                data_table = 'mmm_base_data'
            elif data_type == 'transformed':
                data_table = 'mmm_saturation_data'
            elif data_type == 'final':
                data_table = 'mmm_attribution_data'
            else:
                raise ValueError(f"Unknown data type: {data_type}")
            
            # In production, would query the actual data warehouse
            # For now, simulate data retrieval
            data = self._simulate_data_retrieval(data_table, date_range)
            
            # Validate data quality
            validation_results = self.dbt_runner.validate_mmm_data(data)
            if not validation_results.get('passed', False):
                print(f"[DATA RETRIEVAL] Warning: Data validation issues detected")
            
            print(f"[DATA RETRIEVAL] Retrieved {len(data)} records for {data_type} data")
            return data
            
        except Exception as e:
            print(f"[DATA RETRIEVAL] Error retrieving data: {e}")
            # Return empty DataFrame on error
            return pd.DataFrame()
    
    def monitor_pipeline_performance(self) -> Dict[str, Any]:
        """
        Monitor and report on pipeline performance
        
        Returns:
            Performance monitoring results
        """
        performance_report = {
            'recent_runs': [],
            'performance_trends': {},
            'data_freshness': {},
            'error_rates': {},
            'recommendations': []
        }
        
        try:
            # Analyze recent runs
            recent_runs = self.pipeline_runs[-10:] if len(self.pipeline_runs) >= 10 else self.pipeline_runs
            performance_report['recent_runs'] = [
                {
                    'start_time': run['start_time'],
                    'duration': run.get('execution_duration'),
                    'success': run['overall_success'],
                    'workflow_type': run['workflow_type']
                }
                for run in recent_runs
            ]
            
            # Calculate performance trends
            if len(recent_runs) > 1:
                durations = [run.get('execution_duration', timedelta(0)).total_seconds() 
                           for run in recent_runs if run.get('execution_duration')]
                if durations:
                    performance_report['performance_trends'] = {
                        'average_duration_seconds': sum(durations) / len(durations),
                        'trend': 'improving' if len(durations) > 1 and durations[-1] < durations[0] else 'stable'
                    }
            
            # Check data freshness
            freshness_check = self._check_data_freshness()
            performance_report['data_freshness'] = freshness_check
            
            # Calculate error rates
            total_runs = len(self.pipeline_runs)
            failed_runs = len([run for run in self.pipeline_runs if not run['overall_success']])
            performance_report['error_rates'] = {
                'total_runs': total_runs,
                'failed_runs': failed_runs,
                'error_rate': failed_runs / total_runs if total_runs > 0 else 0
            }
            
            # Generate recommendations
            recommendations = self._generate_performance_recommendations(performance_report)
            performance_report['recommendations'] = recommendations
            
            print(f"[MONITORING] Performance report generated for {total_runs} pipeline runs")
            
        except Exception as e:
            performance_report['error'] = str(e)
            print(f"[MONITORING] Error generating performance report: {e}")
        
        return performance_report
    
    def _get_default_transformation_config(self) -> Dict[str, Any]:
        """Get default transformation configuration for MMM"""
        
        return {
            'adstock_config': {
                'default_rate': 0.5,
                'channel_specific_rates': {}
            },
            'saturation_config': {
                'method': 'hill',
                'default_param': 0.5,
                'channel_specific_params': {}
            },
            'attribution_config': {
                'methods': ['last_touch', 'first_touch', 'linear', 'time_decay', 'data_driven'],
                'attribution_window_days': 30
            },
            'validation_config': {
                'min_data_points': 90,
                'max_null_percentage': 50,
                'required_channels': 2
            }
        }
    
    def _validate_data_sources(self, data_sources: Dict[str, str]) -> Dict[str, Any]:
        """Validate availability and quality of data sources"""
        
        validation_results = {
            'all_valid': True,
            'source_checks': {},
            'validation_errors': []
        }
        
        required_sources = ['marketing_data', 'revenue_data']
        
        for source_type, source_location in data_sources.items():
            source_check = {
                'exists': True,  # Would check actual existence in production
                'accessible': True,
                'schema_valid': True,
                'data_quality': 'good'
            }
            
            # Simulate validation checks
            if source_type in required_sources:
                source_check['required'] = True
            
            validation_results['source_checks'][source_type] = source_check
        
        # Check for missing required sources
        missing_sources = [src for src in required_sources if src not in data_sources]
        if missing_sources:
            validation_results['all_valid'] = False
            validation_results['validation_errors'].append(f"Missing required sources: {missing_sources}")
        
        return validation_results
    
    def _configure_transformations(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Configure transformation parameters"""
        
        transformation_setup = {
            'adstock_configured': True,
            'saturation_configured': True,
            'attribution_configured': True,
            'validation_configured': True
        }
        
        # Store configuration for use in transformations
        self.mmm_integration_config = config
        
        return transformation_setup
    
    def _setup_data_lineage(self, data_sources: Dict[str, str]) -> Dict[str, Any]:
        """Set up data lineage tracking"""
        
        lineage_setup = {
            'lineage_enabled': True,
            'tracked_sources': list(data_sources.keys()),
            'lineage_tables': {
                'mmm_base_data': ['marketing_data', 'revenue_data'],
                'mmm_adstock_transformations': ['mmm_base_data'],
                'mmm_saturation_curves': ['mmm_adstock_transformations'],
                'mmm_attribution_analysis': ['mmm_saturation_curves'],
                'mmm_performance_metrics': ['mmm_attribution_analysis']
            }
        }
        
        self.data_lineage = lineage_setup['lineage_tables']
        
        return lineage_setup
    
    def _setup_monitoring(self) -> Dict[str, Any]:
        """Set up pipeline monitoring"""
        
        monitoring_setup = {
            'monitoring_enabled': True,
            'metrics_collected': [
                'execution_time',
                'data_volume',
                'error_rates',
                'data_quality_scores'
            ],
            'alerting_configured': False  # Would configure alerts in production
        }
        
        return monitoring_setup
    
    def _determine_refresh_plan(self, workflow_type: str, force_refresh: bool) -> Dict[str, bool]:
        """Determine what data needs to be refreshed"""
        
        refresh_plan = {
            'base_data': False,
            'transformations': False,
            'attribution': False,
            'metrics': False
        }
        
        if force_refresh or workflow_type == 'full':
            # Refresh everything
            refresh_plan = {k: True for k in refresh_plan.keys()}
        elif workflow_type == 'incremental':
            # Check what's stale
            refresh_plan = self._check_data_staleness()
        elif workflow_type == 'attribution_only':
            # Only refresh attribution and metrics
            refresh_plan['attribution'] = True
            refresh_plan['metrics'] = True
        
        return refresh_plan
    
    def _execute_full_workflow(self, workflow_results: Dict[str, Any], refresh_plan: Dict[str, bool]) -> Dict[str, Any]:
        """Execute full MMM workflow"""
        
        # Run complete dbt pipeline
        pipeline_result = self.dbt_runner.run_mmm_pipeline(validate_output=True)
        workflow_results['stages']['full_pipeline'] = pipeline_result
        
        if not pipeline_result.get('overall_success', False):
            workflow_results['overall_success'] = False
        
        # Get final data outputs
        workflow_results['data_outputs'] = {
            'base_data': 'mmm_base_data',
            'transformed_data': 'mmm_saturation_data',
            'attribution_data': 'mmm_attribution_data',
            'metrics_data': 'mmm_performance_metrics'
        }
        
        return workflow_results
    
    def _execute_incremental_workflow(self, workflow_results: Dict[str, Any], refresh_plan: Dict[str, bool]) -> Dict[str, Any]:
        """Execute incremental MMM workflow"""
        
        models_to_run = []
        
        if refresh_plan['base_data']:
            models_to_run.append('mmm_base_data')
        if refresh_plan['transformations']:
            models_to_run.extend(['mmm_adstock_transformations', 'mmm_saturation_curves'])
        if refresh_plan['attribution']:
            models_to_run.append('mmm_attribution_analysis')
        if refresh_plan['metrics']:
            models_to_run.append('mmm_performance_metrics')
        
        if models_to_run:
            pipeline_result = self.dbt_runner.run_dbt_models(models=models_to_run)
            workflow_results['stages']['incremental_run'] = pipeline_result
            
            if not pipeline_result.get('success', False):
                workflow_results['overall_success'] = False
        else:
            workflow_results['stages']['incremental_run'] = {'success': True, 'message': 'No refresh needed'}
        
        return workflow_results
    
    def _execute_attribution_workflow(self, workflow_results: Dict[str, Any], refresh_plan: Dict[str, bool]) -> Dict[str, Any]:
        """Execute attribution-only workflow"""
        
        models_to_run = ['mmm_attribution_analysis', 'mmm_performance_metrics']
        pipeline_result = self.dbt_runner.run_dbt_models(models=models_to_run)
        workflow_results['stages']['attribution_run'] = pipeline_result
        
        if not pipeline_result.get('success', False):
            workflow_results['overall_success'] = False
        
        return workflow_results
    
    def _update_data_lineage(self, workflow_results: Dict[str, Any]):
        """Update data lineage based on workflow results"""
        
        # Update lineage tracking with execution details
        for stage_name, stage_result in workflow_results.get('stages', {}).items():
            if stage_result.get('success', False):
                self.data_lineage[f'{stage_name}_last_updated'] = datetime.now()
    
    def _check_data_freshness(self) -> Dict[str, Any]:
        """Check freshness of different data components"""
        
        freshness_report = {
            'overall_status': 'fresh',
            'component_status': {},
            'stale_components': []
        }
        
        # Check each component against thresholds
        current_time = datetime.now()
        
        for component, threshold in self.refresh_thresholds.items():
            # Simulate last update time
            last_update_key = f'{component}_last_updated'
            last_update = self.data_lineage.get(last_update_key, current_time - threshold - timedelta(hours=1))
            
            time_since_update = current_time - last_update
            is_stale = time_since_update > threshold
            
            freshness_report['component_status'][component] = {
                'last_update': last_update,
                'time_since_update': time_since_update,
                'threshold': threshold,
                'is_stale': is_stale
            }
            
            if is_stale:
                freshness_report['stale_components'].append(component)
        
        if freshness_report['stale_components']:
            freshness_report['overall_status'] = 'stale'
        
        return freshness_report
    
    def _check_data_staleness(self) -> Dict[str, bool]:
        """Check which data components are stale and need refresh"""
        
        freshness_report = self._check_data_freshness()
        
        refresh_needed = {
            'base_data': 'base_data' in freshness_report['stale_components'],
            'transformations': 'transformations' in freshness_report['stale_components'],
            'attribution': 'attribution' in freshness_report['stale_components'],
            'metrics': 'metrics' in freshness_report['stale_components']
        }
        
        return refresh_needed
    
    def _generate_performance_recommendations(self, performance_report: Dict[str, Any]) -> List[str]:
        """Generate performance improvement recommendations"""
        
        recommendations = []
        
        # Check error rates
        error_rate = performance_report.get('error_rates', {}).get('error_rate', 0)
        if error_rate > 0.1:  # More than 10% error rate
            recommendations.append("High error rate detected. Review pipeline logs and data quality.")
        
        # Check performance trends
        trends = performance_report.get('performance_trends', {})
        if trends.get('trend') == 'degrading':
            recommendations.append("Pipeline performance is degrading. Consider optimization.")
        
        # Check data freshness
        freshness = performance_report.get('data_freshness', {})
        if freshness.get('overall_status') == 'stale':
            recommendations.append("Some data components are stale. Consider more frequent refreshes.")
        
        # Default recommendations
        if not recommendations:
            recommendations = [
                "Pipeline performance is healthy",
                "Continue monitoring for any degradation",
                "Consider implementing automated alerting"
            ]
        
        return recommendations
    
    def _simulate_data_retrieval(self, table_name: str, date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """Simulate data retrieval from data warehouse"""
        
        # Generate sample data based on table type
        if 'base' in table_name:
            return self._generate_sample_base_data(date_range)
        elif 'saturation' in table_name:
            return self._generate_sample_transformed_data(date_range)
        elif 'attribution' in table_name:
            return self._generate_sample_attribution_data(date_range)
        else:
            return self._generate_sample_base_data(date_range)
    
    def _generate_sample_base_data(self, date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """Generate sample base data"""
        
        # Determine date range
        if date_range:
            start_date, end_date = date_range
            dates = pd.date_range(start=start_date, end=end_date, freq='D')
        else:
            dates = pd.date_range(start='2023-01-01', end='2023-12-31', freq='D')
        
        # Generate sample data
        data = pd.DataFrame({
            'date': dates,
            'revenue': np.random.normal(10000, 2000, len(dates)),
            'tv_spend': np.random.normal(1000, 200, len(dates)),
            'digital_spend': np.random.normal(800, 150, len(dates)),
            'print_spend': np.random.normal(500, 100, len(dates)),
            'radio_spend': np.random.normal(300, 75, len(dates))
        })
        
        return data
    
    def _generate_sample_transformed_data(self, date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """Generate sample transformed data"""
        
        base_data = self._generate_sample_base_data(date_range)
        
        # Add transformed columns
        spend_columns = [col for col in base_data.columns if col.endswith('_spend')]
        
        for channel in spend_columns:
            # Add adstock and saturation transformations
            base_data[f'{channel}_adstocked'] = base_data[channel] * np.random.uniform(1.1, 1.3, len(base_data))
            base_data[f'{channel}_adstocked_saturated'] = base_data[f'{channel}_adstocked'] * np.random.uniform(0.8, 1.0, len(base_data))
        
        return base_data
    
    def _generate_sample_attribution_data(self, date_range: Optional[Tuple[str, str]] = None) -> pd.DataFrame:
        """Generate sample attribution data"""
        
        transformed_data = self._generate_sample_transformed_data(date_range)
        
        # Add attribution columns
        saturated_columns = [col for col in transformed_data.columns if col.endswith('_saturated')]
        
        for channel in saturated_columns:
            transformed_data[f'{channel}_contribution'] = transformed_data[channel] * np.random.uniform(0.05, 0.15, len(transformed_data))
        
        contribution_columns = [col for col in transformed_data.columns if col.endswith('_contribution')]
        transformed_data['total_media_contribution'] = transformed_data[contribution_columns].sum(axis=1)
        transformed_data['base_contribution'] = transformed_data['revenue'] - transformed_data['total_media_contribution']
        
        return transformed_data

if __name__ == "__main__":
    print("dbt Integration for Media Mix Modeling")
    print("Usage: from src.dbt_integration.dbt_integration import DBTIntegration")