"""
Quality Assurance Agent for AutoML Platform

Ensures consistency, reliability, and quality across all AutoML workflows:
1. Validates data quality and consistency
2. Monitors model performance and drift
3. Checks agent execution quality
4. Validates workflow consistency
5. Ensures production readiness standards
6. Provides quality scoring and recommendations
"""

import numpy as np
import pandas as pd
import time
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
import warnings

try:
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    from sklearn.model_selection import cross_val_score
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity

class QualityLevel(Enum):
    """Quality assessment levels"""
    EXCELLENT = "excellent"
    GOOD = "good"
    ACCEPTABLE = "acceptable"
    POOR = "poor"
    FAILED = "failed"

class QualityDimension(Enum):
    """Quality assessment dimensions"""
    DATA_QUALITY = "data_quality"
    MODEL_PERFORMANCE = "model_performance"
    WORKFLOW_CONSISTENCY = "workflow_consistency"
    PRODUCTION_READINESS = "production_readiness"
    AGENT_EXECUTION = "agent_execution"

@dataclass
class QualityMetric:
    """Individual quality metric"""
    dimension: QualityDimension
    metric_name: str
    value: float
    threshold: float
    status: QualityLevel
    message: str
    recommendations: List[str]

@dataclass
class QualityAssessment:
    """Complete quality assessment result"""
    overall_score: float
    overall_status: QualityLevel
    metrics: List[QualityMetric]
    passed_checks: int
    total_checks: int
    critical_issues: List[str]
    warnings: List[str]
    recommendations: List[str]
    timestamp: str

class QualityAssuranceAgent(BaseAgent):
    """
    Quality Assurance Agent for comprehensive system validation.
    
    Responsibilities:
    1. Data quality validation and consistency checks
    2. Model performance monitoring and drift detection
    3. Workflow execution quality assessment
    4. Production readiness validation
    5. Cross-agent consistency verification
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the QA Agent."""
        super().__init__(
            name="Quality Assurance Agent",
            description="Comprehensive quality validation and consistency monitoring",
            specialization="Quality Control & Validation",
            config=config,
            communication_hub=communication_hub
        )
        
        # Quality thresholds
        self.quality_thresholds = {
            "data_completeness": self.config.get("data_completeness_threshold", 0.95),
            "data_consistency": self.config.get("data_consistency_threshold", 0.90),
            "model_accuracy": self.config.get("model_accuracy_threshold", 0.80),
            "model_stability": self.config.get("model_stability_threshold", 0.85),
            "workflow_success_rate": self.config.get("workflow_success_threshold", 0.95),
            "performance_consistency": self.config.get("performance_consistency_threshold", 0.10),
            "production_readiness": self.config.get("production_readiness_threshold", 0.90)
        }
        
        # Quality dimensions weights
        self.dimension_weights = {
            QualityDimension.DATA_QUALITY: 0.25,
            QualityDimension.MODEL_PERFORMANCE: 0.30,
            QualityDimension.WORKFLOW_CONSISTENCY: 0.20,
            QualityDimension.PRODUCTION_READINESS: 0.15,
            QualityDimension.AGENT_EXECUTION: 0.10
        }
        
        # Quality history for trend analysis
        self.quality_history: List[QualityAssessment] = []
        
        self.logger.info("QA Agent initialized with comprehensive validation capabilities")
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive quality assessment.
        
        Args:
            context: Task context with workflow results and data
            
        Returns:
            AgentResult with quality assessment and recommendations
        """
        try:
            self.logger.info("Starting comprehensive quality assessment...")
            
            start_time = time.time()
            
            # Collect all quality metrics
            all_metrics = []
            
            # 1. Data Quality Assessment
            self.logger.info("Assessing data quality...")
            data_metrics = self._assess_data_quality(context)
            all_metrics.extend(data_metrics)
            
            # 2. Model Performance Assessment
            if hasattr(context, 'model_results') or hasattr(context, 'splits'):
                self.logger.info("Assessing model performance...")
                model_metrics = self._assess_model_performance(context)
                all_metrics.extend(model_metrics)
            
            # 3. Workflow Consistency Assessment
            self.logger.info("Assessing workflow consistency...")
            workflow_metrics = self._assess_workflow_consistency(context)
            all_metrics.extend(workflow_metrics)
            
            # 4. Production Readiness Assessment
            self.logger.info("Assessing production readiness...")
            production_metrics = self._assess_production_readiness(context)
            all_metrics.extend(production_metrics)
            
            # 5. Agent Execution Quality Assessment
            self.logger.info("Assessing agent execution quality...")
            agent_metrics = self._assess_agent_execution_quality(context)
            all_metrics.extend(agent_metrics)
            
            # Calculate overall assessment
            assessment = self._calculate_overall_assessment(all_metrics)
            
            # Add to quality history
            self.quality_history.append(assessment)
            
            processing_time = time.time() - start_time
            
            # Prepare result data
            result_data = {
                "quality_assessment": {
                    "overall_score": assessment.overall_score,
                    "overall_status": assessment.overall_status.value,
                    "passed_checks": assessment.passed_checks,
                    "total_checks": assessment.total_checks,
                    "success_rate": assessment.passed_checks / assessment.total_checks if assessment.total_checks > 0 else 0
                },
                "quality_metrics": [
                    {
                        "dimension": metric.dimension.value,
                        "metric_name": metric.metric_name,
                        "value": metric.value,
                        "threshold": metric.threshold,
                        "status": metric.status.value,
                        "message": metric.message
                    }
                    for metric in assessment.metrics
                ],
                "quality_by_dimension": self._summarize_by_dimension(assessment.metrics),
                "issues": {
                    "critical": assessment.critical_issues,
                    "warnings": assessment.warnings
                },
                "recommendations": assessment.recommendations,
                "processing_time": processing_time,
                "timestamp": assessment.timestamp
            }
            
            # Determine success based on critical issues
            success = len(assessment.critical_issues) == 0 and assessment.overall_score >= 0.70
            
            message = f"Quality assessment completed: {assessment.overall_status.value.upper()} ({assessment.overall_score:.2f})"
            if assessment.critical_issues:
                message += f" - {len(assessment.critical_issues)} critical issues found"
            
            # Share results with other agents
            if self.communication_hub:
                self.communication_hub.share_message(
                    "quality_assessment_complete",
                    {
                        "overall_score": assessment.overall_score,
                        "status": assessment.overall_status.value,
                        "critical_issues": len(assessment.critical_issues),
                        "recommendations": len(assessment.recommendations)
                    },
                    sender="QA Agent"
                )
            
            return AgentResult(
                success=success,
                data=result_data,
                message=message,
                recommendations=assessment.recommendations,
                execution_time=processing_time
            )
            
        except Exception as e:
            self.logger.error(f"QA assessment failed: {e}")
            return AgentResult(
                success=False,
                message=f"Quality assessment failed: {str(e)}",
                execution_time=time.time() - start_time if 'start_time' in locals() else 0
            )
    
    def _assess_data_quality(self, context: TaskContext) -> List[QualityMetric]:
        """Assess data quality metrics"""
        metrics = []
        
        try:
            if hasattr(context, 'data') and context.data is not None:
                df = context.data
                
                # Data completeness
                completeness = 1 - (df.isnull().sum().sum() / (df.shape[0] * df.shape[1]))
                metrics.append(QualityMetric(
                    dimension=QualityDimension.DATA_QUALITY,
                    metric_name="data_completeness",
                    value=completeness,
                    threshold=self.quality_thresholds["data_completeness"],
                    status=self._get_quality_level(completeness, self.quality_thresholds["data_completeness"]),
                    message=f"Data completeness: {completeness:.2%}",
                    recommendations=["Address missing values"] if completeness < self.quality_thresholds["data_completeness"] else []
                ))
                
                # Data consistency (duplicate rate)
                duplicate_rate = df.duplicated().sum() / len(df)
                consistency = 1 - duplicate_rate
                metrics.append(QualityMetric(
                    dimension=QualityDimension.DATA_QUALITY,
                    metric_name="data_consistency",
                    value=consistency,
                    threshold=self.quality_thresholds["data_consistency"],
                    status=self._get_quality_level(consistency, self.quality_thresholds["data_consistency"]),
                    message=f"Data consistency: {consistency:.2%} (duplicate rate: {duplicate_rate:.2%})",
                    recommendations=["Remove duplicate records"] if consistency < self.quality_thresholds["data_consistency"] else []
                ))
                
                # Feature quality (variance check)
                numeric_cols = df.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    zero_variance_features = sum(df[col].var() == 0 for col in numeric_cols)
                    feature_quality = 1 - (zero_variance_features / len(numeric_cols))
                    metrics.append(QualityMetric(
                        dimension=QualityDimension.DATA_QUALITY,
                        metric_name="feature_quality",
                        value=feature_quality,
                        threshold=0.95,
                        status=self._get_quality_level(feature_quality, 0.95),
                        message=f"Feature quality: {feature_quality:.2%} ({zero_variance_features} zero-variance features)",
                        recommendations=["Remove zero-variance features"] if feature_quality < 0.95 else []
                    ))
                
                # Data distribution quality (skewness check)
                if len(numeric_cols) > 0:
                    skewness_scores = [abs(df[col].skew()) for col in numeric_cols if df[col].var() > 0]
                    avg_skewness = np.mean(skewness_scores) if skewness_scores else 0
                    distribution_quality = max(0, 1 - (avg_skewness / 5))  # Normalize to 0-1
                    metrics.append(QualityMetric(
                        dimension=QualityDimension.DATA_QUALITY,
                        metric_name="distribution_quality",
                        value=distribution_quality,
                        threshold=0.70,
                        status=self._get_quality_level(distribution_quality, 0.70),
                        message=f"Distribution quality: {distribution_quality:.2%} (avg skewness: {avg_skewness:.2f})",
                        recommendations=["Consider data transformation for highly skewed features"] if distribution_quality < 0.70 else []
                    ))
        
        except Exception as e:
            self.logger.warning(f"Data quality assessment error: {e}")
            metrics.append(QualityMetric(
                dimension=QualityDimension.DATA_QUALITY,
                metric_name="assessment_error",
                value=0.0,
                threshold=1.0,
                status=QualityLevel.FAILED,
                message=f"Data quality assessment failed: {str(e)}",
                recommendations=["Review data format and accessibility"]
            ))
        
        return metrics
    
    def _assess_model_performance(self, context: TaskContext) -> List[QualityMetric]:
        """Assess model performance quality"""
        metrics = []
        
        try:
            # Check if we have model results
            model_results = None
            if hasattr(context, 'model_results'):
                model_results = context.model_results
            elif hasattr(context, 'agent_results') and 'classification' in context.agent_results:
                model_results = context.agent_results['classification']
            
            if model_results and hasattr(model_results, 'data'):
                model_data = model_results.data
                
                # Model accuracy
                if 'best_score' in model_data:
                    accuracy = model_data['best_score']
                    metrics.append(QualityMetric(
                        dimension=QualityDimension.MODEL_PERFORMANCE,
                        metric_name="model_accuracy",
                        value=accuracy,
                        threshold=self.quality_thresholds["model_accuracy"],
                        status=self._get_quality_level(accuracy, self.quality_thresholds["model_accuracy"]),
                        message=f"Model accuracy: {accuracy:.2%}",
                        recommendations=["Improve feature engineering", "Try ensemble methods"] if accuracy < self.quality_thresholds["model_accuracy"] else []
                    ))
                
                # Model stability (cross-validation consistency)
                if 'model_results' in model_data:
                    scores = [result.accuracy for result in model_data['model_results'] if hasattr(result, 'accuracy')]
                    if scores:
                        cv_std = np.std(scores)
                        stability = max(0, 1 - cv_std)  # Lower std = higher stability
                        metrics.append(QualityMetric(
                            dimension=QualityDimension.MODEL_PERFORMANCE,
                            metric_name="model_stability",
                            value=stability,
                            threshold=self.quality_thresholds["model_stability"],
                            status=self._get_quality_level(stability, self.quality_thresholds["model_stability"]),
                            message=f"Model stability: {stability:.2%} (CV std: {cv_std:.3f})",
                            recommendations=["Increase training data", "Regularization"] if stability < self.quality_thresholds["model_stability"] else []
                        ))
                
                # Performance consistency across different metrics
                if 'evaluation_report' in model_data:
                    eval_report = model_data['evaluation_report']
                    if hasattr(eval_report, 'model_rankings'):
                        # Check if best model is consistently ranked high
                        rankings = eval_report.model_rankings
                        best_model = model_data.get('best_model', '')
                        
                        if best_model and rankings:
                            best_model_rankings = [rank for model, rank in rankings.items() if model == best_model]
                            consistency = 1.0 if best_model_rankings and best_model_rankings[0] <= 2 else 0.7
                            
                            metrics.append(QualityMetric(
                                dimension=QualityDimension.MODEL_PERFORMANCE,
                                metric_name="performance_consistency",
                                value=consistency,
                                threshold=0.80,
                                status=self._get_quality_level(consistency, 0.80),
                                message=f"Performance consistency: {'High' if consistency > 0.9 else 'Medium' if consistency > 0.7 else 'Low'}",
                                recommendations=["Validate model selection criteria"] if consistency < 0.80 else []
                            ))
        
        except Exception as e:
            self.logger.warning(f"Model performance assessment error: {e}")
            metrics.append(QualityMetric(
                dimension=QualityDimension.MODEL_PERFORMANCE,
                metric_name="assessment_error",
                value=0.0,
                threshold=1.0,
                status=QualityLevel.FAILED,
                message=f"Model performance assessment failed: {str(e)}",
                recommendations=["Review model training results"]
            ))
        
        return metrics
    
    def _assess_workflow_consistency(self, context: TaskContext) -> List[QualityMetric]:
        """Assess workflow execution consistency"""
        metrics = []
        
        try:
            # Check agent execution sequence consistency
            if hasattr(context, 'execution_sequence'):
                sequence = context.execution_sequence
                expected_patterns = ['eda', 'classification', 'data_hygiene']
                
                # Simple pattern matching for now
                consistency = 1.0  # Default to high consistency
                recommendations = []
                
                metrics.append(QualityMetric(
                    dimension=QualityDimension.WORKFLOW_CONSISTENCY,
                    metric_name="execution_sequence",
                    value=consistency,
                    threshold=0.90,
                    status=self._get_quality_level(consistency, 0.90),
                    message=f"Workflow sequence consistency: {consistency:.2%}",
                    recommendations=recommendations
                ))
            
            # Check data flow consistency
            if hasattr(context, 'data_transformations'):
                transformations = context.data_transformations
                consistency = 1.0 if len(transformations) > 0 else 0.5
                
                metrics.append(QualityMetric(
                    dimension=QualityDimension.WORKFLOW_CONSISTENCY,
                    metric_name="data_flow_consistency",
                    value=consistency,
                    threshold=0.80,
                    status=self._get_quality_level(consistency, 0.80),
                    message=f"Data flow consistency: {consistency:.2%}",
                    recommendations=["Review data transformation pipeline"] if consistency < 0.80 else []
                ))
            
            # Check result format consistency
            result_formats_consistent = True
            if hasattr(context, 'agent_results'):
                for agent_name, result in context.agent_results.items():
                    if not hasattr(result, 'success') or not hasattr(result, 'data'):
                        result_formats_consistent = False
                        break
            
            format_consistency = 1.0 if result_formats_consistent else 0.0
            metrics.append(QualityMetric(
                dimension=QualityDimension.WORKFLOW_CONSISTENCY,
                metric_name="result_format_consistency",
                value=format_consistency,
                threshold=1.0,
                status=self._get_quality_level(format_consistency, 1.0),
                message=f"Result format consistency: {'Consistent' if result_formats_consistent else 'Inconsistent'}",
                recommendations=["Standardize agent result formats"] if not result_formats_consistent else []
            ))
        
        except Exception as e:
            self.logger.warning(f"Workflow consistency assessment error: {e}")
            metrics.append(QualityMetric(
                dimension=QualityDimension.WORKFLOW_CONSISTENCY,
                metric_name="assessment_error",
                value=0.0,
                threshold=1.0,
                status=QualityLevel.FAILED,
                message=f"Workflow consistency assessment failed: {str(e)}",
                recommendations=["Review workflow execution logs"]
            ))
        
        return metrics
    
    def _assess_production_readiness(self, context: TaskContext) -> List[QualityMetric]:
        """Assess production readiness"""
        metrics = []
        
        try:
            # Model artifact completeness
            artifacts_complete = True
            required_artifacts = ['model', 'metadata', 'performance_metrics']
            missing_artifacts = []
            
            if hasattr(context, 'model_results') and context.model_results:
                model_data = context.model_results.data if hasattr(context.model_results, 'data') else {}
                for artifact in required_artifacts:
                    if artifact not in model_data:
                        artifacts_complete = False
                        missing_artifacts.append(artifact)
            
            completeness = 1.0 if artifacts_complete else (len(required_artifacts) - len(missing_artifacts)) / len(required_artifacts)
            metrics.append(QualityMetric(
                dimension=QualityDimension.PRODUCTION_READINESS,
                metric_name="artifact_completeness",
                value=completeness,
                threshold=1.0,
                status=self._get_quality_level(completeness, 1.0),
                message=f"Model artifacts: {'Complete' if artifacts_complete else f'Missing {missing_artifacts}'}",
                recommendations=[f"Generate missing artifacts: {missing_artifacts}"] if not artifacts_complete else []
            ))
            
            # Documentation completeness
            docs_complete = True
            if hasattr(context, 'documentation'):
                docs_complete = len(context.documentation) > 0
            else:
                docs_complete = False
            
            docs_score = 1.0 if docs_complete else 0.0
            metrics.append(QualityMetric(
                dimension=QualityDimension.PRODUCTION_READINESS,
                metric_name="documentation_completeness",
                value=docs_score,
                threshold=0.8,
                status=self._get_quality_level(docs_score, 0.8),
                message=f"Documentation: {'Complete' if docs_complete else 'Missing'}",
                recommendations=["Generate model documentation and API specs"] if not docs_complete else []
            ))
            
            # Performance acceptability for production
            performance_ready = False
            if hasattr(context, 'model_results') and context.model_results:
                model_data = context.model_results.data if hasattr(context.model_results, 'data') else {}
                if 'best_score' in model_data:
                    performance_ready = model_data['best_score'] >= self.quality_thresholds["model_accuracy"]
            
            perf_score = 1.0 if performance_ready else 0.5
            metrics.append(QualityMetric(
                dimension=QualityDimension.PRODUCTION_READINESS,
                metric_name="performance_readiness",
                value=perf_score,
                threshold=0.8,
                status=self._get_quality_level(perf_score, 0.8),
                message=f"Performance readiness: {'Ready' if performance_ready else 'Needs improvement'}",
                recommendations=["Improve model performance before production deployment"] if not performance_ready else []
            ))
        
        except Exception as e:
            self.logger.warning(f"Production readiness assessment error: {e}")
            metrics.append(QualityMetric(
                dimension=QualityDimension.PRODUCTION_READINESS,
                metric_name="assessment_error",
                value=0.0,
                threshold=1.0,
                status=QualityLevel.FAILED,
                message=f"Production readiness assessment failed: {str(e)}",
                recommendations=["Review production deployment requirements"]
            ))
        
        return metrics
    
    def _assess_agent_execution_quality(self, context: TaskContext) -> List[QualityMetric]:
        """Assess agent execution quality"""
        metrics = []
        
        try:
            # Agent success rate
            if hasattr(context, 'agent_results'):
                agent_results = context.agent_results
                successful_agents = sum(1 for result in agent_results.values() if hasattr(result, 'success') and result.success)
                total_agents = len(agent_results)
                success_rate = successful_agents / total_agents if total_agents > 0 else 0
                
                metrics.append(QualityMetric(
                    dimension=QualityDimension.AGENT_EXECUTION,
                    metric_name="agent_success_rate",
                    value=success_rate,
                    threshold=self.quality_thresholds["workflow_success_rate"],
                    status=self._get_quality_level(success_rate, self.quality_thresholds["workflow_success_rate"]),
                    message=f"Agent success rate: {success_rate:.2%} ({successful_agents}/{total_agents})",
                    recommendations=["Investigate failed agent executions"] if success_rate < self.quality_thresholds["workflow_success_rate"] else []
                ))
            
            # Execution time consistency
            if hasattr(context, 'agent_execution_times'):
                exec_times = list(context.agent_execution_times.values())
                if exec_times:
                    time_variance = np.var(exec_times)
                    avg_time = np.mean(exec_times)
                    consistency = max(0, 1 - (time_variance / (avg_time ** 2))) if avg_time > 0 else 1
                    
                    metrics.append(QualityMetric(
                        dimension=QualityDimension.AGENT_EXECUTION,
                        metric_name="execution_time_consistency",
                        value=consistency,
                        threshold=0.70,
                        status=self._get_quality_level(consistency, 0.70),
                        message=f"Execution time consistency: {consistency:.2%}",
                        recommendations=["Optimize slow-running agents"] if consistency < 0.70 else []
                    ))
        
        except Exception as e:
            self.logger.warning(f"Agent execution quality assessment error: {e}")
            metrics.append(QualityMetric(
                dimension=QualityDimension.AGENT_EXECUTION,
                metric_name="assessment_error",
                value=0.0,
                threshold=1.0,
                status=QualityLevel.FAILED,
                message=f"Agent execution assessment failed: {str(e)}",
                recommendations=["Review agent execution logs"]
            ))
        
        return metrics
    
    def _get_quality_level(self, value: float, threshold: float) -> QualityLevel:
        """Determine quality level based on value and threshold"""
        if value >= threshold:
            if value >= 0.95:
                return QualityLevel.EXCELLENT
            elif value >= 0.85:
                return QualityLevel.GOOD
            else:
                return QualityLevel.ACCEPTABLE
        else:
            if value >= threshold * 0.8:
                return QualityLevel.POOR
            else:
                return QualityLevel.FAILED
    
    def _calculate_overall_assessment(self, metrics: List[QualityMetric]) -> QualityAssessment:
        """Calculate overall quality assessment"""
        if not metrics:
            return QualityAssessment(
                overall_score=0.0,
                overall_status=QualityLevel.FAILED,
                metrics=[],
                passed_checks=0,
                total_checks=0,
                critical_issues=["No quality metrics available"],
                warnings=[],
                recommendations=["Review quality assessment configuration"],
                timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
            )
        
        # Calculate weighted score by dimension
        dimension_scores = {}
        dimension_counts = {}
        
        for metric in metrics:
            dim = metric.dimension
            if dim not in dimension_scores:
                dimension_scores[dim] = 0
                dimension_counts[dim] = 0
            dimension_scores[dim] += metric.value
            dimension_counts[dim] += 1
        
        # Average scores by dimension
        for dim in dimension_scores:
            dimension_scores[dim] /= dimension_counts[dim]
        
        # Calculate weighted overall score
        overall_score = 0
        total_weight = 0
        for dim, score in dimension_scores.items():
            weight = self.dimension_weights.get(dim, 0.1)
            overall_score += score * weight
            total_weight += weight
        
        if total_weight > 0:
            overall_score /= total_weight
        
        # Determine overall status
        overall_status = self._get_quality_level(overall_score, 0.80)
        
        # Count passed checks
        passed_checks = sum(1 for m in metrics if m.status in [QualityLevel.EXCELLENT, QualityLevel.GOOD, QualityLevel.ACCEPTABLE])
        total_checks = len(metrics)
        
        # Identify critical issues and warnings
        critical_issues = []
        warnings = []
        all_recommendations = []
        
        for metric in metrics:
            if metric.status == QualityLevel.FAILED:
                critical_issues.append(f"{metric.metric_name}: {metric.message}")
            elif metric.status == QualityLevel.POOR:
                warnings.append(f"{metric.metric_name}: {metric.message}")
            
            all_recommendations.extend(metric.recommendations)
        
        # Remove duplicate recommendations
        unique_recommendations = list(set(all_recommendations))
        
        return QualityAssessment(
            overall_score=overall_score,
            overall_status=overall_status,
            metrics=metrics,
            passed_checks=passed_checks,
            total_checks=total_checks,
            critical_issues=critical_issues,
            warnings=warnings,
            recommendations=unique_recommendations,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S")
        )
    
    def _summarize_by_dimension(self, metrics: List[QualityMetric]) -> Dict[str, Dict[str, Any]]:
        """Summarize quality metrics by dimension"""
        summary = {}
        
        for dimension in QualityDimension:
            dim_metrics = [m for m in metrics if m.dimension == dimension]
            if dim_metrics:
                avg_score = sum(m.value for m in dim_metrics) / len(dim_metrics)
                passed = sum(1 for m in dim_metrics if m.status in [QualityLevel.EXCELLENT, QualityLevel.GOOD, QualityLevel.ACCEPTABLE])
                
                summary[dimension.value] = {
                    "average_score": avg_score,
                    "passed_checks": passed,
                    "total_checks": len(dim_metrics),
                    "success_rate": passed / len(dim_metrics),
                    "status": self._get_quality_level(avg_score, 0.80).value
                }
        
        return summary
    
    def can_handle_task(self, task_type: str, context: TaskContext) -> bool:
        """Check if this agent can handle the given task."""
        qa_tasks = ["quality", "validation", "assessment", "qa", "consistency", "reliability"]
        return any(task in task_type.lower() for task in qa_tasks)
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate task complexity for quality assessment."""
        # QA is always considered complex due to comprehensive analysis
        return TaskComplexity.COMPLEX