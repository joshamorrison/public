"""
Visualization and Reporting Tools for AutoML Agents

Tools for generating plots, charts, and comprehensive reports
that agents use to communicate insights and results.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings
from pathlib import Path

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

warnings.filterwarnings("ignore")


@dataclass
class PlotConfig:
    """Configuration for plot generation."""
    title: str
    xlabel: str = ""
    ylabel: str = ""
    figsize: Tuple[int, int] = (10, 6)
    style: str = "default"
    save_path: Optional[str] = None


@dataclass
class ReportSection:
    """Section of a generated report."""
    title: str
    content: str
    plots: List[str] = None
    data_tables: List[pd.DataFrame] = None


class PlotGenerator:
    """
    Automated plot generation tool for data visualization and model insights.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the plot generator."""
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/visualizations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_plots = []
    
    def create_data_overview_plots(self, df: pd.DataFrame, target_column: Optional[str] = None) -> List[str]:
        """
        Create overview plots for dataset exploration.
        
        Args:
            df: Input dataframe
            target_column: Name of target variable
            
        Returns:
            List of generated plot file paths
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._create_mock_plots("data_overview")
        
        plots = []
        
        # 1. Missing values heatmap
        if df.isnull().sum().sum() > 0:
            plot_path = self._create_missing_values_plot(df)
            if plot_path:
                plots.append(plot_path)
        
        # 2. Distribution plots for numeric columns
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            plot_path = self._create_distribution_plots(df, numeric_cols)
            if plot_path:
                plots.append(plot_path)
        
        # 3. Categorical distribution plots
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        if len(categorical_cols) > 0:
            plot_path = self._create_categorical_plots(df, categorical_cols)
            if plot_path:
                plots.append(plot_path)
        
        # 4. Correlation heatmap
        if len(numeric_cols) > 1:
            plot_path = self._create_correlation_heatmap(df, numeric_cols)
            if plot_path:
                plots.append(plot_path)
        
        # 5. Target variable analysis
        if target_column and target_column in df.columns:
            plot_path = self._create_target_analysis_plot(df, target_column)
            if plot_path:
                plots.append(plot_path)
        
        self.generated_plots.extend(plots)
        return plots
    
    def create_model_performance_plots(self, model_results: List[Any]) -> List[str]:
        """
        Create plots for model performance comparison.
        
        Args:
            model_results: List of model results
            
        Returns:
            List of generated plot file paths
        """
        if not MATPLOTLIB_AVAILABLE:
            return self._create_mock_plots("model_performance")
        
        plots = []
        
        # 1. Model comparison bar chart
        plot_path = self._create_model_comparison_plot(model_results)
        if plot_path:
            plots.append(plot_path)
        
        # 2. Training time vs performance scatter
        plot_path = self._create_performance_time_plot(model_results)
        if plot_path:
            plots.append(plot_path)
        
        # 3. Feature importance plot (if available)
        best_model = max(model_results, key=lambda x: x.validation_score, default=None)
        if best_model and best_model.feature_importance:
            plot_path = self._create_feature_importance_plot(best_model.feature_importance)
            if plot_path:
                plots.append(plot_path)
        
        self.generated_plots.extend(plots)
        return plots
    
    def _create_missing_values_plot(self, df: pd.DataFrame) -> Optional[str]:
        """Create missing values visualization."""
        try:
            plt.figure(figsize=(12, 8))
            missing_data = df.isnull().sum().sort_values(ascending=False)
            missing_data = missing_data[missing_data > 0]
            
            if len(missing_data) == 0:
                return None
            
            plt.subplot(2, 1, 1)
            missing_data.plot(kind='bar')
            plt.title('Missing Values by Column')
            plt.ylabel('Number of Missing Values')
            plt.xticks(rotation=45)
            
            plt.subplot(2, 1, 2)
            missing_percentage = (missing_data / len(df) * 100)
            missing_percentage.plot(kind='bar', color='orange')
            plt.title('Missing Values Percentage by Column')
            plt.ylabel('Percentage Missing')
            plt.xticks(rotation=45)
            
            plt.tight_layout()
            plot_path = self.output_dir / "missing_values_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        except Exception as e:
            print(f"Error creating missing values plot: {e}")
            return None
    
    def _create_distribution_plots(self, df: pd.DataFrame, numeric_cols: List[str]) -> Optional[str]:
        """Create distribution plots for numeric columns."""
        try:
            n_cols = min(len(numeric_cols), 6)  # Limit to 6 plots
            cols_to_plot = numeric_cols[:n_cols]
            
            fig, axes = plt.subplots(2, 3, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(cols_to_plot):
                if i < len(axes):
                    df[col].hist(bins=30, ax=axes[i], alpha=0.7)
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Frequency')
            
            # Hide unused subplots
            for i in range(len(cols_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = self.output_dir / "numeric_distributions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        except Exception as e:
            print(f"Error creating distribution plots: {e}")
            return None
    
    def _create_categorical_plots(self, df: pd.DataFrame, categorical_cols: List[str]) -> Optional[str]:
        """Create plots for categorical variables."""
        try:
            n_cols = min(len(categorical_cols), 4)  # Limit to 4 plots
            cols_to_plot = categorical_cols[:n_cols]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            axes = axes.ravel()
            
            for i, col in enumerate(cols_to_plot):
                if i < len(axes):
                    value_counts = df[col].value_counts().head(10)  # Top 10 categories
                    value_counts.plot(kind='bar', ax=axes[i])
                    axes[i].set_title(f'Distribution of {col}')
                    axes[i].set_xlabel(col)
                    axes[i].set_ylabel('Count')
                    axes[i].tick_params(axis='x', rotation=45)
            
            # Hide unused subplots
            for i in range(len(cols_to_plot), len(axes)):
                axes[i].set_visible(False)
            
            plt.tight_layout()
            plot_path = self.output_dir / "categorical_distributions.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        except Exception as e:
            print(f"Error creating categorical plots: {e}")
            return None
    
    def _create_correlation_heatmap(self, df: pd.DataFrame, numeric_cols: List[str]) -> Optional[str]:
        """Create correlation heatmap."""
        try:
            plt.figure(figsize=(12, 10))
            corr_matrix = df[numeric_cols].corr()
            
            mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
            sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='coolwarm', center=0,
                       square=True, fmt='.2f', cbar_kws={"shrink": .8})
            plt.title('Feature Correlation Heatmap')
            plt.tight_layout()
            
            plot_path = self.output_dir / "correlation_heatmap.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        except Exception as e:
            print(f"Error creating correlation heatmap: {e}")
            return None
    
    def _create_target_analysis_plot(self, df: pd.DataFrame, target_column: str) -> Optional[str]:
        """Create target variable analysis plot."""
        try:
            plt.figure(figsize=(12, 6))
            
            if df[target_column].dtype in ['object', 'category'] or df[target_column].nunique() < 10:
                # Categorical target
                plt.subplot(1, 2, 1)
                df[target_column].value_counts().plot(kind='bar')
                plt.title(f'Distribution of {target_column}')
                plt.ylabel('Count')
                plt.xticks(rotation=45)
                
                plt.subplot(1, 2, 2)
                df[target_column].value_counts().plot(kind='pie', autopct='%1.1f%%')
                plt.title(f'Proportion of {target_column}')
                plt.ylabel('')
            else:
                # Continuous target
                plt.subplot(1, 2, 1)
                df[target_column].hist(bins=30, alpha=0.7)
                plt.title(f'Distribution of {target_column}')
                plt.xlabel(target_column)
                plt.ylabel('Frequency')
                
                plt.subplot(1, 2, 2)
                df[target_column].plot(kind='box')
                plt.title(f'Box Plot of {target_column}')
                plt.ylabel(target_column)
            
            plt.tight_layout()
            plot_path = self.output_dir / "target_analysis.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        except Exception as e:
            print(f"Error creating target analysis plot: {e}")
            return None
    
    def _create_model_comparison_plot(self, model_results: List[Any]) -> Optional[str]:
        """Create model performance comparison plot."""
        try:
            plt.figure(figsize=(12, 8))
            
            model_names = [result.model_name for result in model_results]
            val_scores = [result.validation_score for result in model_results]
            train_scores = [result.train_score for result in model_results]
            
            x = np.arange(len(model_names))
            width = 0.35
            
            plt.bar(x - width/2, train_scores, width, label='Training Score', alpha=0.8)
            plt.bar(x + width/2, val_scores, width, label='Validation Score', alpha=0.8)
            
            plt.xlabel('Model')
            plt.ylabel('Score')
            plt.title('Model Performance Comparison')
            plt.xticks(x, model_names, rotation=45)
            plt.legend()
            plt.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / "model_comparison.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        except Exception as e:
            print(f"Error creating model comparison plot: {e}")
            return None
    
    def _create_performance_time_plot(self, model_results: List[Any]) -> Optional[str]:
        """Create performance vs training time scatter plot."""
        try:
            plt.figure(figsize=(10, 6))
            
            val_scores = [result.validation_score for result in model_results]
            train_times = [result.training_time for result in model_results]
            model_names = [result.model_name for result in model_results]
            
            plt.scatter(train_times, val_scores, s=100, alpha=0.7)
            
            for i, name in enumerate(model_names):
                plt.annotate(name, (train_times[i], val_scores[i]), 
                           xytext=(5, 5), textcoords='offset points')
            
            plt.xlabel('Training Time (seconds)')
            plt.ylabel('Validation Score')
            plt.title('Model Performance vs Training Time')
            plt.grid(alpha=0.3)
            
            plt.tight_layout()
            plot_path = self.output_dir / "performance_vs_time.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        except Exception as e:
            print(f"Error creating performance vs time plot: {e}")
            return None
    
    def _create_feature_importance_plot(self, feature_importance: Dict[str, float]) -> Optional[str]:
        """Create feature importance plot."""
        try:
            plt.figure(figsize=(10, 8))
            
            # Sort features by importance
            sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
            features, importances = zip(*sorted_features[:15])  # Top 15 features
            
            plt.barh(range(len(features)), importances)
            plt.yticks(range(len(features)), features)
            plt.xlabel('Feature Importance')
            plt.title('Top Feature Importances')
            plt.gca().invert_yaxis()
            
            plt.tight_layout()
            plot_path = self.output_dir / "feature_importance.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return str(plot_path)
        except Exception as e:
            print(f"Error creating feature importance plot: {e}")
            return None
    
    def _create_mock_plots(self, plot_type: str) -> List[str]:
        """Create mock plot paths when matplotlib is not available."""
        mock_plots = {
            "data_overview": [
                "missing_values_analysis.png",
                "numeric_distributions.png", 
                "correlation_heatmap.png",
                "target_analysis.png"
            ],
            "model_performance": [
                "model_comparison.png",
                "performance_vs_time.png",
                "feature_importance.png"
            ]
        }
        
        return [str(self.output_dir / plot) for plot in mock_plots.get(plot_type, [])]


class ReportGenerator:
    """
    Automated report generation tool for creating comprehensive ML analysis reports.
    """
    
    def __init__(self, output_dir: Optional[str] = None):
        """Initialize the report generator."""
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.generated_reports = []
    
    def generate_data_analysis_report(self, data_profile: Any, plots: List[str]) -> str:
        """Generate comprehensive data analysis report."""
        
        sections = [
            self._create_executive_summary_section(data_profile),
            self._create_data_overview_section(data_profile),
            self._create_data_quality_section(data_profile),
            self._create_recommendations_section(data_profile)
        ]
        
        # Generate full report
        report_content = self._compile_report("Data Analysis Report", sections)
        
        # Save report
        report_path = self.output_dir / f"data_analysis_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.generated_reports.append(str(report_path))
        return str(report_path)
    
    def generate_model_performance_report(self, evaluation_report: Any, plots: List[str]) -> str:
        """Generate comprehensive model performance report."""
        
        sections = [
            self._create_model_summary_section(evaluation_report),
            self._create_performance_analysis_section(evaluation_report),
            self._create_model_recommendations_section(evaluation_report)
        ]
        
        # Generate full report
        report_content = self._compile_report("Model Performance Report", sections)
        
        # Save report
        report_path = self.output_dir / f"model_performance_report_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(report_path, 'w') as f:
            f.write(report_content)
        
        self.generated_reports.append(str(report_path))
        return str(report_path)
    
    def _create_executive_summary_section(self, data_profile: Any) -> ReportSection:
        """Create executive summary section."""
        content = f"""
EXECUTIVE SUMMARY
================

Dataset Overview:
- Total Samples: {data_profile.shape[0]:,}
- Total Features: {data_profile.shape[1]:,}
- Data Quality Score: {data_profile.data_quality_score:.2f}/1.0

Key Findings:
- {len(data_profile.numeric_summary)} numeric features identified
- {len(data_profile.categorical_summary)} categorical features identified
- Missing values detected in {sum(1 for v in data_profile.missing_values.values() if v > 0)} columns
- Overall data quality: {'Excellent' if data_profile.data_quality_score > 0.9 else 'Good' if data_profile.data_quality_score > 0.7 else 'Needs Improvement'}

Immediate Actions Required:
{chr(10).join(f"- {rec}" for rec in data_profile.recommendations[:3])}
"""
        return ReportSection("Executive Summary", content)
    
    def _create_data_overview_section(self, data_profile: Any) -> ReportSection:
        """Create data overview section."""
        content = f"""
DATA OVERVIEW
=============

Dataset Dimensions:
- Rows: {data_profile.shape[0]:,}
- Columns: {data_profile.shape[1]:,}

Feature Types:
- Numeric Features: {len(data_profile.numeric_summary)}
- Categorical Features: {len(data_profile.categorical_summary)}

Numeric Features Summary:
"""
        for feature, stats in list(data_profile.numeric_summary.items())[:5]:
            content += f"""
  {feature}:
    - Range: {stats['min']:.2f} to {stats['max']:.2f}
    - Mean: {stats['mean']:.2f}
    - Missing: {stats['missing_pct']:.1f}%
"""
        
        content += "\nCategorical Features Summary:\n"
        for feature, stats in list(data_profile.categorical_summary.items())[:5]:
            content += f"""
  {feature}:
    - Unique Values: {stats['unique_count']}
    - Cardinality: {stats['cardinality']}
    - Missing: {stats['missing_pct']:.1f}%
"""
        
        return ReportSection("Data Overview", content)
    
    def _create_data_quality_section(self, data_profile: Any) -> ReportSection:
        """Create data quality section."""
        total_missing = sum(data_profile.missing_values.values())
        total_cells = data_profile.shape[0] * data_profile.shape[1]
        missing_percentage = (total_missing / total_cells) * 100
        
        content = f"""
DATA QUALITY ASSESSMENT
=======================

Overall Quality Score: {data_profile.data_quality_score:.2f}/1.0

Missing Values Analysis:
- Total Missing Values: {total_missing:,} ({missing_percentage:.2f}% of all data)
- Columns with Missing Values: {sum(1 for v in data_profile.missing_values.values() if v > 0)}

Columns with Significant Missing Values (>5%):
"""
        significant_missing = {k: v for k, v in data_profile.missing_values.items() 
                             if v > data_profile.shape[0] * 0.05}
        
        for col, missing_count in significant_missing.items():
            missing_pct = (missing_count / data_profile.shape[0]) * 100
            content += f"- {col}: {missing_count:,} missing ({missing_pct:.1f}%)\n"
        
        return ReportSection("Data Quality", content)
    
    def _create_recommendations_section(self, data_profile: Any) -> ReportSection:
        """Create recommendations section."""
        content = """
RECOMMENDATIONS
===============

Data Preparation Recommendations:
"""
        for i, rec in enumerate(data_profile.recommendations, 1):
            content += f"{i}. {rec}\n"
        
        return ReportSection("Recommendations", content)
    
    def _create_model_summary_section(self, evaluation_report: Any) -> ReportSection:
        """Create model summary section."""
        content = f"""
MODEL PERFORMANCE SUMMARY
=========================

Best Performing Model: {evaluation_report.best_model}
Best Score: {evaluation_report.best_score:.4f}
Task Type: {evaluation_report.task_type}

Model Rankings:
"""
        for ranking in evaluation_report.model_rankings:
            content += f"{ranking['rank']}. {ranking['model_name']}: {ranking['validation_score']:.4f} (Training time: {ranking['training_time']:.1f}s)\n"
        
        return ReportSection("Model Summary", content)
    
    def _create_performance_analysis_section(self, evaluation_report: Any) -> ReportSection:
        """Create performance analysis section."""
        content = """
DETAILED PERFORMANCE ANALYSIS
=============================

"""
        for model_name, metrics in evaluation_report.detailed_metrics.items():
            content += f"""
{model_name}:
- Training Score: {metrics['train_score']:.4f}
- Validation Score: {metrics['validation_score']:.4f}
- Test Score: {metrics['test_score']:.4f}
- Training Time: {metrics['training_time']:.2f}s
- Overfitting Score: {abs(metrics['train_score'] - metrics['validation_score']):.4f}

"""
        
        return ReportSection("Performance Analysis", content)
    
    def _create_model_recommendations_section(self, evaluation_report: Any) -> ReportSection:
        """Create model recommendations section."""
        content = """
MODEL RECOMMENDATIONS
====================

"""
        for i, rec in enumerate(evaluation_report.recommendations, 1):
            content += f"{i}. {rec}\n"
        
        return ReportSection("Model Recommendations", content)
    
    def _compile_report(self, title: str, sections: List[ReportSection]) -> str:
        """Compile sections into full report."""
        timestamp = pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')
        
        report = f"""
{'='*80}
{title.upper()}
{'='*80}
Generated: {timestamp}
Platform: AutoML Agent Platform

"""
        
        for section in sections:
            report += section.content + "\n\n"
        
        report += f"""
{'='*80}
End of Report
{'='*80}
"""
        
        return report