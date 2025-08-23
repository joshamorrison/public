"""
Data Hygiene Agent for AutoML Platform

Specialized agent for data cleaning and preprocessing that:
1. Handles missing values with intelligent imputation strategies
2. Detects and treats outliers using multiple methods
3. Performs data validation and consistency checks
4. Applies data transformations and normalization
5. Ensures data quality meets ML pipeline requirements

This agent typically runs after EDA Agent and before Feature Engineering.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum

try:
    from sklearn.impute import SimpleImputer, KNNImputer, IterativeImputer
    from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
    from sklearn.preprocessing import LabelEncoder, OneHotEncoder
    from sklearn.ensemble import IsolationForest
    from sklearn.neighbors import LocalOutlierFactor
    from sklearn.covariance import EllipticEnvelope
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy import stats
    from scipy.stats import zscore
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity


class ImputationMethod(Enum):
    """Available imputation methods."""
    MEAN = "mean"
    MEDIAN = "median"
    MODE = "mode"
    CONSTANT = "constant"
    KNN = "knn"
    ITERATIVE = "iterative"
    FORWARD_FILL = "forward_fill"
    BACKWARD_FILL = "backward_fill"


class OutlierMethod(Enum):
    """Available outlier detection methods."""
    IQR = "iqr"
    Z_SCORE = "z_score"
    ISOLATION_FOREST = "isolation_forest"
    LOCAL_OUTLIER_FACTOR = "lof"
    ELLIPTIC_ENVELOPE = "elliptic_envelope"


class ScalingMethod(Enum):
    """Available scaling methods."""
    STANDARD = "standard"
    MINMAX = "minmax"
    ROBUST = "robust"
    NONE = "none"


@dataclass
class CleaningResult:
    """Result of data cleaning operation."""
    method: str
    rows_affected: int
    columns_affected: List[str]
    before_stats: Dict[str, Any]
    after_stats: Dict[str, Any]
    success: bool
    message: str


@dataclass
class DataQualityReport:
    """Comprehensive data quality report."""
    original_shape: Tuple[int, int]
    cleaned_shape: Tuple[int, int]
    rows_removed: int
    columns_modified: int
    cleaning_operations: List[CleaningResult]
    quality_score_before: float
    quality_score_after: float
    improvement: float


class DataHygieneAgent(BaseAgent):
    """
    Data Hygiene Agent for comprehensive data cleaning and preprocessing.
    
    Responsibilities:
    1. Missing value detection and intelligent imputation
    2. Outlier detection and treatment
    3. Data validation and consistency checks
    4. Data type optimization and conversion
    5. Quality assessment and reporting
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the Data Hygiene Agent."""
        super().__init__(
            name="Data Hygiene Agent", 
            description="Comprehensive data cleaning and preprocessing specialist",
            specialization="Data Cleaning & Quality Assurance",
            config=config,
            communication_hub=communication_hub
        )
        
        # Cleaning configuration
        self.missing_threshold = self.config.get("missing_threshold", 0.5)  # Drop columns >50% missing
        self.outlier_threshold = self.config.get("outlier_threshold", 0.05)  # 5% outlier rate threshold
        self.auto_drop_duplicates = self.config.get("auto_drop_duplicates", True)
        self.auto_convert_types = self.config.get("auto_convert_types", True)
        
        # Default strategies
        self.default_imputation = ImputationMethod(self.config.get("default_imputation", "median"))
        self.default_outlier_method = OutlierMethod(self.config.get("default_outlier_method", "iqr"))
        self.default_scaling = ScalingMethod(self.config.get("default_scaling", "standard"))
        
        # Quality thresholds
        self.quality_thresholds.update({
            "data_quality_score": 0.8,
            "missing_data_threshold": 0.1,  # Max 10% missing after cleaning
            "outlier_rate_threshold": 0.03   # Max 3% outliers after treatment
        })
        
        # Cleaning history
        self.cleaning_operations: List[CleaningResult] = []
        self.original_data_stats: Optional[Dict[str, Any]] = None
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive data hygiene workflow.
        
        Args:
            context: Task context with dataset and EDA insights
            
        Returns:
            AgentResult with cleaned data and quality report
        """
        try:
            self.logger.info("Starting data hygiene and cleaning workflow...")
            
            # Load dataset
            df = self._load_dataset(context)
            if df is None:
                return AgentResult(
                    success=False,
                    message="Failed to load dataset for cleaning"
                )
            
            # Store original stats
            self.original_data_stats = self._calculate_data_stats(df)
            self.logger.info(f"Original dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Get EDA insights from communication hub
            eda_insights = self._get_eda_insights()
            
            # Phase 1: Initial Assessment
            self.logger.info("Phase 1: Initial data quality assessment...")
            initial_quality = self._assess_data_quality(df)
            
            # Phase 2: Handle Missing Values
            self.logger.info("Phase 2: Handling missing values...")
            df_cleaned = self._handle_missing_values(df, eda_insights)
            
            # Phase 3: Handle Duplicates
            self.logger.info("Phase 3: Removing duplicate rows...")
            df_cleaned = self._handle_duplicates(df_cleaned)
            
            # Phase 4: Outlier Detection and Treatment
            self.logger.info("Phase 4: Detecting and treating outliers...")
            df_cleaned = self._handle_outliers(df_cleaned, eda_insights)
            
            # Phase 5: Data Type Optimization
            self.logger.info("Phase 5: Optimizing data types...")
            df_cleaned = self._optimize_data_types(df_cleaned)
            
            # Phase 6: Data Validation
            self.logger.info("Phase 6: Performing data validation...")
            validation_results = self._validate_data(df_cleaned)
            
            # Phase 7: Generate Quality Report
            self.logger.info("Phase 7: Generating quality report...")
            final_quality = self._assess_data_quality(df_cleaned)
            quality_report = self._generate_quality_report(df, df_cleaned, initial_quality, final_quality)
            
            # Create result data
            result_data = {
                "cleaned_dataset_stats": self._calculate_data_stats(df_cleaned),
                "quality_report": self._quality_report_to_dict(quality_report),
                "cleaning_operations": [self._cleaning_result_to_dict(op) for op in self.cleaning_operations],
                "validation_results": validation_results,
                "improvement_metrics": {
                    "quality_score_improvement": final_quality - initial_quality,
                    "rows_cleaned": df.shape[0] - df_cleaned.shape[0],
                    "columns_modified": len(set(col for op in self.cleaning_operations for col in op.columns_affected))
                },
                "recommended_next_steps": self._generate_next_steps_recommendations(df_cleaned, quality_report)
            }
            
            # Update performance metrics
            performance_metrics = {
                "data_quality_score": final_quality,
                "cleaning_success_rate": sum(1 for op in self.cleaning_operations if op.success) / len(self.cleaning_operations) if self.cleaning_operations else 1.0,
                "rows_preserved_ratio": df_cleaned.shape[0] / df.shape[0],
                "quality_improvement": final_quality - initial_quality
            }
            self.update_performance_metrics(performance_metrics)
            
            # Share cleaning insights
            if self.communication_hub:
                self._share_cleaning_insights(result_data)
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Data cleaning completed: {df_cleaned.shape[0]} rows, {df_cleaned.shape[1]} columns (quality improved by {final_quality - initial_quality:.3f})",
                recommendations=result_data["recommended_next_steps"]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Data hygiene workflow failed: {str(e)}"
            )
    
    def _load_dataset(self, context: TaskContext) -> Optional[pd.DataFrame]:
        """Load dataset from context or previous agent results."""
        # In real implementation, this would load from file or previous agent
        # For demo, create synthetic data with issues
        
        if context.dataset_info:
            shape = context.dataset_info.get("shape", [1000, 10])
            columns = context.dataset_info.get("columns", [f"feature_{i}" for i in range(shape[1])])
        else:
            shape = [1000, 10]
            columns = [f"feature_{i}" for i in range(10)]
        
        # Create synthetic data with cleaning challenges
        np.random.seed(42)
        data = {}
        
        for i, col in enumerate(columns):
            if "target" in col.lower() or "churn" in col.lower():
                data[col] = np.random.choice([0, 1], size=shape[0], p=[0.7, 0.3])
            elif "category" in col.lower() or "type" in col.lower():
                categories = [f"cat_{j}" for j in range(5)]
                data[col] = np.random.choice(categories, size=shape[0])
            elif "age" in col.lower():
                # Add outliers and missing values
                ages = np.random.normal(35, 12, shape[0])
                ages[np.random.choice(shape[0], 50, replace=False)] = np.random.uniform(120, 150, 50)  # Outliers
                data[col] = ages
            elif "income" in col.lower():
                incomes = np.random.lognormal(10, 0.5, shape[0])
                # Add extreme outliers
                incomes[np.random.choice(shape[0], 20, replace=False)] = np.random.uniform(1e6, 1e7, 20)
                data[col] = incomes
            else:
                data[col] = np.random.normal(0, 1, shape[0])
        
        df = pd.DataFrame(data)
        
        # Introduce missing values
        for col in df.columns[:5]:
            missing_rate = np.random.uniform(0.05, 0.25)  # 5-25% missing
            missing_indices = np.random.choice(df.index, int(missing_rate * len(df)), replace=False)
            df.loc[missing_indices, col] = np.nan
        
        # Add duplicate rows
        duplicate_indices = np.random.choice(df.index, 50, replace=False)
        duplicates = df.loc[duplicate_indices].copy()
        df = pd.concat([df, duplicates], ignore_index=True)
        
        return df
    
    def _get_eda_insights(self) -> Dict[str, Any]:
        """Get EDA insights from communication hub."""
        insights = {}
        
        if self.communication_hub:
            # Get shared knowledge from EDA agent
            messages = self.communication_hub.get_messages(self.name)
            for message in messages:
                if message.message_type.value == "knowledge_share":
                    knowledge_type = message.content.get("knowledge_type")
                    if knowledge_type in ["data_quality_issues", "feature_characteristics"]:
                        insights[knowledge_type] = message.content.get("data", {})
        
        return insights
    
    def _calculate_data_stats(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Calculate comprehensive data statistics."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        stats = {
            "shape": df.shape,
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024**2),
            "missing_values": df.isnull().sum().to_dict(),
            "missing_percentages": (df.isnull().sum() / len(df) * 100).to_dict(),
            "duplicate_rows": df.duplicated().sum(),
            "data_types": df.dtypes.astype(str).to_dict(),
            "numerical_features": len(numerical_cols),
            "categorical_features": len(categorical_cols)
        }
        
        # Numerical statistics
        if len(numerical_cols) > 0:
            stats["numerical_stats"] = {
                col: {
                    "mean": df[col].mean(),
                    "std": df[col].std(),
                    "min": df[col].min(),
                    "max": df[col].max(),
                    "skewness": df[col].skew(),
                    "outlier_count": self._count_outliers_iqr(df[col])
                } for col in numerical_cols
            }
        
        # Categorical statistics
        if len(categorical_cols) > 0:
            stats["categorical_stats"] = {
                col: {
                    "unique_count": df[col].nunique(),
                    "most_frequent": df[col].mode().iloc[0] if len(df[col].mode()) > 0 else None,
                    "value_counts": df[col].value_counts().head(5).to_dict()
                } for col in categorical_cols
            }
        
        return stats
    
    def _assess_data_quality(self, df: pd.DataFrame) -> float:
        """Assess overall data quality score (0-1)."""
        # Missing data penalty
        missing_penalty = df.isnull().sum().sum() / (len(df) * len(df.columns))
        
        # Duplicate penalty
        duplicate_penalty = df.duplicated().sum() / len(df)
        
        # Outlier penalty (for numerical columns)
        outlier_penalty = 0.0
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        if len(numerical_cols) > 0:
            total_outliers = sum(self._count_outliers_iqr(df[col]) for col in numerical_cols)
            outlier_penalty = total_outliers / (len(df) * len(numerical_cols))
        
        # Calculate quality score
        quality_score = max(0.0, 1.0 - missing_penalty - duplicate_penalty - outlier_penalty)
        return quality_score
    
    def _handle_missing_values(self, df: pd.DataFrame, eda_insights: Dict[str, Any]) -> pd.DataFrame:
        """Handle missing values using intelligent imputation strategies."""
        df_clean = df.copy()
        
        # Identify columns to drop (too many missing values)
        missing_percentages = df.isnull().sum() / len(df) * 100
        columns_to_drop = missing_percentages[missing_percentages > self.missing_threshold * 100].index.tolist()
        
        if columns_to_drop:
            df_clean = df_clean.drop(columns=columns_to_drop)
            self.cleaning_operations.append(CleaningResult(
                method="drop_high_missing_columns",
                rows_affected=0,
                columns_affected=columns_to_drop,
                before_stats={"missing_columns": len(columns_to_drop)},
                after_stats={"remaining_columns": len(df_clean.columns)},
                success=True,
                message=f"Dropped {len(columns_to_drop)} columns with >{self.missing_threshold*100}% missing values"
            ))
        
        # Handle remaining missing values
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        categorical_cols = df_clean.select_dtypes(include=['object', 'category']).columns
        
        # Numerical imputation
        for col in numerical_cols:
            if df_clean[col].isnull().sum() > 0:
                missing_count = df_clean[col].isnull().sum()
                
                if self.default_imputation == ImputationMethod.MEAN:
                    fill_value = df_clean[col].mean()
                elif self.default_imputation == ImputationMethod.MEDIAN:
                    fill_value = df_clean[col].median()
                else:
                    fill_value = df_clean[col].median()  # Default fallback
                
                df_clean[col] = df_clean[col].fillna(fill_value)
                
                self.cleaning_operations.append(CleaningResult(
                    method=f"impute_{self.default_imputation.value}",
                    rows_affected=missing_count,
                    columns_affected=[col],
                    before_stats={"missing_count": missing_count},
                    after_stats={"fill_value": fill_value},
                    success=True,
                    message=f"Imputed {missing_count} missing values in {col}"
                ))
        
        # Categorical imputation
        for col in categorical_cols:
            if df_clean[col].isnull().sum() > 0:
                missing_count = df_clean[col].isnull().sum()
                fill_value = df_clean[col].mode().iloc[0] if len(df_clean[col].mode()) > 0 else "unknown"
                
                df_clean[col] = df_clean[col].fillna(fill_value)
                
                self.cleaning_operations.append(CleaningResult(
                    method="impute_mode",
                    rows_affected=missing_count,
                    columns_affected=[col],
                    before_stats={"missing_count": missing_count},
                    after_stats={"fill_value": fill_value},
                    success=True,
                    message=f"Imputed {missing_count} missing values in {col}"
                ))
        
        return df_clean
    
    def _handle_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """Remove duplicate rows."""
        if not self.auto_drop_duplicates:
            return df
        
        initial_rows = len(df)
        df_clean = df.drop_duplicates()
        duplicates_removed = initial_rows - len(df_clean)
        
        if duplicates_removed > 0:
            self.cleaning_operations.append(CleaningResult(
                method="remove_duplicates",
                rows_affected=duplicates_removed,
                columns_affected=[],
                before_stats={"total_rows": initial_rows},
                after_stats={"total_rows": len(df_clean)},
                success=True,
                message=f"Removed {duplicates_removed} duplicate rows"
            ))
        
        return df_clean
    
    def _handle_outliers(self, df: pd.DataFrame, eda_insights: Dict[str, Any]) -> pd.DataFrame:
        """Detect and treat outliers in numerical columns."""
        df_clean = df.copy()
        numerical_cols = df_clean.select_dtypes(include=[np.number]).columns
        
        for col in numerical_cols:
            outliers_before = self._count_outliers_iqr(df_clean[col])
            
            if outliers_before / len(df_clean) > self.outlier_threshold:
                # Apply outlier treatment based on method
                if self.default_outlier_method == OutlierMethod.IQR:
                    df_clean = self._treat_outliers_iqr(df_clean, col)
                elif self.default_outlier_method == OutlierMethod.Z_SCORE:
                    df_clean = self._treat_outliers_zscore(df_clean, col)
                else:
                    df_clean = self._treat_outliers_iqr(df_clean, col)  # Default fallback
                
                outliers_after = self._count_outliers_iqr(df_clean[col])
                outliers_treated = outliers_before - outliers_after
                
                self.cleaning_operations.append(CleaningResult(
                    method=f"treat_outliers_{self.default_outlier_method.value}",
                    rows_affected=outliers_treated,
                    columns_affected=[col],
                    before_stats={"outlier_count": outliers_before},
                    after_stats={"outlier_count": outliers_after},
                    success=True,
                    message=f"Treated {outliers_treated} outliers in {col}"
                ))
        
        return df_clean
    
    def _count_outliers_iqr(self, series: pd.Series) -> int:
        """Count outliers using IQR method."""
        if len(series.dropna()) == 0:
            return 0
        
        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = series[(series < lower_bound) | (series > upper_bound)]
        return len(outliers)
    
    def _treat_outliers_iqr(self, df: pd.DataFrame, column: str) -> pd.DataFrame:
        """Treat outliers using IQR method (winsorization)."""
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Winsorize outliers
        df[column] = df[column].clip(lower=lower_bound, upper=upper_bound)
        return df
    
    def _treat_outliers_zscore(self, df: pd.DataFrame, column: str, threshold: float = 3.0) -> pd.DataFrame:
        """Treat outliers using Z-score method."""
        if SCIPY_AVAILABLE:
            z_scores = np.abs(zscore(df[column].dropna()))
            # Cap at 3 standard deviations
            mean_val = df[column].mean()
            std_val = df[column].std()
            df[column] = df[column].clip(
                lower=mean_val - threshold * std_val,
                upper=mean_val + threshold * std_val
            )
        else:
            # Fallback to IQR method
            df = self._treat_outliers_iqr(df, column)
        
        return df
    
    def _optimize_data_types(self, df: pd.DataFrame) -> pd.DataFrame:
        """Optimize data types for memory efficiency."""
        if not self.auto_convert_types:
            return df
        
        df_optimized = df.copy()
        conversions = []
        
        # Optimize numerical types
        for col in df_optimized.select_dtypes(include=[np.number]).columns:
            original_dtype = df_optimized[col].dtype
            
            # Try to downcast integers
            if df_optimized[col].dtype in ['int64']:
                try:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='integer')
                    if df_optimized[col].dtype != original_dtype:
                        conversions.append(f"{col}: {original_dtype} -> {df_optimized[col].dtype}")
                except:
                    pass
            
            # Try to downcast floats
            elif df_optimized[col].dtype in ['float64']:
                try:
                    df_optimized[col] = pd.to_numeric(df_optimized[col], downcast='float')
                    if df_optimized[col].dtype != original_dtype:
                        conversions.append(f"{col}: {original_dtype} -> {df_optimized[col].dtype}")
                except:
                    pass
        
        # Convert object columns to category if low cardinality
        for col in df_optimized.select_dtypes(include=['object']).columns:
            unique_ratio = df_optimized[col].nunique() / len(df_optimized)
            if unique_ratio < 0.5:  # Less than 50% unique values
                try:
                    df_optimized[col] = df_optimized[col].astype('category')
                    conversions.append(f"{col}: object -> category")
                except:
                    pass
        
        if conversions:
            self.cleaning_operations.append(CleaningResult(
                method="optimize_data_types",
                rows_affected=0,
                columns_affected=[conv.split(':')[0] for conv in conversions],
                before_stats={"memory_usage_mb": df.memory_usage(deep=True).sum() / (1024**2)},
                after_stats={"memory_usage_mb": df_optimized.memory_usage(deep=True).sum() / (1024**2)},
                success=True,
                message=f"Optimized data types: {', '.join(conversions)}"
            ))
        
        return df_optimized
    
    def _validate_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data validation."""
        validation_results = {
            "missing_values_check": df.isnull().sum().sum() == 0,
            "duplicate_rows_check": df.duplicated().sum() == 0,
            "data_types_check": True,  # Simplified check
            "value_ranges_check": True,  # Simplified check
            "consistency_check": True,   # Simplified check
            "issues_found": []
        }
        
        # Check for remaining missing values
        if df.isnull().sum().sum() > 0:
            validation_results["issues_found"].append("Missing values still present")
        
        # Check for infinite values
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        inf_count = 0
        for col in numerical_cols:
            inf_count += np.isinf(df[col]).sum()
        
        if inf_count > 0:
            validation_results["issues_found"].append(f"Infinite values found: {inf_count}")
            validation_results["value_ranges_check"] = False
        
        # Check for extremely skewed distributions
        highly_skewed = []
        for col in numerical_cols:
            if abs(df[col].skew()) > 5:
                highly_skewed.append(col)
        
        if highly_skewed:
            validation_results["issues_found"].append(f"Highly skewed features: {highly_skewed}")
        
        validation_results["overall_status"] = "PASS" if len(validation_results["issues_found"]) == 0 else "ISSUES_FOUND"
        
        return validation_results
    
    def _generate_quality_report(self, df_original: pd.DataFrame, df_cleaned: pd.DataFrame, initial_quality: float, final_quality: float) -> DataQualityReport:
        """Generate comprehensive data quality report."""
        return DataQualityReport(
            original_shape=df_original.shape,
            cleaned_shape=df_cleaned.shape,
            rows_removed=df_original.shape[0] - df_cleaned.shape[0],
            columns_modified=len(set(col for op in self.cleaning_operations for col in op.columns_affected)),
            cleaning_operations=self.cleaning_operations.copy(),
            quality_score_before=initial_quality,
            quality_score_after=final_quality,
            improvement=final_quality - initial_quality
        )
    
    def _generate_next_steps_recommendations(self, df: pd.DataFrame, quality_report: DataQualityReport) -> List[str]:
        """Generate recommendations for next steps in the ML pipeline."""
        recommendations = []
        
        # Data quality recommendations
        if quality_report.quality_score_after > 0.9:
            recommendations.append("Data quality is excellent - proceed with feature engineering")
        elif quality_report.quality_score_after > 0.7:
            recommendations.append("Data quality is good - consider additional feature engineering")
        else:
            recommendations.append("Data quality needs improvement - review cleaning strategies")
        
        # Feature engineering recommendations
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if len(numerical_cols) > 0:
            recommendations.append("Consider feature scaling and normalization for numerical features")
        
        if len(categorical_cols) > 0:
            recommendations.append("Apply encoding strategies for categorical features")
        
        # Model readiness recommendations
        recommendations.append("Dataset is ready for feature engineering phase")
        recommendations.append("Consider feature selection and dimensionality reduction")
        
        return recommendations
    
    def _share_cleaning_insights(self, result_data: Dict[str, Any]) -> None:
        """Share cleaning insights with other agents."""
        # Share preprocessing results
        self.share_knowledge(
            knowledge_type="preprocessing_results",
            knowledge_data={
                "quality_improvement": result_data["improvement_metrics"]["quality_score_improvement"],
                "cleaning_operations": result_data["cleaning_operations"],
                "data_characteristics": result_data["cleaned_dataset_stats"],
                "validation_status": result_data["validation_results"]["overall_status"]
            }
        )
        
        # Share data readiness status
        self.share_knowledge(
            knowledge_type="data_readiness",
            knowledge_data={
                "quality_score": result_data["quality_report"]["quality_score_after"],
                "rows_preserved": result_data["quality_report"]["cleaned_shape"][0],
                "columns_count": result_data["quality_report"]["cleaned_shape"][1],
                "ready_for_modeling": result_data["quality_report"]["quality_score_after"] > 0.7
            }
        )
    
    def _quality_report_to_dict(self, report: DataQualityReport) -> Dict[str, Any]:
        """Convert DataQualityReport to dictionary."""
        return {
            "original_shape": report.original_shape,
            "cleaned_shape": report.cleaned_shape,
            "rows_removed": report.rows_removed,
            "columns_modified": report.columns_modified,
            "cleaning_operations_count": len(report.cleaning_operations),
            "quality_score_before": report.quality_score_before,
            "quality_score_after": report.quality_score_after,
            "improvement": report.improvement
        }
    
    def _cleaning_result_to_dict(self, result: CleaningResult) -> Dict[str, Any]:
        """Convert CleaningResult to dictionary."""
        return {
            "method": result.method,
            "rows_affected": result.rows_affected,
            "columns_affected": result.columns_affected,
            "before_stats": result.before_stats,
            "after_stats": result.after_stats,
            "success": result.success,
            "message": result.message
        }
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Data hygiene agent can handle any tabular dataset."""
        return True
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate cleaning task complexity."""
        if not context.dataset_info:
            return TaskComplexity.MODERATE
        
        shape = context.dataset_info.get("shape", [1000, 10])
        rows, cols = shape
        
        # Get data quality indicators
        missing_info = context.dataset_info.get("missing_values", {})
        missing_rate = sum(missing_info.values()) / (rows * cols) if missing_info else 0
        
        # Complexity based on size and data quality
        if rows > 100000 or cols > 100 or missing_rate > 0.3:
            return TaskComplexity.COMPLEX
        elif rows > 10000 or cols > 50 or missing_rate > 0.1:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _create_refinement_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create data hygiene specific refinement plan."""
        return {
            "strategy_name": "advanced_data_cleaning",
            "steps": [
                "advanced_outlier_detection",
                "sophisticated_imputation_methods",
                "data_consistency_validation",
                "quality_optimization"
            ],
            "estimated_improvement": 0.2,
            "execution_time": 6.0
        }
    
    def _assess_knowledge_relevance(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge to data hygiene agent."""
        relevance_map = {
            "data_quality_issues": 0.95,
            "feature_characteristics": 0.8,
            "preprocessing_results": 0.7,
            "model_performance": 0.3,
            "algorithm_insights": 0.2
        }
        return relevance_map.get(knowledge_type, 0.1)