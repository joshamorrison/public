"""
Data Tools for AutoML Agents

Essential data manipulation and analysis tools used by agents
for exploratory data analysis, cleaning, and feature engineering.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
import warnings

try:
    from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
    from sklearn.impute import SimpleImputer, KNNImputer
    from sklearn.feature_selection import SelectKBest, chi2, f_classif
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

warnings.filterwarnings("ignore")


@dataclass
class DataProfile:
    """Data profiling results."""
    shape: Tuple[int, int]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    numeric_summary: Dict[str, Dict[str, float]]
    categorical_summary: Dict[str, Dict[str, Any]]
    data_quality_score: float
    recommendations: List[str]


@dataclass
class CleaningResult:
    """Data cleaning results."""
    cleaned_data: pd.DataFrame
    changes_made: List[str]
    rows_removed: int
    columns_modified: List[str]
    quality_improvement: float


class DataProfiler:
    """
    Comprehensive data profiling tool for understanding dataset characteristics.
    """
    
    def __init__(self):
        """Initialize the data profiler."""
        self.profile_cache = {}
    
    def profile_dataset(self, df: pd.DataFrame, target_column: Optional[str] = None) -> DataProfile:
        """
        Generate comprehensive data profile.
        
        Args:
            df: Input dataframe
            target_column: Name of target variable (if any)
            
        Returns:
            DataProfile with comprehensive analysis
        """
        # Basic info
        shape = df.shape
        dtypes = {col: str(dtype) for col, dtype in df.dtypes.items()}
        missing_values = df.isnull().sum().to_dict()
        
        # Numeric summary
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_summary = {}
        for col in numeric_cols:
            numeric_summary[col] = {
                'mean': float(df[col].mean()),
                'median': float(df[col].median()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'missing_pct': float(df[col].isnull().mean() * 100)
            }
        
        # Categorical summary
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        categorical_summary = {}
        for col in categorical_cols:
            unique_values = df[col].nunique()
            most_common = df[col].value_counts().head(5).to_dict()
            categorical_summary[col] = {
                'unique_count': int(unique_values),
                'missing_pct': float(df[col].isnull().mean() * 100),
                'most_common': most_common,
                'cardinality': 'high' if unique_values > 50 else 'medium' if unique_values > 10 else 'low'
            }
        
        # Calculate data quality score
        quality_score = self._calculate_quality_score(df, missing_values)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df, missing_values, numeric_summary, categorical_summary)
        
        return DataProfile(
            shape=shape,
            dtypes=dtypes,
            missing_values=missing_values,
            numeric_summary=numeric_summary,
            categorical_summary=categorical_summary,
            data_quality_score=quality_score,
            recommendations=recommendations
        )
    
    def _calculate_quality_score(self, df: pd.DataFrame, missing_values: Dict[str, int]) -> float:
        """Calculate overall data quality score (0-1)."""
        total_cells = df.shape[0] * df.shape[1]
        missing_cells = sum(missing_values.values())
        completeness_score = 1 - (missing_cells / total_cells)
        
        # Additional quality factors
        duplicate_score = 1 - (df.duplicated().sum() / len(df))
        
        # Combine scores
        quality_score = (completeness_score * 0.7 + duplicate_score * 0.3)
        return round(quality_score, 3)
    
    def _generate_recommendations(self, df: pd.DataFrame, missing_values: Dict, 
                                numeric_summary: Dict, categorical_summary: Dict) -> List[str]:
        """Generate data quality recommendations."""
        recommendations = []
        
        # Missing value recommendations
        high_missing_cols = [col for col, count in missing_values.items() if count > len(df) * 0.3]
        if high_missing_cols:
            recommendations.append(f"Consider removing columns with >30% missing: {high_missing_cols}")
        
        # High cardinality categorical variables
        high_card_cols = [col for col, info in categorical_summary.items() 
                         if info['cardinality'] == 'high']
        if high_card_cols:
            recommendations.append(f"High cardinality categorical columns may need encoding: {high_card_cols}")
        
        # Outlier detection for numeric columns
        potential_outlier_cols = []
        for col, stats in numeric_summary.items():
            if stats['std'] > 3 * abs(stats['mean']):
                potential_outlier_cols.append(col)
        if potential_outlier_cols:
            recommendations.append(f"Check for outliers in: {potential_outlier_cols}")
        
        # Duplicate detection
        if df.duplicated().sum() > 0:
            recommendations.append(f"Found {df.duplicated().sum()} duplicate rows")
        
        return recommendations


class DataCleaner:
    """
    Automated data cleaning tool for handling missing values, outliers, and data quality issues.
    """
    
    def __init__(self):
        """Initialize the data cleaner."""
        self.cleaning_history = []
    
    def clean_dataset(self, df: pd.DataFrame, 
                     missing_strategy: str = "auto",
                     remove_duplicates: bool = True,
                     handle_outliers: bool = False) -> CleaningResult:
        """
        Perform comprehensive data cleaning.
        
        Args:
            df: Input dataframe
            missing_strategy: Strategy for handling missing values
            remove_duplicates: Whether to remove duplicate rows
            handle_outliers: Whether to handle outliers
            
        Returns:
            CleaningResult with cleaned data and changes made
        """
        original_shape = df.shape
        changes_made = []
        df_cleaned = df.copy()
        
        # Remove duplicates
        if remove_duplicates:
            initial_rows = len(df_cleaned)
            df_cleaned = df_cleaned.drop_duplicates()
            removed_duplicates = initial_rows - len(df_cleaned)
            if removed_duplicates > 0:
                changes_made.append(f"Removed {removed_duplicates} duplicate rows")
        
        # Handle missing values
        df_cleaned, missing_changes = self._handle_missing_values(df_cleaned, missing_strategy)
        changes_made.extend(missing_changes)
        
        # Handle outliers if requested
        if handle_outliers:
            df_cleaned, outlier_changes = self._handle_outliers(df_cleaned)
            changes_made.extend(outlier_changes)
        
        # Data type optimization
        df_cleaned, dtype_changes = self._optimize_dtypes(df_cleaned)
        changes_made.extend(dtype_changes)
        
        # Calculate quality improvement
        quality_improvement = self._calculate_improvement(df, df_cleaned)
        
        rows_removed = original_shape[0] - df_cleaned.shape[0]
        columns_modified = [col for col in df.columns if not df[col].equals(df_cleaned[col])]
        
        result = CleaningResult(
            cleaned_data=df_cleaned,
            changes_made=changes_made,
            rows_removed=rows_removed,
            columns_modified=columns_modified,
            quality_improvement=quality_improvement
        )
        
        self.cleaning_history.append(result)
        return result
    
    def _handle_missing_values(self, df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Handle missing values based on strategy."""
        changes = []
        df_result = df.copy()
        
        if strategy == "auto":
            # Automatic strategy based on data type and missing percentage
            for col in df.columns:
                missing_pct = df[col].isnull().mean()
                
                if missing_pct > 0.5:
                    # Drop columns with >50% missing
                    df_result = df_result.drop(columns=[col])
                    changes.append(f"Dropped column '{col}' (>{missing_pct:.1%} missing)")
                elif missing_pct > 0:
                    if df[col].dtype in ['object', 'category']:
                        # Fill categorical with mode
                        mode_value = df[col].mode().iloc[0] if len(df[col].mode()) > 0 else 'Unknown'
                        df_result[col] = df_result[col].fillna(mode_value)
                        changes.append(f"Filled '{col}' missing values with mode: {mode_value}")
                    else:
                        # Fill numeric with median
                        median_value = df[col].median()
                        df_result[col] = df_result[col].fillna(median_value)
                        changes.append(f"Filled '{col}' missing values with median: {median_value:.2f}")
        
        return df_result, changes
    
    def _handle_outliers(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Handle outliers using IQR method."""
        changes = []
        df_result = df.copy()
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers = (df[col] < lower_bound) | (df[col] > upper_bound)
            outlier_count = outliers.sum()
            
            if outlier_count > 0:
                # Cap outliers instead of removing
                df_result[col] = df_result[col].clip(lower=lower_bound, upper=upper_bound)
                changes.append(f"Capped {outlier_count} outliers in '{col}'")
        
        return df_result, changes
    
    def _optimize_dtypes(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
        """Optimize data types for memory efficiency."""
        changes = []
        df_result = df.copy()
        
        # Convert object columns that should be categorical
        for col in df.select_dtypes(include=['object']).columns:
            unique_ratio = df[col].nunique() / len(df)
            if unique_ratio < 0.5:  # Less than 50% unique values
                df_result[col] = df_result[col].astype('category')
                changes.append(f"Converted '{col}' to categorical")
        
        return df_result, changes
    
    def _calculate_improvement(self, original_df: pd.DataFrame, cleaned_df: pd.DataFrame) -> float:
        """Calculate quality improvement score."""
        original_missing = original_df.isnull().sum().sum()
        cleaned_missing = cleaned_df.isnull().sum().sum()
        
        if original_missing == 0:
            return 0.0
        
        improvement = (original_missing - cleaned_missing) / original_missing
        return round(improvement, 3)


class FeatureAnalyzer:
    """
    Feature analysis and selection tool for understanding feature importance and relationships.
    """
    
    def __init__(self):
        """Initialize the feature analyzer."""
        self.analysis_cache = {}
    
    def analyze_features(self, df: pd.DataFrame, target_column: str, 
                        task_type: str = "classification") -> Dict[str, Any]:
        """
        Comprehensive feature analysis.
        
        Args:
            df: Input dataframe
            target_column: Name of target variable
            task_type: Type of ML task ("classification" or "regression")
            
        Returns:
            Dictionary with feature analysis results
        """
        features = [col for col in df.columns if col != target_column]
        X = df[features]
        y = df[target_column]
        
        analysis = {
            "feature_count": len(features),
            "numeric_features": list(X.select_dtypes(include=[np.number]).columns),
            "categorical_features": list(X.select_dtypes(include=['object', 'category']).columns),
            "feature_correlations": {},
            "feature_importance": {},
            "recommended_features": [],
            "feature_transformations": []
        }
        
        # Feature correlations (numeric only)
        numeric_features = X.select_dtypes(include=[np.number]).columns
        if len(numeric_features) > 1:
            corr_matrix = X[numeric_features].corr()
            # Find highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    corr_val = abs(corr_matrix.iloc[i, j])
                    if corr_val > 0.8:
                        high_corr_pairs.append({
                            'feature1': corr_matrix.columns[i],
                            'feature2': corr_matrix.columns[j],
                            'correlation': round(corr_val, 3)
                        })
            analysis["high_correlations"] = high_corr_pairs
        
        # Basic feature importance (if sklearn available)
        if SKLEARN_AVAILABLE and len(numeric_features) > 0:
            try:
                if task_type == "classification":
                    selector = SelectKBest(score_func=f_classif, k='all')
                else:
                    selector = SelectKBest(score_func=f_classif, k='all')
                
                # Use only numeric features for now
                X_numeric = X[numeric_features].fillna(0)
                if len(X_numeric.columns) > 0:
                    selector.fit(X_numeric, y)
                    feature_scores = dict(zip(numeric_features, selector.scores_))
                    analysis["feature_importance"] = {
                        k: round(v, 3) for k, v in 
                        sorted(feature_scores.items(), key=lambda x: x[1], reverse=True)
                    }
            except Exception as e:
                analysis["feature_importance_error"] = str(e)
        
        # Feature transformation recommendations
        transformations = []
        
        # High cardinality categorical features
        for col in analysis["categorical_features"]:
            unique_count = df[col].nunique()
            if unique_count > 50:
                transformations.append(f"Consider target encoding for high-cardinality '{col}' ({unique_count} unique)")
            elif unique_count < 10:
                transformations.append(f"One-hot encode low-cardinality '{col}' ({unique_count} unique)")
        
        # Skewed numeric features
        for col in analysis["numeric_features"]:
            skewness = df[col].skew()
            if abs(skewness) > 2:
                transformations.append(f"Consider log transform for skewed '{col}' (skew: {skewness:.2f})")
        
        analysis["feature_transformations"] = transformations
        
        return analysis