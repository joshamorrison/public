"""
Data Processing Pipeline

Comprehensive data processing pipeline that handles data loading,
cleaning, validation, and preparation for ML workflows.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass
from pathlib import Path
import warnings

from ..tools.data_tools import DataProfiler, DataCleaner, FeatureAnalyzer

warnings.filterwarnings("ignore")


@dataclass
class PipelineResult:
    """Result of pipeline execution."""
    success: bool
    data: Optional[pd.DataFrame] = None
    metadata: Optional[Dict[str, Any]] = None
    artifacts: Optional[List[str]] = None
    errors: Optional[List[str]] = None
    processing_time: Optional[float] = None


@dataclass
class DataPipelineConfig:
    """Configuration for data processing pipeline."""
    missing_value_strategy: str = "auto"
    remove_duplicates: bool = True
    handle_outliers: bool = False
    feature_selection: bool = True
    validation_split: float = 0.2
    test_split: float = 0.2
    random_state: int = 42
    min_feature_importance: float = 0.01


class DataProcessingPipeline:
    """
    Comprehensive data processing pipeline for ML workflows.
    
    Handles the complete data preparation process from raw data
    to ML-ready datasets with proper train/validation/test splits.
    """
    
    def __init__(self, config: Optional[DataPipelineConfig] = None):
        """Initialize the data processing pipeline."""
        self.config = config or DataPipelineConfig()
        self.profiler = DataProfiler()
        self.cleaner = DataCleaner()
        self.analyzer = FeatureAnalyzer()
        
        self.pipeline_history = []
        self.artifacts = []
    
    def process_dataset(self, data_source: Union[pd.DataFrame, str, Path], 
                       target_column: str,
                       task_type: str = "classification") -> PipelineResult:
        """
        Execute complete data processing pipeline.
        
        Args:
            data_source: DataFrame or path to data file
            target_column: Name of target variable
            task_type: Type of ML task
            
        Returns:
            PipelineResult with processed data and metadata
        """
        import time
        start_time = time.time()
        
        try:
            # Step 1: Load data
            print("[PIPELINE] Step 1: Loading data...")
            df = self._load_data(data_source)
            if df is None:
                return PipelineResult(success=False, errors=["Failed to load data"])
            
            print(f"[PIPELINE] Loaded dataset: {df.shape[0]} rows, {df.shape[1]} columns")
            
            # Step 2: Data profiling
            print("[PIPELINE] Step 2: Profiling data...")
            data_profile = self.profiler.profile_dataset(df, target_column)
            
            # Step 3: Data cleaning
            print("[PIPELINE] Step 3: Cleaning data...")
            cleaning_result = self.cleaner.clean_dataset(
                df,
                missing_strategy=self.config.missing_value_strategy,
                remove_duplicates=self.config.remove_duplicates,
                handle_outliers=self.config.handle_outliers
            )
            
            df_cleaned = cleaning_result.cleaned_data
            print(f"[PIPELINE] Data cleaning completed: {len(cleaning_result.changes_made)} changes made")
            
            # Step 4: Feature analysis
            print("[PIPELINE] Step 4: Analyzing features...")
            feature_analysis = self.analyzer.analyze_features(df_cleaned, target_column, task_type)
            
            # Step 5: Feature encoding (categorical variables)
            print("[PIPELINE] Step 5: Encoding categorical features...")
            df_encoded = self._encode_categorical_features(df_cleaned, target_column)
            
            # Step 6: Train/validation/test split
            print("[PIPELINE] Step 6: Creating data splits...")
            splits = self._create_data_splits(df_encoded, target_column)
            
            # Step 7: Generate metadata
            metadata = {
                "original_shape": df.shape,
                "processed_shape": df_encoded.shape,
                "target_column": target_column,
                "task_type": task_type,
                "data_profile": data_profile,
                "cleaning_summary": {
                    "changes_made": cleaning_result.changes_made,
                    "rows_removed": cleaning_result.rows_removed,
                    "quality_improvement": cleaning_result.quality_improvement
                },
                "feature_analysis": feature_analysis,
                "data_splits": {
                    "train_size": splits["X_train"].shape[0],
                    "validation_size": splits["X_val"].shape[0], 
                    "test_size": splits["X_test"].shape[0]
                }
            }
            
            processing_time = time.time() - start_time
            print(f"[PIPELINE] Processing completed in {processing_time:.2f} seconds")
            
            result = PipelineResult(
                success=True,
                data=df_encoded,
                metadata=metadata,
                artifacts=self.artifacts,
                processing_time=processing_time
            )
            
            # Add splits to result
            result.metadata["splits"] = splits
            
            self.pipeline_history.append(result)
            return result
            
        except Exception as e:
            processing_time = time.time() - start_time
            return PipelineResult(
                success=False,
                errors=[str(e)],
                processing_time=processing_time
            )
    
    def _load_data(self, data_source: Union[pd.DataFrame, str, Path]) -> Optional[pd.DataFrame]:
        """Load data from various sources."""
        try:
            if isinstance(data_source, pd.DataFrame):
                return data_source.copy()
            
            path = Path(data_source)
            if not path.exists():
                print(f"[ERROR] Data file not found: {path}")
                return None
            
            # Load based on file extension
            if path.suffix.lower() == '.csv':
                return pd.read_csv(path)
            elif path.suffix.lower() in ['.xlsx', '.xls']:
                return pd.read_excel(path)
            elif path.suffix.lower() == '.json':
                return pd.read_json(path)
            elif path.suffix.lower() == '.parquet':
                return pd.read_parquet(path)
            else:
                print(f"[ERROR] Unsupported file format: {path.suffix}")
                return None
                
        except Exception as e:
            print(f"[ERROR] Failed to load data: {e}")
            return None
    
    def _create_data_splits(self, df: pd.DataFrame, target_column: str) -> Dict[str, pd.DataFrame]:
        """Create train/validation/test splits."""
        try:
            from sklearn.model_selection import train_test_split
            
            # Separate features and target
            X = df.drop(columns=[target_column])
            y = df[target_column]
            
            # First split: separate test set
            X_temp, X_test, y_temp, y_test = train_test_split(
                X, y, 
                test_size=self.config.test_split,
                random_state=self.config.random_state,
                stratify=y if y.nunique() < 20 else None  # Stratify for classification
            )
            
            # Second split: separate train and validation
            val_size = self.config.validation_split / (1 - self.config.test_split)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp, y_temp,
                test_size=val_size,
                random_state=self.config.random_state,
                stratify=y_temp if y_temp.nunique() < 20 else None
            )
            
            return {
                "X_train": X_train,
                "X_val": X_val,
                "X_test": X_test,
                "y_train": y_train,
                "y_val": y_val,
                "y_test": y_test
            }
            
        except ImportError:
            # Fallback manual split if sklearn not available
            return self._manual_data_split(df, target_column)
    
    def _manual_data_split(self, df: pd.DataFrame, target_column: str) -> Dict[str, pd.DataFrame]:
        """Manual data splitting when sklearn is not available."""
        # Shuffle data
        df_shuffled = df.sample(frac=1, random_state=self.config.random_state).reset_index(drop=True)
        
        n_total = len(df_shuffled)
        n_test = int(n_total * self.config.test_split)
        n_val = int(n_total * self.config.validation_split)
        n_train = n_total - n_test - n_val
        
        # Create splits
        train_df = df_shuffled[:n_train]
        val_df = df_shuffled[n_train:n_train+n_val]
        test_df = df_shuffled[n_train+n_val:]
        
        return {
            "X_train": train_df.drop(columns=[target_column]),
            "X_val": val_df.drop(columns=[target_column]),
            "X_test": test_df.drop(columns=[target_column]),
            "y_train": train_df[target_column],
            "y_val": val_df[target_column],
            "y_test": test_df[target_column]
        }
    
    def get_pipeline_summary(self) -> Dict[str, Any]:
        """Get summary of pipeline execution history."""
        if not self.pipeline_history:
            return {"message": "No pipeline executions recorded"}
        
        successful_runs = [r for r in self.pipeline_history if r.success]
        failed_runs = [r for r in self.pipeline_history if not r.success]
        
        return {
            "total_runs": len(self.pipeline_history),
            "successful_runs": len(successful_runs),
            "failed_runs": len(failed_runs),
            "average_processing_time": np.mean([r.processing_time for r in successful_runs if r.processing_time]),
            "total_artifacts_generated": len(self.artifacts),
            "last_run_status": "success" if self.pipeline_history[-1].success else "failed"
        }
    
    def validate_data_quality(self, df: pd.DataFrame, target_column: str) -> Dict[str, Any]:
        """Validate data quality and provide recommendations."""
        validation_results = {
            "is_valid": True,
            "warnings": [],
            "errors": [],
            "recommendations": []
        }
        
        # Check for missing target
        if target_column not in df.columns:
            validation_results["errors"].append(f"Target column '{target_column}' not found")
            validation_results["is_valid"] = False
        
        # Check for sufficient data
        if len(df) < 100:
            validation_results["warnings"].append("Dataset is very small (<100 rows)")
        
        # Check for missing values
        missing_pct = df.isnull().sum().sum() / (df.shape[0] * df.shape[1])
        if missing_pct > 0.3:
            validation_results["warnings"].append(f"High percentage of missing values ({missing_pct:.1%})")
        
        # Check for duplicates
        duplicate_pct = df.duplicated().sum() / len(df)
        if duplicate_pct > 0.1:
            validation_results["warnings"].append(f"High percentage of duplicates ({duplicate_pct:.1%})")
        
        # Check feature count
        n_features = df.shape[1] - 1  # Exclude target
        if n_features < 2:
            validation_results["warnings"].append("Very few features available for modeling")
        elif n_features > 1000:
            validation_results["recommendations"].append("Consider feature selection for high-dimensional data")
        
        # Check target distribution
        if target_column in df.columns:
            if df[target_column].nunique() < 2:
                validation_results["errors"].append("Target variable has insufficient variation")
                validation_results["is_valid"] = False
            elif df[target_column].nunique() == 2:
                # Binary classification - check balance
                balance = df[target_column].value_counts(normalize=True).min()
                if balance < 0.05:
                    validation_results["warnings"].append("Highly imbalanced target variable")
        
        return validation_results
    
    def _encode_categorical_features(self, df: pd.DataFrame, target_column: str) -> pd.DataFrame:
        """Encode categorical features for ML compatibility."""
        df_encoded = df.copy()
        
        try:
            from sklearn.preprocessing import LabelEncoder
            sklearn_available = True
        except ImportError:
            sklearn_available = False
        
        # Get categorical columns (excluding target)
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if target_column in categorical_cols:
            categorical_cols.remove(target_column)
        
        if not categorical_cols:
            return df_encoded
        
        print(f"[PIPELINE] Encoding {len(categorical_cols)} categorical features...")
        
        if sklearn_available:
            # Use LabelEncoder for categorical features
            for col in categorical_cols:
                try:
                    # First convert to regular string series to avoid categorical issues
                    series_data = pd.Series(df_encoded[col].astype(str))
                    
                    if df[col].nunique() > 100:
                        # High cardinality - keep top categories, others as 'Other'
                        top_categories = df[col].value_counts().head(50).index.tolist()
                        series_data = series_data.apply(lambda x: x if x in top_categories else 'Other')
                    
                    # Label encode
                    le = LabelEncoder()
                    # Handle NaN values in the series
                    series_data = series_data.fillna('Unknown')
                    df_encoded[col] = le.fit_transform(series_data)
                except Exception as e:
                    print(f"[PIPELINE] Warning: Could not encode {col}: {e}")
                    # Use fallback manual encoding for this column
                    unique_values = df[col].dropna().unique()
                    value_map = {val: idx for idx, val in enumerate(unique_values)}
                    value_map[np.nan] = -1
                    df_encoded[col] = df[col].map(value_map).fillna(-1)
        else:
            # Fallback manual encoding
            for col in categorical_cols:
                # Simple mapping to integers
                unique_values = df[col].dropna().unique()
                value_map = {val: idx for idx, val in enumerate(unique_values)}
                value_map[np.nan] = -1  # Handle NaN
                df_encoded[col] = df[col].map(value_map).fillna(-1)
        
        print(f"[PIPELINE] Categorical encoding completed")
        return df_encoded