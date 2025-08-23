"""
Feature Engineering Agent for AutoML Platform

Specialized agent for automated feature engineering that:
1. Creates new features through mathematical transformations
2. Performs feature selection using multiple methods
3. Applies feature scaling and normalization
4. Handles categorical encoding and text features
5. Optimizes feature sets for downstream ML models

This agent runs after Data Hygiene and before ML model training.
"""

import numpy as np
import pandas as pd
from typing import Any, Dict, List, Optional, Tuple, Union
from dataclasses import dataclass
from enum import Enum
from itertools import combinations

try:
    from sklearn.feature_selection import (
        SelectKBest, SelectPercentile, RFE, RFECV,
        mutual_info_classif, mutual_info_regression,
        f_classif, f_regression, chi2
    )
    from sklearn.preprocessing import (
        StandardScaler, MinMaxScaler, RobustScaler,
        PolynomialFeatures, LabelEncoder, OneHotEncoder,
        TargetEncoder, PowerTransformer
    )
    from sklearn.decomposition import PCA, FastICA
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.linear_model import Lasso, LassoCV
    from sklearn.model_selection import cross_val_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    from scipy.stats import boxcox, yeojohnson
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False

from .base_agent import BaseAgent, AgentResult, TaskContext, TaskComplexity


class FeatureType(Enum):
    """Types of features."""
    NUMERICAL = "numerical"
    CATEGORICAL = "categorical"
    TEXT = "text"
    DATETIME = "datetime"
    BOOLEAN = "boolean"


class TransformationType(Enum):
    """Types of feature transformations."""
    POLYNOMIAL = "polynomial"
    INTERACTION = "interaction"
    LOGARITHMIC = "logarithmic"
    EXPONENTIAL = "exponential"
    POWER = "power"
    BINNING = "binning"
    RATIO = "ratio"
    AGGREGATION = "aggregation"


class SelectionMethod(Enum):
    """Feature selection methods."""
    CORRELATION = "correlation"
    MUTUAL_INFO = "mutual_info"
    UNIVARIATE = "univariate"
    RFE = "rfe"
    LASSO = "lasso"
    TREE_IMPORTANCE = "tree_importance"
    VARIANCE_THRESHOLD = "variance_threshold"


@dataclass
class FeatureTransformation:
    """Result of a feature transformation."""
    transformation_type: TransformationType
    original_features: List[str]
    new_features: List[str]
    parameters: Dict[str, Any]
    performance_impact: Optional[float] = None
    success: bool = True
    message: str = ""


@dataclass
class FeatureSelectionResult:
    """Result of feature selection."""
    method: SelectionMethod
    features_before: int
    features_after: int
    selected_features: List[str]
    feature_scores: Optional[Dict[str, float]] = None
    selection_threshold: Optional[float] = None
    performance_improvement: Optional[float] = None


@dataclass
class FeatureEngineeringReport:
    """Comprehensive feature engineering report."""
    original_feature_count: int
    engineered_feature_count: int
    selected_feature_count: int
    transformations_applied: List[FeatureTransformation]
    selection_results: List[FeatureSelectionResult]
    final_feature_importance: Dict[str, float]
    performance_metrics: Dict[str, float]


class FeatureEngineeringAgent(BaseAgent):
    """
    Feature Engineering Agent for automated feature creation and selection.
    
    Responsibilities:
    1. Automated feature generation and transformation
    2. Intelligent feature selection using multiple methods
    3. Feature scaling and normalization
    4. Categorical and text feature encoding
    5. Feature quality assessment and optimization
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None, communication_hub=None):
        """Initialize the Feature Engineering Agent."""
        super().__init__(
            name="Feature Engineering Agent",
            description="Automated feature creation, transformation, and selection specialist",
            specialization="Feature Engineering & Selection",
            config=config,
            communication_hub=communication_hub
        )
        
        # Feature engineering configuration
        self.max_polynomial_degree = self.config.get("max_polynomial_degree", 2)
        self.max_interaction_degree = self.config.get("max_interaction_degree", 2)
        self.enable_polynomial_features = self.config.get("enable_polynomial_features", True)
        self.enable_interaction_features = self.config.get("enable_interaction_features", True)
        self.enable_mathematical_transforms = self.config.get("enable_mathematical_transforms", True)
        
        # Feature selection configuration
        self.target_feature_count = self.config.get("target_feature_count", None)
        self.feature_selection_methods = self.config.get("feature_selection_methods", [
            "correlation", "mutual_info", "tree_importance"
        ])
        self.correlation_threshold = self.config.get("correlation_threshold", 0.95)
        self.variance_threshold = self.config.get("variance_threshold", 0.01)
        
        # Categorical encoding
        self.categorical_encoding_method = self.config.get("categorical_encoding_method", "target_encoding")
        self.max_categorical_cardinality = self.config.get("max_categorical_cardinality", 50)
        
        # Quality thresholds
        self.quality_thresholds.update({
            "feature_selection_improvement": 0.05,
            "feature_importance_coverage": 0.8,
            "engineered_features_ratio": 0.3
        })
        
        # Feature engineering results
        self.transformations: List[FeatureTransformation] = []
        self.selection_results: List[FeatureSelectionResult] = []
        self.feature_importance: Dict[str, float] = {}
    
    def execute_task(self, context: TaskContext) -> AgentResult:
        """
        Execute comprehensive feature engineering workflow.
        
        Args:
            context: Task context with cleaned dataset and preprocessing insights
            
        Returns:
            AgentResult with engineered features and selection report
        """
        try:
            self.logger.info("Starting feature engineering workflow...")
            
            # Load cleaned dataset
            df = self._load_cleaned_dataset(context)
            if df is None:
                return AgentResult(
                    success=False,
                    message="Failed to load cleaned dataset for feature engineering"
                )
            
            # Get preprocessing insights
            preprocessing_insights = self._get_preprocessing_insights()
            
            # Identify target variable
            target_variable = self._identify_target_variable(df, context)
            self.logger.info(f"Identified target variable: {target_variable}")
            
            original_feature_count = len(df.columns) - (1 if target_variable else 0)
            
            # Phase 1: Feature Type Analysis
            self.logger.info("Phase 1: Analyzing feature types...")
            feature_types = self._analyze_feature_types(df, target_variable)
            
            # Phase 2: Categorical Encoding
            self.logger.info("Phase 2: Encoding categorical features...")
            df_encoded = self._encode_categorical_features(df, feature_types, target_variable)
            
            # Phase 3: Feature Generation
            self.logger.info("Phase 3: Generating new features...")
            df_engineered = self._generate_features(df_encoded, feature_types, target_variable)
            
            # Phase 4: Feature Scaling
            self.logger.info("Phase 4: Scaling numerical features...")
            df_scaled = self._scale_features(df_engineered, feature_types, target_variable)
            
            # Phase 5: Feature Selection
            self.logger.info("Phase 5: Performing feature selection...")
            df_selected, selected_features = self._perform_feature_selection(
                df_scaled, target_variable, context
            )
            
            # Phase 6: Feature Importance Analysis
            self.logger.info("Phase 6: Analyzing feature importance...")
            feature_importance = self._analyze_feature_importance(
                df_selected, target_variable, context
            )
            
            # Phase 7: Quality Assessment
            self.logger.info("Phase 7: Assessing feature engineering quality...")
            quality_metrics = self._assess_feature_quality(
                df, df_selected, target_variable, original_feature_count
            )
            
            # Create comprehensive report
            engineering_report = FeatureEngineeringReport(
                original_feature_count=original_feature_count,
                engineered_feature_count=len(df_engineered.columns) - (1 if target_variable else 0),
                selected_feature_count=len(df_selected.columns) - (1 if target_variable else 0),
                transformations_applied=self.transformations.copy(),
                selection_results=self.selection_results.copy(),
                final_feature_importance=feature_importance,
                performance_metrics=quality_metrics
            )
            
            # Prepare result data
            result_data = {
                "engineered_dataset_info": self._get_dataset_info(df_selected),
                "feature_engineering_report": self._report_to_dict(engineering_report),
                "feature_importance": feature_importance,
                "selected_features": selected_features,
                "feature_types": feature_types,
                "transformations_summary": self._summarize_transformations(),
                "selection_summary": self._summarize_selections(),
                "recommendations": self._generate_recommendations(engineering_report)
            }
            
            # Update performance metrics
            performance_metrics = {
                "feature_engineering_score": quality_metrics.get("engineering_score", 0.8),
                "feature_selection_improvement": quality_metrics.get("selection_improvement", 0.0),
                "feature_importance_coverage": quality_metrics.get("importance_coverage", 0.8),
                "feature_count_reduction": (original_feature_count - len(selected_features)) / original_feature_count if original_feature_count > 0 else 0
            }
            self.update_performance_metrics(performance_metrics)
            
            # Share feature insights
            if self.communication_hub:
                self._share_feature_insights(result_data)
            
            return AgentResult(
                success=True,
                data=result_data,
                message=f"Feature engineering completed: {len(selected_features)} features selected from {original_feature_count} original features",
                recommendations=result_data["recommendations"]
            )
            
        except Exception as e:
            return AgentResult(
                success=False,
                message=f"Feature engineering workflow failed: {str(e)}"
            )
    
    def _load_cleaned_dataset(self, context: TaskContext) -> Optional[pd.DataFrame]:
        """Load cleaned dataset from previous agent or simulate for demo."""
        # In real implementation, this would load from previous agent results
        # For demo, create synthetic cleaned data
        
        if context.dataset_info:
            shape = context.dataset_info.get("shape", [1000, 10])
            columns = context.dataset_info.get("columns", [f"feature_{i}" for i in range(shape[1])])
        else:
            shape = [1000, 10]
            columns = [f"feature_{i}" for i in range(10)]
        
        # Create synthetic cleaned data (no missing values, no outliers)
        np.random.seed(42)
        data = {}
        
        for i, col in enumerate(columns):
            if "target" in col.lower() or "churn" in col.lower() or "label" in col.lower():
                # Binary target
                data[col] = np.random.choice([0, 1], size=shape[0], p=[0.7, 0.3])
            elif "category" in col.lower() or "type" in col.lower():
                # Categorical variable
                categories = [f"cat_{j}" for j in range(5)]
                data[col] = np.random.choice(categories, size=shape[0])
            elif "age" in col.lower():
                # Age distribution (cleaned, no outliers)
                data[col] = np.random.normal(35, 12, shape[0]).clip(18, 80)
            elif "income" in col.lower() or "charges" in col.lower():
                # Income distribution (cleaned)
                data[col] = np.random.lognormal(10, 0.5, shape[0]).clip(0, 200000)
            elif "score" in col.lower() or "rating" in col.lower():
                # Score/rating (0-100)
                data[col] = np.random.uniform(0, 100, shape[0])
            else:
                # General numerical feature
                data[col] = np.random.normal(0, 1, shape[0])
        
        return pd.DataFrame(data)
    
    def _get_preprocessing_insights(self) -> Dict[str, Any]:
        """Get preprocessing insights from communication hub."""
        insights = {}
        
        if self.communication_hub:
            messages = self.communication_hub.get_messages(self.name)
            for message in messages:
                if message.message_type.value == "knowledge_share":
                    knowledge_type = message.content.get("knowledge_type")
                    if knowledge_type in ["preprocessing_results", "data_readiness"]:
                        insights[knowledge_type] = message.content.get("data", {})
        
        return insights
    
    def _identify_target_variable(self, df: pd.DataFrame, context: TaskContext) -> Optional[str]:
        """Identify the target variable in the dataset."""
        # Try from context first
        if context.dataset_info and context.dataset_info.get("target_variable"):
            target = context.dataset_info["target_variable"]
            if target in df.columns:
                return target
        
        # Look for common target variable names
        target_candidates = ["target", "label", "y", "churn", "outcome", "class", "response"]
        for candidate in target_candidates:
            matching_cols = [col for col in df.columns if candidate.lower() in col.lower()]
            if matching_cols:
                return matching_cols[0]
        
        return None
    
    def _analyze_feature_types(self, df: pd.DataFrame, target_variable: Optional[str]) -> Dict[str, FeatureType]:
        """Analyze and categorize feature types."""
        feature_types = {}
        
        for col in df.columns:
            if col == target_variable:
                continue
            
            if df[col].dtype in ['int64', 'float64']:
                # Check if it's actually categorical (low cardinality)
                unique_ratio = df[col].nunique() / len(df)
                if unique_ratio < 0.05 and df[col].nunique() < 20:
                    feature_types[col] = FeatureType.CATEGORICAL
                else:
                    feature_types[col] = FeatureType.NUMERICAL
            elif df[col].dtype == 'bool':
                feature_types[col] = FeatureType.BOOLEAN
            elif df[col].dtype == 'object':
                # Check if it could be datetime
                if any(keyword in col.lower() for keyword in ['date', 'time', 'created', 'updated']):
                    feature_types[col] = FeatureType.DATETIME
                else:
                    feature_types[col] = FeatureType.CATEGORICAL
            else:
                feature_types[col] = FeatureType.CATEGORICAL
        
        return feature_types
    
    def _encode_categorical_features(self, df: pd.DataFrame, feature_types: Dict[str, FeatureType], target_variable: Optional[str]) -> pd.DataFrame:
        """Encode categorical features using appropriate methods."""
        df_encoded = df.copy()
        categorical_features = [col for col, ftype in feature_types.items() if ftype == FeatureType.CATEGORICAL]
        
        for col in categorical_features:
            cardinality = df[col].nunique()
            
            if cardinality > self.max_categorical_cardinality:
                # High cardinality - use target encoding or frequency encoding
                if target_variable and SKLEARN_AVAILABLE:
                    try:
                        # Simple target encoding (mean of target for each category)
                        target_means = df.groupby(col)[target_variable].mean()
                        df_encoded[f"{col}_target_encoded"] = df[col].map(target_means)
                        df_encoded = df_encoded.drop(columns=[col])
                        
                        self.transformations.append(FeatureTransformation(
                            transformation_type=TransformationType.AGGREGATION,
                            original_features=[col],
                            new_features=[f"{col}_target_encoded"],
                            parameters={"method": "target_encoding", "cardinality": cardinality},
                            success=True,
                            message=f"Applied target encoding to high-cardinality feature {col}"
                        ))
                    except Exception as e:
                        self.logger.warning(f"Failed to apply target encoding to {col}: {str(e)}")
                else:
                    # Frequency encoding
                    freq_map = df[col].value_counts()
                    df_encoded[f"{col}_frequency"] = df[col].map(freq_map)
                    df_encoded = df_encoded.drop(columns=[col])
                    
                    self.transformations.append(FeatureTransformation(
                        transformation_type=TransformationType.AGGREGATION,
                        original_features=[col],
                        new_features=[f"{col}_frequency"],
                        parameters={"method": "frequency_encoding", "cardinality": cardinality},
                        success=True,
                        message=f"Applied frequency encoding to high-cardinality feature {col}"
                    ))
            
            elif cardinality <= 10:
                # Low cardinality - use one-hot encoding
                dummy_df = pd.get_dummies(df[col], prefix=col, drop_first=True)
                df_encoded = pd.concat([df_encoded.drop(columns=[col]), dummy_df], axis=1)
                
                self.transformations.append(FeatureTransformation(
                    transformation_type=TransformationType.AGGREGATION,
                    original_features=[col],
                    new_features=list(dummy_df.columns),
                    parameters={"method": "one_hot_encoding", "cardinality": cardinality},
                    success=True,
                    message=f"Applied one-hot encoding to categorical feature {col}"
                ))
            
            else:
                # Medium cardinality - use label encoding
                label_encoder = LabelEncoder()
                df_encoded[f"{col}_encoded"] = label_encoder.fit_transform(df[col].astype(str))
                df_encoded = df_encoded.drop(columns=[col])
                
                self.transformations.append(FeatureTransformation(
                    transformation_type=TransformationType.AGGREGATION,
                    original_features=[col],
                    new_features=[f"{col}_encoded"],
                    parameters={"method": "label_encoding", "cardinality": cardinality},
                    success=True,
                    message=f"Applied label encoding to categorical feature {col}"
                ))
        
        return df_encoded
    
    def _generate_features(self, df: pd.DataFrame, feature_types: Dict[str, FeatureType], target_variable: Optional[str]) -> pd.DataFrame:
        """Generate new features through various transformations."""
        df_engineered = df.copy()
        numerical_features = [col for col in df.columns if col != target_variable and 
                             df[col].dtype in ['int64', 'float64']]
        
        # 1. Polynomial Features
        if self.enable_polynomial_features and len(numerical_features) > 0 and SKLEARN_AVAILABLE:
            df_engineered = self._create_polynomial_features(df_engineered, numerical_features, target_variable)
        
        # 2. Interaction Features
        if self.enable_interaction_features and len(numerical_features) > 1:
            df_engineered = self._create_interaction_features(df_engineered, numerical_features, target_variable)
        
        # 3. Mathematical Transformations
        if self.enable_mathematical_transforms:
            df_engineered = self._create_mathematical_transforms(df_engineered, numerical_features, target_variable)
        
        # 4. Ratio Features
        df_engineered = self._create_ratio_features(df_engineered, numerical_features, target_variable)
        
        # 5. Binning Features
        df_engineered = self._create_binning_features(df_engineered, numerical_features, target_variable)
        
        return df_engineered
    
    def _create_polynomial_features(self, df: pd.DataFrame, numerical_features: List[str], target_variable: Optional[str]) -> pd.DataFrame:
        """Create polynomial features."""
        try:
            # Limit to first 5 numerical features to avoid explosion
            selected_features = numerical_features[:5]
            
            poly_transformer = PolynomialFeatures(
                degree=self.max_polynomial_degree,
                include_bias=False,
                interaction_only=False
            )
            
            X_poly = poly_transformer.fit_transform(df[selected_features])
            poly_feature_names = poly_transformer.get_feature_names_out(selected_features)
            
            # Only keep new polynomial features (not the original ones)
            new_features = [name for name in poly_feature_names if name not in selected_features]
            new_feature_data = X_poly[:, [i for i, name in enumerate(poly_feature_names) if name in new_features]]
            
            # Add new features to dataframe
            for i, feature_name in enumerate(new_features):
                df[f"poly_{feature_name}"] = new_feature_data[:, i]
            
            self.transformations.append(FeatureTransformation(
                transformation_type=TransformationType.POLYNOMIAL,
                original_features=selected_features,
                new_features=[f"poly_{name}" for name in new_features],
                parameters={"degree": self.max_polynomial_degree},
                success=True,
                message=f"Created {len(new_features)} polynomial features"
            ))
            
        except Exception as e:
            self.logger.warning(f"Failed to create polynomial features: {str(e)}")
        
        return df
    
    def _create_interaction_features(self, df: pd.DataFrame, numerical_features: List[str], target_variable: Optional[str]) -> pd.DataFrame:
        """Create interaction features between numerical variables."""
        try:
            # Limit to avoid feature explosion
            selected_features = numerical_features[:6]
            interaction_features = []
            
            for feat1, feat2 in combinations(selected_features, 2):
                # Multiplicative interaction
                interaction_name = f"interact_{feat1}_{feat2}"
                df[interaction_name] = df[feat1] * df[feat2]
                interaction_features.append(interaction_name)
                
                # Additive interaction
                sum_name = f"sum_{feat1}_{feat2}"
                df[sum_name] = df[feat1] + df[feat2]
                interaction_features.append(sum_name)
                
                # Difference
                diff_name = f"diff_{feat1}_{feat2}"
                df[diff_name] = df[feat1] - df[feat2]
                interaction_features.append(diff_name)
            
            self.transformations.append(FeatureTransformation(
                transformation_type=TransformationType.INTERACTION,
                original_features=selected_features,
                new_features=interaction_features,
                parameters={"interaction_types": ["multiply", "add", "subtract"]},
                success=True,
                message=f"Created {len(interaction_features)} interaction features"
            ))
            
        except Exception as e:
            self.logger.warning(f"Failed to create interaction features: {str(e)}")
        
        return df
    
    def _create_mathematical_transforms(self, df: pd.DataFrame, numerical_features: List[str], target_variable: Optional[str]) -> pd.DataFrame:
        """Create mathematical transformations of features."""
        try:
            math_features = []
            
            for feature in numerical_features[:8]:  # Limit to avoid explosion
                # Log transformation (for positive values)
                if (df[feature] > 0).all():
                    log_name = f"log_{feature}"
                    df[log_name] = np.log1p(df[feature])  # log1p handles 0 values
                    math_features.append(log_name)
                
                # Square root transformation
                if (df[feature] >= 0).all():
                    sqrt_name = f"sqrt_{feature}"
                    df[sqrt_name] = np.sqrt(np.abs(df[feature]))
                    math_features.append(sqrt_name)
                
                # Square transformation
                square_name = f"square_{feature}"
                df[square_name] = df[feature] ** 2
                math_features.append(square_name)
                
                # Reciprocal (for non-zero values)
                if (df[feature] != 0).all():
                    recip_name = f"recip_{feature}"
                    df[recip_name] = 1 / df[feature]
                    math_features.append(recip_name)
            
            self.transformations.append(FeatureTransformation(
                transformation_type=TransformationType.LOGARITHMIC,
                original_features=numerical_features[:8],
                new_features=math_features,
                parameters={"transforms": ["log", "sqrt", "square", "reciprocal"]},
                success=True,
                message=f"Created {len(math_features)} mathematical transformation features"
            ))
            
        except Exception as e:
            self.logger.warning(f"Failed to create mathematical transforms: {str(e)}")
        
        return df
    
    def _create_ratio_features(self, df: pd.DataFrame, numerical_features: List[str], target_variable: Optional[str]) -> pd.DataFrame:
        """Create ratio features between numerical variables."""
        try:
            ratio_features = []
            
            # Select features that could make sense as ratios
            for feat1, feat2 in combinations(numerical_features[:5], 2):
                if (df[feat2] != 0).all():  # Avoid division by zero
                    ratio_name = f"ratio_{feat1}_{feat2}"
                    df[ratio_name] = df[feat1] / df[feat2]
                    ratio_features.append(ratio_name)
            
            if ratio_features:
                self.transformations.append(FeatureTransformation(
                    transformation_type=TransformationType.RATIO,
                    original_features=numerical_features[:5],
                    new_features=ratio_features,
                    parameters={},
                    success=True,
                    message=f"Created {len(ratio_features)} ratio features"
                ))
            
        except Exception as e:
            self.logger.warning(f"Failed to create ratio features: {str(e)}")
        
        return df
    
    def _create_binning_features(self, df: pd.DataFrame, numerical_features: List[str], target_variable: Optional[str]) -> pd.DataFrame:
        """Create binned/discretized versions of numerical features."""
        try:
            binned_features = []
            
            for feature in numerical_features[:5]:  # Limit binning
                # Create quantile-based bins
                bin_name = f"binned_{feature}"
                df[bin_name] = pd.qcut(df[feature], q=5, labels=['low', 'low_med', 'medium', 'med_high', 'high'], duplicates='drop')
                
                # Convert to numerical encoding
                label_encoder = LabelEncoder()
                df[bin_name] = label_encoder.fit_transform(df[bin_name])
                binned_features.append(bin_name)
            
            if binned_features:
                self.transformations.append(FeatureTransformation(
                    transformation_type=TransformationType.BINNING,
                    original_features=numerical_features[:5],
                    new_features=binned_features,
                    parameters={"bins": 5, "method": "quantile"},
                    success=True,
                    message=f"Created {len(binned_features)} binned features"
                ))
            
        except Exception as e:
            self.logger.warning(f"Failed to create binned features: {str(e)}")
        
        return df
    
    def _scale_features(self, df: pd.DataFrame, feature_types: Dict[str, FeatureType], target_variable: Optional[str]) -> pd.DataFrame:
        """Scale numerical features."""
        df_scaled = df.copy()
        
        # Get numerical features (including engineered ones)
        numerical_features = [col for col in df.columns if col != target_variable and 
                             df[col].dtype in ['int64', 'float64']]
        
        if len(numerical_features) > 0 and SKLEARN_AVAILABLE:
            try:
                scaler = StandardScaler()
                df_scaled[numerical_features] = scaler.fit_transform(df[numerical_features])
                
                self.transformations.append(FeatureTransformation(
                    transformation_type=TransformationType.POWER,  # Using POWER as closest enum
                    original_features=numerical_features,
                    new_features=numerical_features,
                    parameters={"method": "standard_scaling"},
                    success=True,
                    message=f"Applied standard scaling to {len(numerical_features)} numerical features"
                ))
                
            except Exception as e:
                self.logger.warning(f"Failed to scale features: {str(e)}")
        
        return df_scaled
    
    def _perform_feature_selection(self, df: pd.DataFrame, target_variable: Optional[str], context: TaskContext) -> Tuple[pd.DataFrame, List[str]]:
        """Perform comprehensive feature selection."""
        if not target_variable or target_variable not in df.columns:
            # No target variable - return all features
            feature_columns = [col for col in df.columns if col != target_variable]
            return df, feature_columns
        
        X = df.drop(columns=[target_variable])
        y = df[target_variable]
        selected_features = list(X.columns)
        
        # 1. Remove low variance features
        selected_features = self._remove_low_variance_features(X, selected_features)
        
        # 2. Remove highly correlated features
        selected_features = self._remove_correlated_features(X[selected_features], selected_features)
        
        # 3. Univariate feature selection
        if SKLEARN_AVAILABLE:
            selected_features = self._univariate_feature_selection(X[selected_features], y, selected_features)
        
        # 4. Tree-based feature importance
        if SKLEARN_AVAILABLE:
            selected_features = self._tree_based_selection(X[selected_features], y, selected_features)
        
        # Create final dataset with selected features
        final_columns = selected_features + [target_variable]
        df_selected = df[final_columns]
        
        return df_selected, selected_features
    
    def _remove_low_variance_features(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove features with low variance."""
        try:
            low_variance_features = []
            for feature in features:
                if X[feature].var() < self.variance_threshold:
                    low_variance_features.append(feature)
            
            selected_features = [f for f in features if f not in low_variance_features]
            
            if low_variance_features:
                self.selection_results.append(FeatureSelectionResult(
                    method=SelectionMethod.VARIANCE_THRESHOLD,
                    features_before=len(features),
                    features_after=len(selected_features),
                    selected_features=selected_features,
                    selection_threshold=self.variance_threshold
                ))
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"Failed to remove low variance features: {str(e)}")
            return features
    
    def _remove_correlated_features(self, X: pd.DataFrame, features: List[str]) -> List[str]:
        """Remove highly correlated features."""
        try:
            if len(features) <= 1:
                return features
            
            # Calculate correlation matrix
            corr_matrix = X[features].corr().abs()
            
            # Find pairs of highly correlated features
            high_corr_pairs = []
            for i in range(len(corr_matrix.columns)):
                for j in range(i+1, len(corr_matrix.columns)):
                    if corr_matrix.iloc[i, j] > self.correlation_threshold:
                        high_corr_pairs.append((corr_matrix.columns[i], corr_matrix.columns[j]))
            
            # Remove one feature from each highly correlated pair
            features_to_remove = set()
            for feat1, feat2 in high_corr_pairs:
                # Remove the feature with lower variance (keep more informative one)
                if X[feat1].var() < X[feat2].var():
                    features_to_remove.add(feat1)
                else:
                    features_to_remove.add(feat2)
            
            selected_features = [f for f in features if f not in features_to_remove]
            
            if features_to_remove:
                self.selection_results.append(FeatureSelectionResult(
                    method=SelectionMethod.CORRELATION,
                    features_before=len(features),
                    features_after=len(selected_features),
                    selected_features=selected_features,
                    selection_threshold=self.correlation_threshold
                ))
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"Failed to remove correlated features: {str(e)}")
            return features
    
    def _univariate_feature_selection(self, X: pd.DataFrame, y: pd.Series, features: List[str]) -> List[str]:
        """Perform univariate feature selection."""
        try:
            # Determine if classification or regression
            is_classification = len(y.unique()) <= 20 and y.dtype in ['int64', 'object', 'category']
            
            if is_classification:
                scorer = f_classif
            else:
                scorer = f_regression
            
            # Select top k features or top percentile
            if self.target_feature_count:
                k = min(self.target_feature_count, len(features))
                selector = SelectKBest(score_func=scorer, k=k)
            else:
                selector = SelectPercentile(score_func=scorer, percentile=80)
            
            X_selected = selector.fit_transform(X, y)
            selected_mask = selector.get_support()
            selected_features = [features[i] for i, selected in enumerate(selected_mask) if selected]
            
            # Get feature scores
            feature_scores = dict(zip(features, selector.scores_))
            
            self.selection_results.append(FeatureSelectionResult(
                method=SelectionMethod.UNIVARIATE,
                features_before=len(features),
                features_after=len(selected_features),
                selected_features=selected_features,
                feature_scores=feature_scores
            ))
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"Failed univariate feature selection: {str(e)}")
            return features
    
    def _tree_based_selection(self, X: pd.DataFrame, y: pd.Series, features: List[str]) -> List[str]:
        """Perform tree-based feature selection using feature importance."""
        try:
            # Determine if classification or regression
            is_classification = len(y.unique()) <= 20 and y.dtype in ['int64', 'object', 'category']
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=50, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            
            # Get feature importance
            feature_importance = dict(zip(features, model.feature_importances_))
            
            # Select features above median importance
            importance_threshold = np.median(model.feature_importances_)
            selected_features = [feat for feat, importance in feature_importance.items() 
                               if importance > importance_threshold]
            
            self.selection_results.append(FeatureSelectionResult(
                method=SelectionMethod.TREE_IMPORTANCE,
                features_before=len(features),
                features_after=len(selected_features),
                selected_features=selected_features,
                feature_scores=feature_importance,
                selection_threshold=importance_threshold
            ))
            
            return selected_features
            
        except Exception as e:
            self.logger.warning(f"Failed tree-based feature selection: {str(e)}")
            return features
    
    def _analyze_feature_importance(self, df: pd.DataFrame, target_variable: Optional[str], context: TaskContext) -> Dict[str, float]:
        """Analyze final feature importance."""
        if not target_variable or target_variable not in df.columns or not SKLEARN_AVAILABLE:
            # Return uniform importance if no target
            features = [col for col in df.columns if col != target_variable]
            return {feat: 1.0/len(features) for feat in features}
        
        try:
            X = df.drop(columns=[target_variable])
            y = df[target_variable]
            
            # Use Random Forest for feature importance
            is_classification = len(y.unique()) <= 20 and y.dtype in ['int64', 'object', 'category']
            
            if is_classification:
                model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
            else:
                model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
            
            model.fit(X, y)
            
            feature_importance = dict(zip(X.columns, model.feature_importances_))
            self.feature_importance = feature_importance
            
            return feature_importance
            
        except Exception as e:
            self.logger.warning(f"Failed to analyze feature importance: {str(e)}")
            features = [col for col in df.columns if col != target_variable]
            return {feat: 1.0/len(features) for feat in features}
    
    def _assess_feature_quality(self, df_original: pd.DataFrame, df_final: pd.DataFrame, target_variable: Optional[str], original_count: int) -> Dict[str, float]:
        """Assess the quality of feature engineering."""
        final_feature_count = len(df_final.columns) - (1 if target_variable else 0)
        
        # Basic metrics
        feature_reduction_ratio = (original_count - final_feature_count) / original_count if original_count > 0 else 0
        feature_creation_ratio = len(self.transformations) / original_count if original_count > 0 else 0
        
        # Engineering success rate
        successful_transformations = sum(1 for t in self.transformations if t.success)
        engineering_success_rate = successful_transformations / len(self.transformations) if self.transformations else 1.0
        
        # Selection improvement (simplified)
        selection_improvement = 0.1 if len(self.selection_results) > 0 else 0.0
        
        # Overall engineering score
        engineering_score = (
            0.3 * engineering_success_rate +
            0.3 * min(1.0, feature_creation_ratio * 2) +  # Cap at 1.0
            0.2 * (1.0 - feature_reduction_ratio) +  # Reward keeping useful features
            0.2 * (1.0 if selection_improvement > 0 else 0.8)
        )
        
        return {
            "engineering_score": engineering_score,
            "feature_reduction_ratio": feature_reduction_ratio,
            "feature_creation_ratio": feature_creation_ratio,
            "engineering_success_rate": engineering_success_rate,
            "selection_improvement": selection_improvement,
            "importance_coverage": 0.8  # Simplified
        }
    
    def _get_dataset_info(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Get information about the engineered dataset."""
        numerical_cols = df.select_dtypes(include=[np.number]).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        return {
            "shape": df.shape,
            "numerical_features": len(numerical_cols),
            "categorical_features": len(categorical_cols),
            "memory_usage_mb": df.memory_usage(deep=True).sum() / (1024**2),
            "feature_names": {
                "numerical": list(numerical_cols),
                "categorical": list(categorical_cols)
            }
        }
    
    def _report_to_dict(self, report: FeatureEngineeringReport) -> Dict[str, Any]:
        """Convert FeatureEngineeringReport to dictionary."""
        return {
            "original_feature_count": report.original_feature_count,
            "engineered_feature_count": report.engineered_feature_count,
            "selected_feature_count": report.selected_feature_count,
            "transformations_count": len(report.transformations_applied),
            "selection_methods_used": len(report.selection_results),
            "final_feature_importance": report.final_feature_importance,
            "performance_metrics": report.performance_metrics
        }
    
    def _summarize_transformations(self) -> Dict[str, Any]:
        """Summarize all transformations applied."""
        transformation_counts = {}
        for transformation in self.transformations:
            t_type = transformation.transformation_type.value
            transformation_counts[t_type] = transformation_counts.get(t_type, 0) + 1
        
        return {
            "total_transformations": len(self.transformations),
            "successful_transformations": sum(1 for t in self.transformations if t.success),
            "transformation_types": transformation_counts,
            "features_created": sum(len(t.new_features) for t in self.transformations)
        }
    
    def _summarize_selections(self) -> Dict[str, Any]:
        """Summarize all feature selection results."""
        method_counts = {}
        total_reduction = 0
        
        for result in self.selection_results:
            method = result.method.value
            method_counts[method] = method_counts.get(method, 0) + 1
            total_reduction += result.features_before - result.features_after
        
        return {
            "total_selection_methods": len(self.selection_results),
            "selection_methods_used": method_counts,
            "total_features_removed": total_reduction,
            "average_improvement": np.mean([r.performance_improvement for r in self.selection_results if r.performance_improvement])
        }
    
    def _generate_recommendations(self, report: FeatureEngineeringReport) -> List[str]:
        """Generate recommendations based on feature engineering results."""
        recommendations = []
        
        # Feature count recommendations
        if report.selected_feature_count > report.original_feature_count * 2:
            recommendations.append("Consider more aggressive feature selection to reduce overfitting risk")
        elif report.selected_feature_count < report.original_feature_count * 0.5:
            recommendations.append("Good feature reduction achieved - proceed with model training")
        
        # Transformation recommendations
        if len(report.transformations_applied) == 0:
            recommendations.append("Consider creating additional engineered features for better model performance")
        
        # Importance recommendations
        top_features = sorted(report.final_feature_importance.items(), key=lambda x: x[1], reverse=True)[:5]
        if top_features:
            recommendations.append(f"Focus on top features: {[f[0] for f in top_features[:3]]}")
        
        # Model readiness
        if report.performance_metrics.get("engineering_score", 0) > 0.8:
            recommendations.append("Feature engineering completed successfully - ready for model training")
        else:
            recommendations.append("Consider additional feature engineering iterations")
        
        return recommendations
    
    def _share_feature_insights(self, result_data: Dict[str, Any]) -> None:
        """Share feature engineering insights with other agents."""
        # Share feature importance
        self.share_knowledge(
            knowledge_type="feature_importance",
            knowledge_data={
                "feature_importance": result_data["feature_importance"],
                "selected_features": result_data["selected_features"],
                "feature_count": len(result_data["selected_features"]),
                "engineering_score": result_data["feature_engineering_report"]["performance_metrics"]["engineering_score"]
            }
        )
        
        # Share dataset readiness
        self.share_knowledge(
            knowledge_type="modeling_readiness",
            knowledge_data={
                "dataset_info": result_data["engineered_dataset_info"],
                "feature_types": result_data["feature_types"],
                "transformations_applied": result_data["transformations_summary"],
                "ready_for_training": True
            }
        )
    
    def can_handle_task(self, context: TaskContext) -> bool:
        """Feature engineering agent can handle any cleaned dataset."""
        return True
    
    def estimate_complexity(self, context: TaskContext) -> TaskComplexity:
        """Estimate feature engineering complexity."""
        if not context.dataset_info:
            return TaskComplexity.MODERATE
        
        shape = context.dataset_info.get("shape", [1000, 10])
        rows, cols = shape
        
        # Complexity based on dataset size and feature count
        if rows > 100000 or cols > 100:
            return TaskComplexity.COMPLEX
        elif rows > 10000 or cols > 50:
            return TaskComplexity.MODERATE
        else:
            return TaskComplexity.SIMPLE
    
    def _create_refinement_plan(self, strategy: Dict[str, Any]) -> Dict[str, Any]:
        """Create feature engineering specific refinement plan."""
        return {
            "strategy_name": "advanced_feature_engineering",
            "steps": [
                "deep_feature_interaction_analysis",
                "domain_specific_transformations",
                "ensemble_feature_selection",
                "feature_importance_optimization"
            ],
            "estimated_improvement": 0.25,
            "execution_time": 8.0
        }
    
    def _assess_knowledge_relevance(self, knowledge_type: str, knowledge_data: Dict[str, Any]) -> float:
        """Assess relevance of shared knowledge to feature engineering agent."""
        relevance_map = {
            "data_quality_issues": 0.8,
            "preprocessing_results": 0.9,
            "feature_importance": 0.7,
            "model_performance": 0.6,
            "algorithm_insights": 0.4
        }
        return relevance_map.get(knowledge_type, 0.1)