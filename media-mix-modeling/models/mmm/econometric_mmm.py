#!/usr/bin/env python3
"""
Econometric Media Mix Modeling with MLflow Integration
Advanced MMM with adstock, saturation, and attribution modeling
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple, Any
import warnings
warnings.filterwarnings('ignore')

# Import MLflow integration
try:
    from src.mlflow_integration import MMMMLflowTracker
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    # Create dummy class for type hints when MLflow not available
    class MMMMLflowTracker:
        pass

class EconometricMMM:
    """
    Advanced econometric media mix model with adstock and saturation transformations
    
    Features:
    - Adstock transformation for carryover effects
    - Saturation curves for diminishing returns
    - Base vs incremental decomposition
    - Attribution analysis
    - Cross-channel synergy modeling
    """
    
    def __init__(self,
                 adstock_rate: float = 0.5,
                 saturation_param: float = 0.5,
                 regularization_alpha: float = 0.1,
                 base_contribution: float = 0.3,
                 mlflow_tracker: Optional[MMMMLflowTracker] = None):
        """
        Initialize MMM model
        
        Args:
            adstock_rate: Carryover rate for adstock transformation (0-1)
            saturation_param: Saturation parameter for diminishing returns (0-1)
            regularization_alpha: Ridge regression regularization strength
            base_contribution: Expected base business contribution (0-1)
            mlflow_tracker: Optional MLflow tracker for experiment logging
        """
        self.adstock_rate = adstock_rate
        self.saturation_param = saturation_param
        self.regularization_alpha = regularization_alpha
        self.base_contribution = base_contribution
        
        # Model components
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = []
        self.spend_columns = []
        
        # Transformation tracking
        self.adstock_params = {}
        self.saturation_params = {}
        self.transformations = {}
        
        # Results storage
        self.performance_metrics = {}
        self.attribution_results = {}
        self.contribution_decomposition = {}
        
        # MLflow integration
        self.mlflow_tracker = mlflow_tracker
        
    def apply_adstock_transformation(self, 
                                   data: pd.DataFrame, 
                                   spend_columns: List[str],
                                   custom_rates: Optional[Dict[str, float]] = None) -> pd.DataFrame:
        """
        Apply adstock transformation for carryover effects
        
        Args:
            data: Input dataframe
            spend_columns: List of spend column names
            custom_rates: Optional custom adstock rates per channel
            
        Returns:
            DataFrame with adstocked columns
        """
        adstocked_data = data.copy()
        
        for channel in spend_columns:
            if channel not in data.columns:
                continue
                
            # Use custom rate if provided, otherwise default
            rate = custom_rates.get(channel, self.adstock_rate) if custom_rates else self.adstock_rate
            self.adstock_params[channel] = rate
            
            # Apply geometric adstock transformation
            adstocked_values = []
            carryover = 0
            
            for spend in data[channel]:
                adstocked = spend + carryover * rate
                adstocked_values.append(adstocked)
                carryover = adstocked
            
            adstocked_data[f"{channel}_adstocked"] = adstocked_values
            
        return adstocked_data
    
    def apply_saturation_transformation(self, 
                                      data: pd.DataFrame,
                                      adstocked_columns: List[str],
                                      saturation_method: str = 'hill') -> pd.DataFrame:
        """
        Apply saturation transformation for diminishing returns
        
        Args:
            data: Input dataframe with adstocked columns
            adstocked_columns: List of adstocked column names
            saturation_method: Method for saturation ('hill', 'michaelis_menten')
            
        Returns:
            DataFrame with saturated columns
        """
        saturated_data = data.copy()
        
        for channel in adstocked_columns:
            if channel not in data.columns:
                continue
                
            # Store saturation parameters
            self.saturation_params[channel] = {
                'method': saturation_method,
                'param': self.saturation_param
            }
            
            # Normalize to 0-1 for saturation curve
            channel_data = data[channel]
            max_value = channel_data.max()
            
            if max_value > 0:
                normalized = channel_data / max_value
                
                if saturation_method == 'hill':
                    # Hill saturation: x^n / (x^n + k^n)
                    saturated_norm = (normalized ** self.saturation_param) / \
                                   (normalized ** self.saturation_param + 0.5 ** self.saturation_param)
                elif saturation_method == 'michaelis_menten':
                    # Michaelis-Menten: x / (x + k)
                    saturated_norm = normalized / (normalized + self.saturation_param)
                else:
                    # Default to power transformation
                    saturated_norm = normalized ** self.saturation_param
                
                # Scale back to original range
                saturated_data[f"{channel}_saturated"] = saturated_norm * max_value
            else:
                saturated_data[f"{channel}_saturated"] = channel_data
                
        return saturated_data
    
    def create_synergy_features(self, 
                              data: pd.DataFrame,
                              transformed_channels: List[str]) -> pd.DataFrame:
        """
        Create cross-channel synergy features
        
        Args:
            data: Input dataframe
            transformed_channels: List of transformed channel names
            
        Returns:
            DataFrame with synergy features
        """
        synergy_data = data.copy()
        
        # Create pairwise interaction terms
        for i, channel1 in enumerate(transformed_channels):
            for channel2 in transformed_channels[i+1:]:
                if channel1 in data.columns and channel2 in data.columns:
                    # Multiplicative synergy
                    synergy_name = f"synergy_{channel1.replace('_saturated', '')}_{channel2.replace('_saturated', '')}"
                    synergy_data[synergy_name] = np.sqrt(data[channel1] * data[channel2])
        
        return synergy_data
    
    def fit(self, 
            data: pd.DataFrame,
            target_column: str = 'revenue',
            spend_columns: Optional[List[str]] = None,
            include_synergies: bool = True,
            custom_adstock_rates: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Fit the MMM model with transformations
        
        Args:
            data: Input dataframe
            target_column: Target variable column name
            spend_columns: List of spend columns (auto-detected if None)
            include_synergies: Whether to include cross-channel synergies
            custom_adstock_rates: Custom adstock rates per channel
            
        Returns:
            Dictionary with fit results and metrics
        """
        # Auto-detect spend columns if not provided
        if spend_columns is None:
            spend_columns = [col for col in data.columns if col.endswith('_spend')]
        
        self.spend_columns = spend_columns
        
        print(f"[MMM] Fitting model with {len(spend_columns)} channels: {[ch.replace('_spend', '') for ch in spend_columns]}")
        
        # Step 1: Apply adstock transformation
        print(f"[ADSTOCK] Applying carryover effects (rate: {self.adstock_rate})")
        adstocked_data = self.apply_adstock_transformation(data, spend_columns, custom_adstock_rates)
        
        # Step 2: Apply saturation transformation
        print(f"[SATURATION] Applying diminishing returns (param: {self.saturation_param})")
        adstocked_columns = [f"{ch}_adstocked" for ch in spend_columns]
        saturated_data = self.apply_saturation_transformation(adstocked_data, adstocked_columns)
        
        # Step 3: Create features for modeling
        transformed_channels = [f"{ch}_saturated" for ch in adstocked_columns]
        self.feature_names = transformed_channels.copy()
        
        # Step 4: Add synergy features if requested
        if include_synergies:
            print(f"[SYNERGY] Adding cross-channel interaction effects")
            final_data = self.create_synergy_features(saturated_data, transformed_channels)
            synergy_columns = [col for col in final_data.columns if col.startswith('synergy_')]
            self.feature_names.extend(synergy_columns)
        else:
            final_data = saturated_data
        
        # Step 5: Prepare training data
        X = final_data[self.feature_names]
        y = data[target_column]
        
        # Store original data for decomposition
        self.original_data = data.copy()
        self.transformed_data = final_data.copy()
        
        # Step 6: Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Step 7: Fit Ridge regression model
        print(f"[MODEL] Training Ridge regression (alpha: {self.regularization_alpha})")
        self.model = Ridge(alpha=self.regularization_alpha, random_state=42)
        self.model.fit(X_scaled, y)
        
        # Step 8: Calculate performance metrics
        y_pred = self.model.predict(X_scaled)
        
        self.performance_metrics = {
            'r2_score': r2_score(y, y_pred),
            'mape': mean_absolute_percentage_error(y, y_pred),
            'rmse': np.sqrt(np.mean((y - y_pred) ** 2)),
            'mean_actual': y.mean(),
            'mean_predicted': y_pred.mean()
        }
        
        print(f"[PERFORMANCE] R² = {self.performance_metrics['r2_score']:.3f}, MAPE = {self.performance_metrics['mape']:.1%}")
        
        # Step 9: Calculate attribution and decomposition
        self._calculate_attribution_analysis(X_scaled, y)
        self._calculate_contribution_decomposition(data, target_column)
        
        # Step 10: Log to MLflow if available
        if self.mlflow_tracker and MLFLOW_AVAILABLE:
            self._log_to_mlflow(data, target_column, include_synergies)
        
        return {
            'model': self.model,
            'performance': self.performance_metrics,
            'attribution': self.attribution_results,
            'decomposition': self.contribution_decomposition,
            'feature_names': self.feature_names,
            'adstock_params': self.adstock_params,
            'saturation_params': self.saturation_params
        }
    
    def _calculate_attribution_analysis(self, X_scaled: np.ndarray, y: pd.Series):
        """Calculate detailed attribution analysis"""
        if self.model is None:
            return
        
        # Get model coefficients
        coefficients = self.model.coef_
        
        # Calculate feature contributions
        feature_contributions = X_scaled * coefficients
        total_incremental = np.sum(feature_contributions, axis=1)
        
        # Attribution analysis by channel
        self.attribution_results = {}
        
        for i, feature_name in enumerate(self.feature_names):
            # Skip synergy features for channel-level attribution
            if feature_name.startswith('synergy_'):
                continue
                
            # Extract channel name
            channel_name = feature_name.split('_')[0] + '_spend'
            if channel_name not in self.attribution_results:
                self.attribution_results[channel_name] = {
                    'coefficient': 0,
                    'contribution': 0,
                    'attribution_pct': 0,
                    'spend': 0,
                    'roi': 0
                }
            
            # Accumulate contributions (in case of multiple features per channel)
            contribution = np.sum(feature_contributions[:, i])
            self.attribution_results[channel_name]['coefficient'] += coefficients[i]
            self.attribution_results[channel_name]['contribution'] += contribution
        
        # Calculate percentages and ROI
        total_contribution = sum([attr['contribution'] for attr in self.attribution_results.values()])
        
        for channel_name in self.attribution_results:
            if channel_name in self.original_data.columns:
                channel_spend = self.original_data[channel_name].sum()
                self.attribution_results[channel_name]['spend'] = channel_spend
                
                if total_contribution != 0:
                    self.attribution_results[channel_name]['attribution_pct'] = \
                        self.attribution_results[channel_name]['contribution'] / total_contribution
                
                if channel_spend > 0:
                    self.attribution_results[channel_name]['roi'] = \
                        self.attribution_results[channel_name]['contribution'] / channel_spend
    
    def _calculate_contribution_decomposition(self, data: pd.DataFrame, target_column: str):
        """Calculate base vs incremental contribution decomposition"""
        total_revenue = data[target_column].sum()
        
        # Estimate base revenue (minimum observed + trend)
        base_revenue_per_period = data[target_column].min()
        trend_component = np.mean(np.diff(data[target_column].rolling(4).mean().dropna()))
        base_revenue_total = base_revenue_per_period * len(data) + trend_component * len(data) / 2
        
        # Cap base at expected percentage
        max_base = total_revenue * (1 - self.base_contribution)
        base_revenue_total = min(base_revenue_total, max_base)
        
        incremental_revenue = total_revenue - base_revenue_total
        
        self.contribution_decomposition = {
            'total_revenue': total_revenue,
            'base_revenue': base_revenue_total,
            'incremental_revenue': incremental_revenue,
            'base_percentage': base_revenue_total / total_revenue,
            'incremental_percentage': incremental_revenue / total_revenue,
            'total_media_spend': sum([data[ch].sum() for ch in self.spend_columns]),
            'media_efficiency': incremental_revenue / sum([data[ch].sum() for ch in self.spend_columns]) if sum([data[ch].sum() for ch in self.spend_columns]) > 0 else 0
        }
    
    def _log_to_mlflow(self, data: pd.DataFrame, target_column: str, include_synergies: bool):
        """Log model results to MLflow"""
        try:
            # Log model parameters
            self.mlflow_tracker.log_mmm_model(
                model=self.model,
                model_type="econometric_mmm",
                parameters={
                    "adstock_rate": self.adstock_rate,
                    "saturation_param": self.saturation_param,
                    "regularization_alpha": self.regularization_alpha,
                    "base_contribution": self.base_contribution,
                    "include_synergies": include_synergies,
                    "n_channels": len(self.spend_columns),
                    "n_features": len(self.feature_names)
                }
            )
            
            # Log performance metrics
            self.mlflow_tracker.log_mmm_performance({
                'r2': self.performance_metrics['r2_score'],
                'mape': self.performance_metrics['mape'],
                'rmse': self.performance_metrics['rmse']
            })
            
            # Log attribution results
            self.mlflow_tracker.log_attribution_results(self.attribution_results)
            
            # Log decomposition as artifact
            self.mlflow_tracker.log_artifacts_json(self.contribution_decomposition, "decomposition.json")
            
            print(f"[MLFLOW] Logged econometric MMM results to experiment")
            
        except Exception as e:
            print(f"[MLFLOW] Error logging to MLflow: {e}")
    
    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """
        Make predictions on new data
        
        Args:
            data: Input dataframe with same structure as training data
            
        Returns:
            Predictions array
        """
        if self.model is None:
            raise ValueError("Model must be fitted before prediction")
        
        # Apply same transformations as training
        adstocked_data = self.apply_adstock_transformation(data, self.spend_columns)
        adstocked_columns = [f"{ch}_adstocked" for ch in self.spend_columns]
        saturated_data = self.apply_saturation_transformation(adstocked_data, adstocked_columns)
        
        # Add synergy features if they were used in training
        if any(name.startswith('synergy_') for name in self.feature_names):
            transformed_channels = [f"{ch}_saturated" for ch in adstocked_columns]
            final_data = self.create_synergy_features(saturated_data, transformed_channels)
        else:
            final_data = saturated_data
        
        # Extract features and scale
        X = final_data[self.feature_names]
        X_scaled = self.scaler.transform(X)
        
        return self.model.predict(X_scaled)
    
    def simulate_budget_scenarios(self, 
                                baseline_data: pd.DataFrame,
                                budget_scenarios: Dict[str, float],
                                periods: int = 12) -> Dict[str, Any]:
        """
        Simulate different budget allocation scenarios
        
        Args:
            baseline_data: Historical data for baseline
            budget_scenarios: Dict of {channel: weekly_budget} scenarios
            periods: Number of periods to simulate
            
        Returns:
            Dictionary with scenario results
        """
        if self.model is None:
            raise ValueError("Model must be fitted before simulation")
        
        # Create scenario data
        scenario_data = baseline_data.tail(1).copy()
        
        # Replicate for simulation periods
        scenario_data = pd.concat([scenario_data] * periods, ignore_index=True)
        
        # Apply budget scenarios
        for channel, budget in budget_scenarios.items():
            if f"{channel}_spend" in scenario_data.columns:
                scenario_data[f"{channel}_spend"] = budget
        
        # Generate predictions
        predictions = self.predict(scenario_data)
        
        # Calculate scenario metrics
        total_predicted_revenue = predictions.sum()
        total_spend = sum(budget_scenarios.values()) * periods
        scenario_roi = total_predicted_revenue / total_spend if total_spend > 0 else 0
        
        return {
            'predictions': predictions,
            'total_revenue': total_predicted_revenue,
            'total_spend': total_spend,
            'roi': scenario_roi,
            'average_weekly_revenue': predictions.mean(),
            'budget_allocation': budget_scenarios
        }
    
    def get_model_summary(self) -> Dict[str, Any]:
        """Get comprehensive model summary"""
        if self.model is None:
            return {"error": "Model not fitted"}
        
        return {
            'model_type': 'Econometric MMM with Adstock & Saturation',
            'performance_metrics': self.performance_metrics,
            'attribution_results': self.attribution_results,
            'contribution_decomposition': self.contribution_decomposition,
            'model_parameters': {
                'adstock_rate': self.adstock_rate,
                'saturation_param': self.saturation_param,
                'regularization_alpha': self.regularization_alpha,
                'base_contribution': self.base_contribution
            },
            'channels_analyzed': len(self.spend_columns),
            'features_used': len(self.feature_names),
            'synergy_effects': len([f for f in self.feature_names if f.startswith('synergy_')])
        }