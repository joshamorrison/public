"""
Causal Inference Models for Econometric Forecasting
Implements causal discovery and inference methods for understanding economic relationships.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional
import logging
from datetime import datetime
import warnings

# Causal inference libraries
try:
    from causalml.inference.meta import LRSRegressor, XGBTRegressor
    from causalml.inference.tree import UpliftTreeClassifier, UpliftRandomForestClassifier
    CAUSALML_AVAILABLE = True
except ImportError:
    CAUSALML_AVAILABLE = False
    logging.warning("CausalML not available. Install with: pip install causalml")

try:
    import dowhy
    from dowhy import CausalModel
    DOWHY_AVAILABLE = True
except ImportError:
    DOWHY_AVAILABLE = False
    logging.warning("DoWhy not available. Install with: pip install dowhy")

try:
    from econml.dml import DML, LinearDML, SparseLinearDML
    from econml.dr import DRLearner
    from econml.metalearners import TLearner, SLearner, XLearner
    ECONML_AVAILABLE = True
except ImportError:
    ECONML_AVAILABLE = False
    logging.warning("EconML not available. Install with: pip install econml")

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import scipy.stats as stats

logger = logging.getLogger(__name__)


class CausalDiscovery:
    """
    Discover causal relationships in economic data using various algorithms.
    """
    
    def __init__(self):
        """Initialize causal discovery engine."""
        self.discovered_relationships = {}
        self.causal_graph = None
        
    def pc_algorithm(self, data: pd.DataFrame, alpha: float = 0.05) -> Dict[str, List[str]]:
        """
        PC (Peter-Clark) algorithm for causal discovery.
        
        Args:
            data: Economic time series data
            alpha: Significance level for independence tests
            
        Returns:
            Dictionary of causal relationships
        """
        logger.info("Running PC algorithm for causal discovery...")
        
        variables = list(data.columns)
        n_vars = len(variables)
        
        # Initialize adjacency matrix (fully connected)
        adj_matrix = np.ones((n_vars, n_vars)) - np.eye(n_vars)
        
        # Phase 1: Remove edges based on conditional independence
        for i in range(n_vars):
            for j in range(i + 1, n_vars):
                if adj_matrix[i, j] == 1:
                    # Test independence
                    var_i, var_j = variables[i], variables[j]
                    
                    # Simple correlation test (can be enhanced with partial correlation)
                    corr, p_value = stats.pearsonr(data[var_i], data[var_j])
                    
                    if p_value > alpha:
                        adj_matrix[i, j] = adj_matrix[j, i] = 0
        
        # Convert to causal relationships
        causal_graph = {}
        for i, var_i in enumerate(variables):
            causal_graph[var_i] = []
            for j, var_j in enumerate(variables):
                if adj_matrix[i, j] == 1:
                    causal_graph[var_i].append(var_j)
        
        self.causal_graph = causal_graph
        logger.info(f"Discovered {sum(len(v) for v in causal_graph.values())} causal relationships")
        
        return causal_graph
    
    def granger_causality_test(self, data: pd.DataFrame, max_lag: int = 4) -> Dict[str, Dict[str, float]]:
        """
        Granger causality test for time series data.
        
        Args:
            data: Time series data
            max_lag: Maximum lag to test
            
        Returns:
            Dictionary of Granger causality p-values
        """
        from statsmodels.tsa.stattools import grangercausalitytests
        
        logger.info("Running Granger causality tests...")
        
        variables = list(data.columns)
        granger_results = {}
        
        for i, var_x in enumerate(variables):
            granger_results[var_x] = {}
            for j, var_y in enumerate(variables):
                if i != j:
                    try:
                        # Prepare data for Granger test
                        test_data = data[[var_y, var_x]].dropna()
                        
                        if len(test_data) > 2 * max_lag:
                            # Run Granger causality test
                            result = grangercausalitytests(test_data, max_lag, verbose=False)
                            
                            # Extract minimum p-value across lags
                            min_p_value = min([result[lag][0]['ssr_ftest'][1] for lag in range(1, max_lag + 1)])
                            granger_results[var_x][var_y] = min_p_value
                        else:
                            granger_results[var_x][var_y] = 1.0  # Not enough data
                            
                    except Exception as e:
                        logger.warning(f"Granger test failed for {var_x} -> {var_y}: {e}")
                        granger_results[var_x][var_y] = 1.0
        
        return granger_results


class CausalInferenceEngine:
    """
    Causal inference for policy impact analysis and counterfactual forecasting.
    """
    
    def __init__(self):
        """Initialize causal inference engine."""
        self.models = {}
        self.treatment_effects = {}
        
    def estimate_treatment_effect(self, 
                                data: pd.DataFrame,
                                treatment_col: str,
                                outcome_col: str,
                                confounders: List[str],
                                method: str = "double_ml") -> Dict[str, Any]:
        """
        Estimate causal treatment effects using various methods.
        
        Args:
            data: Dataset with treatment, outcome, and confounders
            treatment_col: Treatment variable column
            outcome_col: Outcome variable column  
            confounders: List of confounder variables
            method: Estimation method ('double_ml', 'propensity_score', 'iv')
            
        Returns:
            Treatment effect estimates and confidence intervals
        """
        logger.info(f"Estimating treatment effect using {method}...")
        
        # Prepare data
        X = data[confounders].values
        T = data[treatment_col].values
        Y = data[outcome_col].values
        
        if method == "double_ml" and ECONML_AVAILABLE:
            return self._double_ml_estimation(X, T, Y, confounders)
        elif method == "propensity_score":
            return self._propensity_score_estimation(X, T, Y, confounders)
        elif method == "instrumental_variable":
            return self._iv_estimation(data, treatment_col, outcome_col, confounders)
        else:
            return self._simple_regression_estimation(X, T, Y, confounders)
    
    def _double_ml_estimation(self, X, T, Y, confounders) -> Dict[str, Any]:
        """Double Machine Learning estimation."""
        try:
            # Use LinearDML from EconML
            dml_model = LinearDML(
                model_y=RandomForestRegressor(n_estimators=100, random_state=42),
                model_t=RandomForestRegressor(n_estimators=100, random_state=42),
                linear_first_stages=False,
                random_state=42
            )
            
            dml_model.fit(Y, T, X=X)
            
            # Get treatment effect
            treatment_effect = dml_model.effect(X)
            conf_int = dml_model.effect_interval(X, alpha=0.05)
            
            return {
                "method": "double_ml",
                "average_treatment_effect": np.mean(treatment_effect),
                "treatment_effect_std": np.std(treatment_effect),
                "confidence_interval": [np.mean(conf_int[0]), np.mean(conf_int[1])],
                "individual_effects": treatment_effect.tolist(),
                "model": dml_model
            }
            
        except Exception as e:
            logger.error(f"Double ML estimation failed: {e}")
            return {"method": "double_ml", "error": str(e)}
    
    def _propensity_score_estimation(self, X, T, Y, confounders) -> Dict[str, Any]:
        """Propensity score matching estimation."""
        from sklearn.linear_model import LogisticRegression
        from sklearn.neighbors import NearestNeighbors
        
        try:
            # Estimate propensity scores
            ps_model = LogisticRegression(random_state=42)
            ps_model.fit(X, T)
            propensity_scores = ps_model.predict_proba(X)[:, 1]
            
            # Simple matching estimator
            treated_indices = np.where(T == 1)[0]
            control_indices = np.where(T == 0)[0]
            
            # For each treated unit, find nearest control by propensity score
            treated_outcomes = []
            control_outcomes = []
            
            for treated_idx in treated_indices:
                treated_ps = propensity_scores[treated_idx]
                
                # Find closest control unit
                distances = np.abs(propensity_scores[control_indices] - treated_ps)
                closest_control_idx = control_indices[np.argmin(distances)]
                
                treated_outcomes.append(Y[treated_idx])
                control_outcomes.append(Y[closest_control_idx])
            
            # Calculate average treatment effect
            ate = np.mean(treated_outcomes) - np.mean(control_outcomes)
            
            return {
                "method": "propensity_score",
                "average_treatment_effect": ate,
                "treated_mean": np.mean(treated_outcomes),
                "control_mean": np.mean(control_outcomes),
                "n_matched_pairs": len(treated_outcomes),
                "propensity_score_overlap": {
                    "treated_min": np.min(propensity_scores[treated_indices]),
                    "treated_max": np.max(propensity_scores[treated_indices]),
                    "control_min": np.min(propensity_scores[control_indices]),
                    "control_max": np.max(propensity_scores[control_indices])
                }
            }
            
        except Exception as e:
            logger.error(f"Propensity score estimation failed: {e}")
            return {"method": "propensity_score", "error": str(e)}
    
    def _iv_estimation(self, data: pd.DataFrame, treatment_col: str, outcome_col: str, confounders: List[str]) -> Dict[str, Any]:
        """Instrumental variables estimation."""
        try:
            # This is a simplified IV estimation
            # In practice, you would need a valid instrument
            
            # Use lagged treatment as instrument (simplified approach)
            data_with_lag = data.copy()
            data_with_lag[f'{treatment_col}_lag'] = data_with_lag[treatment_col].shift(1)
            data_with_lag = data_with_lag.dropna()
            
            # Two-stage least squares
            # Stage 1: Regress treatment on instrument and confounders
            from sklearn.linear_model import LinearRegression
            
            X_stage1 = data_with_lag[[f'{treatment_col}_lag'] + confounders].values
            T = data_with_lag[treatment_col].values
            
            stage1_model = LinearRegression()
            stage1_model.fit(X_stage1, T)
            T_hat = stage1_model.predict(X_stage1)
            
            # Stage 2: Regress outcome on predicted treatment and confounders
            X_stage2 = np.column_stack([T_hat, data_with_lag[confounders].values])
            Y = data_with_lag[outcome_col].values
            
            stage2_model = LinearRegression()
            stage2_model.fit(X_stage2, Y)
            
            iv_coefficient = stage2_model.coef_[0]  # Treatment effect
            
            return {
                "method": "instrumental_variable",
                "treatment_effect": iv_coefficient,
                "instrument": f"{treatment_col}_lag",
                "stage1_r2": stage1_model.score(X_stage1, T),
                "stage2_r2": stage2_model.score(X_stage2, Y),
                "n_observations": len(data_with_lag)
            }
            
        except Exception as e:
            logger.error(f"IV estimation failed: {e}")
            return {"method": "instrumental_variable", "error": str(e)}
    
    def _simple_regression_estimation(self, X, T, Y, confounders) -> Dict[str, Any]:
        """Simple regression-based estimation (baseline)."""
        try:
            # Linear regression with treatment and confounders
            X_full = np.column_stack([T, X])
            
            model = LinearRegression()
            model.fit(X_full, Y)
            
            treatment_effect = model.coef_[0]
            
            # Calculate confidence interval (simplified)
            residuals = Y - model.predict(X_full)
            mse = np.mean(residuals**2)
            se = np.sqrt(mse / len(Y))
            
            ci_lower = treatment_effect - 1.96 * se
            ci_upper = treatment_effect + 1.96 * se
            
            return {
                "method": "linear_regression",
                "treatment_effect": treatment_effect,
                "standard_error": se,
                "confidence_interval": [ci_lower, ci_upper],
                "r_squared": model.score(X_full, Y),
                "n_observations": len(Y)
            }
            
        except Exception as e:
            logger.error(f"Linear regression estimation failed: {e}")
            return {"method": "linear_regression", "error": str(e)}
    
    def policy_impact_analysis(self, 
                             data: pd.DataFrame,
                             policy_start_date: str,
                             outcome_variables: List[str],
                             control_variables: List[str]) -> Dict[str, Any]:
        """
        Analyze the impact of a policy intervention using causal methods.
        
        Args:
            data: Time series data with policy period
            policy_start_date: When the policy was implemented
            outcome_variables: Variables potentially affected by policy
            control_variables: Control variables
            
        Returns:
            Policy impact analysis results
        """
        logger.info(f"Analyzing policy impact starting from {policy_start_date}")
        
        # Create treatment indicator
        data = data.copy()
        data['policy_treatment'] = (data.index >= pd.to_datetime(policy_start_date)).astype(int)
        
        results = {}
        
        for outcome_var in outcome_variables:
            try:
                # Estimate treatment effect for this outcome
                effect_result = self.estimate_treatment_effect(
                    data=data.dropna(),
                    treatment_col='policy_treatment',
                    outcome_col=outcome_var,
                    confounders=control_variables,
                    method='double_ml' if ECONML_AVAILABLE else 'linear_regression'
                )
                
                results[outcome_var] = effect_result
                
            except Exception as e:
                logger.error(f"Policy impact analysis failed for {outcome_var}: {e}")
                results[outcome_var] = {"error": str(e)}
        
        return {
            "policy_start_date": policy_start_date,
            "outcome_variables": outcome_variables,
            "control_variables": control_variables,
            "treatment_effects": results,
            "analysis_timestamp": datetime.utcnow().isoformat()
        }
    
    def counterfactual_forecasting(self,
                                 data: pd.DataFrame,
                                 treatment_scenarios: Dict[str, Any],
                                 outcome_variable: str,
                                 confounders: List[str],
                                 forecast_horizon: int = 12) -> Dict[str, Any]:
        """
        Generate counterfactual forecasts under different treatment scenarios.
        
        Args:
            data: Historical data
            treatment_scenarios: Different treatment scenarios to evaluate
            outcome_variable: Variable to forecast
            confounders: Control variables
            forecast_horizon: Number of periods to forecast
            
        Returns:
            Counterfactual forecasts for each scenario
        """
        logger.info("Generating counterfactual forecasts...")
        
        counterfactual_results = {}
        
        for scenario_name, scenario_treatment in treatment_scenarios.items():
            try:
                # Create scenario data
                scenario_data = data.copy()
                
                # Apply treatment scenario
                if isinstance(scenario_treatment, (int, float)):
                    scenario_data['treatment'] = scenario_treatment
                else:
                    # More complex treatment assignment
                    scenario_data['treatment'] = scenario_treatment
                
                # Estimate causal model
                causal_model = self.estimate_treatment_effect(
                    data=scenario_data.dropna(),
                    treatment_col='treatment',
                    outcome_col=outcome_variable,
                    confounders=confounders
                )
                
                # Generate counterfactual forecast
                if 'model' in causal_model:
                    # Use the fitted model to predict future outcomes
                    last_confounders = scenario_data[confounders].iloc[-1:].values
                    future_treatment = np.full((forecast_horizon, 1), scenario_treatment)
                    future_X = np.tile(last_confounders, (forecast_horizon, 1))
                    
                    if hasattr(causal_model['model'], 'effect'):
                        future_effects = causal_model['model'].effect(future_X)
                        baseline_forecast = scenario_data[outcome_variable].iloc[-1]
                        
                        counterfactual_forecast = [
                            baseline_forecast + effect for effect in future_effects
                        ]
                    else:
                        # Fallback to simple projection
                        treatment_effect = causal_model.get('treatment_effect', 0)
                        baseline_forecast = scenario_data[outcome_variable].iloc[-1]
                        
                        counterfactual_forecast = [
                            baseline_forecast + treatment_effect * scenario_treatment
                            for _ in range(forecast_horizon)
                        ]
                else:
                    # Simple projection based on treatment effect
                    treatment_effect = causal_model.get('treatment_effect', 0)
                    baseline_forecast = scenario_data[outcome_variable].iloc[-1]
                    
                    counterfactual_forecast = [
                        baseline_forecast + treatment_effect * scenario_treatment
                        for _ in range(forecast_horizon)
                    ]
                
                counterfactual_results[scenario_name] = {
                    "treatment_scenario": scenario_treatment,
                    "forecast": counterfactual_forecast,
                    "treatment_effect": causal_model.get('treatment_effect', 0),
                    "confidence_interval": causal_model.get('confidence_interval', [0, 0])
                }
                
            except Exception as e:
                logger.error(f"Counterfactual forecasting failed for scenario {scenario_name}: {e}")
                counterfactual_results[scenario_name] = {"error": str(e)}
        
        return {
            "outcome_variable": outcome_variable,
            "forecast_horizon": forecast_horizon,
            "scenarios": counterfactual_results,
            "generated_at": datetime.utcnow().isoformat()
        }


# Convenience functions
def discover_causal_relationships(data: pd.DataFrame, method: str = "granger") -> Dict[str, Any]:
    """Discover causal relationships in economic data."""
    discovery = CausalDiscovery()
    
    if method == "granger":
        return discovery.granger_causality_test(data)
    elif method == "pc":
        return discovery.pc_algorithm(data)
    else:
        raise ValueError(f"Unknown causal discovery method: {method}")

def estimate_policy_impact(data: pd.DataFrame, 
                         policy_date: str,
                         outcomes: List[str],
                         controls: List[str]) -> Dict[str, Any]:
    """Estimate the causal impact of a policy intervention."""
    engine = CausalInferenceEngine()
    return engine.policy_impact_analysis(data, policy_date, outcomes, controls)

def generate_counterfactual_scenarios(data: pd.DataFrame,
                                    scenarios: Dict[str, Any],
                                    outcome: str,
                                    controls: List[str]) -> Dict[str, Any]:
    """Generate counterfactual forecasts under different scenarios."""
    engine = CausalInferenceEngine()
    return engine.counterfactual_forecasting(data, scenarios, outcome, controls)