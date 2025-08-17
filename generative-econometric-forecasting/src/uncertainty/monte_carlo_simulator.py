"""
Monte Carlo Simulator
Implements Monte Carlo methods for economic scenario simulation and uncertainty analysis.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize
import concurrent.futures
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class EconomicScenario:
    """Structure for economic scenarios."""
    scenario_id: str
    probability: float
    gdp_growth: float
    inflation_rate: float
    unemployment_rate: float
    interest_rate: float
    description: str
    risk_factors: List[str]


@dataclass
class SimulationParameters:
    """Parameters for Monte Carlo simulation."""
    n_simulations: int = 10000
    time_horizon: int = 12
    random_seed: Optional[int] = None
    confidence_levels: List[float] = None
    correlation_matrix: Optional[np.ndarray] = None
    
    def __post_init__(self):
        if self.confidence_levels is None:
            self.confidence_levels = [0.80, 0.90, 0.95, 0.99]


class MonteCarloSimulator:
    """Monte Carlo simulation for economic forecasting and scenario analysis."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize Monte Carlo simulator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Economic variable distributions
        self.variable_distributions = {
            'gdp_growth': {
                'type': 'normal',
                'params': {'mean': 0.025, 'std': 0.015},  # 2.5% ± 1.5%
                'bounds': (-0.10, 0.08)  # -10% to 8%
            },
            'inflation': {
                'type': 'normal',
                'params': {'mean': 0.02, 'std': 0.01},   # 2% ± 1%
                'bounds': (-0.02, 0.10)  # -2% to 10%
            },
            'unemployment': {
                'type': 'beta',
                'params': {'a': 2, 'b': 8, 'scale': 0.15, 'loc': 0.03},  # 3-18%
                'bounds': (0.01, 0.20)
            },
            'interest_rate': {
                'type': 'gamma',
                'params': {'a': 2, 'scale': 0.02, 'loc': 0.01},  # Skewed positive
                'bounds': (0.0, 0.15)
            }
        }
        
        # Default correlation matrix (can be updated)
        self.correlation_matrix = np.array([
            [1.0, -0.3, -0.7, 0.5],   # GDP: neg corr with inflation/unemployment, pos with rates
            [-0.3, 1.0, 0.4, 0.6],    # Inflation: pos corr with unemployment/rates
            [-0.7, 0.4, 1.0, -0.2],   # Unemployment: neg corr with GDP/rates
            [0.5, 0.6, -0.2, 1.0]     # Interest rates: pos corr with GDP/inflation
        ])
        
        self.variable_names = ['gdp_growth', 'inflation', 'unemployment', 'interest_rate']
        
        logger.info("Monte Carlo simulator initialized")
    
    def simulate_economic_paths(self, 
                               params: SimulationParameters,
                               initial_conditions: Optional[Dict[str, float]] = None) -> Dict[str, Any]:
        """
        Simulate multiple economic paths using Monte Carlo.
        
        Args:
            params: Simulation parameters
            initial_conditions: Starting values for economic variables
        
        Returns:
            Dictionary with simulation results
        """
        logger.info(f"Running {params.n_simulations} Monte Carlo simulations over {params.time_horizon} periods")
        
        if params.random_seed:
            np.random.seed(params.random_seed)
        
        # Initialize results storage
        results = {
            'paths': {},
            'summary_statistics': {},
            'confidence_intervals': {},
            'correlations': {},
            'parameters': params
        }
        
        # Set initial conditions
        if initial_conditions is None:
            initial_conditions = {
                'gdp_growth': 0.025,
                'inflation': 0.02,
                'unemployment': 0.05,
                'interest_rate': 0.03
            }
        
        # Generate correlated random shocks
        if params.correlation_matrix is not None:
            correlation_matrix = params.correlation_matrix
        else:
            correlation_matrix = self.correlation_matrix
        
        # Simulate paths for each variable
        for i, var_name in enumerate(self.variable_names):
            logger.info(f"Simulating {var_name}")
            
            # Generate correlated innovations
            innovations = self._generate_correlated_innovations(
                params.n_simulations,
                params.time_horizon,
                correlation_matrix,
                variable_index=i
            )
            
            # Convert innovations to variable paths
            paths = self._innovations_to_paths(
                innovations,
                var_name,
                initial_conditions[var_name],
                params.time_horizon
            )
            
            results['paths'][var_name] = paths
            
            # Calculate summary statistics
            results['summary_statistics'][var_name] = self._calculate_path_statistics(
                paths, params.confidence_levels
            )
        
        # Calculate cross-variable correlations
        results['correlations'] = self._calculate_path_correlations(results['paths'])
        
        # Calculate confidence intervals for each time step
        for var_name in self.variable_names:
            results['confidence_intervals'][var_name] = self._calculate_time_series_confidence_intervals(
                results['paths'][var_name], params.confidence_levels
            )
        
        logger.info("Monte Carlo simulation completed")
        return results
    
    def _generate_correlated_innovations(self, 
                                       n_simulations: int,
                                       time_horizon: int,
                                       correlation_matrix: np.ndarray,
                                       variable_index: int) -> np.ndarray:
        """Generate correlated random innovations."""
        # Generate independent standard normal variables
        independent_shocks = np.random.standard_normal((n_simulations, time_horizon, len(self.variable_names)))
        
        # Apply correlation structure using Cholesky decomposition
        try:
            chol_matrix = np.linalg.cholesky(correlation_matrix)
            correlated_shocks = np.zeros_like(independent_shocks)
            
            for sim in range(n_simulations):
                for t in range(time_horizon):
                    correlated_shocks[sim, t, :] = chol_matrix @ independent_shocks[sim, t, :]
            
            return correlated_shocks[:, :, variable_index]
            
        except np.linalg.LinAlgError:
            logger.warning("Correlation matrix not positive definite, using independent shocks")
            return independent_shocks[:, :, variable_index]
    
    def _innovations_to_paths(self, 
                            innovations: np.ndarray,
                            variable_name: str,
                            initial_value: float,
                            time_horizon: int) -> np.ndarray:
        """Convert random innovations to variable paths."""
        n_simulations = innovations.shape[0]
        paths = np.zeros((n_simulations, time_horizon + 1))
        paths[:, 0] = initial_value
        
        # Get distribution parameters
        dist_config = self.variable_distributions[variable_name]
        
        for sim in range(n_simulations):
            for t in range(time_horizon):
                # Current value
                current_value = paths[sim, t]
                
                # Generate shock based on distribution type
                if dist_config['type'] == 'normal':
                    shock = innovations[sim, t] * dist_config['params']['std']
                elif dist_config['type'] == 'beta':
                    # Transform normal innovation to beta-distributed change
                    shock = stats.norm.ppf(stats.norm.cdf(innovations[sim, t])) * 0.005
                elif dist_config['type'] == 'gamma':
                    # Transform to gamma-distributed change
                    shock = innovations[sim, t] * 0.005
                else:
                    shock = innovations[sim, t] * 0.01  # Default
                
                # Apply mean reversion for some variables
                if variable_name in ['inflation', 'unemployment', 'interest_rate']:
                    mean_level = dist_config['params'].get('mean', current_value)
                    if variable_name == 'unemployment':
                        mean_level = 0.05  # Natural rate
                    elif variable_name == 'interest_rate':
                        mean_level = 0.03  # Neutral rate
                    
                    # Mean reversion parameter
                    reversion_speed = 0.1
                    mean_reversion = reversion_speed * (mean_level - current_value)
                    shock += mean_reversion
                
                # Update value
                new_value = current_value + shock
                
                # Apply bounds
                bounds = dist_config['bounds']
                new_value = np.clip(new_value, bounds[0], bounds[1])
                
                paths[sim, t + 1] = new_value
        
        return paths
    
    def _calculate_path_statistics(self, 
                                 paths: np.ndarray,
                                 confidence_levels: List[float]) -> Dict[str, Any]:
        """Calculate summary statistics for simulated paths."""
        final_values = paths[:, -1]
        
        stats_dict = {
            'mean': np.mean(final_values),
            'median': np.median(final_values),
            'std': np.std(final_values),
            'min': np.min(final_values),
            'max': np.max(final_values),
            'skewness': stats.skew(final_values),
            'kurtosis': stats.kurtosis(final_values),
            'percentiles': {},
            'var': {}  # Value at Risk
        }
        
        # Calculate percentiles and VaR
        for conf_level in confidence_levels:
            percentile = (1 - conf_level) * 100
            stats_dict['percentiles'][f'p{percentile:.1f}'] = np.percentile(final_values, percentile)
            stats_dict['var'][f'{conf_level:.0%}'] = np.percentile(final_values, (1 - conf_level) * 100)
        
        return stats_dict
    
    def _calculate_path_correlations(self, paths_dict: Dict[str, np.ndarray]) -> Dict[str, Any]:
        """Calculate correlations between simulated paths."""
        # Extract final values for each variable
        final_values = {}
        for var_name, paths in paths_dict.items():
            final_values[var_name] = paths[:, -1]
        
        # Create correlation matrix
        var_names = list(final_values.keys())
        n_vars = len(var_names)
        correlation_matrix = np.zeros((n_vars, n_vars))
        
        for i, var1 in enumerate(var_names):
            for j, var2 in enumerate(var_names):
                correlation_matrix[i, j] = np.corrcoef(
                    final_values[var1], final_values[var2]
                )[0, 1]
        
        return {
            'correlation_matrix': correlation_matrix,
            'variable_names': var_names,
            'pairwise_correlations': {
                f'{var1}_vs_{var2}': correlation_matrix[i, j]
                for i, var1 in enumerate(var_names)
                for j, var2 in enumerate(var_names)
                if i < j
            }
        }
    
    def _calculate_time_series_confidence_intervals(self, 
                                                  paths: np.ndarray,
                                                  confidence_levels: List[float]) -> Dict[str, np.ndarray]:
        """Calculate confidence intervals for each time step."""
        confidence_intervals = {}
        
        for conf_level in confidence_levels:
            alpha = (1 - conf_level) / 2
            lower_percentile = alpha * 100
            upper_percentile = (1 - alpha) * 100
            
            lower_bound = np.percentile(paths, lower_percentile, axis=0)
            upper_bound = np.percentile(paths, upper_percentile, axis=0)
            
            confidence_intervals[f'{conf_level:.0%}'] = {
                'lower': lower_bound,
                'upper': upper_bound
            }
        
        return confidence_intervals
    
    def generate_stress_scenarios(self, 
                                base_conditions: Dict[str, float],
                                stress_types: List[str] = None) -> Dict[str, EconomicScenario]:
        """
        Generate stress test scenarios for economic analysis.
        
        Args:
            base_conditions: Baseline economic conditions
            stress_types: Types of stress scenarios to generate
        
        Returns:
            Dictionary of stress scenarios
        """
        if stress_types is None:
            stress_types = ['recession', 'inflation_shock', 'financial_crisis', 'stagflation']
        
        scenarios = {}
        
        for stress_type in stress_types:
            if stress_type == 'recession':
                scenario = EconomicScenario(
                    scenario_id='recession_stress',
                    probability=0.15,
                    gdp_growth=base_conditions.get('gdp_growth', 0.025) - 0.06,  # -6% shock
                    inflation_rate=base_conditions.get('inflation', 0.02) - 0.01,
                    unemployment_rate=base_conditions.get('unemployment', 0.05) + 0.04,
                    interest_rate=base_conditions.get('interest_rate', 0.03) - 0.015,
                    description="Severe economic recession with negative growth",
                    risk_factors=['Demand collapse', 'Business failures', 'Job losses']
                )
                
            elif stress_type == 'inflation_shock':
                scenario = EconomicScenario(
                    scenario_id='inflation_shock',
                    probability=0.20,
                    gdp_growth=base_conditions.get('gdp_growth', 0.025) - 0.02,
                    inflation_rate=base_conditions.get('inflation', 0.02) + 0.05,  # +5% shock
                    unemployment_rate=base_conditions.get('unemployment', 0.05) + 0.015,
                    interest_rate=base_conditions.get('interest_rate', 0.03) + 0.04,
                    description="High inflation with aggressive monetary tightening",
                    risk_factors=['Supply chain disruption', 'Commodity price surge', 'Wage spiral']
                )
                
            elif stress_type == 'financial_crisis':
                scenario = EconomicScenario(
                    scenario_id='financial_crisis',
                    probability=0.10,
                    gdp_growth=base_conditions.get('gdp_growth', 0.025) - 0.08,
                    inflation_rate=base_conditions.get('inflation', 0.02) - 0.005,
                    unemployment_rate=base_conditions.get('unemployment', 0.05) + 0.06,
                    interest_rate=base_conditions.get('interest_rate', 0.03) + 0.02,
                    description="Financial system crisis with credit freeze",
                    risk_factors=['Bank failures', 'Credit crunch', 'Asset price collapse']
                )
                
            elif stress_type == 'stagflation':
                scenario = EconomicScenario(
                    scenario_id='stagflation',
                    probability=0.12,
                    gdp_growth=base_conditions.get('gdp_growth', 0.025) - 0.035,
                    inflation_rate=base_conditions.get('inflation', 0.02) + 0.04,
                    unemployment_rate=base_conditions.get('unemployment', 0.05) + 0.03,
                    interest_rate=base_conditions.get('interest_rate', 0.03) + 0.025,
                    description="Stagnant growth with persistent high inflation",
                    risk_factors=['Supply constraints', 'Policy dilemma', 'Expectations anchoring']
                )
            
            scenarios[stress_type] = scenario
        
        return scenarios
    
    def simulate_scenario_outcomes(self, 
                                 scenarios: Dict[str, EconomicScenario],
                                 time_horizon: int = 12,
                                 n_simulations: int = 5000) -> Dict[str, Dict[str, Any]]:
        """
        Simulate outcomes under different stress scenarios.
        
        Args:
            scenarios: Dictionary of scenarios to simulate
            time_horizon: Simulation time horizon
            n_simulations: Number of simulations per scenario
        
        Returns:
            Simulation results for each scenario
        """
        results = {}
        
        for scenario_name, scenario in scenarios.items():
            logger.info(f"Simulating scenario: {scenario_name}")
            
            # Set initial conditions from scenario
            initial_conditions = {
                'gdp_growth': scenario.gdp_growth,
                'inflation': scenario.inflation_rate,
                'unemployment': scenario.unemployment_rate,
                'interest_rate': scenario.interest_rate
            }
            
            # Create simulation parameters
            params = SimulationParameters(
                n_simulations=n_simulations,
                time_horizon=time_horizon,
                random_seed=self.random_seed
            )
            
            # Run simulation
            scenario_results = self.simulate_economic_paths(params, initial_conditions)
            
            # Add scenario metadata
            scenario_results['scenario_info'] = {
                'scenario_id': scenario.scenario_id,
                'probability': scenario.probability,
                'description': scenario.description,
                'risk_factors': scenario.risk_factors
            }
            
            results[scenario_name] = scenario_results
        
        return results
    
    def calculate_portfolio_impact(self, 
                                 simulation_results: Dict[str, Any],
                                 portfolio_sensitivities: Dict[str, float]) -> Dict[str, Any]:
        """
        Calculate portfolio impact based on economic simulations.
        
        Args:
            simulation_results: Results from Monte Carlo simulation
            portfolio_sensitivities: Portfolio sensitivities to economic variables
        
        Returns:
            Portfolio impact analysis
        """
        portfolio_returns = []
        
        # Extract paths
        paths = simulation_results['paths']
        n_simulations = list(paths.values())[0].shape[0]
        
        for sim in range(n_simulations):
            portfolio_return = 0.0
            
            for var_name, sensitivity in portfolio_sensitivities.items():
                if var_name in paths:
                    # Calculate change in variable
                    initial_value = paths[var_name][sim, 0]
                    final_value = paths[var_name][sim, -1]
                    variable_change = (final_value - initial_value) / initial_value
                    
                    # Apply sensitivity
                    portfolio_return += sensitivity * variable_change
            
            portfolio_returns.append(portfolio_return)
        
        portfolio_returns = np.array(portfolio_returns)
        
        # Calculate portfolio statistics
        portfolio_stats = {
            'expected_return': np.mean(portfolio_returns),
            'volatility': np.std(portfolio_returns),
            'skewness': stats.skew(portfolio_returns),
            'kurtosis': stats.kurtosis(portfolio_returns),
            'var_95': np.percentile(portfolio_returns, 5),
            'var_99': np.percentile(portfolio_returns, 1),
            'cvar_95': np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 5)]),
            'cvar_99': np.mean(portfolio_returns[portfolio_returns <= np.percentile(portfolio_returns, 1)]),
            'probability_negative': np.mean(portfolio_returns < 0),
            'max_loss': np.min(portfolio_returns),
            'max_gain': np.max(portfolio_returns)
        }
        
        return {
            'portfolio_returns': portfolio_returns,
            'portfolio_statistics': portfolio_stats,
            'sensitivities': portfolio_sensitivities
        }
    
    def export_simulation_results(self, 
                                results: Dict[str, Any],
                                filename: str) -> str:
        """Export simulation results to file."""
        import json
        
        # Convert numpy arrays to lists for JSON serialization
        export_data = {}
        
        for key, value in results.items():
            if key == 'paths':
                export_data[key] = {
                    var_name: paths.tolist() 
                    for var_name, paths in value.items()
                }
            elif key == 'confidence_intervals':
                export_data[key] = {}
                for var_name, intervals in value.items():
                    export_data[key][var_name] = {}
                    for conf_level, bounds in intervals.items():
                        export_data[key][var_name][conf_level] = {
                            'lower': bounds['lower'].tolist(),
                            'upper': bounds['upper'].tolist()
                        }
            elif isinstance(value, np.ndarray):
                export_data[key] = value.tolist()
            else:
                export_data[key] = value
        
        try:
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            logger.info(f"Simulation results exported to {filename}")
            return filename
            
        except Exception as e:
            logger.error(f"Error exporting results: {e}")
            return ""


if __name__ == "__main__":
    # Example usage
    simulator = MonteCarloSimulator()
    
    # Set up simulation parameters
    params = SimulationParameters(
        n_simulations=1000,
        time_horizon=12,
        random_seed=42
    )
    
    # Run baseline simulation
    print("Running baseline Monte Carlo simulation...")
    baseline_results = simulator.simulate_economic_paths(params)
    
    print("Simulation Results:")
    for var_name, stats in baseline_results['summary_statistics'].items():
        print(f"\n{var_name.upper()}:")
        print(f"  Mean: {stats['mean']:.4f}")
        print(f"  Std Dev: {stats['std']:.4f}")
        print(f"  95% VaR: {stats['var']['95%']:.4f}")
    
    # Generate stress scenarios
    base_conditions = {
        'gdp_growth': 0.025,
        'inflation': 0.02,
        'unemployment': 0.05,
        'interest_rate': 0.03
    }
    
    stress_scenarios = simulator.generate_stress_scenarios(base_conditions)
    print(f"\nGenerated {len(stress_scenarios)} stress scenarios:")
    for name, scenario in stress_scenarios.items():
        print(f"  {name}: {scenario.description} (prob: {scenario.probability:.1%})")
    
    # Simulate stress scenarios
    print("\nSimulating stress scenarios...")
    stress_results = simulator.simulate_scenario_outcomes(
        stress_scenarios, 
        time_horizon=12, 
        n_simulations=500
    )
    
    # Portfolio impact analysis
    portfolio_sensitivities = {
        'gdp_growth': 2.0,      # 2% portfolio return per 1% GDP growth
        'inflation': -1.5,      # -1.5% portfolio return per 1% inflation
        'unemployment': -0.8,   # -0.8% portfolio return per 1% unemployment
        'interest_rate': -1.2   # -1.2% portfolio return per 1% interest rate
    }
    
    portfolio_impact = simulator.calculate_portfolio_impact(
        baseline_results, 
        portfolio_sensitivities
    )
    
    print(f"\nPortfolio Impact Analysis:")
    print(f"  Expected Return: {portfolio_impact['portfolio_statistics']['expected_return']:.3f}")
    print(f"  Volatility: {portfolio_impact['portfolio_statistics']['volatility']:.3f}")
    print(f"  95% VaR: {portfolio_impact['portfolio_statistics']['var_95']:.3f}")
    print(f"  Probability of Loss: {portfolio_impact['portfolio_statistics']['probability_negative']:.1%}")
    
    # Export results
    export_file = simulator.export_simulation_results(
        baseline_results, 
        'monte_carlo_results.json'
    )
    print(f"\nResults exported to: {export_file}")
    
    print("\nMonte Carlo simulation example completed")