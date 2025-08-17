"""
Economic Scenario Simulator
Comprehensive economic scenario generation using multiple approaches including stress testing and rare event simulation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple, Callable
import logging
from datetime import datetime, timedelta
from dataclasses import dataclass
import warnings
warnings.filterwarnings('ignore')

from scipy import stats
from scipy.optimize import minimize
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.mixture import GaussianMixture

logger = logging.getLogger(__name__)


@dataclass
class EconomicScenario:
    """Comprehensive economic scenario structure."""
    scenario_id: str
    scenario_name: str
    description: str
    probability: float
    time_horizon: int
    economic_variables: Dict[str, np.ndarray]
    metadata: Dict[str, Any]
    risk_factors: List[str]
    policy_implications: List[str]
    confidence_level: float


@dataclass
class RareEventConfiguration:
    """Configuration for rare event simulation."""
    event_type: str
    probability: float
    magnitude: float
    duration: int
    affected_variables: List[str]
    correlation_effects: Dict[str, float]


class EconomicScenarioSimulator:
    """Advanced economic scenario simulation with stress testing and rare events."""
    
    def __init__(self, random_seed: int = 42):
        """
        Initialize scenario simulator.
        
        Args:
            random_seed: Random seed for reproducibility
        """
        self.random_seed = random_seed
        np.random.seed(random_seed)
        
        # Economic variable configurations
        self.variable_configs = {
            'gdp_growth': {
                'mean': 0.025,
                'std': 0.015,
                'min_bound': -0.10,
                'max_bound': 0.08,
                'persistence': 0.3,
                'seasonality': 0.005
            },
            'inflation': {
                'mean': 0.02,
                'std': 0.01,
                'min_bound': -0.02,
                'max_bound': 0.12,
                'persistence': 0.5,
                'seasonality': 0.002
            },
            'unemployment': {
                'mean': 0.05,
                'std': 0.015,
                'min_bound': 0.02,
                'max_bound': 0.15,
                'persistence': 0.7,
                'seasonality': 0.003
            },
            'interest_rate': {
                'mean': 0.03,
                'std': 0.02,
                'min_bound': 0.0,
                'max_bound': 0.15,
                'persistence': 0.8,
                'seasonality': 0.001
            },
            'stock_market': {
                'mean': 0.08,
                'std': 0.20,
                'min_bound': -0.50,
                'max_bound': 0.50,
                'persistence': 0.1,
                'seasonality': 0.02
            },
            'commodity_prices': {
                'mean': 0.05,
                'std': 0.25,
                'min_bound': -0.60,
                'max_bound': 0.80,
                'persistence': 0.2,
                'seasonality': 0.05
            }
        }
        
        # Correlation matrix for economic variables
        self.correlation_matrix = np.array([
            [1.0, -0.3, -0.6, 0.4, 0.7, 0.3],   # GDP
            [-0.3, 1.0, 0.3, 0.6, -0.2, 0.4],   # Inflation
            [-0.6, 0.3, 1.0, -0.1, -0.5, 0.1],  # Unemployment
            [0.4, 0.6, -0.1, 1.0, 0.3, 0.2],    # Interest rate
            [0.7, -0.2, -0.5, 0.3, 1.0, 0.5],   # Stock market
            [0.3, 0.4, 0.1, 0.2, 0.5, 1.0]      # Commodities
        ])
        
        self.variable_names = list(self.variable_configs.keys())
        
        # Rare event configurations
        self.rare_events = {
            'financial_crisis': RareEventConfiguration(
                event_type='financial_crisis',
                probability=0.05,
                magnitude=3.0,
                duration=8,
                affected_variables=['gdp_growth', 'unemployment', 'stock_market'],
                correlation_effects={'gdp_growth': -0.8, 'unemployment': 0.6, 'stock_market': -0.4}
            ),
            'oil_shock': RareEventConfiguration(
                event_type='oil_shock',
                probability=0.08,
                magnitude=2.5,
                duration=6,
                affected_variables=['inflation', 'commodity_prices', 'gdp_growth'],
                correlation_effects={'inflation': 0.7, 'commodity_prices': 0.9, 'gdp_growth': -0.3}
            ),
            'pandemic': RareEventConfiguration(
                event_type='pandemic',
                probability=0.03,
                magnitude=4.0,
                duration=12,
                affected_variables=['gdp_growth', 'unemployment', 'stock_market', 'interest_rate'],
                correlation_effects={'gdp_growth': -0.9, 'unemployment': 0.8, 'stock_market': -0.6, 'interest_rate': -0.5}
            ),
            'geopolitical_crisis': RareEventConfiguration(
                event_type='geopolitical_crisis',
                probability=0.10,
                magnitude=2.0,
                duration=4,
                affected_variables=['commodity_prices', 'stock_market', 'inflation'],
                correlation_effects={'commodity_prices': 0.5, 'stock_market': -0.3, 'inflation': 0.4}
            )
        }
        
        logger.info("Economic scenario simulator initialized")
    
    def generate_baseline_scenarios(self, 
                                  time_horizon: int = 24,
                                  n_scenarios: int = 1000,
                                  include_seasonality: bool = True) -> List[EconomicScenario]:
        """
        Generate baseline economic scenarios without rare events.
        
        Args:
            time_horizon: Forecast horizon in periods
            n_scenarios: Number of scenarios to generate
            include_seasonality: Whether to include seasonal patterns
        
        Returns:
            List of baseline economic scenarios
        """
        logger.info(f"Generating {n_scenarios} baseline scenarios over {time_horizon} periods")
        
        scenarios = []
        
        # Generate correlated innovations
        innovations = self._generate_correlated_innovations(n_scenarios, time_horizon)
        
        for i in range(n_scenarios):
            scenario_data = {}
            
            for j, var_name in enumerate(self.variable_names):
                config = self.variable_configs[var_name]
                
                # Generate variable path
                path = self._generate_variable_path(
                    innovations[i, :, j],
                    config,
                    time_horizon,
                    include_seasonality
                )
                
                scenario_data[var_name] = path
            
            # Create scenario object
            scenario = EconomicScenario(
                scenario_id=f"baseline_{i:04d}",
                scenario_name=f"Baseline Scenario {i+1}",
                description="Baseline economic scenario without rare events",
                probability=1.0 / n_scenarios,
                time_horizon=time_horizon,
                economic_variables=scenario_data,
                metadata={
                    'scenario_type': 'baseline',
                    'generation_method': 'correlated_multivariate',
                    'includes_seasonality': include_seasonality
                },
                risk_factors=[],
                policy_implications=[],
                confidence_level=0.8
            )
            
            scenarios.append(scenario)
        
        logger.info(f"Generated {len(scenarios)} baseline scenarios")
        return scenarios
    
    def generate_stress_test_scenarios(self, 
                                     time_horizon: int = 24,
                                     stress_types: List[str] = None) -> List[EconomicScenario]:
        """
        Generate stress test scenarios for regulatory compliance and risk assessment.
        
        Args:
            time_horizon: Forecast horizon in periods
            stress_types: Types of stress scenarios to generate
        
        Returns:
            List of stress test scenarios
        """
        if stress_types is None:
            stress_types = ['severe_recession', 'hyperinflation', 'deflation', 'stagflation', 'financial_meltdown']
        
        logger.info(f"Generating stress test scenarios: {stress_types}")
        
        scenarios = []
        
        for stress_type in stress_types:
            scenario_data = self._generate_stress_scenario(stress_type, time_horizon)
            
            if scenario_data:
                scenario = EconomicScenario(
                    scenario_id=f"stress_{stress_type}",
                    scenario_name=f"Stress Test: {stress_type.replace('_', ' ').title()}",
                    description=self._get_stress_description(stress_type),
                    probability=self._get_stress_probability(stress_type),
                    time_horizon=time_horizon,
                    economic_variables=scenario_data,
                    metadata={
                        'scenario_type': 'stress_test',
                        'stress_type': stress_type,
                        'regulatory_compliant': True
                    },
                    risk_factors=self._get_stress_risk_factors(stress_type),
                    policy_implications=self._get_stress_policy_implications(stress_type),
                    confidence_level=0.95
                )
                
                scenarios.append(scenario)
        
        logger.info(f"Generated {len(scenarios)} stress test scenarios")
        return scenarios
    
    def generate_rare_event_scenarios(self, 
                                     time_horizon: int = 24,
                                     event_types: List[str] = None,
                                     n_scenarios_per_event: int = 50) -> List[EconomicScenario]:
        """
        Generate scenarios incorporating rare economic events.
        
        Args:
            time_horizon: Forecast horizon in periods
            event_types: Types of rare events to simulate
            n_scenarios_per_event: Number of scenarios per event type
        
        Returns:
            List of rare event scenarios
        """
        if event_types is None:
            event_types = list(self.rare_events.keys())
        
        logger.info(f"Generating rare event scenarios: {event_types}")
        
        scenarios = []
        
        for event_type in event_types:
            if event_type not in self.rare_events:
                logger.warning(f"Unknown rare event type: {event_type}")
                continue
            
            event_config = self.rare_events[event_type]
            
            for i in range(n_scenarios_per_event):
                scenario_data = self._generate_rare_event_scenario(
                    event_config, time_horizon, i
                )
                
                scenario = EconomicScenario(
                    scenario_id=f"rare_{event_type}_{i:03d}",
                    scenario_name=f"Rare Event: {event_type.replace('_', ' ').title()} #{i+1}",
                    description=f"Economic scenario with {event_type.replace('_', ' ')} rare event",
                    probability=event_config.probability / n_scenarios_per_event,
                    time_horizon=time_horizon,
                    economic_variables=scenario_data,
                    metadata={
                        'scenario_type': 'rare_event',
                        'event_type': event_type,
                        'event_magnitude': event_config.magnitude,
                        'event_duration': event_config.duration
                    },
                    risk_factors=self._get_rare_event_risk_factors(event_type),
                    policy_implications=self._get_rare_event_policy_implications(event_type),
                    confidence_level=0.6
                )
                
                scenarios.append(scenario)
        
        logger.info(f"Generated {len(scenarios)} rare event scenarios")
        return scenarios
    
    def _generate_correlated_innovations(self, 
                                       n_scenarios: int,
                                       time_horizon: int) -> np.ndarray:
        """Generate correlated random innovations."""
        # Generate independent standard normal variables
        independent_shocks = np.random.standard_normal((n_scenarios, time_horizon, len(self.variable_names)))
        
        # Apply correlation structure using Cholesky decomposition
        try:
            chol_matrix = np.linalg.cholesky(self.correlation_matrix)
            correlated_shocks = np.zeros_like(independent_shocks)
            
            for scenario in range(n_scenarios):
                for t in range(time_horizon):
                    correlated_shocks[scenario, t, :] = chol_matrix @ independent_shocks[scenario, t, :]\n        \n            return correlated_shocks\n            \n        except np.linalg.LinAlgError:\n            logger.warning(\"Correlation matrix not positive definite, using independent shocks\")\n            return independent_shocks\n    \n    def _generate_variable_path(self, \n                               innovations: np.ndarray,\n                               config: Dict[str, float],\n                               time_horizon: int,\n                               include_seasonality: bool) -> np.ndarray:\n        \"\"\"Generate path for a single economic variable.\"\"\"\n        path = np.zeros(time_horizon + 1)\n        path[0] = config['mean']  # Initial value\n        \n        for t in range(time_horizon):\n            # Base change\n            change = innovations[t] * config['std']\n            \n            # Add persistence (AR(1) component)\n            if t > 0:\n                persistence_effect = config['persistence'] * (path[t] - config['mean'])\n                change += persistence_effect\n            \n            # Add seasonality\n            if include_seasonality:\n                seasonal_effect = config['seasonality'] * np.sin(2 * np.pi * t / 12)\n                change += seasonal_effect\n            \n            # Update path\n            new_value = path[t] + change\n            \n            # Apply bounds\n            new_value = np.clip(new_value, config['min_bound'], config['max_bound'])\n            path[t + 1] = new_value\n        \n        return path[1:]  # Return without initial value\n    \n    def _generate_stress_scenario(self, \n                                 stress_type: str,\n                                 time_horizon: int) -> Dict[str, np.ndarray]:\n        \"\"\"Generate specific stress scenario.\"\"\"\n        scenario_data = {}\n        \n        if stress_type == 'severe_recession':\n            scenario_data['gdp_growth'] = np.linspace(-0.08, -0.02, time_horizon)\n            scenario_data['unemployment'] = np.linspace(0.12, 0.08, time_horizon)\n            scenario_data['inflation'] = np.linspace(0.01, 0.015, time_horizon)\n            scenario_data['interest_rate'] = np.linspace(0.001, 0.005, time_horizon)\n            scenario_data['stock_market'] = np.linspace(-0.4, -0.1, time_horizon)\n            scenario_data['commodity_prices'] = np.linspace(-0.3, 0.0, time_horizon)\n            \n        elif stress_type == 'hyperinflation':\n            scenario_data['inflation'] = np.linspace(0.15, 0.25, time_horizon)\n            scenario_data['interest_rate'] = np.linspace(0.12, 0.20, time_horizon)\n            scenario_data['gdp_growth'] = np.linspace(-0.05, 0.01, time_horizon)\n            scenario_data['unemployment'] = np.linspace(0.08, 0.12, time_horizon)\n            scenario_data['stock_market'] = np.linspace(-0.3, 0.05, time_horizon)\n            scenario_data['commodity_prices'] = np.linspace(0.3, 0.6, time_horizon)\n            \n        elif stress_type == 'deflation':\n            scenario_data['inflation'] = np.linspace(-0.03, -0.01, time_horizon)\n            scenario_data['gdp_growth'] = np.linspace(-0.04, 0.005, time_horizon)\n            scenario_data['unemployment'] = np.linspace(0.08, 0.06, time_horizon)\n            scenario_data['interest_rate'] = np.linspace(0.0, 0.001, time_horizon)\n            scenario_data['stock_market'] = np.linspace(-0.2, 0.1, time_horizon)\n            scenario_data['commodity_prices'] = np.linspace(-0.4, -0.1, time_horizon)\n            \n        elif stress_type == 'stagflation':\n            scenario_data['inflation'] = np.linspace(0.08, 0.12, time_horizon)\n            scenario_data['gdp_growth'] = np.linspace(-0.02, 0.005, time_horizon)\n            scenario_data['unemployment'] = np.linspace(0.08, 0.10, time_horizon)\n            scenario_data['interest_rate'] = np.linspace(0.06, 0.10, time_horizon)\n            scenario_data['stock_market'] = np.linspace(-0.15, 0.02, time_horizon)\n            scenario_data['commodity_prices'] = np.linspace(0.2, 0.4, time_horizon)\n            \n        elif stress_type == 'financial_meltdown':\n            scenario_data['stock_market'] = np.linspace(-0.6, -0.2, time_horizon)\n            scenario_data['interest_rate'] = np.linspace(0.001, 0.15, time_horizon)\n            scenario_data['gdp_growth'] = np.linspace(-0.10, -0.03, time_horizon)\n            scenario_data['unemployment'] = np.linspace(0.15, 0.10, time_horizon)\n            scenario_data['inflation'] = np.linspace(0.005, 0.02, time_horizon)\n            scenario_data['commodity_prices'] = np.linspace(-0.5, -0.1, time_horizon)\n        \n        else:\n            logger.warning(f\"Unknown stress type: {stress_type}\")\n            return None\n        \n        # Add noise to make scenarios more realistic\n        for var_name, path in scenario_data.items():\n            config = self.variable_configs[var_name]\n            noise = np.random.normal(0, config['std'] * 0.2, time_horizon)\n            scenario_data[var_name] = np.clip(\n                path + noise, \n                config['min_bound'], \n                config['max_bound']\n            )\n        \n        return scenario_data\n    \n    def _generate_rare_event_scenario(self, \n                                     event_config: RareEventConfiguration,\n                                     time_horizon: int,\n                                     scenario_index: int) -> Dict[str, np.ndarray]:\n        \"\"\"Generate scenario with rare event.\"\"\"\n        # Start with baseline scenario\n        baseline_innovations = np.random.standard_normal((time_horizon, len(self.variable_names)))\n        \n        # Apply correlation\n        try:\n            chol_matrix = np.linalg.cholesky(self.correlation_matrix)\n            correlated_innovations = np.array([chol_matrix @ baseline_innovations[t, :] for t in range(time_horizon)])\n        except:\n            correlated_innovations = baseline_innovations\n        \n        scenario_data = {}\n        \n        for j, var_name in enumerate(self.variable_names):\n            config = self.variable_configs[var_name]\n            \n            # Generate baseline path\n            path = self._generate_variable_path(\n                correlated_innovations[:, j],\n                config,\n                time_horizon,\n                include_seasonality=True\n            )\n            \n            # Apply rare event if variable is affected\n            if var_name in event_config.affected_variables:\n                event_start = np.random.randint(0, max(1, time_horizon - event_config.duration))\n                event_end = min(event_start + event_config.duration, time_horizon)\n                \n                # Calculate event impact\n                correlation_effect = event_config.correlation_effects.get(var_name, 0.0)\n                magnitude = event_config.magnitude * correlation_effect\n                \n                # Apply event impact with decay\n                for t in range(event_start, event_end):\n                    decay_factor = 1.0 - (t - event_start) / event_config.duration\n                    impact = magnitude * config['std'] * decay_factor\n                    path[t] += impact\n                \n                # Apply bounds\n                path = np.clip(path, config['min_bound'], config['max_bound'])\n            \n            scenario_data[var_name] = path\n        \n        return scenario_data\n    \n    def cluster_scenarios(self, \n                         scenarios: List[EconomicScenario],\n                         n_clusters: int = 5,\n                         variables: List[str] = None) -> Dict[int, List[EconomicScenario]]:\n        \"\"\"Cluster scenarios based on economic variable patterns.\"\"\"\n        if variables is None:\n            variables = self.variable_names\n        \n        # Extract features for clustering\n        features = []\n        for scenario in scenarios:\n            scenario_features = []\n            for var_name in variables:\n                if var_name in scenario.economic_variables:\n                    var_data = scenario.economic_variables[var_name]\n                    # Use statistical features\n                    scenario_features.extend([\n                        np.mean(var_data),\n                        np.std(var_data),\n                        np.min(var_data),\n                        np.max(var_data),\n                        var_data[-1] - var_data[0]  # Total change\n                    ])\n            features.append(scenario_features)\n        \n        features = np.array(features)\n        \n        # Standardize features\n        scaler = StandardScaler()\n        features_scaled = scaler.fit_transform(features)\n        \n        # Perform clustering\n        kmeans = KMeans(n_clusters=n_clusters, random_state=self.random_seed)\n        cluster_labels = kmeans.fit_predict(features_scaled)\n        \n        # Group scenarios by cluster\n        clustered_scenarios = {}\n        for i in range(n_clusters):\n            clustered_scenarios[i] = []\n        \n        for scenario, label in zip(scenarios, cluster_labels):\n            clustered_scenarios[label].append(scenario)\n        \n        logger.info(f\"Clustered {len(scenarios)} scenarios into {n_clusters} groups\")\n        return clustered_scenarios\n    \n    def calculate_scenario_statistics(self, \n                                    scenarios: List[EconomicScenario]) -> Dict[str, Dict[str, float]]:\n        \"\"\"Calculate summary statistics across scenarios.\"\"\"\n        stats = {}\n        \n        for var_name in self.variable_names:\n            var_data = []\n            for scenario in scenarios:\n                if var_name in scenario.economic_variables:\n                    var_data.extend(scenario.economic_variables[var_name])\n            \n            if var_data:\n                var_array = np.array(var_data)\n                stats[var_name] = {\n                    'mean': np.mean(var_array),\n                    'std': np.std(var_array),\n                    'min': np.min(var_array),\n                    'max': np.max(var_array),\n                    'p5': np.percentile(var_array, 5),\n                    'p95': np.percentile(var_array, 95),\n                    'skewness': stats.skew(var_array),\n                    'kurtosis': stats.kurtosis(var_array)\n                }\n        \n        return stats\n    \n    def export_scenarios(self, \n                        scenarios: List[EconomicScenario],\n                        filename: str,\n                        format: str = 'json') -> str:\n        \"\"\"Export scenarios to file.\"\"\"\n        if format == 'json':\n            import json\n            \n            export_data = []\n            for scenario in scenarios:\n                scenario_dict = {\n                    'scenario_id': scenario.scenario_id,\n                    'scenario_name': scenario.scenario_name,\n                    'description': scenario.description,\n                    'probability': scenario.probability,\n                    'time_horizon': scenario.time_horizon,\n                    'economic_variables': {k: v.tolist() for k, v in scenario.economic_variables.items()},\n                    'metadata': scenario.metadata,\n                    'risk_factors': scenario.risk_factors,\n                    'policy_implications': scenario.policy_implications,\n                    'confidence_level': scenario.confidence_level\n                }\n                export_data.append(scenario_dict)\n            \n            with open(filename, 'w') as f:\n                json.dump(export_data, f, indent=2, default=str)\n            \n        elif format == 'csv':\n            # Export as CSV with scenarios as rows\n            rows = []\n            for scenario in scenarios:\n                row = {\n                    'scenario_id': scenario.scenario_id,\n                    'scenario_name': scenario.scenario_name,\n                    'probability': scenario.probability,\n                    'confidence_level': scenario.confidence_level\n                }\n                \n                # Add variable statistics\n                for var_name, var_data in scenario.economic_variables.items():\n                    row[f'{var_name}_mean'] = np.mean(var_data)\n                    row[f'{var_name}_std'] = np.std(var_data)\n                    row[f'{var_name}_final'] = var_data[-1]\n                \n                rows.append(row)\n            \n            df = pd.DataFrame(rows)\n            df.to_csv(filename, index=False)\n        \n        logger.info(f\"Exported {len(scenarios)} scenarios to {filename}\")\n        return filename\n    \n    # Helper methods for stress scenario metadata\n    def _get_stress_description(self, stress_type: str) -> str:\n        descriptions = {\n            'severe_recession': 'Severe economic recession with significant GDP contraction',\n            'hyperinflation': 'Hyperinflationary environment with rapidly rising prices',\n            'deflation': 'Deflationary spiral with falling prices and economic stagnation',\n            'stagflation': 'Stagflation with high inflation and low growth',\n            'financial_meltdown': 'Financial system collapse with market crashes'\n        }\n        return descriptions.get(stress_type, f'Stress test scenario: {stress_type}')\n    \n    def _get_stress_probability(self, stress_type: str) -> float:\n        probabilities = {\n            'severe_recession': 0.05,\n            'hyperinflation': 0.02,\n            'deflation': 0.03,\n            'stagflation': 0.04,\n            'financial_meltdown': 0.01\n        }\n        return probabilities.get(stress_type, 0.05)\n    \n    def _get_stress_risk_factors(self, stress_type: str) -> List[str]:\n        risk_factors = {\n            'severe_recession': ['Demand collapse', 'Business failures', 'Mass unemployment'],\n            'hyperinflation': ['Currency devaluation', 'Supply chain disruption', 'Wage-price spiral'],\n            'deflation': ['Debt deflation', 'Liquidity trap', 'Deflationary expectations'],\n            'stagflation': ['Supply shocks', 'Policy ineffectiveness', 'Economic uncertainty'],\n            'financial_meltdown': ['Bank failures', 'Credit freeze', 'Market panic']\n        }\n        return risk_factors.get(stress_type, [])\n    \n    def _get_stress_policy_implications(self, stress_type: str) -> List[str]:\n        implications = {\n            'severe_recession': ['Fiscal stimulus', 'Monetary easing', 'Unemployment support'],\n            'hyperinflation': ['Monetary tightening', 'Price controls', 'Currency stabilization'],\n            'deflation': ['Quantitative easing', 'Fiscal expansion', 'Negative interest rates'],\n            'stagflation': ['Targeted interventions', 'Supply-side policies', 'Expectations management'],\n            'financial_meltdown': ['Bank bailouts', 'Liquidity support', 'Regulatory intervention']\n        }\n        return implications.get(stress_type, [])\n    \n    def _get_rare_event_risk_factors(self, event_type: str) -> List[str]:\n        risk_factors = {\n            'financial_crisis': ['Systemic risk', 'Contagion effects', 'Liquidity crisis'],\n            'oil_shock': ['Energy security', 'Supply disruption', 'Price volatility'],\n            'pandemic': ['Health crisis', 'Lockdown measures', 'Supply chain disruption'],\n            'geopolitical_crisis': ['Political instability', 'Trade disruption', 'Security concerns']\n        }\n        return risk_factors.get(event_type, [])\n    \n    def _get_rare_event_policy_implications(self, event_type: str) -> List[str]:\n        implications = {\n            'financial_crisis': ['Financial regulation', 'Systemic risk monitoring', 'Crisis management'],\n            'oil_shock': ['Energy diversification', 'Strategic reserves', 'Alternative energy'],\n            'pandemic': ['Health preparedness', 'Economic support', 'Supply chain resilience'],\n            'geopolitical_crisis': ['Diplomatic engagement', 'Economic sanctions', 'Security measures']\n        }\n        return implications.get(event_type, [])\n\n\nif __name__ == \"__main__\":\n    # Example usage\n    simulator = EconomicScenarioSimulator()\n    \n    print(\"Generating economic scenarios...\")\n    \n    # Generate baseline scenarios\n    baseline_scenarios = simulator.generate_baseline_scenarios(\n        time_horizon=24, \n        n_scenarios=100\n    )\n    print(f\"Generated {len(baseline_scenarios)} baseline scenarios\")\n    \n    # Generate stress test scenarios\n    stress_scenarios = simulator.generate_stress_test_scenarios(\n        time_horizon=24,\n        stress_types=['severe_recession', 'hyperinflation', 'stagflation']\n    )\n    print(f\"Generated {len(stress_scenarios)} stress test scenarios\")\n    \n    # Generate rare event scenarios\n    rare_event_scenarios = simulator.generate_rare_event_scenarios(\n        time_horizon=24,\n        event_types=['financial_crisis', 'pandemic'],\n        n_scenarios_per_event=25\n    )\n    print(f\"Generated {len(rare_event_scenarios)} rare event scenarios\")\n    \n    # Combine all scenarios\n    all_scenarios = baseline_scenarios + stress_scenarios + rare_event_scenarios\n    \n    # Calculate statistics\n    scenario_stats = simulator.calculate_scenario_statistics(all_scenarios)\n    print(\"\\nScenario statistics:\")\n    for var_name, stats in scenario_stats.items():\n        print(f\"  {var_name}: mean={stats['mean']:.4f}, std={stats['std']:.4f}\")\n    \n    # Cluster scenarios\n    clustered = simulator.cluster_scenarios(all_scenarios, n_clusters=5)\n    print(f\"\\nClustered scenarios:\")\n    for cluster_id, scenarios in clustered.items():\n        print(f\"  Cluster {cluster_id}: {len(scenarios)} scenarios\")\n    \n    # Export scenarios\n    export_file = simulator.export_scenarios(\n        all_scenarios[:10],  # Export first 10 for demo\n        'sample_scenarios.json'\n    )\n    print(f\"\\nExported sample scenarios to: {export_file}\")\n    \n    print(\"\\nEconomic scenario simulation completed\")"