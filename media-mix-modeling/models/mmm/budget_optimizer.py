#!/usr/bin/env python3
"""
Budget Optimization for Media Mix Models
Multi-objective optimization with constraints
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple, Any, Callable
from scipy.optimize import minimize, differential_evolution
import warnings
warnings.filterwarnings('ignore')

class BudgetOptimizer:
    """
    Advanced budget optimization for media mix models
    
    Features:
    - Multi-objective optimization (ROI, reach, frequency)
    - Constraint handling for budget limits
    - Scenario analysis and what-if modeling
    - Real-time bid adjustment algorithms
    - Cross-channel synergy optimization
    """
    
    def __init__(self,
                 optimization_method: str = 'scipy',
                 max_budget_change: float = 0.3,
                 min_budget_change: float = -0.2):
        """
        Initialize budget optimizer
        
        Args:
            optimization_method: Optimization algorithm ('scipy', 'differential_evolution')
            max_budget_change: Maximum allowed budget increase per channel
            min_budget_change: Maximum allowed budget decrease per channel
        """
        self.optimization_method = optimization_method
        self.max_budget_change = max_budget_change
        self.min_budget_change = min_budget_change
        
        # Optimization results storage
        self.optimization_results = {}
        self.scenario_results = {}
        
    def optimize_budget_allocation(self,
                                 mmm_model,
                                 current_budgets: Dict[str, float],
                                 total_budget: float,
                                 objective: str = 'roi',
                                 constraints: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """
        Optimize budget allocation across channels
        
        Args:
            mmm_model: Fitted MMM model with predict method
            current_budgets: Current budget allocation per channel
            total_budget: Total available budget
            objective: Optimization objective ('roi', 'revenue', 'efficiency')
            constraints: Additional constraints dictionary
            
        Returns:
            Dictionary with optimization results
        """
        channels = list(current_budgets.keys())
        current_allocation = np.array([current_budgets[ch] for ch in channels])
        
        print(f"[OPTIMIZATION] Optimizing {len(channels)} channels for {objective}")
        print(f"[BUDGET] Total budget: ${total_budget:,.0f}")
        
        # Define objective function
        def objective_function(allocation):
            """Calculate objective value for given allocation"""
            try:
                # Create scenario data for prediction
                scenario_data = self._create_scenario_data(dict(zip(channels, allocation)))
                
                # Get predictions from MMM model
                predictions = mmm_model.predict(scenario_data)
                total_revenue = predictions.sum()
                total_spend = allocation.sum()
                
                if objective == 'roi':
                    if total_spend > 0:
                        return -(total_revenue / total_spend)  # Negative for minimization
                    else:
                        return -0
                elif objective == 'revenue':
                    return -total_revenue  # Negative for minimization
                elif objective == 'efficiency':
                    if total_spend > 0:
                        return -(total_revenue - total_spend) / total_spend  # Net ROI
                    else:
                        return -0
                else:
                    return -total_revenue
                    
            except Exception as e:
                print(f"[ERROR] Objective function error: {e}")
                return 1e10  # Large penalty for errors
        
        # Set up constraints
        optimization_constraints = []
        
        # Budget constraint: sum equals total budget
        optimization_constraints.append({
            'type': 'eq',
            'fun': lambda x: x.sum() - total_budget
        })
        
        # Individual channel constraints
        bounds = []
        for i, channel in enumerate(channels):
            current_budget = current_allocation[i]
            
            # Default bounds based on change limits
            min_budget = max(0, current_budget * (1 + self.min_budget_change))
            max_budget = current_budget * (1 + self.max_budget_change)
            
            # Apply custom constraints if provided
            if constraints and channel in constraints:
                if 'min_budget' in constraints[channel]:
                    min_budget = max(min_budget, constraints[channel]['min_budget'])
                if 'max_budget' in constraints[channel]:
                    max_budget = min(max_budget, constraints[channel]['max_budget'])
            
            bounds.append((min_budget, max_budget))
        
        # Add custom constraints
        if constraints:
            if 'required_channels' in constraints:
                # Ensure required channels have minimum spend
                for req_channel in constraints['required_channels']:
                    if req_channel in channels:
                        idx = channels.index(req_channel)
                        min_required = constraints['required_channels'][req_channel]
                        optimization_constraints.append({
                            'type': 'ineq',
                            'fun': lambda x, i=idx, min_val=min_required: x[i] - min_val
                        })
        
        # Run optimization
        try:
            if self.optimization_method == 'scipy':
                result = minimize(
                    objective_function,
                    current_allocation,
                    method='SLSQP',
                    bounds=bounds,
                    constraints=optimization_constraints,
                    options={'maxiter': 1000, 'ftol': 1e-6}
                )
                optimal_allocation = result.x
                success = result.success
                
            elif self.optimization_method == 'differential_evolution':
                result = differential_evolution(
                    objective_function,
                    bounds,
                    maxiter=300,
                    seed=42
                )
                optimal_allocation = result.x
                success = result.success
                
                # Apply budget constraint manually for differential evolution
                optimal_allocation = optimal_allocation * (total_budget / optimal_allocation.sum())
            
            else:
                raise ValueError(f"Unknown optimization method: {self.optimization_method}")
            
            if success:
                print(f"[SUCCESS] Optimization converged")
            else:
                print(f"[WARNING] Optimization may not have converged")
            
        except Exception as e:
            print(f"[ERROR] Optimization failed: {e}")
            optimal_allocation = current_allocation
            success = False
        
        # Calculate optimization results
        optimal_budgets = dict(zip(channels, optimal_allocation))
        
        # Calculate performance metrics
        current_scenario = self._create_scenario_data(current_budgets)
        optimal_scenario = self._create_scenario_data(optimal_budgets)
        
        try:
            current_predictions = mmm_model.predict(current_scenario)
            optimal_predictions = mmm_model.predict(optimal_scenario)
            
            current_revenue = current_predictions.sum()
            optimal_revenue = optimal_predictions.sum()
            
            current_roi = current_revenue / sum(current_budgets.values()) if sum(current_budgets.values()) > 0 else 0
            optimal_roi = optimal_revenue / total_budget if total_budget > 0 else 0
            
        except Exception as e:
            print(f"[ERROR] Performance calculation failed: {e}")
            current_revenue = optimal_revenue = 0
            current_roi = optimal_roi = 0
        
        # Calculate budget changes
        budget_changes = {}
        for channel in channels:
            current = current_budgets[channel]
            optimal = optimal_budgets[channel]
            change_amount = optimal - current
            change_pct = change_amount / current if current > 0 else 0
            
            budget_changes[channel] = {
                'current_budget': current,
                'optimal_budget': optimal,
                'change_amount': change_amount,
                'change_percentage': change_pct
            }
        
        self.optimization_results = {
            'success': success,
            'optimization_method': self.optimization_method,
            'objective': objective,
            'current_budgets': current_budgets,
            'optimal_budgets': optimal_budgets,
            'budget_changes': budget_changes,
            'performance_improvement': {
                'current_revenue': current_revenue,
                'optimal_revenue': optimal_revenue,
                'revenue_lift': optimal_revenue - current_revenue,
                'revenue_lift_pct': (optimal_revenue - current_revenue) / current_revenue if current_revenue > 0 else 0,
                'current_roi': current_roi,
                'optimal_roi': optimal_roi,
                'roi_improvement': optimal_roi - current_roi,
                'roi_improvement_pct': (optimal_roi - current_roi) / current_roi if current_roi > 0 else 0
            },
            'constraints_applied': {
                'total_budget': total_budget,
                'max_budget_change': self.max_budget_change,
                'min_budget_change': self.min_budget_change,
                'custom_constraints': constraints is not None
            }
        }
        
        return self.optimization_results
    
    def _create_scenario_data(self, budget_allocation: Dict[str, float]) -> pd.DataFrame:
        """Create scenario data for MMM prediction"""
        # Create a single period scenario
        scenario_data = pd.DataFrame({
            'date': [pd.Timestamp.now()],
        })
        
        # Add budget allocations as spend
        for channel, budget in budget_allocation.items():
            if not channel.endswith('_spend'):
                spend_column = f"{channel}_spend"
            else:
                spend_column = channel
            scenario_data[spend_column] = budget
        
        return scenario_data
    
    def run_scenario_analysis(self,
                            mmm_model,
                            baseline_budgets: Dict[str, float],
                            scenarios: Dict[str, Dict[str, float]],
                            periods: int = 12) -> Dict[str, Any]:
        """
        Run multiple budget scenarios and compare results
        
        Args:
            mmm_model: Fitted MMM model
            baseline_budgets: Baseline budget allocation
            scenarios: Dictionary of scenario name -> budget allocation
            periods: Number of periods to simulate
            
        Returns:
            Dictionary with scenario analysis results
        """
        print(f"[SCENARIOS] Analyzing {len(scenarios)} budget scenarios over {periods} periods")
        
        scenario_results = {}
        
        # Add baseline scenario
        all_scenarios = {'baseline': baseline_budgets}
        all_scenarios.update(scenarios)
        
        for scenario_name, budget_allocation in all_scenarios.items():
            try:
                # Create multi-period scenario data
                scenario_data = pd.DataFrame({
                    'date': pd.date_range(start='2024-01-01', periods=periods, freq='W')
                })
                
                # Add budget allocations
                for channel, budget in budget_allocation.items():
                    if not channel.endswith('_spend'):
                        spend_column = f"{channel}_spend"
                    else:
                        spend_column = channel
                    scenario_data[spend_column] = budget
                
                # Get predictions
                predictions = mmm_model.predict(scenario_data)
                
                # Calculate metrics
                total_revenue = predictions.sum()
                total_spend = sum(budget_allocation.values()) * periods
                roi = total_revenue / total_spend if total_spend > 0 else 0
                
                scenario_results[scenario_name] = {
                    'budget_allocation': budget_allocation,
                    'total_spend': total_spend,
                    'total_revenue': total_revenue,
                    'roi': roi,
                    'average_weekly_revenue': predictions.mean(),
                    'revenue_variance': predictions.var(),
                    'periods_simulated': periods
                }
                
                print(f"   [SCENARIO] {scenario_name}: ROI = {roi:.2f}x, Revenue = ${total_revenue:,.0f}")
                
            except Exception as e:
                print(f"   [ERROR] Scenario {scenario_name} failed: {e}")
                scenario_results[scenario_name] = {'error': str(e)}
        
        # Calculate scenario comparisons
        if 'baseline' in scenario_results and not 'error' in scenario_results['baseline']:
            baseline_roi = scenario_results['baseline']['roi']
            baseline_revenue = scenario_results['baseline']['total_revenue']
            
            for scenario_name in scenario_results:
                if scenario_name != 'baseline' and 'error' not in scenario_results[scenario_name]:
                    scenario_roi = scenario_results[scenario_name]['roi']
                    scenario_revenue = scenario_results[scenario_name]['total_revenue']
                    
                    scenario_results[scenario_name]['vs_baseline'] = {
                        'roi_lift': scenario_roi - baseline_roi,
                        'roi_lift_pct': (scenario_roi - baseline_roi) / baseline_roi if baseline_roi > 0 else 0,
                        'revenue_lift': scenario_revenue - baseline_revenue,
                        'revenue_lift_pct': (scenario_revenue - baseline_revenue) / baseline_revenue if baseline_revenue > 0 else 0
                    }
        
        self.scenario_results = {
            'scenarios': scenario_results,
            'best_scenario': max(
                [name for name in scenario_results.keys() if 'error' not in scenario_results[name]], 
                key=lambda x: scenario_results[x].get('roi', 0)
            ) if scenario_results else None,
            'analysis_summary': {
                'total_scenarios': len(scenarios),
                'periods_analyzed': periods,
                'baseline_included': 'baseline' in scenario_results
            }
        }
        
        return self.scenario_results
    
    def calculate_marginal_roi(self,
                             mmm_model,
                             current_budgets: Dict[str, float],
                             budget_increments: Dict[str, float]) -> Dict[str, float]:
        """
        Calculate marginal ROI for budget increments
        
        Args:
            mmm_model: Fitted MMM model
            current_budgets: Current budget allocation
            budget_increments: Budget increment to test per channel
            
        Returns:
            Dictionary with marginal ROI per channel
        """
        marginal_rois = {}
        
        # Get baseline performance
        baseline_scenario = self._create_scenario_data(current_budgets)
        baseline_predictions = mmm_model.predict(baseline_scenario)
        baseline_revenue = baseline_predictions.sum()
        
        # Test marginal impact of each channel
        for channel, increment in budget_increments.items():
            if channel in current_budgets:
                # Create incremented budget
                incremented_budgets = current_budgets.copy()
                incremented_budgets[channel] += increment
                
                # Get incremented performance
                incremented_scenario = self._create_scenario_data(incremented_budgets)
                incremented_predictions = mmm_model.predict(incremented_scenario)
                incremented_revenue = incremented_predictions.sum()
                
                # Calculate marginal ROI
                marginal_revenue = incremented_revenue - baseline_revenue
                marginal_roi = marginal_revenue / increment if increment > 0 else 0
                
                marginal_rois[channel] = marginal_roi
        
        return marginal_rois
    
    def suggest_budget_reallocation(self,
                                  current_budgets: Dict[str, float],
                                  marginal_rois: Dict[str, float],
                                  reallocation_amount: float) -> Dict[str, float]:
        """
        Suggest budget reallocation based on marginal ROIs
        
        Args:
            current_budgets: Current budget allocation
            marginal_rois: Marginal ROI per channel
            reallocation_amount: Total amount to reallocate
            
        Returns:
            Dictionary with suggested new budget allocation
        """
        # Sort channels by marginal ROI
        sorted_channels = sorted(marginal_rois.items(), key=lambda x: x[1], reverse=True)
        
        # Identify source and target channels
        n_channels = len(sorted_channels)
        target_channels = sorted_channels[:n_channels//2]  # Top performing channels
        source_channels = sorted_channels[n_channels//2:]  # Lower performing channels
        
        # Calculate reallocation
        new_budgets = current_budgets.copy()
        
        # Take from lower performing channels
        total_taken = 0
        for channel, roi in source_channels:
            if total_taken < reallocation_amount:
                take_amount = min(
                    reallocation_amount - total_taken,
                    current_budgets[channel] * 0.2  # Max 20% from any channel
                )
                new_budgets[channel] -= take_amount
                total_taken += take_amount
        
        # Give to higher performing channels proportionally
        if total_taken > 0:
            target_total_roi = sum([roi for _, roi in target_channels])
            
            for channel, roi in target_channels:
                if target_total_roi > 0:
                    allocation_proportion = roi / target_total_roi
                    give_amount = total_taken * allocation_proportion
                    new_budgets[channel] += give_amount
        
        return new_budgets
    
    def get_optimization_summary(self) -> Dict[str, Any]:
        """Get summary of optimization results"""
        if not self.optimization_results:
            return {"error": "No optimization results available"}
        
        summary = {
            'optimization_success': self.optimization_results.get('success', False),
            'objective_optimized': self.optimization_results.get('objective', 'unknown'),
            'total_channels': len(self.optimization_results.get('current_budgets', {})),
            'performance_improvement': self.optimization_results.get('performance_improvement', {}),
            'top_budget_changes': [],
            'constraints_applied': self.optimization_results.get('constraints_applied', {})
        }
        
        # Get top budget changes
        budget_changes = self.optimization_results.get('budget_changes', {})
        if budget_changes:
            sorted_changes = sorted(
                budget_changes.items(), 
                key=lambda x: abs(x[1]['change_percentage']), 
                reverse=True
            )
            
            summary['top_budget_changes'] = [
                {
                    'channel': channel,
                    'change_percentage': data['change_percentage'],
                    'change_amount': data['change_amount'],
                    'direction': 'increase' if data['change_amount'] > 0 else 'decrease'
                }
                for channel, data in sorted_changes[:5]
            ]
        
        return summary