"""
Tests for budget optimization functionality
"""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock

from models.mmm.budget_optimizer import BudgetOptimizer
from models.mmm.econometric_mmm import EconometricMMM

class TestBudgetOptimizer:
    """Test cases for BudgetOptimizer"""
    
    def test_init(self):
        """Test initialization"""
        optimizer = BudgetOptimizer()
        
        # Should initialize without error
        assert isinstance(optimizer, BudgetOptimizer)
    
    def test_optimize_budget_allocation_basic(self, sample_marketing_data, sample_budget_allocation):
        """Test basic budget optimization"""
        # First train a model
        mmm_model = EconometricMMM()
        spend_columns = ['tv_spend', 'digital_spend', 'radio_spend']
        
        mmm_model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        # Create budget allocation for channels
        budget_allocation = {
            'tv': 50000,
            'digital': 30000,
            'radio': 20000
        }
        
        optimizer = BudgetOptimizer()
        
        results = optimizer.optimize_budget_allocation(
            mmm_model=mmm_model,
            current_budgets=budget_allocation,
            total_budget=sum(budget_allocation.values())
        )
        
        # Check results structure
        assert isinstance(results, dict)
        assert 'optimized_allocation' in results
        assert 'improvement_metrics' in results
        assert 'optimization_summary' in results
    
    def test_marginal_roi_calculation(self, sample_marketing_data):
        """Test marginal ROI calculation"""
        # Train model
        mmm_model = EconometricMMM()
        spend_columns = ['tv_spend', 'digital_spend']
        
        mmm_model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        current_budgets = {'tv': 50000, 'digital': 30000}
        budget_increments = {'tv': 5000, 'digital': 3000}
        
        optimizer = BudgetOptimizer()
        
        marginal_rois = optimizer.calculate_marginal_roi(
            mmm_model=mmm_model,
            current_budgets=current_budgets,
            budget_increments=budget_increments
        )
        
        # Should return dict with ROI for each channel
        assert isinstance(marginal_rois, dict)
        assert 'tv' in marginal_rois
        assert 'digital' in marginal_rois
        
        # ROI values should be numeric
        for channel, roi in marginal_rois.items():
            assert isinstance(roi, (int, float, np.number))
    
    def test_budget_reallocation_suggestions(self, sample_marketing_data):
        """Test budget reallocation suggestions"""
        optimizer = BudgetOptimizer()
        
        current_budgets = {'tv': 50000, 'digital': 30000, 'radio': 20000}
        marginal_rois = {'tv': 1.2, 'digital': 1.8, 'radio': 0.9}
        reallocation_amount = 10000
        
        suggestions = optimizer.suggest_budget_reallocation(
            current_budgets=current_budgets,
            marginal_rois=marginal_rois,
            reallocation_amount=reallocation_amount
        )
        
        # Should return dict with suggested allocations
        assert isinstance(suggestions, dict)
        
        # Should have same channels
        assert set(suggestions.keys()) == set(current_budgets.keys())
        
        # Total should remain approximately the same
        original_total = sum(current_budgets.values())
        suggested_total = sum(suggestions.values())
        assert abs(suggested_total - original_total) < 1  # Allow for rounding
    
    def test_scenario_analysis(self, sample_marketing_data):
        """Test scenario analysis functionality"""
        # Train model
        mmm_model = EconometricMMM()
        spend_columns = ['tv_spend', 'digital_spend']
        
        mmm_model.fit(
            data=sample_marketing_data,
            target_column='revenue', 
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        baseline_budgets = {'tv': 50000, 'digital': 30000}
        
        scenarios = {
            'increase_tv': {'tv': 60000, 'digital': 30000},
            'increase_digital': {'tv': 50000, 'digital': 40000},
            'balanced_increase': {'tv': 55000, 'digital': 35000}
        }
        
        optimizer = BudgetOptimizer()
        
        scenario_results = optimizer.run_scenario_analysis(
            mmm_model=mmm_model,
            baseline_budgets=baseline_budgets,
            scenarios=scenarios,
            periods=12
        )
        
        # Should return results for each scenario
        assert isinstance(scenario_results, dict)
        assert 'baseline' in scenario_results
        
        for scenario_name in scenarios.keys():
            assert scenario_name in scenario_results
        
        # Each scenario should have metrics
        for scenario, results in scenario_results.items():
            assert isinstance(results, dict)
    
    def test_optimization_constraints(self, sample_marketing_data):
        """Test optimization with constraints"""
        # Train model
        mmm_model = EconometricMMM()
        spend_columns = ['tv_spend', 'digital_spend', 'radio_spend']
        
        mmm_model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        current_budgets = {'tv': 50000, 'digital': 30000, 'radio': 20000}
        
        # Define constraints
        constraints = {
            'min_budget_pct': {'tv': 0.3, 'digital': 0.2},  # Minimum budget percentages
            'max_budget_pct': {'tv': 0.6, 'digital': 0.5},  # Maximum budget percentages
        }
        
        optimizer = BudgetOptimizer()
        
        results = optimizer.optimize_budget_allocation(
            mmm_model=mmm_model,
            current_budgets=current_budgets,
            total_budget=sum(current_budgets.values()),
            constraints=constraints
        )
        
        # Should return results respecting constraints
        assert isinstance(results, dict)
        
        if 'optimized_allocation' in results:
            optimized = results['optimized_allocation']
            total_budget = sum(optimized.values())
            
            # Check minimum constraints
            if 'min_budget_pct' in constraints:
                for channel, min_pct in constraints['min_budget_pct'].items():
                    if channel in optimized:
                        actual_pct = optimized[channel] / total_budget
                        assert actual_pct >= min_pct - 0.01  # Small tolerance for rounding
            
            # Check maximum constraints
            if 'max_budget_pct' in constraints:
                for channel, max_pct in constraints['max_budget_pct'].items():
                    if channel in optimized:
                        actual_pct = optimized[channel] / total_budget
                        assert actual_pct <= max_pct + 0.01  # Small tolerance for rounding
    
    def test_optimization_summary(self):
        """Test optimization summary generation"""
        optimizer = BudgetOptimizer()
        
        # Should be able to get summary (even if empty initially)
        summary = optimizer.get_optimization_summary()
        
        assert isinstance(summary, dict)

class TestOptimizationValidation:
    """Test optimization validation and quality checks"""
    
    def test_budget_conservation(self, sample_marketing_data):
        """Test that total budget is conserved in optimization"""
        # Train model
        mmm_model = EconometricMMM()
        spend_columns = ['tv_spend', 'digital_spend']
        
        mmm_model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        current_budgets = {'tv': 50000, 'digital': 30000}
        total_budget = sum(current_budgets.values())
        
        optimizer = BudgetOptimizer()
        
        results = optimizer.optimize_budget_allocation(
            mmm_model=mmm_model,
            current_budgets=current_budgets,
            total_budget=total_budget
        )
        
        if 'optimized_allocation' in results:
            optimized_total = sum(results['optimized_allocation'].values())
            # Total should be conserved (within small tolerance for rounding)
            assert abs(optimized_total - total_budget) < 1
    
    def test_positive_allocations(self, sample_marketing_data):
        """Test that optimized allocations are positive"""
        # Train model
        mmm_model = EconometricMMM()
        spend_columns = ['tv_spend', 'digital_spend', 'radio_spend']
        
        mmm_model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        current_budgets = {'tv': 50000, 'digital': 30000, 'radio': 20000}
        
        optimizer = BudgetOptimizer()
        
        results = optimizer.optimize_budget_allocation(
            mmm_model=mmm_model,
            current_budgets=current_budgets,
            total_budget=sum(current_budgets.values())
        )
        
        if 'optimized_allocation' in results:
            for channel, allocation in results['optimized_allocation'].items():
                assert allocation >= 0, f"Negative allocation for {channel}: {allocation}"
    
    def test_roi_improvement_direction(self, sample_marketing_data):
        """Test that optimization improves or maintains ROI"""
        # Train model
        mmm_model = EconometricMMM()
        spend_columns = ['tv_spend', 'digital_spend']
        
        mmm_model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        current_budgets = {'tv': 50000, 'digital': 30000}
        
        optimizer = BudgetOptimizer()
        
        results = optimizer.optimize_budget_allocation(
            mmm_model=mmm_model,
            current_budgets=current_budgets,
            total_budget=sum(current_budgets.values()),
            objective='roi'
        )
        
        # Should not make ROI worse (though improvement may be minimal)
        if 'improvement_metrics' in results:
            improvement = results['improvement_metrics']
            if 'roi_improvement' in improvement:
                # Allow for small negative improvements due to optimization constraints
                assert improvement['roi_improvement'] >= -0.05

class TestErrorHandling:
    """Test error handling and edge cases"""
    
    def test_empty_budget_allocation(self):
        """Test handling of empty budget allocation"""
        optimizer = BudgetOptimizer()
        
        # This should handle gracefully or raise appropriate error
        try:
            results = optimizer.suggest_budget_reallocation(
                current_budgets={},
                marginal_rois={},
                reallocation_amount=1000
            )
            # If it succeeds, should return empty dict
            assert isinstance(results, dict)
        except (ValueError, KeyError):
            # Or it should raise an appropriate error
            pass
    
    def test_zero_total_budget(self, sample_marketing_data):
        """Test handling of zero total budget"""
        # Train minimal model
        mmm_model = EconometricMMM()
        spend_columns = ['tv_spend']
        
        mmm_model.fit(
            data=sample_marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        optimizer = BudgetOptimizer()
        
        # This should handle gracefully or raise appropriate error
        try:
            results = optimizer.optimize_budget_allocation(
                mmm_model=mmm_model,
                current_budgets={'tv': 0},
                total_budget=0
            )
            assert isinstance(results, dict)
        except (ValueError, ZeroDivisionError):
            # Should raise appropriate error for zero budget
            pass
    
    def test_negative_marginal_rois(self):
        """Test handling of negative marginal ROIs"""
        optimizer = BudgetOptimizer()
        
        current_budgets = {'tv': 50000, 'digital': 30000}
        marginal_rois = {'tv': -0.5, 'digital': 1.2}  # Negative ROI for TV
        reallocation_amount = 10000
        
        # Should handle negative ROIs gracefully
        results = optimizer.suggest_budget_reallocation(
            current_budgets=current_budgets,
            marginal_rois=marginal_rois,
            reallocation_amount=reallocation_amount
        )
        
        assert isinstance(results, dict)
        # Should reallocate away from negative ROI channels
        assert results['tv'] <= current_budgets['tv']  # Should decrease TV
        assert results['digital'] >= current_budgets['digital']  # Should increase digital