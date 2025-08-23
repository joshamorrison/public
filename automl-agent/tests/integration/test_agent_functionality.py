#!/usr/bin/env python3
"""
Integration tests for AutoML Agent functionality

Tests the core agents with real data to ensure they work end-to-end.
"""

import sys
import os
import pytest
import pandas as pd
import numpy as np
from sklearn.datasets import make_classification, make_regression

# Add src to path for testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from agents.base_agent import TaskContext
from agents.eda_agent import EDAAgent
from agents.data_hygiene_agent import DataHygieneAgent
from agents.feature_engineering_agent import FeatureEngineeringAgent
from agents.classification_agent import ClassificationAgent
from agents.regression_agent import RegressionAgent


class TestAgentFunctionality:
    """Test suite for agent functionality"""
    
    @pytest.fixture
    def classification_data(self):
        """Create sample classification dataset"""
        X, y = make_classification(n_samples=300, n_features=8, n_informative=6, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(8)])
        df['target'] = y
        
        # Add some data quality issues
        df.loc[np.random.choice(df.index, 20), 'feature_0'] = np.nan
        
        return df
    
    @pytest.fixture
    def regression_data(self):
        """Create sample regression dataset"""
        X, y = make_regression(n_samples=250, n_features=6, noise=0.1, random_state=42)
        df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(6)])
        df['target'] = y
        return df
    
    def test_eda_agent(self, classification_data):
        """Test EDA Agent functionality"""
        context = TaskContext(
            task_id='test_eda',
            user_input='Analyze this classification dataset',
            dataset_info={'shape': classification_data.shape, 'target_column': 'target'}
        )
        
        agent = EDAAgent()
        assert agent.can_handle_task(context)
        
        result = agent.execute_task(context)
        assert result.success
        assert "analyzed" in result.message.lower()
        assert result.execution_time >= 0
    
    def test_data_hygiene_agent(self, classification_data):
        """Test Data Hygiene Agent functionality"""
        context = TaskContext(
            task_id='test_hygiene',
            user_input='Clean this dataset',
            dataset_info={'shape': classification_data.shape}
        )
        
        agent = DataHygieneAgent()
        assert agent.can_handle_task(context)
        
        result = agent.execute_task(context)
        assert result.success
        assert "cleaning" in result.message.lower() or "quality" in result.message.lower()
    
    def test_feature_engineering_agent(self, classification_data):
        """Test Feature Engineering Agent functionality"""
        context = TaskContext(
            task_id='test_features',
            user_input='Engineer features for this dataset',
            dataset_info={'shape': classification_data.shape, 'target_column': 'target'}
        )
        
        agent = FeatureEngineeringAgent()
        assert agent.can_handle_task(context)
        
        result = agent.execute_task(context)
        assert result.success
        assert "feature" in result.message.lower()
    
    def test_classification_agent(self, classification_data):
        """Test Classification Agent functionality"""
        context = TaskContext(
            task_id='test_classification',
            user_input='Build a classification model',
            dataset_info={
                'shape': classification_data.shape,
                'target_column': 'target',
                'problem_type': 'binary_classification'
            }
        )
        
        agent = ClassificationAgent()
        assert agent.can_handle_task(context)
        
        result = agent.execute_task(context)
        assert result.success
        assert "classification" in result.message.lower() or "accuracy" in result.message.lower()
    
    def test_regression_agent(self, regression_data):
        """Test Regression Agent functionality"""
        context = TaskContext(
            task_id='test_regression',
            user_input='Build a regression model',
            dataset_info={
                'shape': regression_data.shape,
                'target_column': 'target',
                'problem_type': 'regression'
            }
        )
        
        agent = RegressionAgent()
        assert agent.can_handle_task(context)
        
        result = agent.execute_task(context)
        assert result.success
        assert "regression" in result.message.lower() or "score" in result.message.lower()


if __name__ == "__main__":
    # Run basic functionality test
    print("Running AutoML Agent Integration Tests...")
    
    # Create test data
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    df = pd.DataFrame(X, columns=[f'feature_{i}' for i in range(5)])
    df['target'] = y
    
    # Test core agents
    agents_to_test = [
        (EDAAgent(), "EDA Agent"),
        (DataHygieneAgent(), "Data Hygiene Agent"),
        (FeatureEngineeringAgent(), "Feature Engineering Agent"),
        (ClassificationAgent(), "Classification Agent")
    ]
    
    results = []
    for agent, name in agents_to_test:
        try:
            context = TaskContext(
                task_id=f'test_{name.lower().replace(" ", "_")}',
                user_input=f'Test {name}',
                dataset_info={'shape': df.shape, 'target_column': 'target'}
            )
            
            if agent.can_handle_task(context):
                result = agent.execute_task(context)
                status = "PASS" if result.success else "FAIL"
                print(f"  {name}: {status}")
                results.append(result.success)
            else:
                print(f"  {name}: SKIP (cannot handle task)")
                results.append(False)
                
        except Exception as e:
            print(f"  {name}: ERROR ({str(e)[:50]}...)")
            results.append(False)
    
    # Test regression separately
    try:
        X_reg, y_reg = make_regression(n_samples=200, n_features=4, random_state=42)
        df_reg = pd.DataFrame(X_reg, columns=[f'feature_{i}' for i in range(4)])
        df_reg['target'] = y_reg
        
        context_reg = TaskContext(
            task_id='test_regression',
            user_input='Test Regression Agent',
            dataset_info={'shape': df_reg.shape, 'target_column': 'target'}
        )
        
        regression_agent = RegressionAgent()
        if regression_agent.can_handle_task(context_reg):
            result = regression_agent.execute_task(context_reg)
            status = "PASS" if result.success else "FAIL"
            print(f"  Regression Agent: {status}")
            results.append(result.success)
        else:
            print(f"  Regression Agent: SKIP")
            results.append(False)
            
    except Exception as e:
        print(f"  Regression Agent: ERROR ({str(e)[:50]}...)")
        results.append(False)
    
    print(f"\nTest Summary: {sum(results)}/{len(results)} agents working")
    
    if all(results):
        print("All agents are functional!")
    else:
        print("Some agents need attention - check logs above")