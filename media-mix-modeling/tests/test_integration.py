"""
Integration tests for the complete MMM pipeline
"""

import pytest
import pandas as pd
import numpy as np
import tempfile
import json
from pathlib import Path
from unittest.mock import patch, MagicMock

from data.media_data_client import MediaDataClient
from models.mmm.econometric_mmm import EconometricMMM
from models.mmm.budget_optimizer import BudgetOptimizer

@pytest.mark.integration
class TestEndToEndPipeline:
    """Test complete end-to-end MMM pipeline"""
    
    def test_full_mmm_pipeline(self):
        """Test complete pipeline from data to optimization"""
        print("\n[INTEGRATION] Testing full MMM pipeline...")
        
        # Step 1: Data extraction
        data_client = MediaDataClient()
        marketing_data, data_info, source_type = data_client.get_best_available_data()
        
        assert isinstance(marketing_data, pd.DataFrame)
        assert len(marketing_data) > 0
        assert source_type == 'SYNTHETIC'
        
        print(f"[DATA] Extracted {len(marketing_data)} records from {source_type}")
        
        # Step 2: Model training
        mmm_model = EconometricMMM(
            adstock_rate=0.5,
            saturation_param=0.6,
            regularization_alpha=0.1
        )
        
        spend_columns = [col for col in marketing_data.columns if col.endswith('_spend')]
        assert len(spend_columns) >= 2, "Need at least 2 spend columns for meaningful test"
        
        # Use subset of channels for faster testing
        test_spend_columns = spend_columns[:3]
        
        results = mmm_model.fit(
            data=marketing_data,
            target_column='revenue',
            spend_columns=test_spend_columns,
            include_synergies=True
        )
        
        assert isinstance(results, dict)
        assert 'performance' in results
        assert 'coefficients' in results
        
        print(f"[MODEL] Trained MMM - R²: {results['performance']['r2_score']:.3f}")
        
        # Step 3: Budget optimization
        optimizer = BudgetOptimizer()
        
        # Create current budget allocation
        current_budgets = {
            channel.replace('_spend', ''): marketing_data[channel].mean()
            for channel in test_spend_columns
        }
        
        optimization_results = optimizer.optimize_budget_allocation(
            mmm_model=mmm_model,
            current_budgets=current_budgets,
            total_budget=sum(current_budgets.values()),
            objective='roi'
        )
        
        assert isinstance(optimization_results, dict)
        assert 'optimized_allocation' in optimization_results
        
        print(f"[OPTIMIZATION] Budget optimization completed")
        
        # Step 4: Validation
        # Ensure total budget is conserved
        if 'optimized_allocation' in optimization_results:
            original_total = sum(current_budgets.values())
            optimized_total = sum(optimization_results['optimized_allocation'].values())
            assert abs(optimized_total - original_total) < 1
        
        # Ensure predictions work
        predictions = mmm_model.predict(marketing_data[test_spend_columns])
        assert len(predictions) == len(marketing_data)
        
        print("[INTEGRATION] Full pipeline test completed successfully")
        
        return {
            'data_extraction': {'source': source_type, 'records': len(marketing_data)},
            'model_training': results['performance'],
            'optimization': optimization_results,
            'pipeline_status': 'success'
        }
    
    def test_multi_model_comparison(self):
        """Test comparison of multiple model configurations"""
        print("\n[INTEGRATION] Testing multi-model comparison...")
        
        # Get data
        data_client = MediaDataClient()
        marketing_data, _, _ = data_client.get_best_available_data()
        
        spend_columns = [col for col in marketing_data.columns if col.endswith('_spend')][:2]
        
        # Test different model configurations
        model_configs = [
            {'adstock_rate': 0.3, 'saturation_param': 0.5, 'name': 'low_carryover'},
            {'adstock_rate': 0.7, 'saturation_param': 0.8, 'name': 'high_carryover'},
            {'adstock_rate': 0.5, 'saturation_param': 0.6, 'name': 'baseline'}
        ]
        
        model_results = {}
        
        for config in model_configs:
            model = EconometricMMM(
                adstock_rate=config['adstock_rate'],
                saturation_param=config['saturation_param']
            )
            
            results = model.fit(
                data=marketing_data,
                target_column='revenue',
                spend_columns=spend_columns,
                include_synergies=False
            )
            
            model_results[config['name']] = {
                'config': config,
                'performance': results['performance'],
                'coefficients': results['coefficients']
            }
        
        # All models should complete
        assert len(model_results) == len(model_configs)
        
        # Compare performance
        for name, result in model_results.items():
            assert 'performance' in result
            assert 'r2_score' in result['performance']
            print(f"[MODEL] {name}: R² = {result['performance']['r2_score']:.3f}")
        
        print("[INTEGRATION] Multi-model comparison completed")
        return model_results
    
    def test_cross_validation_simulation(self):
        """Test cross-validation simulation"""
        print("\n[INTEGRATION] Testing cross-validation simulation...")
        
        # Get data
        data_client = MediaDataClient()
        marketing_data, _, _ = data_client.get_best_available_data()
        
        spend_columns = [col for col in marketing_data.columns if col.endswith('_spend')][:2]
        
        # Simulate 3-fold cross-validation
        n_folds = 3
        fold_size = len(marketing_data) // n_folds
        
        cv_results = []
        
        for fold in range(n_folds):
            # Create train/test split
            start_idx = fold * fold_size
            end_idx = start_idx + fold_size
            
            test_data = marketing_data.iloc[start_idx:end_idx]
            train_data = pd.concat([
                marketing_data.iloc[:start_idx],
                marketing_data.iloc[end_idx:]
            ])
            
            if len(train_data) < 5:  # Need minimum data for training
                continue
            
            # Train model
            model = EconometricMMM()
            
            try:
                results = model.fit(
                    data=train_data,
                    target_column='revenue',
                    spend_columns=spend_columns,
                    include_synergies=False
                )
                
                # Test predictions
                predictions = model.predict(test_data[spend_columns])
                actual = test_data['revenue'].values
                
                # Calculate test metrics
                mse = np.mean((predictions - actual) ** 2)
                mae = np.mean(np.abs(predictions - actual))
                
                cv_results.append({
                    'fold': fold,
                    'train_size': len(train_data),
                    'test_size': len(test_data),
                    'train_r2': results['performance']['r2_score'],
                    'test_mse': mse,
                    'test_mae': mae
                })
                
            except Exception as e:
                print(f"[CV] Fold {fold} failed: {e}")
                continue
        
        # Should have some successful folds
        assert len(cv_results) > 0
        
        # Calculate average performance
        avg_train_r2 = np.mean([r['train_r2'] for r in cv_results])
        avg_test_mse = np.mean([r['test_mse'] for r in cv_results])
        
        print(f"[CV] Average train R²: {avg_train_r2:.3f}")
        print(f"[CV] Average test MSE: {avg_test_mse:.0f}")
        print("[INTEGRATION] Cross-validation simulation completed")
        
        return cv_results

@pytest.mark.integration
class TestDataIntegration:
    """Test data integration across different sources"""
    
    def test_data_consistency_across_calls(self):
        """Test that data client provides consistent data"""
        data_client = MediaDataClient()
        
        # Get data multiple times
        results = []
        for i in range(3):
            data, info, source = data_client.get_best_available_data()
            results.append({
                'call': i,
                'shape': data.shape,
                'columns': list(data.columns),
                'source': source,
                'total_spend': sum(data[col].sum() for col in data.columns if col.endswith('_spend'))
            })
        
        # All calls should return same structure
        base_result = results[0]
        for result in results[1:]:
            assert result['shape'] == base_result['shape']
            assert result['columns'] == base_result['columns']
            assert result['source'] == base_result['source']
    
    @patch.dict('os.environ', {'KAGGLE_USERNAME': 'test', 'KAGGLE_KEY': 'test'})
    def test_fallback_mechanism(self):
        """Test fallback from real sources to synthetic"""
        data_client = MediaDataClient()
        
        # Should still work even if credentials are fake
        data, info, source = data_client.get_best_available_data()
        
        # Should fall back to synthetic
        assert source == 'SYNTHETIC'
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0

@pytest.mark.integration
class TestReportingIntegration:
    """Test reporting and output integration"""
    
    def test_end_to_end_reporting(self, temp_directory):
        """Test complete pipeline with reporting"""
        print("\n[INTEGRATION] Testing end-to-end reporting...")
        
        # Run pipeline
        data_client = MediaDataClient()
        marketing_data, data_info, source_type = data_client.get_best_available_data()
        
        # Train model
        mmm_model = EconometricMMM()
        spend_columns = [col for col in marketing_data.columns if col.endswith('_spend')][:2]
        
        results = mmm_model.fit(
            data=marketing_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=False
        )
        
        # Run optimization
        optimizer = BudgetOptimizer()
        current_budgets = {
            channel.replace('_spend', ''): marketing_data[channel].mean()
            for channel in spend_columns
        }
        
        optimization_results = optimizer.optimize_budget_allocation(
            mmm_model=mmm_model,
            current_budgets=current_budgets,
            total_budget=sum(current_budgets.values())
        )
        
        # Generate comprehensive report
        report = {
            'timestamp': pd.Timestamp.now().isoformat(),
            'data_summary': {
                'source': source_type,
                'records': len(marketing_data),
                'channels': len(spend_columns),
                'date_range': f"{marketing_data['date'].min()} to {marketing_data['date'].max()}" if 'date' in marketing_data.columns else 'N/A'
            },
            'model_performance': results['performance'],
            'model_coefficients': results['coefficients'],
            'optimization_results': optimization_results,
            'current_budgets': current_budgets,
            'total_budget': sum(current_budgets.values())
        }
        
        # Save report to multiple formats
        report_base = temp_directory / "mmm_report"
        
        # JSON report
        json_path = report_base.with_suffix('.json')
        with open(json_path, 'w') as f:
            json.dump(report, f, indent=2, default=str)
        
        # CSV summary
        csv_data = []
        for channel in spend_columns:
            channel_name = channel.replace('_spend', '')
            csv_data.append({
                'channel': channel_name,
                'current_budget': current_budgets.get(channel_name, 0),
                'optimized_budget': optimization_results.get('optimized_allocation', {}).get(channel_name, 0),
                'coefficient': results['coefficients'].get(channel_name, 0)
            })
        
        csv_path = report_base.with_suffix('.csv')
        pd.DataFrame(csv_data).to_csv(csv_path, index=False)
        
        # Verify files were created
        assert json_path.exists()
        assert csv_path.exists()
        
        # Verify content
        with open(json_path, 'r') as f:
            loaded_report = json.load(f)
            assert 'data_summary' in loaded_report
            assert 'model_performance' in loaded_report
        
        csv_df = pd.read_csv(csv_path)
        assert len(csv_df) == len(spend_columns)
        assert 'channel' in csv_df.columns
        
        print(f"[REPORTING] Generated reports: {json_path.name}, {csv_path.name}")
        print("[INTEGRATION] End-to-end reporting test completed")
        
        return report

@pytest.mark.slow
@pytest.mark.integration
class TestPerformanceIntegration:
    """Test performance characteristics of the integrated system"""
    
    def test_large_dataset_performance(self):
        """Test performance with larger synthetic dataset"""
        # Generate larger dataset
        np.random.seed(42)
        large_size = 200  # 200 weeks ~ 4 years
        
        dates = pd.date_range('2020-01-01', periods=large_size, freq='W')
        
        large_data = pd.DataFrame({
            'date': dates,
            'tv_spend': np.random.normal(50000, 10000, large_size),
            'digital_spend': np.random.normal(30000, 8000, large_size),
            'radio_spend': np.random.normal(20000, 5000, large_size),
            'print_spend': np.random.normal(15000, 3000, large_size),
            'social_spend': np.random.normal(25000, 6000, large_size),
        })
        
        # Create realistic revenue relationship
        large_data['revenue'] = (
            large_data['tv_spend'] * 0.8 +
            large_data['digital_spend'] * 1.2 +
            large_data['radio_spend'] * 0.6 +
            large_data['print_spend'] * 0.4 +
            large_data['social_spend'] * 1.0 +
            np.random.normal(20000, 5000, large_size)
        )
        
        # Ensure positive values
        for col in large_data.columns:
            if col != 'date':
                large_data[col] = np.maximum(large_data[col], 0)
        
        print(f"\n[PERFORMANCE] Testing with {len(large_data)} records...")
        
        # Time the model training
        import time
        
        start_time = time.time()
        
        model = EconometricMMM()
        spend_columns = [col for col in large_data.columns if col.endswith('_spend')]
        
        results = model.fit(
            data=large_data,
            target_column='revenue',
            spend_columns=spend_columns,
            include_synergies=True
        )
        
        training_time = time.time() - start_time
        
        print(f"[PERFORMANCE] Training completed in {training_time:.2f} seconds")
        print(f"[PERFORMANCE] Model R²: {results['performance']['r2_score']:.3f}")
        
        # Training should complete in reasonable time
        assert training_time < 60  # Should be under 1 minute
        
        # Performance should be reasonable for synthetic data
        assert results['performance']['r2_score'] > 0.1
        
        return {
            'dataset_size': len(large_data),
            'training_time_seconds': training_time,
            'performance': results['performance']
        }

if __name__ == "__main__":
    # Run integration tests
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == "run":
        test_class = TestEndToEndPipeline()
        
        print("Running integration tests...")
        
        try:
            pipeline_result = test_class.test_full_mmm_pipeline()
            print("✓ Full pipeline test passed")
            
            model_comparison = test_class.test_multi_model_comparison()
            print("✓ Multi-model comparison passed")
            
            cv_results = test_class.test_cross_validation_simulation()
            print("✓ Cross-validation simulation passed")
            
            print("\n🚀 All integration tests passed!")
            
        except Exception as e:
            print(f"\n❌ Integration test failed: {e}")
            sys.exit(1)