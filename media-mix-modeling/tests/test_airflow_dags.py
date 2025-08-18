"""
Tests for Airflow DAG validation and structure
Validates DAGs without running full Airflow server
"""

import pytest
import os
import sys
import traceback
from pathlib import Path

# Add project paths
project_root = Path(__file__).parent.parent  # Go up one level from tests folder
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "infrastructure" / "airflow" / "dags"))

class TestAirflowDAGs:
    """Test class for Airflow DAG validation"""
    
    @pytest.fixture
    def dag_files(self):
        """List of DAG files to test"""
        return [
            "infrastructure/airflow/dags/daily_mmm_pipeline.py",
            "infrastructure/airflow/dags/data_ingestion_workflow.py", 
            "infrastructure/airflow/dags/model_retraining_schedule.py"
        ]
    
    def _import_dag(self, dag_file_path):
        """Helper method to import a DAG file"""
        dag_name = Path(dag_file_path).stem
        
        # Import the DAG module
        spec = __import__(dag_name)
        
        # Check if DAG object exists
        assert hasattr(spec, 'dag'), f"No 'dag' object found in {dag_name}"
        
        dag = spec.dag
        return dag
    
    @pytest.mark.parametrize("dag_file", [
        "infrastructure/airflow/dags/daily_mmm_pipeline.py",
        "infrastructure/airflow/dags/data_ingestion_workflow.py", 
        "infrastructure/airflow/dags/model_retraining_schedule.py"
    ])
    def test_dag_import(self, dag_file):
        """Test if a DAG file can be imported without errors"""
        # Check if file exists
        dag_path = project_root / dag_file
        assert dag_path.exists(), f"DAG file not found: {dag_file}"
        
        # Import the DAG
        dag = self._import_dag(dag_file)
        
        # Validate DAG properties
        assert hasattr(dag, 'dag_id'), "DAG should have dag_id"
        assert hasattr(dag, 'schedule_interval'), "DAG should have schedule_interval"
        assert hasattr(dag, 'tasks'), "DAG should have tasks"
        assert len(dag.tasks) > 0, "DAG should have at least one task"
    
    def test_dag_structure_validation(self, dag_files):
        """Test overall DAG structure and consistency"""
        imported_dags = []
        
        for dag_file in dag_files:
            dag_path = project_root / dag_file
            if dag_path.exists():
                dag = self._import_dag(dag_file)
                imported_dags.append(dag)
        
        # Ensure we have at least one DAG
        assert len(imported_dags) > 0, "Should have at least one valid DAG"
        
        # Check for unique DAG IDs
        dag_ids = [dag.dag_id for dag in imported_dags]
        assert len(dag_ids) == len(set(dag_ids)), "All DAG IDs should be unique"
    
    def test_dag_task_structure(self, dag_files):
        """Test that DAG tasks have proper structure"""
        for dag_file in dag_files:
            dag_path = project_root / dag_file
            if dag_path.exists():
                dag = self._import_dag(dag_file)
                
                # Check each task has required properties
                for task in dag.tasks:
                    assert hasattr(task, 'task_id'), f"Task should have task_id: {task}"
                    assert hasattr(task, 'dag'), f"Task should reference DAG: {task}"
                    assert task.dag == dag, f"Task should belong to correct DAG: {task}"

# Standalone function for non-pytest usage
def validate_airflow_dags():
    """Standalone function to validate all DAGs (for CI/CD)"""
    dag_files = [
        "infrastructure/airflow/dags/daily_mmm_pipeline.py",
        "infrastructure/airflow/dags/data_ingestion_workflow.py", 
        "infrastructure/airflow/dags/model_retraining_schedule.py"
    ]
    
    results = []
    for dag_file in dag_files:
        dag_path = project_root / dag_file
        try:
            if dag_path.exists():
                dag_name = Path(dag_file).stem
                spec = __import__(dag_name)
                
                if hasattr(spec, 'dag'):
                    dag = spec.dag
                    results.append({
                        'file': dag_file,
                        'dag_id': dag.dag_id,
                        'tasks': len(dag.tasks),
                        'status': 'valid'
                    })
                else:
                    results.append({
                        'file': dag_file,
                        'status': 'error',
                        'message': 'No dag object found'
                    })
            else:
                results.append({
                    'file': dag_file,
                    'status': 'missing',
                    'message': 'File not found'
                })
        except Exception as e:
            results.append({
                'file': dag_file,
                'status': 'error', 
                'message': str(e)
            })
    
    return results

if __name__ == "__main__":
    # Run validation when script is executed directly
    results = validate_airflow_dags()
    
    print("Airflow DAG Validation Results:")
    print("=" * 40)
    
    for result in results:
        status = result['status']
        file_name = Path(result['file']).name
        
        if status == 'valid':
            print(f"✓ {file_name}: {result['dag_id']} ({result['tasks']} tasks)")
        else:
            print(f"✗ {file_name}: {result['message']}")
    
    valid_count = sum(1 for r in results if r['status'] == 'valid')
    total_count = len(results)
    
    print(f"\nResult: {valid_count}/{total_count} DAGs valid")
    
    if valid_count == total_count:
        print("All Airflow DAGs are valid!")
        sys.exit(0)
    else:
        print("Some DAGs failed validation")
        sys.exit(1)