#!/usr/bin/env python3
"""
Test runner script for the MMM platform
Comprehensive testing with different test suites
"""

import subprocess
import sys
import os
import argparse
from pathlib import Path

def run_command(command, description):
    """Run a command and capture output"""
    print(f"\n{'='*60}")
    print(f"RUNNING: {description}")
    print(f"COMMAND: {command}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run(
            command,
            shell=True,
            capture_output=True,
            text=True,
            timeout=300  # 5 minute timeout
        )
        
        print(result.stdout)
        if result.stderr:
            print("STDERR:", result.stderr)
        
        if result.returncode == 0:
            print(f"[PASS] {description} - PASSED")
            return True
        else:
            print(f"[FAIL] {description} - FAILED (exit code: {result.returncode})")
            return False
            
    except subprocess.TimeoutExpired:
        print(f"[TIMEOUT] {description} - TIMEOUT")
        return False
    except Exception as e:
        print(f"[ERROR] {description} - ERROR: {e}")
        return False

def run_unit_tests():
    """Run unit tests"""
    command = "python -m pytest tests/ -m 'not slow and not integration and not aws and not r_integration' -v"
    return run_command(command, "Unit Tests")

def run_integration_tests():
    """Run integration tests"""
    command = "python -m pytest tests/ -m 'integration' -v"
    return run_command(command, "Integration Tests")

def run_aws_tests():
    """Run AWS-related tests"""
    command = "python -m pytest tests/ -m 'aws' -v"
    return run_command(command, "AWS Deployment Tests")

def run_r_integration_tests():
    """Run R integration tests"""
    command = "python -m pytest tests/ -m 'r_integration' -v"
    return run_command(command, "R Integration Tests")

def run_all_tests():
    """Run all tests"""
    command = "python -m pytest tests/ -v"
    return run_command(command, "All Tests")

def run_smoke_tests():
    """Run smoke tests (quick validation)"""
    smoke_tests = [
        "python -c \"from data.media_data_client import MediaDataClient; print('Data client import: OK')\"",
        "python -c \"from models.mmm.econometric_mmm import EconometricMMM; print('MMM model import: OK')\"",
        "python -c \"from models.mmm.budget_optimizer import BudgetOptimizer; print('Budget optimizer import: OK')\"",
    ]
    
    print(f"\n{'='*60}")
    print("RUNNING: Smoke Tests (Quick Validation)")
    print(f"{'='*60}")
    
    all_passed = True
    
    for i, command in enumerate(smoke_tests, 1):
        print(f"\n[SMOKE {i}] {command}")
        try:
            result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=30)
            if result.returncode == 0:
                print(f"[PASS] Smoke test {i} passed")
                if result.stdout.strip():
                    print(f"   Output: {result.stdout.strip()}")
            else:
                print(f"[FAIL] Smoke test {i} failed")
                if result.stderr:
                    print(f"   Error: {result.stderr.strip()}")
                all_passed = False
        except subprocess.TimeoutExpired:
            print(f"[TIMEOUT] Smoke test {i} timed out")
            all_passed = False
        except Exception as e:
            print(f"[ERROR] Smoke test {i} error: {e}")
            all_passed = False
    
    return all_passed

def run_coverage_report():
    """Generate coverage report"""
    command = "python -m pytest tests/ --cov=src --cov=models --cov=data --cov-report=html --cov-report=term"
    return run_command(command, "Coverage Report")

def check_dependencies():
    """Check if test dependencies are available"""
    dependencies = [
        ('pytest', 'pytest'),
        ('pytest-cov', 'coverage reporting'),
        ('pandas', 'data processing'),
        ('numpy', 'numerical computing'),
        ('scikit-learn', 'machine learning')
    ]
    
    print(f"\n{'='*60}")
    print("CHECKING: Test Dependencies")
    print(f"{'='*60}")
    
    missing_deps = []
    
    for package, description in dependencies:
        try:
            __import__(package.replace('-', '_'))
            print(f"[OK] {package} - {description}")
        except ImportError:
            print(f"[MISSING] {package} - {description} (MISSING)")
            missing_deps.append(package)
    
    if missing_deps:
        print(f"\n[WARNING] Missing dependencies: {', '.join(missing_deps)}")
        print("Install with: pip install " + " ".join(missing_deps))
        return False
    
    print("\n[SUCCESS] All test dependencies available")
    return True

def main():
    """Main test runner function"""
    parser = argparse.ArgumentParser(description="Run MMM platform tests")
    parser.add_argument('--suite', choices=[
        'unit', 'integration', 'aws', 'r', 'all', 'smoke', 'coverage'
    ], default='unit', help='Test suite to run')
    parser.add_argument('--check-deps', action='store_true', help='Check test dependencies')
    parser.add_argument('--quick', action='store_true', help='Run quick smoke tests only')
    
    args = parser.parse_args()
    
    print("MMM PLATFORM TEST RUNNER")
    print("="*60)
    
    # Check dependencies first
    if args.check_deps or args.suite != 'smoke':
        if not check_dependencies():
            print("\n[FAIL] Dependency check failed")
            sys.exit(1)
    
    # Change to project directory
    project_root = Path(__file__).parent.parent
    os.chdir(project_root)
    print(f"\nWorking directory: {project_root}")
    
    # Run quick tests if requested
    if args.quick:
        success = run_smoke_tests()
        sys.exit(0 if success else 1)
    
    # Run selected test suite
    success = False
    
    if args.suite == 'unit':
        success = run_unit_tests()
    elif args.suite == 'integration':
        success = run_integration_tests()
    elif args.suite == 'aws':
        success = run_aws_tests()
    elif args.suite == 'r':
        success = run_r_integration_tests()
    elif args.suite == 'all':
        success = run_all_tests()
    elif args.suite == 'smoke':
        success = run_smoke_tests()
    elif args.suite == 'coverage':
        success = run_coverage_report()
    
    # Summary
    print(f"\n{'='*60}")
    if success:
        print("[SUCCESS] ALL TESTS PASSED!")
        print("[READY] MMM platform is ready for production")
    else:
        print("[FAILURE] SOME TESTS FAILED!")
        print("[ACTION] Review test output and fix issues")
    print(f"{'='*60}")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()