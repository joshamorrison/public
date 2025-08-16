#!/usr/bin/env python3
"""
Test runner for Vision Object Classifier

This script runs all unit tests and provides a summary of results.
Can be run with: python tests/run_tests.py
"""

import unittest
import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

def run_all_tests():
    """Run all unit tests and return results"""
    # Discover and run all tests
    loader = unittest.TestLoader()
    test_suite = loader.discover('tests', pattern='test_*.py')
    
    # Run tests with detailed output
    runner = unittest.TextTestRunner(verbosity=2, buffer=True)
    result = runner.run(test_suite)
    
    return result

def print_test_summary(result):
    """Print a summary of test results"""
    total_tests = result.testsRun
    failures = len(result.failures)
    errors = len(result.errors)
    skipped = len(result.skipped) if hasattr(result, 'skipped') else 0
    passed = total_tests - failures - errors - skipped
    
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    print(f"Total Tests Run: {total_tests}")
    print(f"Passed: {passed}")
    print(f"Failed: {failures}")
    print(f"Errors: {errors}")
    print(f"Skipped: {skipped}")
    print("=" * 60)
    
    if failures > 0:
        print("\nFAILURES:")
        for test, traceback in result.failures:
            print(f"- {test}: {traceback.split('AssertionError:')[-1].strip()}")
    
    if errors > 0:
        print("\nERRORS:")
        for test, traceback in result.errors:
            print(f"- {test}: {traceback.split('Error:')[-1].strip()}")
    
    success_rate = (passed / total_tests) * 100 if total_tests > 0 else 0
    print(f"\nSuccess Rate: {success_rate:.1f}%")
    
    if success_rate == 100:
        print("SUCCESS: All tests passed!")
    elif success_rate >= 80:
        print("GOOD: Most tests passed - good job!")
    elif success_rate >= 60:
        print("WARNING: Some tests need attention")
    else:
        print("ERROR: Many tests failing - needs work")

def run_specific_test_module(module_name):
    """Run tests from a specific module"""
    try:
        # Import the specific test module
        test_module = __import__(f'test_{module_name}', fromlist=[''])
        
        # Create test suite from module
        loader = unittest.TestLoader()
        test_suite = loader.loadTestsFromModule(test_module)
        
        # Run tests
        runner = unittest.TextTestRunner(verbosity=2)
        result = runner.run(test_suite)
        
        return result
    except ImportError:
        print(f"Test module 'test_{module_name}' not found")
        return None

def main():
    """Main test runner function"""
    print("Vision Object Classifier - Test Runner")
    print("=" * 60)
    
    # Check command line arguments
    if len(sys.argv) > 1:
        module_name = sys.argv[1]
        print(f"Running tests for module: {module_name}")
        result = run_specific_test_module(module_name)
    else:
        print("Running all tests...")
        result = run_all_tests()
    
    if result:
        print_test_summary(result)
        
        # Exit with error code if tests failed
        if result.failures or result.errors:
            sys.exit(1)
        else:
            sys.exit(0)
    else:
        print("No tests were run")
        sys.exit(1)

if __name__ == '__main__':
    main()