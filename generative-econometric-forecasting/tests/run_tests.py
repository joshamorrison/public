#!/usr/bin/env python3
"""
🧪 Test Runner
Runs all tests for the platform
"""

import unittest
import sys
import os
from pathlib import Path

def discover_and_run_tests():
    """Discover and run all tests"""
    print("🔍 Discovering tests...")
    
    # Set up test discovery
    test_dir = Path(__file__).parent
    loader = unittest.TestLoader()
    suite = loader.discover(str(test_dir), pattern='test_*.py')
    
    # Run tests
    print("🧪 Running tests...")
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    # Print summary
    print("\n" + "=" * 50)
    print("📊 TEST SUMMARY")
    print("=" * 50)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    
    if result.failures:
        print("\n❌ FAILURES:")
        for test, failure in result.failures:
            print(f"  - {test}: {failure}")
    
    if result.errors:
        print("\n💥 ERRORS:")
        for test, error in result.errors:
            print(f"  - {test}: {error}")
    
    if result.wasSuccessful():
        print("\n🎉 ALL TESTS PASSED!")
        return True
    else:
        print("\n❌ SOME TESTS FAILED")
        return False

if __name__ == '__main__':
    print("🚀 GENERATIVE ECONOMETRIC FORECASTING - TEST SUITE")
    print("=" * 60)
    
    success = discover_and_run_tests()
    sys.exit(0 if success else 1)