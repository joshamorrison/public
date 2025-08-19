#!/usr/bin/env python3
"""
Test Runner Script

Runs comprehensive test suite with reporting and coverage analysis.
"""

import subprocess
import sys
import argparse
from pathlib import Path
import json
from datetime import datetime


def run_command(cmd, cwd=None):
    """Run shell command and return result."""
    try:
        result = subprocess.run(cmd, shell=True, cwd=cwd, capture_output=True, text=True)
        return result.returncode == 0, result.stdout, result.stderr
    except Exception as e:
        return False, "", str(e)


def run_unit_tests(coverage=True, verbose=False):
    """Run unit tests."""
    print("🧪 Running Unit Tests...")
    
    cmd = "pytest tests/unit/"
    if coverage:
        cmd += " --cov=src --cov-report=html --cov-report=term"
    if verbose:
        cmd += " -v"
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("✅ Unit tests passed!")
    else:
        print("❌ Unit tests failed!")
        print(f"Error: {stderr}")
    
    return success


def run_integration_tests(verbose=False):
    """Run integration tests."""
    print("🔗 Running Integration Tests...")
    
    cmd = "pytest tests/integration/"
    if verbose:
        cmd += " -v"
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("✅ Integration tests passed!")
    else:
        print("❌ Integration tests failed!")
        print(f"Error: {stderr}")
    
    return success


def run_e2e_tests(verbose=False):
    """Run end-to-end tests."""
    print("🚀 Running End-to-End Tests...")
    
    cmd = "pytest tests/e2e/"
    if verbose:
        cmd += " -v"
    cmd += " -m 'not slow'"  # Skip slow tests by default
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("✅ E2E tests passed!")
    else:
        print("❌ E2E tests failed!")
        print(f"Error: {stderr}")
    
    return success


def run_slow_tests(verbose=False):
    """Run slow tests."""
    print("🐌 Running Slow Tests...")
    
    cmd = "pytest tests/ -m slow"
    if verbose:
        cmd += " -v"
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("✅ Slow tests passed!")
    else:
        print("❌ Slow tests failed!")
        print(f"Error: {stderr}")
    
    return success


def run_network_tests(verbose=False):
    """Run tests that require network access."""
    print("🌐 Running Network Tests...")
    
    cmd = "pytest tests/ -m network"
    if verbose:
        cmd += " -v"
    
    success, stdout, stderr = run_command(cmd)
    
    if success:
        print("✅ Network tests passed!")
    else:
        print("❌ Network tests failed!")
        print(f"Error: {stderr}")
    
    return success


def run_all_tests(include_slow=False, include_network=False, coverage=True, verbose=False):
    """Run all test suites."""
    print("🎯 Running Complete Test Suite...")
    print("=" * 50)
    
    results = {}
    
    # Unit tests
    results["unit"] = run_unit_tests(coverage=coverage, verbose=verbose)
    
    # Integration tests
    results["integration"] = run_integration_tests(verbose=verbose)
    
    # E2E tests
    results["e2e"] = run_e2e_tests(verbose=verbose)
    
    # Optional test suites
    if include_slow:
        results["slow"] = run_slow_tests(verbose=verbose)
    
    if include_network:
        results["network"] = run_network_tests(verbose=verbose)
    
    return results


def generate_test_report(results):
    """Generate test report."""
    print("\\n" + "=" * 50)
    print("📊 Test Results Summary")
    print("=" * 50)
    
    total_suites = len(results)
    passed_suites = sum(1 for success in results.values() if success)
    
    for suite, success in results.items():
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"{suite.ljust(12)}: {status}")
    
    print("-" * 50)
    print(f"Total: {passed_suites}/{total_suites} test suites passed")
    
    overall_success = passed_suites == total_suites
    if overall_success:
        print("🎉 All tests passed!")
    else:
        print("⚠️  Some tests failed!")
    
    return overall_success


def run_linting():
    """Run code linting checks."""
    print("🔍 Running Code Quality Checks...")
    
    # Check if linting tools are available
    linting_results = {}
    
    # Run flake8 if available
    success, stdout, stderr = run_command("flake8 --version")
    if success:
        print("Running flake8...")
        success, stdout, stderr = run_command("flake8 src/ tests/ --max-line-length=100 --ignore=E203,W503")
        linting_results["flake8"] = success
        if not success:
            print(f"Flake8 issues: {stderr}")
    
    # Run black check if available
    success, stdout, stderr = run_command("black --version")
    if success:
        print("Running black (check only)...")
        success, stdout, stderr = run_command("black --check src/ tests/")
        linting_results["black"] = success
        if not success:
            print("Black formatting issues found. Run 'black src/ tests/' to fix.")
    
    # Run isort check if available
    success, stdout, stderr = run_command("isort --version")
    if success:
        print("Running isort...")
        success, stdout, stderr = run_command("isort --check-only src/ tests/")
        linting_results["isort"] = success
        if not success:
            print("Import sorting issues found. Run 'isort src/ tests/' to fix.")
    
    if linting_results:
        passed = sum(1 for success in linting_results.values() if success)
        total = len(linting_results)
        print(f"Code quality: {passed}/{total} checks passed")
        return passed == total
    else:
        print("No linting tools available")
        return True


def check_dependencies():
    """Check if required test dependencies are installed."""
    required_packages = [
        "pytest",
        "pytest-asyncio", 
        "pytest-cov",
        "httpx"
    ]
    
    missing_packages = []
    
    for package in required_packages:
        success, stdout, stderr = run_command(f"python -c \\"import {package.replace('-', '_')}\\"")
        if not success:
            missing_packages.append(package)
    
    if missing_packages:
        print(f"❌ Missing test dependencies: {', '.join(missing_packages)}")
        print("Install with: pip install " + " ".join(missing_packages))
        return False
    
    print("✅ All test dependencies are installed")
    return True


def main():
    """Main test runner."""
    parser = argparse.ArgumentParser(description="Multi-Agent Platform Test Runner")
    parser.add_argument("--suite", choices=["unit", "integration", "e2e", "all"], 
                       default="all", help="Test suite to run")
    parser.add_argument("--slow", action="store_true", help="Include slow tests")
    parser.add_argument("--network", action="store_true", help="Include network tests")
    parser.add_argument("--no-coverage", action="store_true", help="Skip coverage reporting")
    parser.add_argument("--lint", action="store_true", help="Run linting checks")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser.add_argument("--check-deps", action="store_true", help="Check test dependencies")
    
    args = parser.parse_args()
    
    print("🚀 Multi-Agent Platform Test Runner")
    print("=" * 50)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Check dependencies if requested
    if args.check_deps:
        if not check_dependencies():
            sys.exit(1)
        return
    
    # Run linting if requested
    if args.lint:
        if not run_linting():
            sys.exit(1)
    
    # Run tests based on suite selection
    coverage = not args.no_coverage
    
    if args.suite == "unit":
        success = run_unit_tests(coverage=coverage, verbose=args.verbose)
        results = {"unit": success}
    elif args.suite == "integration":
        success = run_integration_tests(verbose=args.verbose)
        results = {"integration": success}
    elif args.suite == "e2e":
        success = run_e2e_tests(verbose=args.verbose)
        results = {"e2e": success}
    else:  # all
        results = run_all_tests(
            include_slow=args.slow,
            include_network=args.network,
            coverage=coverage,
            verbose=args.verbose
        )
    
    # Generate report
    overall_success = generate_test_report(results)
    
    # Coverage report info
    if coverage and "unit" in results:
        print("\\n📈 Coverage report generated in htmlcov/index.html")
    
    print(f"\\nCompleted at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Exit with appropriate code
    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    main()