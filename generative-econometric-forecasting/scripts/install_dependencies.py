#!/usr/bin/env python3
"""
🔧 Dependency Installation Script
Ensures all required packages are installed for the platform
"""

import subprocess
import sys
import os

def install_requirements():
    """Install requirements.txt dependencies"""
    print("📦 Installing core dependencies...")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])
        print("✅ Core dependencies installed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Failed to install dependencies: {e}")
        return False

def test_imports():
    """Test critical imports"""
    print("\n🧪 Testing critical imports...")
    
    test_packages = [
        'pandas', 'numpy', 'matplotlib', 'sklearn',
        'statsmodels', 'fredapi', 'langchain', 'openai'
    ]
    
    failed_imports = []
    for package in test_packages:
        try:
            __import__(package)
            print(f"✅ {package}")
        except ImportError:
            print(f"❌ {package}")
            failed_imports.append(package)
    
    if failed_imports:
        print(f"\n⚠️  Failed imports: {', '.join(failed_imports)}")
        return False
    else:
        print("\n🎉 All critical packages imported successfully!")
        return True

if __name__ == "__main__":
    print("🚀 DEPENDENCY INSTALLATION SCRIPT")
    print("=" * 40)
    
    success = install_requirements()
    if success:
        test_imports()
    else:
        print("❌ Installation failed")
        sys.exit(1)