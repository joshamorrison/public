#!/usr/bin/env python3
"""
Setup script for R integration
Installs required R packages and configures rpy2
"""

import subprocess
import sys
import os
from pathlib import Path

def check_r_installation():
    """Check if R is installed and accessible"""
    try:
        result = subprocess.run(['R', '--version'], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"[R] R is installed: {result.stdout.split()[2]}")
            return True
        else:
            print("[R] R is not accessible via command line")
            return False
    except (subprocess.TimeoutExpired, FileNotFoundError):
        print("[R] R is not installed or not in PATH")
        return False

def install_rpy2():
    """Install rpy2 package for Python-R interface"""
    try:
        print("[INSTALL] Installing rpy2...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', 'rpy2'])
        print("[INSTALL] rpy2 installed successfully")
        return True
    except subprocess.CalledProcessError:
        print("[INSTALL] Failed to install rpy2")
        return False

def install_r_packages():
    """Install required R packages"""
    r_packages = [
        'forecast',    # Time series forecasting
        'vars',        # Vector autoregression  
        'urca',        # Unit root and cointegration tests
        'tseries',     # Time series analysis
        'lmtest',      # Linear model diagnostic tests
        'car',         # Companion to applied regression
        'mgcv',        # Generalized additive models
        'MASS'         # Modern applied statistics (usually pre-installed)
    ]
    
    print("[R-PACKAGES] Installing required R packages...")
    
    # Create R script for package installation
    r_script = f'''
# Install required packages for MMM
packages_to_install <- c({", ".join([f'"{pkg}"' for pkg in r_packages])})

for(pkg in packages_to_install) {{
    if(!require(pkg, character.only = TRUE)) {{
        cat("Installing", pkg, "\\n")
        install.packages(pkg, repos="https://cloud.r-project.org/", dependencies=TRUE)
        
        if(require(pkg, character.only = TRUE)) {{
            cat("Successfully installed", pkg, "\\n")
        }} else {{
            cat("Failed to install", pkg, "\\n")
        }}
    }} else {{
        cat("Package", pkg, "already installed\\n")
    }}
}}

# Verify installation
cat("\\n=== PACKAGE VERIFICATION ===\\n")
for(pkg in packages_to_install) {{
    if(require(pkg, character.only = TRUE)) {{
        cat("✓", pkg, "- OK\\n")
    }} else {{
        cat("✗", pkg, "- FAILED\\n")
    }}
}}

cat("\\nR package installation completed.\\n")
'''
    
    # Write R script to temporary file
    script_path = Path('./temp_install_packages.R')
    with open(script_path, 'w') as f:
        f.write(r_script)
    
    try:
        # Run R script
        result = subprocess.run(['R', '--slave', '--no-restore', '--file', str(script_path)], 
                              capture_output=True, text=True, timeout=300)
        
        print(result.stdout)
        if result.stderr:
            print(f"[R-PACKAGES] Warnings: {result.stderr}")
        
        # Clean up
        script_path.unlink()
        
        return result.returncode == 0
        
    except subprocess.TimeoutExpired:
        print("[R-PACKAGES] Installation timed out")
        return False
    except Exception as e:
        print(f"[R-PACKAGES] Installation failed: {e}")
        return False

def test_integration():
    """Test the R integration setup"""
    try:
        print("[TEST] Testing R integration...")
        
        # Test rpy2 import
        import rpy2.robjects as robjects
        from rpy2.robjects.packages import importr
        
        # Test basic R functionality
        r_base = importr('base')
        result = robjects.r('2 + 2')
        
        if float(result[0]) == 4.0:
            print("[TEST] ✓ Basic R functionality working")
        else:
            print("[TEST] ✗ Basic R functionality failed")
            return False
        
        # Test package loading
        successful_packages = []
        failed_packages = []
        
        test_packages = ['forecast', 'vars', 'lmtest', 'MASS']
        
        for package in test_packages:
            try:
                importr(package)
                successful_packages.append(package)
                print(f"[TEST] ✓ {package} loaded successfully")
            except:
                failed_packages.append(package)
                print(f"[TEST] ✗ {package} failed to load")
        
        print(f"\n[SUMMARY] {len(successful_packages)}/{len(test_packages)} packages working")
        
        if len(successful_packages) >= len(test_packages) // 2:
            print("[TEST] ✓ R integration setup successful!")
            return True
        else:
            print("[TEST] ⚠ Partial R integration - some packages missing")
            return False
            
    except ImportError:
        print("[TEST] ✗ rpy2 not available")
        return False
    except Exception as e:
        print(f"[TEST] ✗ Integration test failed: {e}")
        return False

def create_setup_guide():
    """Create setup guide for manual installation"""
    guide = '''
# R Integration Setup Guide

## Prerequisites
1. Install R from https://cran.r-project.org/
2. Add R to your system PATH
3. Install Python rpy2 package

## Manual Installation Steps

### 1. Install R (if not already installed)
- Windows: Download from https://cran.r-project.org/bin/windows/base/
- macOS: Download from https://cran.r-project.org/bin/macosx/
- Linux: Use package manager (e.g., sudo apt-get install r-base)

### 2. Install rpy2
```bash
pip install rpy2
```

### 3. Install R packages
Open R console and run:
```r
install.packages(c("forecast", "vars", "urca", "tseries", 
                  "lmtest", "car", "mgcv", "MASS"))
```

### 4. Test integration
```python
from models.r_integration import test_r_integration
test_r_integration()
```

## Troubleshooting

### Common Issues:
1. **R not found**: Ensure R is installed and in PATH
2. **Package installation fails**: Try installing packages individually in R
3. **rpy2 import error**: May need to install additional system dependencies

### Windows Specific:
- Install Rtools if package compilation fails
- Ensure R_HOME environment variable is set

### macOS Specific:  
- Install XCode command line tools: xcode-select --install
- May need to install gfortran

### Linux Specific:
- Install R development packages: sudo apt-get install r-base-dev
- Install system dependencies as needed

For more help, see: https://rpy2.github.io/doc/latest/html/introduction.html
'''
    
    guide_path = Path('./R_INTEGRATION_SETUP.md')
    with open(guide_path, 'w') as f:
        f.write(guide)
    
    print(f"[GUIDE] Setup guide created: {guide_path}")

def main():
    """Main setup function"""
    print("=" * 60)
    print("R INTEGRATION SETUP FOR MMM PLATFORM")
    print("=" * 60)
    
    # Step 1: Check R installation
    print("\n[STEP 1] Checking R installation...")
    r_available = check_r_installation()
    
    if not r_available:
        print("\n⚠️  R is not installed or not accessible")
        print("Please install R from https://cran.r-project.org/")
        create_setup_guide()
        return False
    
    # Step 2: Install rpy2
    print("\n[STEP 2] Installing rpy2...")
    rpy2_success = install_rpy2()
    
    if not rpy2_success:
        print("\n⚠️  Failed to install rpy2")
        print("Try manual installation: pip install rpy2")
        create_setup_guide()
        return False
    
    # Step 3: Install R packages
    print("\n[STEP 3] Installing R packages...")
    packages_success = install_r_packages()
    
    if not packages_success:
        print("\n⚠️  Some R packages may have failed to install")
        print("Check the output above for specific failures")
    
    # Step 4: Test integration
    print("\n[STEP 4] Testing integration...")
    test_success = test_integration()
    
    # Step 5: Create guide regardless
    create_setup_guide()
    
    # Summary
    print("\n" + "=" * 60)
    if test_success:
        print("🚀 R INTEGRATION SETUP COMPLETED SUCCESSFULLY!")
        print("\nYou can now use advanced econometric models:")
        print("- Vector Autoregression (VAR)")
        print("- Advanced Adstock modeling")  
        print("- Bayesian MMM with uncertainty quantification")
    else:
        print("⚠️  R INTEGRATION SETUP INCOMPLETE")
        print("\nSome components may not work. Check R_INTEGRATION_SETUP.md for manual setup.")
    
    print("=" * 60)
    
    return test_success

if __name__ == "__main__":
    main()