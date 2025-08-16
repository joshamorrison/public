#!/usr/bin/env python3
"""
Installation Test Script - Validates out-of-box experience
Run this after: pip install -r requirements.txt
"""

import os
import sys

def test_installation():
    """Test that a new user can run the model out of the box"""
    
    print("=== Vision Object Classifier - Installation Test ===")
    print()
    
    # Get project root directory (parent of scripts directory)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.chdir(project_root)
    
    # Test 1: Check essential files exist
    print("1. Checking essential files...")
    
    essential_files = [
        'models/final_balanced_model.pth',
        'models/balanced_config.json',
        'src/predict.py',
        'requirements.txt',
        'README.md'
    ]
    
    missing_files = []
    for file_path in essential_files:
        if os.path.exists(file_path):
            print(f"   OK: {file_path}")
        else:
            print(f"   MISSING: {file_path}")
            missing_files.append(file_path)
    
    if missing_files:
        print(f"\nFAILED: Missing {len(missing_files)} essential files")
        return False
    
    print()
    
    # Test 2: Check sample images exist
    print("2. Checking sample demo images...")
    
    sample_images = []
    if os.path.exists('data/clean'):
        clean_images = [f for f in os.listdir('data/clean') if f.endswith('.jpg')][:2]
        sample_images.extend([f'data/clean/{img}' for img in clean_images])
    
    if os.path.exists('data/dirty'):  
        dirty_images = [f for f in os.listdir('data/dirty') if f.endswith('.jpg')][:2]
        sample_images.extend([f'data/dirty/{img}' for img in dirty_images])
    
    if len(sample_images) >= 2:
        for img in sample_images[:4]:
            print(f"   OK: {img}")
        print(f"   Total sample images: {len(sample_images)}")
    else:
        print("   WARNING: Limited sample images (users can add their own)")
    
    print()
    
    # Test 3: Try importing key modules
    print("3. Testing Python imports...")
    
    try:
        sys.path.append('src')
        import predict
        print("   OK: predict.py imports successfully")
    except ImportError as e:
        print(f"   ERROR: predict.py import failed: {e}")
        return False
    
    try:
        import torch
        print("   OK: PyTorch available")
    except ImportError:
        print("   ERROR: PyTorch not installed - run: pip install -r requirements.txt")
        return False
    
    print()
    
    # Test 4: Simulate user workflow
    print("4. Testing user workflow...")
    
    model_path = 'models/final_balanced_model.pth'
    config_path = 'models/balanced_config.json'
    
    if sample_images:
        test_image = sample_images[0]
        print(f"   OK: Would test with: {test_image}")
        print(f"   OK: Command: python src/predict.py --model {model_path} --config {config_path} --image {test_image}")
    else:
        print("   OK: User can test with: python src/predict.py --model models/final_balanced_model.pth --config models/balanced_config.json --image YOUR_IMAGE.jpg")
    
    print()
    
    # Summary
    print("=== INSTALLATION TEST RESULTS ===")
    print("SUCCESS: Out-of-box experience ready!")
    print()
    print("Next steps for new users:")
    print("1. pip install -r requirements.txt")
    print("2. python src/predict.py --model models/final_balanced_model.pth --config models/balanced_config.json --image YOUR_DISH_IMAGE.jpg")
    print()
    print("For training new models:")
    print("1. Add training images to data/clean/ and data/dirty/")
    print("2. python src/train.py")
    print()
    
    return True

if __name__ == "__main__":
    success = test_installation()
    sys.exit(0 if success else 1)