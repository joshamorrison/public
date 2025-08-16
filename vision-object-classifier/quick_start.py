#!/usr/bin/env python3
"""
Quick Start Demo - Test the vision classifier with sample images
Run this after: pip install -r requirements.txt
"""

import sys
import subprocess
import os

def run_demo():
    """Run quick demo with pre-included sample images"""
    
    print("=== Vision Object Classifier - Quick Start Demo ===")
    print()
    
    # Check if we're ready to run
    if not os.path.exists('models/final_balanced_model.pth'):
        print("ERROR: Model file not found. Make sure you're in the project root directory.")
        return
    
    if not os.path.exists('src/predict.py'):
        print("ERROR: Prediction script not found. Make sure you're in the project root directory.")
        return
    
    # Test samples
    test_cases = [
        {
            'name': 'Clean Plate (Synthetic)',
            'image': 'data/clean/plate_01.jpg',
            'expected': 'Clean'
        },
        {
            'name': 'Dirty Plate (Synthetic)', 
            'image': 'data/dirty/plate_01_dirty_medium_02.jpg',
            'expected': 'Dirty'
        },
        {
            'name': 'Real-World Dirty Dish (Pasta Stains)',
            'image': 'data/dirty/real_dirty_pasta_plate.jpg',
            'expected': 'Dirty'
        }
    ]
    
    print("Running demos with included sample images...")
    print()
    
    success_count = 0
    
    for i, test in enumerate(test_cases, 1):
        print(f"{i}. Testing {test['name']} (expecting: {test['expected']})")
        
        if not os.path.exists(test['image']):
            print(f"   SKIP: {test['image']} not found")
            continue
            
        # Run prediction
        cmd = [
            sys.executable, 'src/predict.py',
            '--model', 'models/final_balanced_model.pth',
            '--config', 'models/balanced_config.json', 
            '--image', test['image']
        ]
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                # Parse output for class prediction
                lines = result.stdout.strip().split('\n')
                prediction_line = [line for line in lines if line.startswith('Class:')]
                
                if prediction_line:
                    predicted_class = prediction_line[0].split(':', 1)[1].strip()
                    confidence_line = [line for line in lines if line.startswith('Confidence:')]
                    confidence = confidence_line[0].split(':', 1)[1].strip() if confidence_line else "Unknown"
                    
                    print(f"   Result: {predicted_class} (confidence: {confidence})")
                    
                    if predicted_class.lower() == test['expected'].lower():
                        print(f"   SUCCESS: Correct prediction!")
                        success_count += 1
                    else:
                        print(f"   WARNING: Expected {test['expected']}, got {predicted_class}")
                else:
                    print(f"   ERROR: Could not parse prediction result")
            else:
                print(f"   ERROR: Prediction failed - {result.stderr.strip()}")
                
        except subprocess.TimeoutExpired:
            print(f"   ERROR: Prediction timed out")
        except Exception as e:
            print(f"   ERROR: {e}")
        
        print()
    
    # Summary
    print("=== DEMO RESULTS ===")
    print(f"Successful predictions: {success_count}/{len(test_cases)}")
    
    if success_count == len(test_cases):
        print("SUCCESS: All demo tests passed! The system is working correctly.")
    elif success_count > 0:
        print("PARTIAL: Some tests passed. The system is mostly working.")
    else:
        print("FAILED: No tests passed. Check your installation.")
    
    print()
    print("=== NEXT STEPS ===")
    print("To test with your own images:")
    print("python src/predict.py --model models/final_balanced_model.pth --config models/balanced_config.json --image YOUR_IMAGE.jpg")
    print()
    print("To train with your own data:")
    print("1. Add images to data/clean/ and data/dirty/")
    print("2. python src/train.py")

if __name__ == "__main__":
    run_demo()