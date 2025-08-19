#!/usr/bin/env python3
"""
Test script for refactored vision classifier components
Windows-compatible version without Unicode characters
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

def test_imports():
    """Test if all refactored components can be imported"""
    print("Testing imports...")
    
    try:
        from vision_classifier.predict import DishCleanlinessPredictor
        print("  [OK] DishCleanlinessPredictor imported successfully")
    except ImportError as e:
        print(f"  [ERROR] Failed to import DishCleanlinessPredictor: {e}")
        return False
        
    try:
        sys.path.append(str(project_root))
        from api.models.request_models import ModelType, ClassificationRequest
        from api.models.response_models import ClassificationResponse
        print("  [OK] API models imported successfully")
    except ImportError as e:
        print(f"  [ERROR] Failed to import API models: {e}")
        return False
    
    return True

def test_model_availability():
    """Test if trained models are available"""
    print("\nTesting model availability...")
    
    models_dir = project_root / "models"
    if not models_dir.exists():
        print("  [ERROR] Models directory not found")
        return False
    
    model_files = list(models_dir.glob("*.pth"))
    if not model_files:
        print("  [ERROR] No trained models found (.pth files)")
        return False
    
    for model_file in model_files:
        print(f"  [OK] Found model: {model_file.name}")
    
    return True

def test_sample_data():
    """Test if sample data is properly organized"""
    print("\nTesting sample data organization...")
    
    # Test samples folder
    samples_dir = project_root / "data" / "samples" / "demo_images"
    if samples_dir.exists():
        sample_files = list(samples_dir.glob("*.jpg"))
        print(f"  [OK] Found {len(sample_files)} sample images")
        for sample in sample_files:
            print(f"    - {sample.name}")
    else:
        print("  [WARNING] No demo images found in samples folder")
    
    # Test schemas
    schemas_dir = project_root / "data" / "schemas"
    if schemas_dir.exists():
        schema_files = list(schemas_dir.glob("*.json"))
        print(f"  [OK] Found {len(schema_files)} schema files")
    else:
        print("  [WARNING] Schemas directory not found")
    
    return True

def test_classification():
    """Test actual classification with a sample image"""
    print("\nTesting classification functionality...")
    
    try:
        from vision_classifier.predict import DishCleanlinessPredictor
        
        # Find model - try specific models in order of preference
        models_dir = project_root / "models"
        preferred_models = ["final_balanced_model.pth", "fast_model.pth", "test_model.pth"]
        model_path = None
        
        for preferred in preferred_models:
            candidate = models_dir / preferred
            if candidate.exists():
                model_path = candidate
                break
        
        if not model_path:
            # Fallback to any available model
            model_files = list(models_dir.glob("*.pth"))
            if not model_files:
                print("  [SKIP] No models available for testing")
                return True
            model_path = model_files[0]
            
        print(f"  Using model: {model_path.name}")
        
        # Find sample image
        samples_dir = project_root / "data" / "samples" / "demo_images"
        sample_images = list(samples_dir.glob("*.jpg")) if samples_dir.exists() else []
        
        if not sample_images:
            print("  [SKIP] No sample images available for testing")
            return True
        
        sample_image = sample_images[0]
        print(f"  Testing with image: {sample_image.name}")
        
        # Try multiple models if needed
        for model_candidate in preferred_models:
            model_file = models_dir / model_candidate
            if not model_file.exists():
                continue
                
            try:
                print(f"  Trying model: {model_candidate}")
                
                # Load predictor
                predictor = DishCleanlinessPredictor(
                    model_path=str(model_file),
                    config_path=None  # Optional config
                )
                
                # Make prediction
                result = predictor.predict_single(str(sample_image))
                model_path = model_file
                break
                
            except Exception as model_error:
                print(f"    Model {model_candidate} failed: {str(model_error)[:100]}...")
                continue
        else:
            print("  [ERROR] All models failed to load")
            return False
        
        # Display results
        prediction = "clean" if result['prediction'] == 0 else "dirty"
        confidence = result['confidence']
        
        print(f"  [OK] Classification successful!")
        print(f"    Predicted: {prediction}")
        print(f"    Confidence: {confidence:.3f}")
        print(f"    Clean prob: {result['clean_prob']:.3f}")
        print(f"    Dirty prob: {result['dirty_prob']:.3f}")
        
        return True
        
    except Exception as e:
        print(f"  [ERROR] Classification failed: {e}")
        return False

def main():
    """Run all tests"""
    print("Vision Object Classifier - Refactored Components Test")
    print("=" * 55)
    
    tests = [
        ("Import Test", test_imports),
        ("Model Availability", test_model_availability), 
        ("Sample Data", test_sample_data),
        ("Classification", test_classification)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        try:
            if test_func():
                passed += 1
                print(f"\n{test_name}: PASSED")
            else:
                print(f"\n{test_name}: FAILED")
        except Exception as e:
            print(f"\n{test_name}: ERROR - {e}")
    
    print("\n" + "=" * 55)
    print(f"Tests Results: {passed}/{total} passed")
    
    if passed == total:
        print("All tests passed! Refactored components are working correctly.")
        return True
    else:
        print("Some tests failed. Check the output above for details.")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)