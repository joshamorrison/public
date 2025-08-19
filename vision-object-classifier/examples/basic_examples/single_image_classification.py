#!/usr/bin/env python3
"""
Single Image Classification Example

Demonstrates basic usage of the vision object classifier
to classify a single image as clean or dirty.
"""

import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

try:
    from src.vision_classifier.predict import DishCleanlinessPredictor
except ImportError as e:
    print(f"⚠️  Import error: {e}")
    print("Make sure you're running from the project root and have installed dependencies")
    sys.exit(1)

def main():
    """
    Classify a single image and display results
    """
    print("Single Image Classification Example")
    print("=" * 45)
    
    try:
        # Use demo images from samples folder
        samples_dir = project_root / "data" / "samples" / "demo_images"
        
        # Find sample images
        sample_images = []
        
        if samples_dir.exists():
            clean_sample = samples_dir / "clean_plate_sample.jpg"
            dirty_sample = samples_dir / "real_dirty_pasta_plate.jpg"
            
            if clean_sample.exists():
                sample_images.append((clean_sample, "clean"))
            if dirty_sample.exists():
                sample_images.append((dirty_sample, "dirty"))
        
        # Fallback to processed data if samples not available
        if not sample_images:
            data_dir = project_root / "data" / "processed"
            clean_dir = data_dir / "clean_labeled"
            dirty_dir = data_dir / "dirty_labeled" 
            
            if clean_dir.exists():
                clean_samples = list(clean_dir.glob("plate_*.jpg"))[:1]
                sample_images.extend([(img, "clean") for img in clean_samples])
            
            if dirty_dir.exists():
                dirty_samples = list(dirty_dir.glob("*pasta*.jpg"))[:1]
                if not dirty_samples:
                    dirty_samples = list(dirty_dir.glob("*dirty*.jpg"))[:1]
                sample_images.extend([(img, "dirty") for img in dirty_samples])
        
        if not sample_images:
            print("ERROR: No sample images found in data/processed/")
            print("   Expected images in:")
            print(f"   - {clean_dir}")
            print(f"   - {dirty_dir}")
            return False
        
        print(f"📸 Found {len(sample_images)} sample images for classification")
        
        # Load model
        print("\n🤖 Loading classification model...")
        predictor = None
        try:
            models_dir = project_root / "models"
            model_path = None
            
            # Try different model files in order of preference
            for model_name in ["balanced_model.pth", "fast_model.pth", "final_balanced_model.pth", "trained_model.pth"]:
                candidate_path = models_dir / model_name
                if candidate_path.exists():
                    model_path = candidate_path
                    break
            
            if model_path:
                print(f"   ✅ Found model: {model_path.name}")
                config_path = models_dir / model_path.name.replace("_model.pth", "_config.json").replace(".pth", "_config.json")
                
                predictor = DishCleanlinessPredictor(
                    model_path=str(model_path),
                    config_path=str(config_path) if config_path.exists() else None
                )
                print("   ✅ Model loaded successfully")
            else:
                print("   ❌ No trained models found!")
                available_models = list(models_dir.glob("*.pth"))
                if available_models:
                    print(f"   Available models: {[m.name for m in available_models]}")
                return False
                
        except Exception as e:
            print(f"   ❌ Model loading error: {e}")
            return False
        
        # Classify each sample image
        print("\n🔍 Classification Results:")
        print("-" * 60)
        
        results = []
        
        for i, (image_path, true_label) in enumerate(sample_images, 1):
            try:
                # Get actual prediction from model
                print(f"   🔍 Processing {image_path.name}...")
                result = predictor.predict_single(str(image_path))
                
                # Extract prediction and confidence
                prediction = "clean" if result['prediction'] == 0 else "dirty"
                confidence = result['confidence']
                
                # Determine correctness
                correct = prediction == true_label
                accuracy_emoji = "✅" if correct else "❌"
                
                # Confidence level emoji
                if confidence > 0.9:
                    conf_emoji = "🟢"  # High confidence
                elif confidence > 0.7:
                    conf_emoji = "🟡"  # Medium confidence
                else:
                    conf_emoji = "🔴"  # Low confidence
                
                print(f"{i}. {image_path.name[:30]:<30} | "
                      f"Predicted: {prediction.upper():<5} | "
                      f"Confidence: {confidence:.1%} {conf_emoji} | "
                      f"Actual: {true_label.upper():<5} {accuracy_emoji}")
                
                results.append({
                    "image": image_path.name,
                    "predicted": prediction,
                    "confidence": confidence,
                    "actual": true_label,
                    "correct": correct
                })
                
            except Exception as e:
                print(f"{i}. {image_path.name} | ❌ Classification failed: {e}")
                results.append({
                    "image": image_path.name,
                    "predicted": "error",
                    "confidence": 0.0,
                    "actual": true_label,
                    "correct": False
                })
        
        # Calculate summary statistics
        print("-" * 60)
        
        total_predictions = len(results)
        successful_predictions = len([r for r in results if r["predicted"] != "error"])
        correct_predictions = len([r for r in results if r["correct"]])
        
        if successful_predictions > 0:
            accuracy = correct_predictions / successful_predictions
            avg_confidence = sum(r["confidence"] for r in results if r["confidence"] > 0) / successful_predictions
            
            print(f"📊 Summary Statistics:")
            print(f"   Total Images: {total_predictions}")
            print(f"   Successful Predictions: {successful_predictions}")
            print(f"   Accuracy: {accuracy:.1%} ({correct_predictions}/{successful_predictions})")
            print(f"   Average Confidence: {avg_confidence:.1%}")
            
            # Performance assessment
            if accuracy >= 0.8 and avg_confidence >= 0.8:
                assessment = "🟢 Excellent performance"
            elif accuracy >= 0.7 and avg_confidence >= 0.7:
                assessment = "🟡 Good performance"
            else:
                assessment = "🔴 Needs improvement"
            
            print(f"   Assessment: {assessment}")
        
        # Save results
        outputs_dir = project_root / "outputs"
        outputs_dir.mkdir(exist_ok=True)
        
        results_summary = {
            "analysis_type": "single_image_classification",
            "timestamp": datetime.now().isoformat(),
            "model_used": str(model_path) if model_path else "mock_predictions",
            "results": results,
            "summary": {
                "total_images": total_predictions,
                "successful_predictions": successful_predictions,
                "accuracy": accuracy if successful_predictions > 0 else 0,
                "average_confidence": avg_confidence if successful_predictions > 0 else 0
            }
        }
        
        import json
        report_path = outputs_dir / f"classification_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_path, 'w') as f:
            json.dump(results_summary, f, indent=2)
        
        print(f"\n✅ Results saved to: {report_path}")
        
        # Usage tips
        print("\n💡 Usage Tips:")
        print("   • For best results, ensure good lighting and clear image quality")
        print("   • Model performs best on plates, bowls, and cups")
        print("   • Confidence scores above 80% are generally reliable")
        print("   • Consider ensemble predictions for critical applications")
        
        print("\n🎉 Single image classification completed!")
        return True
        
    except Exception as e:
        print(f"❌ Error during classification: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)