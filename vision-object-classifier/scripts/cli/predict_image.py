#!/usr/bin/env python3
"""
CLI script for predicting dish cleanliness from images.
"""

import sys
import os
import argparse

# Add the src directory to the path so we can import our package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from vision_classifier.predict import DishCleanlinessPredictor, predict_single_image


def main():
    parser = argparse.ArgumentParser(description='Predict dish cleanliness from image')
    parser.add_argument('image_path', type=str, help='Path to image file')
    parser.add_argument('--model_path', type=str, default='models/final_balanced_model.pth',
                        help='Path to trained model')
    parser.add_argument('--config_path', type=str, default='models/training_config.json',
                        help='Path to training config (optional)')
    parser.add_argument('--show_confidence', action='store_true',
                        help='Show prediction confidence score')
    parser.add_argument('--show_image', action='store_true',
                        help='Display the image being classified')
    
    args = parser.parse_args()
    
    # Check if image exists
    if not os.path.exists(args.image_path):
        print(f"Error: Image file not found: {args.image_path}")
        return
    
    # Check if model exists
    if not os.path.exists(args.model_path):
        print(f"Error: Model file not found: {args.model_path}")
        print("Please train a model first using train_model.py")
        return
    
    try:
        # Make prediction
        result = predict_single_image(args.image_path, args.model_path, args.config_path)
        
        # Display results
        prediction = "Clean" if result['prediction'] == 0 else "Dirty"
        print(f"Image: {args.image_path}")
        print(f"Prediction: {prediction}")
        
        if args.show_confidence:
            confidence = result['confidence']
            print(f"Confidence: {confidence:.2%}")
            print(f"Probabilities: Clean={result['probabilities'][0]:.3f}, Dirty={result['probabilities'][1]:.3f}")
        
        if args.show_image:
            try:
                import matplotlib.pyplot as plt
                from PIL import Image
                
                img = Image.open(args.image_path)
                plt.figure(figsize=(8, 6))
                plt.imshow(img)
                plt.title(f"Prediction: {prediction}")
                plt.axis('off')
                plt.show()
            except ImportError:
                print("Note: Install matplotlib to display images with --show_image")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)