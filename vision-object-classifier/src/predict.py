import torch
import cv2
import numpy as np
from PIL import Image
import argparse
import os
import json
from torchvision import transforms
import matplotlib.pyplot as plt

try:
    from .model import load_model, ModelEvaluator
    from .data_utils import get_data_transforms
except ImportError:
    from model import load_model, ModelEvaluator
    from data_utils import get_data_transforms


class DishCleanlinessPredictor:
    def __init__(self, model_path, config_path=None, device=None):
        """
        Initialize the predictor with a trained model
        
        Args:
            model_path: Path to the saved model checkpoint
            config_path: Path to the training config (optional)
            device: Device to run inference on (optional)
        """
        self.device = device if device else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load configuration if available
        self.config = {}
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                self.config = json.load(f)
        
        # Default model parameters
        model_name = self.config.get('model_name', 'resnet50')
        num_classes = self.config.get('num_classes', 2)
        
        # Load model
        print(f"Loading model from {model_path}")
        self.model = load_model(model_path, model_name, num_classes, self.device)
        print(f"Model loaded successfully on {self.device}")
        
        # Setup data transforms
        _, self.transform = get_data_transforms(augment=False)
        
        # Class labels
        self.class_names = ['Clean', 'Dirty']
        
        # Initialize evaluator
        self.evaluator = ModelEvaluator(self.model, self.device)
    
    def preprocess_image(self, image_input):
        """
        Preprocess image for inference
        
        Args:
            image_input: Can be a file path (str), PIL Image, or numpy array
            
        Returns:
            Preprocessed tensor
        """
        if isinstance(image_input, str):
            # Load from file path
            if not os.path.exists(image_input):
                raise FileNotFoundError(f"Image file not found: {image_input}")
            image = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Convert numpy array to PIL
            if image_input.shape[2] == 3:  # BGR to RGB
                image_input = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image_input)
        elif isinstance(image_input, Image.Image):
            # Already PIL Image
            image = image_input.convert('RGB')
        else:
            raise ValueError("Unsupported image input type")
        
        # Apply transforms
        tensor = self.transform(image)
        return tensor
    
    def predict_single(self, image_input, return_image=False):
        """
        Predict cleanliness of a single dish image
        
        Args:
            image_input: Image to classify
            return_image: Whether to return the processed image
            
        Returns:
            Dictionary with prediction results
        """
        # Preprocess image
        image_tensor = self.preprocess_image(image_input)
        
        # Get prediction
        result = self.evaluator.predict_single_image(image_tensor)
        
        # Add class name
        result['class_name'] = self.class_names[result['prediction']]
        result['clean_prob'] = result['probabilities'][0]
        result['dirty_prob'] = result['probabilities'][1]
        
        if return_image:
            # Return original image for visualization
            if isinstance(image_input, str):
                original_image = cv2.imread(image_input)
            elif isinstance(image_input, np.ndarray):
                original_image = image_input.copy()
            else:
                original_image = np.array(image_input)
            
            result['original_image'] = original_image
        
        return result
    
    def predict_batch(self, image_paths, batch_size=32):
        """
        Predict cleanliness for multiple images
        
        Args:
            image_paths: List of image file paths
            batch_size: Batch size for processing
            
        Returns:
            List of prediction results
        """
        results = []
        
        print(f"Processing {len(image_paths)} images...")
        
        for i in range(0, len(image_paths), batch_size):
            batch_paths = image_paths[i:i+batch_size]
            batch_tensors = []
            batch_results = []
            
            # Preprocess batch
            for path in batch_paths:
                try:
                    tensor = self.preprocess_image(path)
                    batch_tensors.append(tensor)
                    batch_results.append({'image_path': path, 'success': True})
                except Exception as e:
                    batch_results.append({'image_path': path, 'success': False, 'error': str(e)})
            
            # Process valid images
            if batch_tensors:
                batch_tensor = torch.stack(batch_tensors).to(self.device)
                
                self.model.eval()
                with torch.no_grad():
                    outputs = self.model(batch_tensor)
                    probabilities = torch.nn.functional.softmax(outputs, dim=1)
                    predictions = torch.argmax(outputs, dim=1)
                
                # Update results
                tensor_idx = 0
                for j, result in enumerate(batch_results):
                    if result['success']:
                        result.update({
                            'prediction': predictions[tensor_idx].item(),
                            'class_name': self.class_names[predictions[tensor_idx].item()],
                            'clean_prob': probabilities[tensor_idx][0].item(),
                            'dirty_prob': probabilities[tensor_idx][1].item(),
                            'confidence': probabilities[tensor_idx].max().item()
                        })
                        tensor_idx += 1
            
            results.extend(batch_results)
            
            print(f"Processed {min(i + batch_size, len(image_paths))}/{len(image_paths)} images")
        
        return results
    
    def visualize_prediction(self, image_input, save_path=None):
        """
        Visualize prediction with image and results
        
        Args:
            image_input: Image to classify
            save_path: Path to save visualization (optional)
        """
        result = self.predict_single(image_input, return_image=True)
        
        # Setup plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Display original image
        if 'original_image' in result:
            img = result['original_image']
            if len(img.shape) == 3 and img.shape[2] == 3:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            ax1.imshow(img)
        ax1.set_title('Input Image')
        ax1.axis('off')
        
        # Display prediction results
        classes = ['Clean', 'Dirty']
        probabilities = [result['clean_prob'], result['dirty_prob']]
        colors = ['green' if result['prediction'] == 0 else 'red', 
                 'red' if result['prediction'] == 1 else 'green']
        
        bars = ax2.bar(classes, probabilities, color=colors, alpha=0.7)
        ax2.set_ylim(0, 1)
        ax2.set_ylabel('Probability')
        ax2.set_title(f'Prediction: {result["class_name"]} '
                     f'(Confidence: {result["confidence"]:.3f})')
        
        # Add probability labels on bars
        for bar, prob in zip(bars, probabilities):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{prob:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
        
        return result


def main():
    parser = argparse.ArgumentParser(description='Dish Cleanliness Prediction')
    parser.add_argument('--model', required=True, help='Path to trained model')
    parser.add_argument('--config', help='Path to training config file')
    parser.add_argument('--image', help='Single image to classify')
    parser.add_argument('--image_dir', help='Directory of images to classify')
    parser.add_argument('--output', help='Output directory for results')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size for multiple images')
    parser.add_argument('--visualize', action='store_true', help='Show visualization')
    parser.add_argument('--device', help='Device to use (cuda/cpu)')
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = DishCleanlinessPredictor(
        model_path=args.model,
        config_path=args.config,
        device=args.device
    )
    
    # Create output directory if specified
    if args.output:
        os.makedirs(args.output, exist_ok=True)
    
    if args.image:
        # Single image prediction
        print(f"Classifying single image: {args.image}")
        
        if args.visualize:
            result = predictor.visualize_prediction(args.image, 
                                                  save_path=os.path.join(args.output, 'prediction_viz.png') if args.output else None)
        else:
            result = predictor.predict_single(args.image)
        
        print(f"\nPrediction Results:")
        print(f"Class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.4f}")
        print(f"Clean Probability: {result['clean_prob']:.4f}")
        print(f"Dirty Probability: {result['dirty_prob']:.4f}")
        
        # Save results if output directory specified
        if args.output:
            result_file = os.path.join(args.output, 'single_prediction.json')
            with open(result_file, 'w') as f:
                json.dump({k: v for k, v in result.items() if k != 'original_image'}, f, indent=2)
            print(f"Results saved to: {result_file}")
    
    elif args.image_dir:
        # Batch prediction
        if not os.path.exists(args.image_dir):
            print(f"Error: Image directory {args.image_dir} not found")
            return
        
        # Get all image files
        image_extensions = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
        image_paths = []
        
        for filename in os.listdir(args.image_dir):
            if filename.lower().endswith(image_extensions):
                image_paths.append(os.path.join(args.image_dir, filename))
        
        if not image_paths:
            print(f"No image files found in {args.image_dir}")
            return
        
        print(f"Found {len(image_paths)} images to classify")
        
        # Run batch prediction
        results = predictor.predict_batch(image_paths, batch_size=args.batch_size)
        
        # Print summary
        successful = [r for r in results if r.get('success', False)]
        clean_count = len([r for r in successful if r.get('prediction') == 0])
        dirty_count = len([r for r in successful if r.get('prediction') == 1])
        
        print(f"\nBatch Prediction Summary:")
        print(f"Total images processed: {len(successful)}")
        print(f"Clean dishes: {clean_count}")
        print(f"Dirty dishes: {dirty_count}")
        print(f"Failed predictions: {len(results) - len(successful)}")
        
        # Save detailed results
        if args.output:
            results_file = os.path.join(args.output, 'batch_predictions.json')
            with open(results_file, 'w') as f:
                json.dump(results, f, indent=2)
            print(f"Detailed results saved to: {results_file}")
            
            # Create summary CSV
            import csv
            csv_file = os.path.join(args.output, 'predictions_summary.csv')
            with open(csv_file, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Image Path', 'Prediction', 'Confidence', 'Clean Prob', 'Dirty Prob'])
                
                for result in successful:
                    writer.writerow([
                        result['image_path'],
                        result['class_name'],
                        f"{result['confidence']:.4f}",
                        f"{result['clean_prob']:.4f}",
                        f"{result['dirty_prob']:.4f}"
                    ])
            
            print(f"Summary CSV saved to: {csv_file}")
    
    else:
        print("Error: Must specify either --image or --image_dir")
        parser.print_help()


if __name__ == "__main__":
    main()