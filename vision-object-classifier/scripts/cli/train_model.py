#!/usr/bin/env python3
"""
CLI script for training the vision classifier model.
"""

import sys
import os
import argparse
import json

# Add the src directory to the path so we can import our package
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src'))

from vision_classifier.train import DishClassifierTrainer


def main():
    parser = argparse.ArgumentParser(description='Train Vision Object Classifier')
    parser.add_argument('--config', type=str, default='models/training_config.json',
                        help='Path to training configuration file')
    parser.add_argument('--data_dir', type=str, default='data',
                        help='Path to data directory')
    parser.add_argument('--output_dir', type=str, default='models',
                        help='Path to save trained model')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Number of training epochs (overrides config)')
    parser.add_argument('--batch_size', type=int, default=None,
                        help='Batch size (overrides config)')
    parser.add_argument('--learning_rate', type=float, default=None,
                        help='Learning rate (overrides config)')
    
    args = parser.parse_args()
    
    # Load config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        # Default configuration
        config = {
            "model_name": "resnet50",
            "num_classes": 2,
            "pretrained": True,
            "epochs": 50,
            "batch_size": 32,
            "learning_rate": 0.001,
            "optimizer": "adam",
            "weight_decay": 1e-4,
            "use_focal_loss": True,
            "focal_alpha": [1.0, 2.0],
            "focal_gamma": 2.0,
            "patience": 10,
            "min_delta": 0.001,
            "save_best_only": True
        }
    
    # Override config with command line arguments
    if args.epochs is not None:
        config['epochs'] = args.epochs
    if args.batch_size is not None:
        config['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        config['learning_rate'] = args.learning_rate
    
    # Set data directory
    config['data_dir'] = args.data_dir
    config['output_dir'] = args.output_dir
    
    # Create trainer and start training
    trainer = DishClassifierTrainer(config)
    trainer.train()
    
    print(f"Training completed! Model saved in {args.output_dir}")


if __name__ == "__main__":
    main()