import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import os
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import json
from datetime import datetime

from .model import create_model, save_model, FocalLoss, ModelEvaluator
from .data_utils import create_data_loaders, prepare_dataset_structure
from .synthetic_data import SyntheticDirtyDishGenerator


class DishClassifierTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Using device: {self.device}")
        
        # Create model
        self.model = create_model(
            model_name=config['model_name'],
            num_classes=config['num_classes'],
            pretrained=config['pretrained']
        ).to(self.device)
        
        # Setup loss function
        if config['use_focal_loss']:
            self.criterion = FocalLoss(alpha=config['focal_alpha'], gamma=config['focal_gamma'])
        else:
            # Use class weights if provided
            if config.get('class_weights'):
                weights = torch.FloatTensor(config['class_weights']).to(self.device)
                self.criterion = nn.CrossEntropyLoss(weight=weights)
            else:
                self.criterion = nn.CrossEntropyLoss()
        
        # Setup optimizer
        if config['optimizer'] == 'adam':
            self.optimizer = optim.Adam(self.model.parameters(), lr=config['learning_rate'])
        elif config['optimizer'] == 'sgd':
            self.optimizer = optim.SGD(self.model.parameters(), lr=config['learning_rate'], 
                                     momentum=0.9, weight_decay=config['weight_decay'])
        else:
            raise ValueError(f"Unsupported optimizer: {config['optimizer']}")
        
        # Learning rate scheduler
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, 
                                                 step_size=config['lr_step_size'], 
                                                 gamma=config['lr_gamma'])
        
        # Training history
        self.train_losses = []
        self.train_accuracies = []
        self.val_losses = []
        self.val_accuracies = []
        
        # Model evaluator
        self.evaluator = ModelEvaluator(self.model, self.device)
    
    def train_epoch(self, train_loader):
        """Train for one epoch"""
        self.model.train()
        running_loss = 0.0
        correct_predictions = 0
        total_samples = 0
        
        progress_bar = tqdm(train_loader, desc="Training")
        
        for batch_idx, (inputs, targets) in enumerate(progress_bar):
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            
            # Zero gradients
            self.optimizer.zero_grad()
            
            # Forward pass
            outputs = self.model(inputs)
            loss = self.criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            self.optimizer.step()
            
            # Statistics
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += targets.size(0)
            correct_predictions += (predicted == targets).sum().item()
            
            # Update progress bar
            accuracy = correct_predictions / total_samples
            progress_bar.set_postfix({
                'loss': running_loss / (batch_idx + 1),
                'acc': f"{accuracy:.4f}"
            })
        
        epoch_loss = running_loss / len(train_loader)
        epoch_accuracy = correct_predictions / total_samples
        
        return epoch_loss, epoch_accuracy
    
    def validate(self, val_loader):
        """Validate the model"""
        results = self.evaluator.evaluate(val_loader)
        return results['loss'], results['accuracy']
    
    def train(self, train_loader, val_loader, num_epochs):
        """Main training loop"""
        best_val_accuracy = 0.0
        best_model_path = None
        
        print(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print("-" * 50)
            
            # Train
            train_loss, train_acc = self.train_epoch(train_loader)
            
            # Validate
            val_loss, val_acc = self.validate(val_loader)
            
            # Update learning rate
            self.scheduler.step()
            
            # Store history
            self.train_losses.append(train_loss)
            self.train_accuracies.append(train_acc)
            self.val_losses.append(val_loss)
            self.val_accuracies.append(val_acc)
            
            print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
            print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
            print(f"Learning Rate: {self.scheduler.get_last_lr()[0]:.6f}")
            
            # Save best model
            if val_acc > best_val_accuracy:
                best_val_accuracy = val_acc
                best_model_path = os.path.join(self.config['save_dir'], 
                                             f"best_model_epoch_{epoch+1}.pth")
                save_model(self.model, self.optimizer, epoch, val_loss, val_acc, best_model_path)
                print(f"New best model saved: {best_model_path}")
            
            # Save checkpoint every few epochs
            if (epoch + 1) % self.config['save_frequency'] == 0:
                checkpoint_path = os.path.join(self.config['save_dir'], 
                                             f"checkpoint_epoch_{epoch+1}.pth")
                save_model(self.model, self.optimizer, epoch, val_loss, val_acc, checkpoint_path)
        
        print(f"\nTraining completed! Best validation accuracy: {best_val_accuracy:.4f}")
        return best_model_path
    
    def plot_training_history(self, save_path=None):
        """Plot training history"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
        
        # Plot losses
        ax1.plot(self.train_losses, label='Train Loss')
        ax1.plot(self.val_losses, label='Validation Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Plot accuracies
        ax2.plot(self.train_accuracies, label='Train Accuracy')
        ax2.plot(self.val_accuracies, label='Validation Accuracy')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
    
    def save_training_config(self, filepath):
        """Save training configuration"""
        config_copy = self.config.copy()
        config_copy['device'] = str(self.device)
        config_copy['timestamp'] = datetime.now().isoformat()
        
        with open(filepath, 'w') as f:
            json.dump(config_copy, f, indent=2)


def generate_synthetic_data(clean_data_dir, synthetic_output_dir, num_variations=3):
    """Generate synthetic dirty dishes from clean images"""
    print("Generating synthetic dirty dish data...")
    
    generator = SyntheticDirtyDishGenerator(synthetic_output_dir)
    
    if os.path.exists(clean_data_dir):
        total_generated = generator.batch_generate(clean_data_dir, num_variations)
        print(f"Generated {total_generated} synthetic dirty images")
        return total_generated
    else:
        print(f"Clean data directory {clean_data_dir} not found")
        return 0


def main():
    # Training configuration
    config = {
        'model_name': 'resnet50',  # 'resnet50', 'efficientnet_b0', 'mobilenet_v3', 'custom_cnn'
        'num_classes': 2,
        'pretrained': True,
        'batch_size': 32,
        'learning_rate': 0.001,
        'num_epochs': 50,
        'optimizer': 'adam',  # 'adam' or 'sgd'
        'weight_decay': 1e-4,
        'lr_step_size': 15,
        'lr_gamma': 0.1,
        'use_focal_loss': False,
        'focal_alpha': 1.0,
        'focal_gamma': 2.0,
        'class_weights': None,  # [1.0, 1.0] for balanced
        'test_split': 0.2,
        'save_frequency': 10,
        'data_dir': '../data',
        'save_dir': '../models',
        'generate_synthetic': True,
        'synthetic_variations': 5
    }
    
    # Create directories
    os.makedirs(config['save_dir'], exist_ok=True)
    prepare_dataset_structure(config['data_dir'])
    
    # Generate synthetic data if requested
    if config['generate_synthetic']:
        clean_dir = os.path.join(config['data_dir'], 'clean')
        synthetic_dir = os.path.join(config['data_dir'], 'dirty')
        
        if os.path.exists(clean_dir) and len(os.listdir(clean_dir)) > 0:
            generate_synthetic_data(clean_dir, synthetic_dir, config['synthetic_variations'])
        else:
            print("Warning: No clean images found for synthetic generation")
    
    # Check if we have data
    clean_dir = os.path.join(config['data_dir'], 'clean')
    dirty_dir = os.path.join(config['data_dir'], 'dirty')
    
    if not (os.path.exists(clean_dir) and os.path.exists(dirty_dir)):
        print("Error: Clean and dirty data directories not found!")
        print("Please place clean dish images in data/clean/ and dirty images in data/dirty/")
        return
    
    clean_count = len([f for f in os.listdir(clean_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    dirty_count = len([f for f in os.listdir(dirty_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    
    print(f"Found {clean_count} clean images and {dirty_count} dirty images")
    
    if clean_count == 0 or dirty_count == 0:
        print("Error: Need both clean and dirty images to train!")
        return
    
    # Create data loaders
    try:
        train_loader, val_loader = create_data_loaders(
            config['data_dir'], 
            batch_size=config['batch_size'],
            test_split=config['test_split']
        )
        
        print(f"Training samples: {len(train_loader.dataset)}")
        print(f"Validation samples: {len(val_loader.dataset)}")
        
    except Exception as e:
        print(f"Error creating data loaders: {e}")
        return
    
    # Initialize trainer
    trainer = DishClassifierTrainer(config)
    
    # Save configuration
    config_path = os.path.join(config['save_dir'], 'training_config.json')
    trainer.save_training_config(config_path)
    
    # Train model
    try:
        best_model_path = trainer.train(train_loader, val_loader, config['num_epochs'])
        
        # Plot and save training history
        history_plot_path = os.path.join(config['save_dir'], 'training_history.png')
        trainer.plot_training_history(history_plot_path)
        
        # Final evaluation
        print("\nFinal model evaluation:")
        results = trainer.evaluator.evaluate(val_loader)
        
        print(f"Final Validation Accuracy: {results['accuracy']:.4f}")
        print(f"Final Validation Loss: {results['loss']:.4f}")
        
        # Confusion matrix
        cm = confusion_matrix(results['targets'], results['predictions'])
        print(f"\nConfusion Matrix:")
        print(cm)
        
        # Classification report
        print(f"\nClassification Report:")
        print(classification_report(results['targets'], results['predictions'], 
                                  target_names=['Clean', 'Dirty']))
        
        print(f"\nBest model saved at: {best_model_path}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        raise


if __name__ == "__main__":
    main()