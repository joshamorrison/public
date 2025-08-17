# Training Guide & Optimization Strategies

Complete guide to training the Vision Object Classifier with advanced optimization techniques for various computational constraints.

## Training Overview

The Vision Object Classifier uses sophisticated training strategies to overcome data limitations and achieve high accuracy with efficient resource usage. This guide covers everything from basic training to advanced optimization techniques.

## Quick Start Training

### Basic Training Pipeline
```bash
# 1. Prepare your dataset
python -c "
from src.vision_classifier.synthetic_data import SyntheticDirtyDishGenerator
generator = SyntheticDirtyDishGenerator('data/dirty')
generator.batch_generate('data/clean', num_variations=2)
"

# 2. Train with default settings (recommended for beginners)
python src/train.py

# 3. Validate your model
python src/predict.py --model models/final_balanced_model.pth --config models/balanced_config.json --image data/clean/plate_01.jpg
```

### Expected Training Output
```
Epoch 1/10: Train Loss: 0.6234, Train Acc: 65.5%, Val Loss: 0.5123, Val Acc: 75.0%
Epoch 2/10: Train Loss: 0.4456, Train Acc: 78.2%, Val Loss: 0.3891, Val Acc: 82.5%
Epoch 3/10: Train Loss: 0.3234, Train Acc: 85.1%, Val Loss: 0.2945, Val Acc: 87.5%
...
Training completed! Best validation accuracy: 90.0%
Model saved: models/final_balanced_model.pth
```

## Dataset Preparation Strategies

### 1. Balanced Dataset Creation
**Optimal for general training**

```python
def create_balanced_dataset(clean_dir, dirty_dir, target_ratio=2):
    """Create balanced dataset with specified clean:dirty ratio."""
    
    # Count existing images
    clean_count = len([f for f in os.listdir(clean_dir) if f.endswith(('.jpg', '.png'))])
    dirty_count = len([f for f in os.listdir(dirty_dir) if f.endswith(('.jpg', '.png'))])
    
    target_dirty = clean_count * target_ratio
    
    if dirty_count < target_dirty:
        # Generate additional synthetic dirty images
        generator = SyntheticDirtyDishGenerator(dirty_dir)
        needed = target_dirty - dirty_count
        variations_per_clean = math.ceil(needed / clean_count)
        
        generator.batch_generate(
            clean_images_dir=clean_dir,
            num_variations=variations_per_clean,
            intensity_distribution={
                'light': 0.3,
                'medium': 0.5,
                'heavy': 0.2
            }
        )
    
    print(f"Dataset balanced: {clean_count} clean, {target_dirty} dirty images")
    return clean_count, target_dirty
```

### 2. Stratified Data Splitting
```python
def create_stratified_splits(data_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1):
    """Create stratified train/validation/test splits."""
    
    # Collect all images with labels
    all_images = []
    for class_name in ['clean', 'dirty']:
        class_dir = os.path.join(data_dir, class_name)
        for img_file in os.listdir(class_dir):
            if img_file.endswith(('.jpg', '.png')):
                all_images.append((os.path.join(class_dir, img_file), class_name))
    
    # Separate by class
    clean_images = [img for img, label in all_images if label == 'clean']
    dirty_images = [img for img, label in all_images if label == 'dirty']
    
    # Create splits for each class
    splits = {}
    for class_name, images in [('clean', clean_images), ('dirty', dirty_images)]:
        random.shuffle(images)
        
        n_train = int(len(images) * train_ratio)
        n_val = int(len(images) * val_ratio)
        
        splits[f'{class_name}_train'] = images[:n_train]
        splits[f'{class_name}_val'] = images[n_train:n_train+n_val]
        splits[f'{class_name}_test'] = images[n_train+n_val:]
    
    return splits
```

## Training Configuration

### Configuration File Structure
```json
{
    "model": {
        "architecture": "custom_cnn",
        "num_classes": 2,
        "input_size": [224, 224],
        "pretrained": false
    },
    "training": {
        "epochs": 10,
        "batch_size": 16,
        "learning_rate": 0.001,
        "weight_decay": 1e-4,
        "optimizer": "adam",
        "scheduler": "step",
        "step_size": 5,
        "gamma": 0.1
    },
    "data": {
        "train_ratio": 0.8,
        "val_ratio": 0.1,
        "test_ratio": 0.1,
        "augmentation": true,
        "normalize": true
    },
    "class_weighting": {
        "enabled": true,
        "strategy": "inverse_frequency"
    },
    "early_stopping": {
        "enabled": true,
        "patience": 5,
        "min_delta": 0.001
    },
    "checkpointing": {
        "save_best": true,
        "save_last": true,
        "monitor": "val_accuracy"
    }
}
```

### Model Architecture Selection
```python
def select_model_architecture(config, computational_constraint='medium'):
    """Select optimal model architecture based on constraints."""
    
    architectures = {
        'low': {
            'model': 'mobilenet_v3',
            'batch_size': 8,
            'epochs': 15,
            'description': 'Lightweight for CPU-only training'
        },
        'medium': {
            'model': 'custom_cnn',
            'batch_size': 16,
            'epochs': 10,
            'description': 'Balanced performance and speed'
        },
        'high': {
            'model': 'resnet50',
            'batch_size': 32,
            'epochs': 20,
            'description': 'Maximum accuracy with GPU'
        }
    }
    
    selected = architectures[computational_constraint]
    print(f"Selected: {selected['model']} - {selected['description']}")
    
    return selected
```

## Advanced Training Techniques

### 1. Class-Weighted Loss Function
```python
class BalancedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights=None, focal_alpha=1.0, focal_gamma=2.0, 
                 use_focal=False):
        super().__init__()
        self.class_weights = class_weights
        self.use_focal = use_focal
        self.focal_alpha = focal_alpha
        self.focal_gamma = focal_gamma
    
    def forward(self, predictions, targets):
        # Standard cross-entropy with class weights
        ce_loss = F.cross_entropy(predictions, targets, weight=self.class_weights)
        
        if self.use_focal:
            # Apply focal loss for hard example mining
            pt = torch.exp(-ce_loss)
            focal_loss = self.focal_alpha * (1-pt)**self.focal_gamma * ce_loss
            return focal_loss
        
        return ce_loss

# Usage
def get_class_weights(dataset):
    """Calculate class weights for imbalanced datasets."""
    from collections import Counter
    
    labels = [dataset[i][1] for i in range(len(dataset))]
    class_counts = Counter(labels)
    total_samples = len(labels)
    
    weights = {
        cls: total_samples / (len(class_counts) * count)
        for cls, count in class_counts.items()
    }
    
    return torch.FloatTensor([weights[i] for i in sorted(weights.keys())])
```

### 2. Learning Rate Scheduling
```python
class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, base_lr, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.base_lr = base_lr
        self.min_lr = min_lr
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * epoch / self.warmup_epochs
        else:
            # Cosine annealing
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr
```

### 3. Data Augmentation Pipeline
```python
def get_advanced_transforms(image_size=224, training=True):
    """Get advanced data augmentation pipeline."""
    
    if training:
        return transforms.Compose([
            transforms.Resize((int(image_size * 1.1), int(image_size * 1.1))),
            transforms.RandomCrop(image_size),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=15),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.1
            ),
            transforms.RandomApply([
                transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
            ], p=0.2),
            transforms.RandomApply([
                transforms.RandomAffine(
                    degrees=0,
                    translate=(0.1, 0.1),
                    scale=(0.9, 1.1),
                    shear=5
                )
            ], p=0.3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            ),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1))
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])
```

## Training Strategies by Computational Constraint

### Strategy 1: CPU-Only Training (Low Resources)
```python
def cpu_optimized_training(config):
    """Optimized training configuration for CPU-only systems."""
    
    optimized_config = config.copy()
    optimized_config.update({
        'model_architecture': 'mobilenet_v3',
        'batch_size': 8,
        'num_workers': 2,
        'epochs': 15,
        'mixed_precision': False,
        'gradient_accumulation_steps': 4,  # Simulate larger batch
        'max_train_batches': 50,  # Limit for faster epochs
        'early_stopping_patience': 3
    })
    
    return optimized_config

def train_cpu_optimized(model, train_loader, val_loader, config):
    """CPU-optimized training loop with gradient accumulation."""
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config['learning_rate'])
    criterion = nn.CrossEntropyLoss()
    
    accumulation_steps = config['gradient_accumulation_steps']
    
    for epoch in range(config['epochs']):
        model.train()
        running_loss = 0.0
        
        optimizer.zero_grad()
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            # Forward pass
            outputs = model(data)
            loss = criterion(outputs, targets)
            
            # Scale loss for gradient accumulation
            loss = loss / accumulation_steps
            loss.backward()
            
            # Update weights every accumulation_steps
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
            
            running_loss += loss.item() * accumulation_steps
            
            # Early break for fast iteration
            if batch_idx >= config.get('max_train_batches', float('inf')):
                break
        
        # Validation
        val_accuracy = evaluate_model(model, val_loader)
        print(f"Epoch {epoch+1}: Loss: {running_loss/len(train_loader):.4f}, "
              f"Val Accuracy: {val_accuracy:.4f}")
```

### Strategy 2: Balanced Training (Medium Resources)
```python
def balanced_training_strategy(config):
    """Balanced training for systems with moderate GPU/CPU power."""
    
    return {
        'model_architecture': 'custom_cnn',
        'batch_size': 16,
        'num_workers': 4,
        'epochs': 10,
        'mixed_precision': True,
        'lr_scheduler': 'cosine',
        'data_augmentation': 'advanced',
        'class_weighting': True,
        'early_stopping': True
    }

class MixedPrecisionTrainer:
    def __init__(self, model, optimizer, device):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.scaler = torch.cuda.amp.GradScaler() if device.type == 'cuda' else None
    
    def train_epoch(self, train_loader, criterion):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        
        for data, targets in train_loader:
            data, targets = data.to(self.device), targets.to(self.device)
            
            self.optimizer.zero_grad()
            
            if self.scaler:
                # Mixed precision training
                with torch.cuda.amp.autocast():
                    outputs = self.model(data)
                    loss = criterion(outputs, targets)
                
                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                # Standard training
                outputs = self.model(data)
                loss = criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()
            
            total_loss += loss.item()
            predicted = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
        
        return total_loss / len(train_loader), 100.0 * correct / total
```

### Strategy 3: High-Performance Training (GPU Clusters)
```python
def high_performance_training_config():
    """Configuration for high-end GPU training."""
    
    return {
        'model_architecture': 'resnet50',
        'batch_size': 64,
        'num_workers': 8,
        'epochs': 50,
        'mixed_precision': True,
        'gradient_clipping': True,
        'lr_scheduler': 'warmup_cosine',
        'data_augmentation': 'heavy',
        'test_time_augmentation': True,
        'ensemble_training': True
    }

class DistributedTrainer:
    """Trainer for multi-GPU distributed training."""
    
    def __init__(self, model, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        
        # Wrap model for distributed training
        self.model = nn.parallel.DistributedDataParallel(
            model.to(rank),
            device_ids=[rank]
        )
    
    def setup_distributed_training(self):
        """Initialize distributed training environment."""
        dist.init_process_group(
            backend='nccl',
            init_method='env://',
            world_size=self.world_size,
            rank=self.rank
        )
    
    def cleanup_distributed_training(self):
        """Clean up distributed training."""
        dist.destroy_process_group()
```

## Training Monitoring & Debugging

### Real-Time Training Monitoring
```python
class TrainingMonitor:
    def __init__(self, log_dir='logs', use_tensorboard=True):
        self.log_dir = log_dir
        self.use_tensorboard = use_tensorboard
        
        if use_tensorboard:
            from torch.utils.tensorboard import SummaryWriter
            self.writer = SummaryWriter(log_dir)
        
        self.metrics_history = {
            'train_loss': [],
            'train_accuracy': [],
            'val_loss': [],
            'val_accuracy': [],
            'learning_rate': []
        }
    
    def log_epoch(self, epoch, train_metrics, val_metrics, lr):
        """Log metrics for one epoch."""
        
        # Store in history
        self.metrics_history['train_loss'].append(train_metrics['loss'])
        self.metrics_history['train_accuracy'].append(train_metrics['accuracy'])
        self.metrics_history['val_loss'].append(val_metrics['loss'])
        self.metrics_history['val_accuracy'].append(val_metrics['accuracy'])
        self.metrics_history['learning_rate'].append(lr)
        
        # Log to tensorboard
        if self.use_tensorboard:
            self.writer.add_scalar('Loss/Train', train_metrics['loss'], epoch)
            self.writer.add_scalar('Loss/Validation', val_metrics['loss'], epoch)
            self.writer.add_scalar('Accuracy/Train', train_metrics['accuracy'], epoch)
            self.writer.add_scalar('Accuracy/Validation', val_metrics['accuracy'], epoch)
            self.writer.add_scalar('Learning_Rate', lr, epoch)
        
        # Console output
        print(f"Epoch {epoch+1}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.2f}%")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.2f}%")
        print(f"  LR: {lr:.6f}")
    
    def plot_training_history(self, save_path=None):
        """Plot training history."""
        import matplotlib.pyplot as plt
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))
        
        epochs = range(1, len(self.metrics_history['train_loss']) + 1)
        
        # Loss plot
        ax1.plot(epochs, self.metrics_history['train_loss'], 'b-', label='Train Loss')
        ax1.plot(epochs, self.metrics_history['val_loss'], 'r-', label='Val Loss')
        ax1.set_title('Training and Validation Loss')
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.legend()
        ax1.grid(True)
        
        # Accuracy plot
        ax2.plot(epochs, self.metrics_history['train_accuracy'], 'b-', label='Train Acc')
        ax2.plot(epochs, self.metrics_history['val_accuracy'], 'r-', label='Val Acc')
        ax2.set_title('Training and Validation Accuracy')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Accuracy (%)')
        ax2.legend()
        ax2.grid(True)
        
        # Learning rate plot
        ax3.plot(epochs, self.metrics_history['learning_rate'], 'g-')
        ax3.set_title('Learning Rate Schedule')
        ax3.set_xlabel('Epoch')
        ax3.set_ylabel('Learning Rate')
        ax3.grid(True)
        
        # Performance metrics
        best_val_acc = max(self.metrics_history['val_accuracy'])
        best_epoch = self.metrics_history['val_accuracy'].index(best_val_acc) + 1
        
        ax4.text(0.1, 0.8, f'Best Val Accuracy: {best_val_acc:.2f}%', fontsize=14)
        ax4.text(0.1, 0.6, f'Best Epoch: {best_epoch}', fontsize=14)
        ax4.text(0.1, 0.4, f'Final Train Acc: {self.metrics_history["train_accuracy"][-1]:.2f}%', fontsize=14)
        ax4.text(0.1, 0.2, f'Total Epochs: {len(epochs)}', fontsize=14)
        ax4.set_title('Training Summary')
        ax4.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        
        plt.show()
```

### Model Debugging Tools
```python
def debug_model_training(model, train_loader, device):
    """Debug common training issues."""
    
    print("=== MODEL DEBUGGING ===")
    
    # 1. Check model forward pass
    model.eval()
    sample_batch = next(iter(train_loader))
    sample_input, sample_targets = sample_batch[0][:2], sample_batch[1][:2]
    
    try:
        with torch.no_grad():
            sample_input = sample_input.to(device)
            outputs = model(sample_input)
            print(f"✓ Forward pass successful")
            print(f"  Input shape: {sample_input.shape}")
            print(f"  Output shape: {outputs.shape}")
            print(f"  Output range: [{outputs.min().item():.3f}, {outputs.max().item():.3f}]")
    except Exception as e:
        print(f"✗ Forward pass failed: {e}")
        return False
    
    # 2. Check gradient flow
    model.train()
    criterion = nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    optimizer.zero_grad()
    
    outputs = model(sample_input)
    loss = criterion(outputs, sample_targets.to(device))
    loss.backward()
    
    # Check for gradient issues
    total_grad_norm = 0
    param_count = 0
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_norm = param.grad.data.norm(2)
            total_grad_norm += grad_norm.item() ** 2
            param_count += 1
        else:
            print(f"  Warning: No gradient for {name}")
    
    total_grad_norm = total_grad_norm ** (1. / 2)
    
    print(f"✓ Gradient computation successful")
    print(f"  Total gradient norm: {total_grad_norm:.6f}")
    print(f"  Parameters with gradients: {param_count}")
    
    if total_grad_norm < 1e-6:
        print("  Warning: Very small gradients - check learning rate")
    elif total_grad_norm > 100:
        print("  Warning: Large gradients - consider gradient clipping")
    
    # 3. Check data distribution
    all_targets = []
    for batch in train_loader:
        all_targets.extend(batch[1].tolist())
        if len(all_targets) > 1000:  # Sample check
            break
    
    from collections import Counter
    class_dist = Counter(all_targets)
    print(f"✓ Data distribution check")
    for class_id, count in class_dist.items():
        print(f"  Class {class_id}: {count} samples ({count/len(all_targets)*100:.1f}%)")
    
    return True
```

## Hyperparameter Optimization

### Grid Search for Optimal Parameters
```python
def hyperparameter_grid_search(train_loader, val_loader, device):
    """Perform grid search for optimal hyperparameters."""
    
    param_grid = {
        'learning_rate': [0.001, 0.003, 0.01],
        'batch_size': [8, 16, 32],
        'weight_decay': [1e-4, 1e-3, 1e-2],
        'optimizer': ['adam', 'sgd'],
        'scheduler': ['step', 'cosine']
    }
    
    best_score = 0
    best_params = None
    results = []
    
    # Generate all combinations
    from itertools import product
    param_combinations = list(product(*param_grid.values()))
    
    for i, params in enumerate(param_combinations):
        param_dict = dict(zip(param_grid.keys(), params))
        
        print(f"Testing combination {i+1}/{len(param_combinations)}: {param_dict}")
        
        # Quick training run
        score = quick_train_eval(train_loader, val_loader, param_dict, device)
        
        results.append({
            'params': param_dict,
            'score': score
        })
        
        if score > best_score:
            best_score = score
            best_params = param_dict
        
        print(f"  Score: {score:.4f}")
    
    print(f"\nBest parameters: {best_params}")
    print(f"Best score: {best_score:.4f}")
    
    return best_params, results

def quick_train_eval(train_loader, val_loader, params, device, epochs=3):
    """Quick training for hyperparameter evaluation."""
    from src.vision_classifier.model import CustomCNN
    
    model = CustomCNN(num_classes=2).to(device)
    
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay']
        )
    else:
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=params['learning_rate'],
            weight_decay=params['weight_decay'],
            momentum=0.9
        )
    
    criterion = nn.CrossEntropyLoss()
    
    # Quick training
    for epoch in range(epochs):
        model.train()
        for batch_idx, (data, targets) in enumerate(train_loader):
            if batch_idx > 10:  # Limit batches for speed
                break
                
            data, targets = data.to(device), targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
    
    # Evaluate
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for data, targets in val_loader:
            data, targets = data.to(device), targets.to(device)
            outputs = model(data)
            predicted = outputs.argmax(dim=1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    return 100.0 * correct / total
```

## Troubleshooting Common Issues

### Issue 1: Low Training Accuracy
```python
def diagnose_low_accuracy(model, train_loader, val_loader, device):
    """Diagnose and suggest fixes for low training accuracy."""
    
    print("=== DIAGNOSING LOW ACCURACY ===")
    
    # Check 1: Data loading
    sample_batch = next(iter(train_loader))
    print(f"Batch size: {sample_batch[0].shape[0]}")
    print(f"Input shape: {sample_batch[0].shape}")
    print(f"Target distribution: {torch.bincount(sample_batch[1])}")
    
    # Check 2: Model complexity
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Check 3: Learning rate
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    
    # Suggestions
    suggestions = []
    
    if total_params < 100000:
        suggestions.append("Model may be too simple - try ResNet50 or increase CNN layers")
    
    if torch.bincount(sample_batch[1]).min() < 2:
        suggestions.append("Severe class imbalance - implement class weighting")
    
    print("\nSuggestions:")
    for i, suggestion in enumerate(suggestions, 1):
        print(f"{i}. {suggestion}")
```

### Issue 2: Overfitting
```python
def prevent_overfitting(config):
    """Configuration adjustments to prevent overfitting."""
    
    return {
        'dropout_rate': 0.5,  # Increase dropout
        'weight_decay': 1e-3,  # Increase regularization
        'data_augmentation': 'heavy',  # More augmentation
        'early_stopping_patience': 3,  # Earlier stopping
        'batch_size': 8,  # Smaller batches
        'learning_rate': 0.0001,  # Lower learning rate
        'label_smoothing': 0.1,  # Label smoothing
        'mixup_alpha': 0.2  # Mixup augmentation
    }
```

This comprehensive training guide provides strategies for achieving optimal performance across different computational constraints while addressing common training challenges in computer vision.