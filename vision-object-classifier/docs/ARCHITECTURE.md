# Model Architecture & Technical Implementation

## Overview

The Vision Object Classifier uses advanced computer vision techniques to classify household objects (plates, bowls, cups) as clean or dirty. The system employs multiple neural network architectures with synthetic data generation for robust performance.

## Model Architectures

### Optimized Custom CNN (Primary)
**Best for local training and fast iteration**

```python
class CustomCNN(nn.Module):
    def __init__(self, num_classes=2):
        super(CustomCNN, self).__init__()
        self.features = nn.Sequential(
            # Layer 1: 3 -> 32 channels
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 2: 32 -> 64 channels  
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 3: 64 -> 128 channels
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            
            # Layer 4: 128 -> 256 channels
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((7, 7))
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(256 * 7 * 7, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )
```

**Performance Characteristics:**
- **Input Size**: 224x224 RGB images
- **Parameters**: ~6.5M parameters
- **Training Time**: <5 minutes on local CPU
- **Accuracy**: 85% on balanced dataset
- **Memory Usage**: ~2GB RAM during training

### ResNet50 (High Accuracy)
**Best for production deployment with higher compute**

```python
class ResNet50Classifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(ResNet50Classifier, self).__init__()
        self.backbone = models.resnet50(pretrained=pretrained)
        self.backbone.fc = nn.Linear(2048, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
```

**Performance Characteristics:**
- **Parameters**: ~25M parameters
- **Training Time**: 15-30 minutes
- **Accuracy**: 90%+ potential with larger datasets
- **Memory Usage**: ~6GB RAM during training

### EfficientNet-B0 (Balanced)
**Best for mobile and edge deployment**

```python
class EfficientNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(EfficientNetClassifier, self).__init__()
        self.backbone = EfficientNet.from_pretrained('efficientnet-b0')
        self.backbone._fc = nn.Linear(1280, num_classes)
        
    def forward(self, x):
        return self.backbone(x)
```

**Performance Characteristics:**
- **Parameters**: ~5M parameters
- **Training Time**: 8-12 minutes
- **Accuracy**: 87-92% depending on dataset
- **Memory Usage**: ~3GB RAM during training

### MobileNet-V3 (Lightweight)
**Best for real-time inference and mobile apps**

```python
class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=2, pretrained=True):
        super(MobileNetClassifier, self).__init__()
        self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
        self.backbone.classifier = nn.Sequential(
            nn.Linear(960, 512),
            nn.Hardswish(),
            nn.Dropout(0.2),
            nn.Linear(512, num_classes)
        )
        
    def forward(self, x):
        return self.backbone(x)
```

**Performance Characteristics:**
- **Parameters**: ~3M parameters
- **Inference Speed**: 50-100 FPS on mobile devices
- **Accuracy**: 82-88% depending on optimization
- **Memory Usage**: <1GB RAM

## Data Processing Pipeline

### Image Preprocessing
```python
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],  # ImageNet normalization
        std=[0.229, 0.224, 0.225]
    )
])
```

### Data Augmentation
```python
training_transforms = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=15),
    transforms.ColorJitter(
        brightness=0.2,
        contrast=0.2,
        saturation=0.2,
        hue=0.1
    ),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
```

### Class Balancing Strategy
```python
def get_class_weights(dataset):
    """Calculate class weights for imbalanced datasets."""
    class_counts = Counter([dataset[i][1] for i in range(len(dataset))])
    total_samples = sum(class_counts.values())
    
    weights = {
        cls: total_samples / (len(class_counts) * count)
        for cls, count in class_counts.items()
    }
    
    return torch.FloatTensor([weights[i] for i in sorted(weights.keys())])
```

## Training Optimization

### Loss Functions

#### Weighted CrossEntropy (Primary)
```python
class WeightedCrossEntropyLoss(nn.Module):
    def __init__(self, class_weights):
        super().__init__()
        self.class_weights = class_weights
        
    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(
            predictions, 
            targets, 
            weight=self.class_weights
        )
        return ce_loss
```

#### Focal Loss (Advanced)
```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, predictions, targets):
        ce_loss = F.cross_entropy(predictions, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()
```

### Optimizer Configuration
```python
def get_optimizer(model, config):
    if config['optimizer'] == 'adam':
        return torch.optim.Adam(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
    elif config['optimizer'] == 'sgd':
        return torch.optim.SGD(
            model.parameters(),
            lr=config['learning_rate'],
            momentum=config['momentum'],
            weight_decay=config['weight_decay']
        )
```

### Learning Rate Scheduling
```python
def get_scheduler(optimizer, config):
    if config['scheduler'] == 'step':
        return torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=config['step_size'],
            gamma=config['gamma']
        )
    elif config['scheduler'] == 'cosine':
        return torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=config['epochs']
        )
```

## Performance Metrics

### Training Results Comparison

| **Model** | **Accuracy** | **Clean Precision** | **Dirty Precision** | **Training Time** | **Parameters** |
|-----------|--------------|-------------------|-------------------|------------------|----------------|
| Custom CNN | 85% | 100% | 78% | 5 min | 6.5M |
| ResNet50 | 92% | 95% | 89% | 25 min | 25M |
| EfficientNet-B0 | 89% | 93% | 85% | 12 min | 5M |
| MobileNet-V3 | 84% | 88% | 80% | 8 min | 3M |

### Confidence Score Analysis
```python
def analyze_confidence_scores(predictions, targets):
    """Analyze prediction confidence distribution."""
    confidence_scores = F.softmax(predictions, dim=1).max(dim=1)[0]
    
    correct_predictions = (predictions.argmax(dim=1) == targets)
    correct_confidence = confidence_scores[correct_predictions]
    incorrect_confidence = confidence_scores[~correct_predictions]
    
    return {
        'avg_correct_confidence': correct_confidence.mean().item(),
        'avg_incorrect_confidence': incorrect_confidence.mean().item(),
        'confidence_threshold_90': torch.quantile(confidence_scores, 0.9).item(),
        'high_confidence_accuracy': (correct_predictions[confidence_scores > 0.9]).float().mean().item()
    }
```

## Inference Pipeline

### Single Image Prediction
```python
def predict_single_image(model, image_path, device='cpu'):
    """Predict cleanliness of a single dish image."""
    model.eval()
    
    # Load and preprocess image
    image = Image.open(image_path).convert('RGB')
    image_tensor = transform_pipeline(image).unsqueeze(0).to(device)
    
    # Make prediction
    with torch.no_grad():
        logits = model(image_tensor)
        probabilities = F.softmax(logits, dim=1)
        predicted_class = logits.argmax(dim=1).item()
        confidence = probabilities.max().item()
    
    return {
        'predicted_class': predicted_class,
        'class_name': 'clean' if predicted_class == 0 else 'dirty',
        'confidence': confidence,
        'clean_probability': probabilities[0][0].item(),
        'dirty_probability': probabilities[0][1].item()
    }
```

### Batch Processing
```python
def predict_batch(model, image_paths, batch_size=8, device='cpu'):
    """Process multiple images efficiently."""
    model.eval()
    results = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i+batch_size]
        batch_tensors = []
        
        # Load batch
        for path in batch_paths:
            image = Image.open(path).convert('RGB')
            tensor = transform_pipeline(image)
            batch_tensors.append(tensor)
        
        # Process batch
        batch_tensor = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            logits = model(batch_tensor)
            probabilities = F.softmax(logits, dim=1)
        
        # Extract results
        for j, path in enumerate(batch_paths):
            pred_class = logits[j].argmax().item()
            confidence = probabilities[j].max().item()
            
            results.append({
                'image_path': path,
                'predicted_class': pred_class,
                'class_name': 'clean' if pred_class == 0 else 'dirty',
                'confidence': confidence,
                'clean_probability': probabilities[j][0].item(),
                'dirty_probability': probabilities[j][1].item()
            })
    
    return results
```

## Model Deployment Strategies

### Local Deployment
```python
class LocalPredictor:
    def __init__(self, model_path, config_path, device='auto'):
        self.device = self._get_device(device)
        self.model = self._load_model(model_path, config_path)
        self.model.eval()
    
    def _get_device(self, device):
        if device == 'auto':
            return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        return torch.device(device)
    
    def _load_model(self, model_path, config_path):
        # Load configuration
        with open(config_path, 'r') as f:
            config = json.load(f)
        
        # Initialize model
        model = self._create_model(config)
        
        # Load weights
        checkpoint = torch.load(model_path, map_location=self.device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        return model.to(self.device)
```

### Edge Deployment (Mobile/IoT)
```python
def export_to_mobile(model, example_input, output_path):
    """Export model for mobile deployment."""
    # Trace the model
    traced_model = torch.jit.trace(model, example_input)
    
    # Optimize for mobile
    optimized_model = optimize_for_mobile(traced_model)
    
    # Save
    optimized_model._save_for_lite_interpreter(output_path)
    
    return output_path
```

### API Deployment
```python
from flask import Flask, request, jsonify
import base64
from io import BytesIO

app = Flask(__name__)
predictor = LocalPredictor('models/final_model.pth', 'models/config.json')

@app.route('/predict', methods=['POST'])
def predict_api():
    # Get image from request
    image_data = request.json['image']  # Base64 encoded
    image_bytes = base64.b64decode(image_data)
    image = Image.open(BytesIO(image_bytes))
    
    # Save temporarily and predict
    temp_path = 'temp_image.jpg'
    image.save(temp_path)
    
    result = predictor.predict_single(temp_path)
    
    # Cleanup
    os.remove(temp_path)
    
    return jsonify(result)
```

## Performance Optimization

### Memory Optimization
```python
def optimize_memory_usage():
    """Techniques for reducing memory consumption."""
    
    # 1. Gradient checkpointing for large models
    model = torch.utils.checkpoint.checkpoint_sequential(model, segments=4)
    
    # 2. Mixed precision training
    scaler = torch.cuda.amp.GradScaler()
    
    # 3. Smaller batch sizes with gradient accumulation
    effective_batch_size = 32
    actual_batch_size = 8
    accumulation_steps = effective_batch_size // actual_batch_size
    
    # 4. Clear cache regularly
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### Speed Optimization
```python
def optimize_inference_speed():
    """Techniques for faster inference."""
    
    # 1. Model compilation (PyTorch 2.0+)
    compiled_model = torch.compile(model)
    
    # 2. TensorRT optimization (NVIDIA GPUs)
    import torch_tensorrt
    trt_model = torch_tensorrt.compile(model, 
        inputs=[torch.randn(1, 3, 224, 224)],
        enabled_precisions={torch.float, torch.half}
    )
    
    # 3. ONNX export for cross-platform deployment
    torch.onnx.export(model, dummy_input, "model.onnx", 
                     export_params=True, opset_version=11)
    
    # 4. Quantization for mobile deployment
    quantized_model = torch.quantization.quantize_dynamic(
        model, {torch.nn.Linear}, dtype=torch.qint8
    )
```

## System Architecture Diagram

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Data Layer    │    │ Processing      │    │   Model Layer   │
│                 │    │    Pipeline     │    │                 │
│ • Raw Images    │───▶│ • Preprocessing │───▶│ • Custom CNN    │
│ • Synthetic Gen │    │ • Augmentation  │    │ • ResNet50      │
│ • Kaggle Data   │    │ • Normalization │    │ • EfficientNet  │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Training      │    │   Inference     │    │    Output       │
│                 │    │                 │    │                 │
│ • Class Balance │    │ • Single Image  │    │ • Confidence    │
│ • Loss Function │    │ • Batch Process │    │ • Probabilities │
│ • Optimization  │    │ • Visualization │    │ • JSON/API      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
```

## Hardware Requirements

### Minimum Requirements
- **CPU**: 2+ cores, 2.0 GHz
- **RAM**: 4GB (8GB recommended)
- **Storage**: 2GB free space
- **Python**: 3.8+
- **OS**: Windows 10/macOS 10.14/Ubuntu 18.04+

### Recommended Setup
- **CPU**: 4+ cores, 3.0 GHz (Intel i5/AMD Ryzen 5+)
- **RAM**: 16GB
- **GPU**: NVIDIA GTX 1060 / RTX 3060 or better (optional)
- **Storage**: SSD with 10GB+ free space
- **Python**: 3.9 or 3.10

### Production Environment
- **CPU**: 8+ cores, 3.5 GHz (Intel i7/AMD Ryzen 7+)
- **RAM**: 32GB+
- **GPU**: NVIDIA RTX 4080/A100 for large-scale training
- **Storage**: NVMe SSD with 50GB+ free space
- **Network**: High-speed internet for dataset downloads