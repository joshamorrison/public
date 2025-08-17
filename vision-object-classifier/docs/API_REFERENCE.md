# API Reference

Complete reference for all classes, functions, and command-line interfaces in the Vision Object Classifier.

## Command Line Interface

### quick_start.py
Main entry point for platform demonstration and testing.

```bash
python quick_start.py
```

**Description:** Runs comprehensive demo testing clean and dirty dish classification with pre-trained models.

**Output:** Test results for 3 scenarios with confidence scores and success/failure status.

### Training Scripts

#### src/train.py
Main training script for model training with various configurations.

```bash
python src/train.py [OPTIONS]
```

**Options:**
- `--config CONFIG_PATH`: Path to training configuration file
- `--data_dir DATA_DIR`: Directory containing clean/ and dirty/ subdirectories
- `--model_type MODEL`: Model architecture (custom_cnn, resnet50, efficientnet, mobilenet)
- `--epochs EPOCHS`: Number of training epochs (default: 10)
- `--batch_size BATCH_SIZE`: Training batch size (default: 16)
- `--learning_rate LR`: Learning rate (default: 0.001)
- `--device DEVICE`: Training device (cpu, cuda, auto)
- `--save_path PATH`: Model save path (default: models/)

**Examples:**
```bash
# Basic training with default settings
python src/train.py

# Custom configuration
python src/train.py --epochs 20 --batch_size 32 --learning_rate 0.003

# Train ResNet50 with GPU
python src/train.py --model_type resnet50 --device cuda --epochs 15
```

#### src/predict.py
Prediction script for single images and batch processing.

```bash
python src/predict.py --model MODEL_PATH --config CONFIG_PATH --image IMAGE_PATH [OPTIONS]
```

**Required Arguments:**
- `--model MODEL_PATH`: Path to trained model (.pth file)
- `--config CONFIG_PATH`: Path to model configuration (.json file)
- `--image IMAGE_PATH`: Path to image file for prediction

**Optional Arguments:**
- `--image_dir DIR_PATH`: Directory of images for batch processing
- `--output OUTPUT_DIR`: Output directory for results
- `--visualize`: Generate prediction visualization
- `--batch_size BATCH_SIZE`: Batch size for processing (default: 8)
- `--device DEVICE`: Device for inference (cpu, cuda, auto)
- `--confidence_threshold THRESHOLD`: Minimum confidence threshold (default: 0.5)

**Examples:**
```bash
# Single image prediction
python src/predict.py --model models/final_balanced_model.pth --config models/balanced_config.json --image dish.jpg

# Batch processing with visualization
python src/predict.py --model models/final_balanced_model.pth --config models/balanced_config.json --image_dir test_images/ --output results/ --visualize

# High confidence predictions only
python src/predict.py --model models/final_balanced_model.pth --config models/balanced_config.json --image dish.jpg --confidence_threshold 0.8
```

## Core Classes and Functions

### Model Architecture Classes

#### CustomCNN
Custom convolutional neural network optimized for dish classification.

```python
from src.vision_classifier.model import CustomCNN

model = CustomCNN(num_classes=2, dropout_rate=0.5)
```

**Parameters:**
- `num_classes` (int): Number of output classes (default: 2)
- `dropout_rate` (float): Dropout probability (default: 0.5)

**Methods:**

##### `forward(x)`
Forward pass through the network.

**Parameters:**
- `x` (torch.Tensor): Input tensor of shape (batch_size, 3, 224, 224)

**Returns:**
- `torch.Tensor`: Output logits of shape (batch_size, num_classes)

**Example:**
```python
import torch
model = CustomCNN(num_classes=2)
input_tensor = torch.randn(1, 3, 224, 224)
output = model(input_tensor)
print(f"Output shape: {output.shape}")  # torch.Size([1, 2])
```

#### ResNet50Classifier
ResNet50-based classifier with custom head.

```python
from src.vision_classifier.model import ResNet50Classifier

model = ResNet50Classifier(num_classes=2, pretrained=True)
```

**Parameters:**
- `num_classes` (int): Number of output classes (default: 2)
- `pretrained` (bool): Use ImageNet pre-trained weights (default: True)
- `freeze_backbone` (bool): Freeze backbone weights (default: False)

### Data Processing Classes

#### DishDataset
Custom dataset class for loading dish images.

```python
from src.vision_classifier.data_utils import DishDataset

dataset = DishDataset(
    data_dir='data',
    transform=transforms.ToTensor(),
    train=True
)
```

**Parameters:**
- `data_dir` (str): Root directory containing clean/ and dirty/ subdirectories
- `transform` (callable): Optional transform to apply to images
- `train` (bool): Whether this is training dataset (affects augmentation)

**Methods:**

##### `__len__()`
Returns the total number of samples in the dataset.

**Returns:**
- `int`: Number of samples

##### `__getitem__(idx)`
Get a sample from the dataset.

**Parameters:**
- `idx` (int): Index of the sample

**Returns:**
- `tuple`: (image, label) where image is PIL Image and label is int

**Example:**
```python
dataset = DishDataset('data', transform=transforms.ToTensor())
print(f"Dataset size: {len(dataset)}")

image, label = dataset[0]
print(f"Image shape: {image.shape}")
print(f"Label: {label} ({'clean' if label == 0 else 'dirty'})")
```

#### DataLoader Creation
```python
from src.vision_classifier.data_utils import create_data_loaders

train_loader, val_loader, test_loader = create_data_loaders(
    data_dir='data',
    batch_size=16,
    train_ratio=0.8,
    val_ratio=0.1,
    num_workers=4
)
```

**Parameters:**
- `data_dir` (str): Root data directory
- `batch_size` (int): Batch size for data loaders
- `train_ratio` (float): Fraction of data for training
- `val_ratio` (float): Fraction of data for validation
- `num_workers` (int): Number of worker processes for data loading

**Returns:**
- `tuple`: (train_loader, val_loader, test_loader)

### Prediction Classes

#### DishCleanlinessPredictor
High-level interface for dish cleanliness prediction.

```python
from src.vision_classifier.predict import DishCleanlinessPredictor

predictor = DishCleanlinessPredictor(
    model_path='models/final_balanced_model.pth',
    config_path='models/balanced_config.json',
    device='auto'
)
```

**Parameters:**
- `model_path` (str): Path to trained model file
- `config_path` (str): Path to model configuration file
- `device` (str): Device for inference ('cpu', 'cuda', 'auto')

**Methods:**

##### `predict_single(image_path)`
Predict cleanliness of a single image.

**Parameters:**
- `image_path` (str): Path to image file

**Returns:**
- `dict`: Prediction results with keys:
  - `predicted_class` (int): 0 for clean, 1 for dirty
  - `class_name` (str): 'clean' or 'dirty'
  - `confidence` (float): Maximum class probability
  - `clean_probability` (float): Probability of clean class
  - `dirty_probability` (float): Probability of dirty class

**Example:**
```python
predictor = DishCleanlinessPredictor(
    'models/final_balanced_model.pth',
    'models/balanced_config.json'
)

result = predictor.predict_single('dish_image.jpg')
print(f"Prediction: {result['class_name']}")
print(f"Confidence: {result['confidence']:.3f}")
```

##### `predict_batch(image_paths, batch_size=8)`
Predict cleanliness for multiple images efficiently.

**Parameters:**
- `image_paths` (list): List of image file paths
- `batch_size` (int): Batch size for processing

**Returns:**
- `list`: List of prediction dictionaries (same format as predict_single)

**Example:**
```python
image_paths = ['img1.jpg', 'img2.jpg', 'img3.jpg']
results = predictor.predict_batch(image_paths, batch_size=4)

for i, result in enumerate(results):
    print(f"Image {i+1}: {result['class_name']} ({result['confidence']:.3f})")
```

##### `visualize_prediction(image_path, save_path=None)`
Create visualization of prediction with confidence scores.

**Parameters:**
- `image_path` (str): Path to image file
- `save_path` (str, optional): Path to save visualization

**Returns:**
- `matplotlib.figure.Figure`: Figure object with visualization

**Example:**
```python
fig = predictor.visualize_prediction('dish.jpg', 'prediction_viz.png')
plt.show()
```

### Synthetic Data Generation

#### SyntheticDirtyDishGenerator
Generate realistic dirty dish images from clean images.

```python
from src.vision_classifier.synthetic_data import SyntheticDirtyDishGenerator

generator = SyntheticDirtyDishGenerator(output_dir='data/dirty')
```

**Parameters:**
- `output_dir` (str): Directory to save generated images

**Methods:**

##### `generate_dirty_dish(clean_image_path, output_path, dirty_level='medium')`
Generate a single dirty dish variant.

**Parameters:**
- `clean_image_path` (str): Path to clean dish image
- `output_path` (str): Path to save dirty variant
- `dirty_level` (str): Intensity level ('light', 'medium', 'heavy')

**Returns:**
- `bool`: True if successful, False otherwise

**Example:**
```python
generator = SyntheticDirtyDishGenerator('output_dirty')
success = generator.generate_dirty_dish(
    'clean_plate.jpg',
    'dirty_plate.jpg',
    'medium'
)
print(f"Generation successful: {success}")
```

##### `batch_generate(clean_images_dir, num_variations=5, intensity_distribution=None)`
Generate multiple dirty variants for all clean images.

**Parameters:**
- `clean_images_dir` (str): Directory containing clean images
- `num_variations` (int): Number of dirty variants per clean image
- `intensity_distribution` (dict): Distribution of intensity levels

**Returns:**
- `int`: Total number of images generated

**Example:**
```python
total_generated = generator.batch_generate(
    clean_images_dir='data/clean',
    num_variations=3,
    intensity_distribution={
        'light': 0.3,
        'medium': 0.5,
        'heavy': 0.2
    }
)
print(f"Generated {total_generated} dirty images")
```

### Training and Evaluation

#### ModelTrainer
Comprehensive model training with monitoring and checkpointing.

```python
from src.vision_classifier.train import ModelTrainer

trainer = ModelTrainer(
    model=model,
    train_loader=train_loader,
    val_loader=val_loader,
    config=training_config
)
```

**Parameters:**
- `model` (torch.nn.Module): Model to train
- `train_loader` (DataLoader): Training data loader
- `val_loader` (DataLoader): Validation data loader
- `config` (dict): Training configuration

**Methods:**

##### `train(epochs=10, save_best=True)`
Train the model for specified number of epochs.

**Parameters:**
- `epochs` (int): Number of training epochs
- `save_best` (bool): Save best model based on validation accuracy

**Returns:**
- `dict`: Training history with losses and accuracies

**Example:**
```python
trainer = ModelTrainer(model, train_loader, val_loader, config)
history = trainer.train(epochs=15, save_best=True)

print(f"Best validation accuracy: {max(history['val_accuracy']):.2f}%")
```

##### `evaluate(test_loader)`
Evaluate model on test dataset.

**Parameters:**
- `test_loader` (DataLoader): Test data loader

**Returns:**
- `dict`: Evaluation metrics including accuracy, precision, recall, F1-score

#### Model Evaluation Functions

##### `calculate_metrics(y_true, y_pred, class_names=None)`
Calculate comprehensive classification metrics.

**Parameters:**
- `y_true` (array-like): True labels
- `y_pred` (array-like): Predicted labels
- `class_names` (list, optional): Names of classes

**Returns:**
- `dict`: Dictionary containing accuracy, precision, recall, F1-score, confusion matrix

**Example:**
```python
from src.vision_classifier.evaluation import calculate_metrics

metrics = calculate_metrics(
    y_true=[0, 1, 0, 1, 1],
    y_pred=[0, 1, 1, 1, 0],
    class_names=['clean', 'dirty']
)

print(f"Accuracy: {metrics['accuracy']:.3f}")
print(f"Precision: {metrics['precision']}")
print(f"Recall: {metrics['recall']}")
```

##### `plot_confusion_matrix(y_true, y_pred, class_names, save_path=None)`
Plot confusion matrix with customizable styling.

**Parameters:**
- `y_true` (array-like): True labels
- `y_pred` (array-like): Predicted labels
- `class_names` (list): Names of classes
- `save_path` (str, optional): Path to save plot

**Returns:**
- `matplotlib.figure.Figure`: Figure object

## Utility Functions

### Image Processing

#### `load_and_preprocess_image(image_path, transform=None)`
Load and preprocess image for model input.

**Parameters:**
- `image_path` (str): Path to image file
- `transform` (callable, optional): Preprocessing transform

**Returns:**
- `torch.Tensor`: Preprocessed image tensor

#### `save_prediction_visualization(image, prediction, save_path)`
Save image with prediction overlay.

**Parameters:**
- `image` (PIL.Image or torch.Tensor): Input image
- `prediction` (dict): Prediction results
- `save_path` (str): Path to save visualization

### Configuration Management

#### `load_config(config_path)`
Load training/model configuration from JSON file.

**Parameters:**
- `config_path` (str): Path to configuration file

**Returns:**
- `dict`: Configuration dictionary

#### `save_config(config, save_path)`
Save configuration dictionary to JSON file.

**Parameters:**
- `config` (dict): Configuration dictionary
- `save_path` (str): Path to save configuration

### Model Management

#### `save_model_checkpoint(model, optimizer, epoch, metrics, save_path)`
Save complete model checkpoint.

**Parameters:**
- `model` (torch.nn.Module): Model to save
- `optimizer` (torch.optim.Optimizer): Optimizer state
- `epoch` (int): Current epoch number
- `metrics` (dict): Current metrics
- `save_path` (str): Path to save checkpoint

#### `load_model_checkpoint(checkpoint_path, model, optimizer=None)`
Load model checkpoint.

**Parameters:**
- `checkpoint_path` (str): Path to checkpoint file
- `model` (torch.nn.Module): Model to load weights into
- `optimizer` (torch.optim.Optimizer, optional): Optimizer to load state into

**Returns:**
- `dict`: Checkpoint metadata (epoch, metrics, etc.)

## Error Handling

### Common Exceptions

#### `ModelLoadError`
Raised when model loading fails.

```python
try:
    predictor = DishCleanlinessPredictor('invalid_model.pth', 'config.json')
except ModelLoadError as e:
    print(f"Failed to load model: {e}")
```

#### `ImageProcessingError`
Raised when image processing fails.

```python
try:
    result = predictor.predict_single('corrupted_image.jpg')
except ImageProcessingError as e:
    print(f"Image processing failed: {e}")
```

#### `ConfigurationError`
Raised when configuration is invalid.

```python
try:
    config = load_config('invalid_config.json')
except ConfigurationError as e:
    print(f"Configuration error: {e}")
```

### Error Recovery Patterns

#### Graceful Degradation
```python
def robust_prediction(image_path, model_paths, config_paths):
    """Try multiple models for robust prediction."""
    for model_path, config_path in zip(model_paths, config_paths):
        try:
            predictor = DishCleanlinessPredictor(model_path, config_path)
            return predictor.predict_single(image_path)
        except Exception as e:
            print(f"Model {model_path} failed: {e}")
            continue
    
    raise RuntimeError("All models failed")
```

#### Batch Processing with Error Handling
```python
def robust_batch_processing(image_paths, predictor):
    """Process batch with individual error handling."""
    results = []
    failed_images = []
    
    for image_path in image_paths:
        try:
            result = predictor.predict_single(image_path)
            results.append({
                'image_path': image_path,
                'prediction': result,
                'status': 'success'
            })
        except Exception as e:
            failed_images.append(image_path)
            results.append({
                'image_path': image_path,
                'error': str(e),
                'status': 'failed'
            })
    
    return results, failed_images
```

## Performance Optimization

### Batch Processing Optimization
```python
def optimized_batch_prediction(predictor, image_paths, max_batch_size=32):
    """Optimized batch processing for large image sets."""
    import torch
    
    results = []
    
    for i in range(0, len(image_paths), max_batch_size):
        batch_paths = image_paths[i:i + max_batch_size]
        
        # Process batch
        batch_results = predictor.predict_batch(
            batch_paths, 
            batch_size=min(len(batch_paths), max_batch_size)
        )
        
        results.extend(batch_results)
        
        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    return results
```

### Memory Management
```python
def memory_efficient_training(model, train_loader, config):
    """Training with memory optimization."""
    
    # Enable gradient checkpointing for large models
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    # Use smaller batch sizes with gradient accumulation
    effective_batch_size = config.get('effective_batch_size', 32)
    actual_batch_size = config.get('actual_batch_size', 8)
    accumulation_steps = effective_batch_size // actual_batch_size
    
    optimizer = torch.optim.Adam(model.parameters())
    
    for epoch in range(config['epochs']):
        model.train()
        optimizer.zero_grad()
        
        for batch_idx, (data, targets) in enumerate(train_loader):
            outputs = model(data)
            loss = criterion(outputs, targets) / accumulation_steps
            loss.backward()
            
            if (batch_idx + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()
```

## Integration Examples

### Flask API Integration
```python
from flask import Flask, request, jsonify
import base64
from io import BytesIO
from PIL import Image

app = Flask(__name__)
predictor = DishCleanlinessPredictor(
    'models/final_balanced_model.pth',
    'models/balanced_config.json'
)

@app.route('/predict', methods=['POST'])
def predict_api():
    try:
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
        
        return jsonify({
            'success': True,
            'prediction': result
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 400

if __name__ == '__main__':
    app.run(debug=True)
```

### Streamlit Web App
```python
import streamlit as st
from PIL import Image

st.title("Dish Cleanliness Classifier")

# Initialize predictor
@st.cache_resource
def load_predictor():
    return DishCleanlinessPredictor(
        'models/final_balanced_model.pth',
        'models/balanced_config.json'
    )

predictor = load_predictor()

# File uploader
uploaded_file = st.file_uploader(
    "Choose a dish image...", 
    type=['jpg', 'jpeg', 'png']
)

if uploaded_file is not None:
    # Display image
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Make prediction
    if st.button('Classify Dish'):
        with st.spinner('Analyzing...'):
            # Save temporarily
            temp_path = 'temp_upload.jpg'
            image.save(temp_path)
            
            # Predict
            result = predictor.predict_single(temp_path)
            
            # Display results
            st.success(f"Prediction: {result['class_name'].title()}")
            st.info(f"Confidence: {result['confidence']:.1%}")
            
            # Progress bars
            st.write("Class Probabilities:")
            st.progress(result['clean_probability'], text=f"Clean: {result['clean_probability']:.1%}")
            st.progress(result['dirty_probability'], text=f"Dirty: {result['dirty_probability']:.1%}")
            
            # Cleanup
            os.remove(temp_path)
```

This comprehensive API reference provides complete documentation for integrating and extending the Vision Object Classifier in various applications and environments.