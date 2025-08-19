# Examples Guide

## Overview

This guide covers all available examples in the Vision Object Classifier, organized by complexity and use case. Examples demonstrate different ways to use the classifier, from simple single-image classification to complex batch processing workflows.

## Example Structure

```
examples/
├── basic_examples/          # Simple, straightforward examples
│   ├── single_image_classification.py
│   └── batch_classification.py
├── advanced_examples/       # Complex scenarios and comparisons
│   └── model_comparison.py
└── integration_examples/    # Real-world API integration
```

## Basic Examples

### Single Image Classification

**File**: `examples/basic_examples/single_image_classification.py`

**Purpose**: Demonstrates basic usage of the vision classifier on a single image.

**Usage**:
```bash
# Activate virtual environment
source venv/bin/activate  # Linux/Mac
# or venv\Scripts\activate  # Windows

# Run example
python examples/basic_examples/single_image_classification.py
```

**What it does**:
- Loads available trained models
- Finds sample images in `data/samples/demo_images/`
- Classifies each image as clean or dirty
- Displays confidence scores and probabilities
- Saves results to `outputs/` folder

**Expected Output**:
```
Single Image Classification Example
=============================================

Found 2 sample images for classification

Loading classification model...
  ✓ Found model: balanced_model.pth
  ✓ Model loaded successfully

Classification Results:
------------------------------------------------------------
   Processing real_dirty_pasta_plate.jpg...
1. real_dirty_pasta_plate.jpg      | Predicted: DIRTY | Confidence: 87.3% 🟢 | Actual: DIRTY ✓
   Processing clean_plate_sample.jpg...
2. clean_plate_sample.jpg          | Predicted: CLEAN | Confidence: 84.1% 🟢 | Actual: CLEAN ✓
------------------------------------------------------------
Summary Statistics:
   Total Images: 2
   Successful Predictions: 2
   Accuracy: 100% (2/2)
   Average Confidence: 85.7%
   Assessment: 🟢 Excellent performance

✓ Results saved to: outputs/classification_results_20250819_143025.json
```

**Customization**:
```python
# Use different model
predictor = DishCleanlinessPredictor(
    model_path="models/fast_model.pth",
    config_path="models/fast_config.json"
)

# Process your own images
image_path = "path/to/your/image.jpg"
result = predictor.predict_single(image_path)
```

### Batch Classification

**File**: `examples/basic_examples/batch_classification.py`

**Purpose**: Process multiple images efficiently in batches.

**Usage**:
```bash
python examples/basic_examples/batch_classification.py
```

**Features**:
- Batch processing for better performance
- Progress tracking
- Error handling for individual images
- Detailed reporting
- Export to CSV format

**Example Usage**:
```python
from src.vision_classifier.predict import DishCleanlinessPredictor

# Initialize predictor
predictor = DishCleanlinessPredictor("models/balanced_model.pth")

# Batch process images
image_paths = [
    "data/samples/image1.jpg",
    "data/samples/image2.jpg",
    "data/samples/image3.jpg"
]

results = predictor.predict_batch(image_paths, batch_size=8)
```

## Advanced Examples

### Model Comparison

**File**: `examples/advanced_examples/model_comparison.py`

**Purpose**: Compare performance of different model variants (fast vs balanced vs accurate).

**Usage**:
```bash
python examples/advanced_examples/model_comparison.py
```

**What it does**:
- Tests all available models on the same dataset
- Measures accuracy, confidence, and processing time
- Generates comparison charts and reports
- Helps choose the best model for your use case

**Expected Output**:
```
Model Performance Comparison
============================

Testing Models: fast_model.pth, balanced_model.pth

Processing with fast_model.pth...
  Average accuracy: 84.2%
  Average confidence: 79.8%
  Average processing time: 45ms

Processing with balanced_model.pth...  
  Average accuracy: 91.7%
  Average confidence: 87.3%
  Average processing time: 89ms

Recommendation: Use balanced_model.pth for best accuracy/performance balance
```

**Customization**:
```python
# Add custom metrics
def calculate_precision_recall(results):
    # Your implementation
    pass

# Test specific models
models_to_test = [
    "models/custom_model.pth",
    "models/fine_tuned_model.pth"
]
```

## Integration Examples

### API Integration Example

**Purpose**: Demonstrates how to integrate the classifier with existing applications via the API.

**Prerequisites**:
```bash
# Start API server
python -m api.main
```

**Python Integration**:
```python
import requests
from pathlib import Path

# API endpoint
API_URL = "http://localhost:8000"

# Single image classification
def classify_image_api(image_path):
    with open(image_path, 'rb') as f:
        files = {'image': f}
        response = requests.post(f"{API_URL}/api/v1/classify/single", files=files)
    return response.json()

# Example usage
result = classify_image_api("data/samples/demo_images/real_dirty_pasta_plate.jpg")
print(f"Classification: {result['result']['predicted_class']}")
print(f"Confidence: {result['result']['confidence']}")
```

**JavaScript Integration**:
```javascript
// Browser JavaScript example
async function classifyImage(imageFile) {
    const formData = new FormData();
    formData.append('image', imageFile);
    
    const response = await fetch('/api/v1/classify/single', {
        method: 'POST',
        body: formData
    });
    
    const result = await response.json();
    return result;
}

// Usage with file input
document.getElementById('imageInput').addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (file) {
        const result = await classifyImage(file);
        console.log('Classification:', result.result.predicted_class);
    }
});
```

**cURL Examples**:
```bash
# Single image classification
curl -X POST "http://localhost:8000/api/v1/classify/single" \
  -F "image=@data/samples/demo_images/real_dirty_pasta_plate.jpg" \
  -F "model_type=balanced" \
  -F "return_confidence=true"

# Batch classification
curl -X POST "http://localhost:8000/api/v1/batch/classify" \
  -F "images=@image1.jpg" \
  -F "images=@image2.jpg" \
  -F "model_type=fast"

# Health check
curl http://localhost:8000/health/status

# Model information
curl http://localhost:8000/api/v1/classify/model-info
```

## Running Examples

### Prerequisites Check
Before running examples, ensure:

1. **Virtual Environment**:
   ```bash
   source venv/bin/activate  # Linux/Mac
   venv\Scripts\activate     # Windows
   ```

2. **Dependencies Installed**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Models Available**:
   ```bash
   ls models/*.pth  # Should show trained models
   ```

4. **Sample Data Present**:
   ```bash
   ls data/samples/demo_images/*.jpg
   ```

### Troubleshooting Examples

#### Import Errors
```bash
# Ensure PYTHONPATH is set
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"

# Or add to your shell profile
echo 'export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"' >> ~/.bashrc
```

#### Model Not Found
```bash
# Check available models
ls models/

# Train a model if none available
python scripts/cli/train_model.py
```

#### Permission Errors
```bash
# Fix file permissions
chmod 644 models/*.pth
chmod 644 data/samples/demo_images/*.jpg
```

## Creating Custom Examples

### Example Template
```python
#!/usr/bin/env python3
"""
Custom Example: Your Use Case
Description of what this example demonstrates
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

from vision_classifier.predict import DishCleanlinessPredictor

def main():
    """Main example function"""
    print("Custom Example: Your Use Case")
    print("=" * 40)
    
    try:
        # Initialize predictor
        predictor = DishCleanlinessPredictor(
            model_path="models/balanced_model.pth"
        )
        
        # Your custom logic here
        
        print("Example completed successfully!")
        return True
        
    except Exception as e:
        print(f"Error: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
```

### Best Practices for Examples

1. **Clear Documentation**: Include docstrings and comments
2. **Error Handling**: Handle common failure cases gracefully  
3. **Progress Indication**: Show progress for long-running operations
4. **Output Formatting**: Make results easy to understand
5. **Resource Cleanup**: Clean up temporary files and resources
6. **Flexible Input**: Allow different input sources when possible

### Example Categories

**Basic Examples**: Simple, single-purpose demonstrations
- Focus on one core feature
- Minimal dependencies
- Clear, straightforward code
- Good for learning and testing

**Advanced Examples**: Complex, real-world scenarios
- Multiple features combined
- Performance optimizations
- Error handling and edge cases
- Production-ready patterns

**Integration Examples**: How to use with other systems
- API integrations
- Database connections
- File processing workflows
- External service integration

## Performance Considerations

### Optimizing Examples for Speed

```python
# Use appropriate batch sizes
batch_size = 16  # Adjust based on available memory

# Pre-load models
predictor = DishCleanlinessPredictor("models/fast_model.pth")

# Process images in parallel (if CPU-bound)
from concurrent.futures import ThreadPoolExecutor
with ThreadPoolExecutor(max_workers=4) as executor:
    results = list(executor.map(predictor.predict_single, image_paths))
```

### Memory Management
```python
import torch

# Clear GPU cache between predictions
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use context managers for resource management
with torch.no_grad():
    predictions = model(batch)
```

## Example Output Formats

Examples generate outputs in multiple formats:

### JSON Results
```json
{
  "analysis_type": "single_image_classification",
  "timestamp": "2025-08-19T14:30:25",
  "model_used": "models/balanced_model.pth",
  "results": [
    {
      "image": "real_dirty_pasta_plate.jpg",
      "predicted": "dirty",
      "confidence": 0.873,
      "actual": "dirty", 
      "correct": true
    }
  ],
  "summary": {
    "total_images": 1,
    "accuracy": 1.0,
    "average_confidence": 0.873
  }
}
```

### CSV Reports
```csv
Image,Predicted,Confidence,Actual,Correct
real_dirty_pasta_plate.jpg,dirty,0.873,dirty,true
clean_plate_sample.jpg,clean,0.841,clean,true
```

### Console Output
```
Classification Results:
----------------------
✓ real_dirty_pasta_plate.jpg: DIRTY (87.3% confidence)
✓ clean_plate_sample.jpg: CLEAN (84.1% confidence)

Summary: 2/2 correct (100% accuracy)
```