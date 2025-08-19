# Troubleshooting Guide

## Common Issues and Solutions

### Installation Problems

#### Issue: Module Import Errors
```
ImportError: No module named 'torch'
ModuleNotFoundError: No module named 'torchvision'
```

**Solution:**
```bash
# Ensure virtual environment is activated
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Verify installation
python -c "import torch, torchvision; print('Success')"
```

#### Issue: CUDA/GPU Problems
```
RuntimeError: CUDA out of memory
UserWarning: CUDA initialization: No CUDA-capable device is detected
```

**Solution:**
```bash
# Check GPU availability
python -c "import torch; print(torch.cuda.is_available())"

# Force CPU usage if needed
export DEVICE=cpu

# Or in .env file
DEVICE=cpu
```

#### Issue: Virtual Environment Problems
```
bash: venv/Scripts/activate: No such file or directory
```

**Solution:**
```bash
# Recreate virtual environment
rm -rf venv
python -m venv venv

# Activate (Linux/Mac)
source venv/bin/activate

# Activate (Windows)
venv\Scripts\activate

# Reinstall dependencies
pip install -r requirements.txt
```

### Model Loading Issues

#### Issue: Model File Not Found
```
FileNotFoundError: [Errno 2] No such file or directory: 'models/balanced_model.pth'
```

**Solution:**
```bash
# Check if models directory exists
ls -la models/

# Download or train models if missing
python scripts/cli/train_model.py

# Verify model files
ls -la models/*.pth
```

#### Issue: Model Compatibility Errors
```
RuntimeError: Error(s) in loading state_dict for ResNet50
```

**Solution:**
```bash
# Check PyTorch version compatibility
python -c "import torch; print(torch.__version__)"

# If version mismatch, reinstall compatible version
pip install torch==1.12.0 torchvision==0.13.0

# Or retrain model with current PyTorch version
python scripts/cli/train_model.py --model resnet50
```

#### Issue: Model Performance Problems
```
Prediction confidence consistently low
Classification results seem random
```

**Solution:**
```python
# Check model evaluation metrics
python examples/advanced_examples/model_comparison.py

# Validate on known good images
python examples/basic_examples/single_image_classification.py

# Retrain if necessary
python scripts/cli/train_model.py --epochs 50 --learning-rate 0.0001
```

### API Issues

#### Issue: API Server Won't Start
```
Address already in use: 0.0.0.0:8000
uvicorn.error: Error loading ASGI app
```

**Solution:**
```bash
# Check if port is in use
netstat -an | grep 8000
# or on Windows:
netstat -an | findstr 8000

# Kill existing processes
pkill -f "python.*api.main"

# Use different port
python -m api.main --port 8080

# Or set in environment
export API_PORT=8080
```

#### Issue: API Endpoints Return 500 Errors
```
Internal Server Error
FastAPI automatic error handling
```

**Solution:**
```bash
# Check API logs
python -m api.main --log-level DEBUG

# Test individual components
python tests/integration/test_api_endpoints.py

# Verify model loading in API
curl http://localhost:8000/health/models
```

#### Issue: Image Upload Failures
```
HTTP 413: Request Entity Too Large
HTTP 400: Invalid image format
```

**Solution:**
```python
# Check image size limits
MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB

# Supported formats
SUPPORTED_FORMATS = ['jpg', 'jpeg', 'png']

# Test with small sample image
curl -X POST "http://localhost:8000/api/v1/classify/single" \
  -F "image=@data/samples/demo_images/clean_plate_sample.jpg"
```

### Data Issues

#### Issue: No Training Data Found
```
ERROR: No sample images found in data/processed/
FileNotFoundError: data/samples/demo_images/
```

**Solution:**
```bash
# Check data directory structure
tree data/

# Download sample data
python scripts/download_data.py

# Generate synthetic data
python data/synthetic/data_generator.py

# Verify data structure
ls -la data/samples/demo_images/
```

#### Issue: Poor Classification Results
```
Low confidence scores consistently
Misclassification of obvious cases
```

**Solution:**
```bash
# Verify image quality
python examples/basic_examples/single_image_classification.py

# Check data balance
python -c "
import os
clean = len(os.listdir('data/processed/clean_labeled'))
dirty = len(os.listdir('data/processed/dirty_labeled'))
print(f'Clean: {clean}, Dirty: {dirty}, Ratio: {clean/dirty:.2f}')
"

# Retrain with balanced data
python src/vision_classifier/train.py --balance-dataset
```

### Performance Issues

#### Issue: Slow Predictions
```
Processing time > 5 seconds per image
API timeout errors
Memory usage growing continuously
```

**Solution:**
```python
# Use faster model
DEFAULT_MODEL=fast_model.pth

# Optimize image preprocessing
# Resize images before processing
from PIL import Image
image = Image.open('path/to/image.jpg')
image = image.resize((224, 224))

# Use GPU if available
import torch
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
```

#### Issue: Memory Leaks
```
MemoryError: Unable to allocate array
Gradual memory increase over time
```

**Solution:**
```python
# Clear GPU cache
import torch
if torch.cuda.is_available():
    torch.cuda.empty_cache()

# Use context managers
with torch.no_grad():
    prediction = model(image)

# Limit batch size
BATCH_SIZE = 8  # Reduce if memory issues persist
```

### Docker Issues

#### Issue: Container Build Failures
```
ERROR: failed to solve: process "/bin/sh -c pip install -r requirements.txt" did not complete successfully
```

**Solution:**
```bash
# Update base image
docker pull python:3.11-slim

# Clear Docker cache
docker system prune -a

# Build with verbose output
docker build --no-cache -f docker/Dockerfile -t vision-classifier .

# Check Docker resources
docker system df
```

#### Issue: Container Runtime Errors
```
container exits immediately
Health check failed
```

**Solution:**
```bash
# Check container logs
docker logs vision-classifier

# Run interactively for debugging
docker run -it --entrypoint /bin/bash vision-classifier

# Test health endpoints
curl http://localhost:8000/health/status
```

### Testing Issues

#### Issue: Tests Failing
```
ImportError during test collection
AssertionError in test_model_prediction
```

**Solution:**
```bash
# Run tests with verbose output
python -m pytest tests/ -v

# Run specific test category
python -m pytest tests/unit/ -v

# Check test environment
python -m pytest tests/conftest.py -v

# Update test dependencies
pip install pytest pytest-cov
```

### Environment Issues

#### Issue: Windows-Specific Problems
```
UnicodeEncodeError: 'charmap' codec can't encode character
FileNotFoundError: [WinError 2] The system cannot find the file specified
```

**Solution:**
```bash
# Set encoding for Windows
set PYTHONIOENCODING=utf-8

# Use Windows-compatible paths
python examples/basic_examples/single_image_classification.py

# Install Windows-specific dependencies
pip install pywin32
```

## Diagnostic Commands

### System Diagnostics
```bash
# Check Python environment
python --version
pip list

# Check system resources
df -h          # Disk space
free -h        # Memory usage
top            # CPU usage

# Check GPU (if applicable)
nvidia-smi
```

### Application Diagnostics
```bash
# Test basic functionality
python -c "
import sys, torch
print(f'Python: {sys.version}')
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
"

# Test model loading
python tests/integration/test_refactored_components.py

# Test API endpoints
curl -X GET http://localhost:8000/health/status
```

## Getting Help

### Debug Information to Collect

When reporting issues, include:

1. **System Information:**
   - OS and version
   - Python version
   - PyTorch version
   - Available memory/GPU

2. **Error Messages:**
   - Full error traceback
   - Log files
   - Console output

3. **Configuration:**
   - Environment variables
   - Model files being used
   - API endpoint being called

4. **Reproduction Steps:**
   - Commands run
   - Files used
   - Expected vs actual behavior

### Support Channels

1. **Documentation**: Check this guide and API reference
2. **Examples**: Run provided examples to verify setup
3. **Tests**: Run test suite to identify specific issues
4. **Logs**: Enable debug logging for detailed output

### Quick Health Check Script

```python
# health_check.py
import sys
import os
from pathlib import Path

def health_check():
    print("=== Vision Object Classifier Health Check ===")
    
    # Python version
    print(f"Python: {sys.version}")
    
    # Dependencies
    try:
        import torch, torchvision, cv2, PIL
        print("✓ All dependencies available")
    except ImportError as e:
        print(f"✗ Missing dependency: {e}")
        return False
    
    # Model files
    models_dir = Path("models")
    if models_dir.exists():
        models = list(models_dir.glob("*.pth"))
        print(f"✓ Found {len(models)} model files")
    else:
        print("✗ Models directory not found")
        return False
    
    # Sample data
    samples_dir = Path("data/samples/demo_images")
    if samples_dir.exists():
        samples = list(samples_dir.glob("*.jpg"))
        print(f"✓ Found {len(samples)} sample images")
    else:
        print("✗ Sample images not found")
        return False
    
    print("✓ Health check passed!")
    return True

if __name__ == "__main__":
    health_check()
```