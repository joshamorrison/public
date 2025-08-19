"""
Pytest configuration and shared fixtures for Vision Object Classifier tests
"""

import pytest
import tempfile
import shutil
from pathlib import Path
import sys
import os

# Add src to path for testing
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

@pytest.fixture(scope="session")
def project_root_path():
    """Provide project root path"""
    return Path(__file__).parent.parent

@pytest.fixture(scope="session")
def models_dir(project_root_path):
    """Provide models directory path"""
    return project_root_path / "models"

@pytest.fixture(scope="session")
def data_dir(project_root_path):
    """Provide data directory path"""
    return project_root_path / "data"

@pytest.fixture(scope="session")
def sample_images_dir(data_dir):
    """Provide sample images directory"""
    return data_dir / "samples" / "demo_images"

@pytest.fixture(scope="function")
def temp_dir():
    """Create temporary directory for tests"""
    temp_path = tempfile.mkdtemp()
    yield Path(temp_path)
    shutil.rmtree(temp_path)

@pytest.fixture(scope="session")
def sample_clean_image(sample_images_dir):
    """Provide path to sample clean image"""
    clean_image = sample_images_dir / "clean_plate_sample.jpg"
    if clean_image.exists():
        return clean_image
    return None

@pytest.fixture(scope="session") 
def sample_dirty_image(sample_images_dir):
    """Provide path to sample dirty image"""
    dirty_image = sample_images_dir / "real_dirty_pasta_plate.jpg"
    if dirty_image.exists():
        return dirty_image
    return None

@pytest.fixture(scope="session")
def available_models(models_dir):
    """Get list of available trained models"""
    if not models_dir.exists():
        return []
    return list(models_dir.glob("*.pth"))

@pytest.fixture(scope="function")
def mock_model_predictor():
    """Mock predictor for unit tests that don't need actual models"""
    class MockPredictor:
        def __init__(self, model_path=None, config_path=None):
            self.model_path = model_path
            self.config_path = config_path
            
        def predict_single(self, image_path):
            # Mock prediction based on filename
            if "dirty" in str(image_path).lower():
                return {
                    'prediction': 1,
                    'class_name': 'Dirty',
                    'confidence': 0.85,
                    'clean_prob': 0.15,
                    'dirty_prob': 0.85,
                    'probabilities': [0.15, 0.85]
                }
            else:
                return {
                    'prediction': 0,
                    'class_name': 'Clean',
                    'confidence': 0.82,
                    'clean_prob': 0.82,
                    'dirty_prob': 0.18,
                    'probabilities': [0.82, 0.18]
                }
    
    return MockPredictor

@pytest.fixture(scope="session", autouse=True)
def setup_test_environment():
    """Set up test environment variables"""
    os.environ["TESTING"] = "1"
    os.environ["LOG_LEVEL"] = "WARNING"
    yield
    # Cleanup if needed
    if "TESTING" in os.environ:
        del os.environ["TESTING"]

# Markers for different test types
def pytest_configure(config):
    """Configure pytest with custom markers"""
    config.addinivalue_line("markers", "unit: Unit tests")
    config.addinivalue_line("markers", "integration: Integration tests") 
    config.addinivalue_line("markers", "e2e: End-to-end tests")
    config.addinivalue_line("markers", "performance: Performance tests")
    config.addinivalue_line("markers", "slow: Slow running tests")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "model: Model-related tests")
    config.addinivalue_line("markers", "data: Data processing tests")