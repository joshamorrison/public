"""
Synthetic data generation utilities for vision classifier
Moved from src/vision_classifier/synthetic_data.py for better organization
"""

import sys
from pathlib import Path

# Add src to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root / "src"))

# Import the actual generator
from vision_classifier.synthetic_data import SyntheticDirtyDishGenerator

def generate_sample_data():
    """Generate sample synthetic data for demos"""
    # Create generator pointing to processed data
    data_dir = Path(__file__).parent.parent
    generator = SyntheticDirtyDishGenerator(str(data_dir / "processed" / "dirty_labeled"))
    
    # Generate from clean samples if they exist
    clean_samples = data_dir / "samples" / "demo_images"
    if clean_samples.exists():
        print("Generating synthetic dirty variants from clean samples...")
        generator.batch_generate(str(clean_samples), num_variations=2)
    else:
        print("No clean samples found for synthetic generation")

if __name__ == "__main__":
    generate_sample_data()