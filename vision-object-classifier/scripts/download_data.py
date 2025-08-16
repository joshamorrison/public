#!/usr/bin/env python3
"""
Script to download and prepare datasets for dish cleanliness classification
"""

import os
import sys
import argparse
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from data_utils import download_kaggle_dataset, prepare_dataset_structure
from synthetic_data import SyntheticDirtyDishGenerator


def download_kaggle_data(output_dir):
    """Download relevant datasets from Kaggle"""
    datasets_to_download = [
        "platesv2",  # Cleaned vs Dirty V2 competition
        "plates",    # Original Cleaned vs Dirty competition  
    ]
    
    kaggle_dir = os.path.join(output_dir, 'kaggle_downloads')
    os.makedirs(kaggle_dir, exist_ok=True)
    
    for dataset in datasets_to_download:
        print(f"Downloading {dataset} from Kaggle...")
        try:
            download_kaggle_dataset(dataset, kaggle_dir)
            print(f"Successfully downloaded {dataset}")
        except Exception as e:
            print(f"Failed to download {dataset}: {e}")
    
    return kaggle_dir


def setup_data_structure(base_dir):
    """Setup the data directory structure"""
    print(f"Setting up data structure in {base_dir}")
    prepare_dataset_structure(base_dir)
    
    # Create additional directories
    additional_dirs = [
        'kaggle_downloads',
        'external_datasets', 
        'augmented',
        'validation',
        'test'
    ]
    
    for dir_name in additional_dirs:
        dir_path = os.path.join(base_dir, dir_name)
        os.makedirs(dir_path, exist_ok=True)
    
    print("Data structure created successfully!")


def generate_sample_synthetic_data(data_dir):
    """Generate some sample synthetic data for testing"""
    clean_dir = os.path.join(data_dir, 'clean')
    synthetic_dir = os.path.join(data_dir, 'synthetic')
    
    if not os.path.exists(clean_dir) or len(os.listdir(clean_dir)) == 0:
        print("No clean images found. Please add clean dish images to data/clean/ first.")
        return
    
    print("Generating sample synthetic dirty dishes...")
    generator = SyntheticDirtyDishGenerator(synthetic_dir)
    
    try:
        total_generated = generator.batch_generate(clean_dir, num_variations=2)
        print(f"Generated {total_generated} synthetic dirty dish images")
    except Exception as e:
        print(f"Error generating synthetic data: {e}")


def create_sample_instructions(data_dir):
    """Create instruction file for data preparation"""
    instructions_file = os.path.join(data_dir, 'DATA_PREPARATION.md')
    
    instructions = """# Data Preparation Instructions

## Directory Structure
```
data/
├── clean/          # Clean dish images
├── dirty/          # Dirty dish images (real + synthetic)
├── synthetic/      # Generated synthetic dirty images
├── kaggle_downloads/  # Downloaded Kaggle datasets
├── external_datasets/ # Other external datasets
├── augmented/      # Augmented training data
├── validation/     # Validation set
└── test/          # Test set
```

## Data Collection Steps

### 1. Kaggle Datasets
Run this script with `--download-kaggle` to automatically download:
- Cleaned vs Dirty V2 competition data
- Original Cleaned vs Dirty competition data

### 2. Manual Data Collection
- Place clean dish images in `data/clean/`
- Place dirty dish images in `data/dirty/`
- Supported formats: JPG, PNG, JPEG

### 3. Synthetic Data Generation
- Run this script with `--generate-synthetic` to create synthetic dirty images
- Or use: `python src/synthetic_data.py`

### 4. External Datasets
Consider downloading from:
- Roboflow Universe (dirty/clean object datasets)
- COCO dataset (for general object detection)
- ImageNet (for transfer learning)

## Quality Guidelines

### Clean Images
- Well-lit, clear images
- Various dish types (plates, bowls, cups)
- Different angles and backgrounds
- No visible food residue or stains

### Dirty Images  
- Visible food stains, grease, or residue
- Various levels of dirtiness (light to heavy)
- Same dish types as clean images
- Consistent lighting and quality

## Data Validation
After collecting data, run:
```bash
python scripts/validate_data.py --data-dir data/
```
"""
    
    with open(instructions_file, 'w') as f:
        f.write(instructions)
    
    print(f"Created data preparation instructions: {instructions_file}")


def main():
    parser = argparse.ArgumentParser(description='Download and prepare datasets')
    parser.add_argument('--data-dir', default='../data', 
                       help='Base directory for data (default: ../data)')
    parser.add_argument('--download-kaggle', action='store_true',
                       help='Download datasets from Kaggle')
    parser.add_argument('--generate-synthetic', action='store_true',
                       help='Generate synthetic dirty dish data')
    parser.add_argument('--setup-only', action='store_true',
                       help='Only setup directory structure')
    
    args = parser.parse_args()
    
    # Convert to absolute path
    data_dir = os.path.abspath(args.data_dir)
    print(f"Using data directory: {data_dir}")
    
    # Setup directory structure
    setup_data_structure(data_dir)
    
    if args.setup_only:
        print("Setup complete!")
        return
    
    # Download Kaggle data if requested
    if args.download_kaggle:
        try:
            download_kaggle_data(data_dir)
        except Exception as e:
            print(f"Kaggle download failed: {e}")
            print("Make sure you have kaggle API configured:")
            print("1. pip install kaggle")
            print("2. Setup kaggle API credentials")
    
    # Generate synthetic data if requested
    if args.generate_synthetic:
        generate_sample_synthetic_data(data_dir)
    
    # Create instructions
    create_sample_instructions(data_dir)
    
    print("\nData preparation complete!")
    print(f"Next steps:")
    print(f"1. Add clean dish images to: {os.path.join(data_dir, 'clean')}")
    print(f"2. Add dirty dish images to: {os.path.join(data_dir, 'dirty')}")
    print(f"3. Run training: python src/train.py")


if __name__ == "__main__":
    main()