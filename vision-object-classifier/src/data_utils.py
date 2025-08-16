import os
import cv2
import numpy as np
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import random
from typing import Tuple, List, Optional


class DishDataset(Dataset):
    def __init__(self, data_dir: str, transform=None, target_size=(224, 224)):
        self.data_dir = data_dir
        self.transform = transform
        self.target_size = target_size
        self.images = []
        self.labels = []
        
        # Load clean and dirty dish images
        self._load_images()
        
    def _load_images(self):
        """Load images from clean and dirty subdirectories"""
        clean_dir = os.path.join(self.data_dir, 'clean')
        dirty_dir = os.path.join(self.data_dir, 'dirty')
        
        # Load clean images (label = 0)
        if os.path.exists(clean_dir):
            for filename in os.listdir(clean_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(clean_dir, filename))
                    self.labels.append(0)
        
        # Load dirty images (label = 1)
        if os.path.exists(dirty_dir):
            for filename in os.listdir(dirty_dir):
                if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
                    self.images.append(os.path.join(dirty_dir, filename))
                    self.labels.append(1)
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image_path = self.images[idx]
        label = self.labels[idx]
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        image = image.resize(self.target_size)
        
        if self.transform:
            image = self.transform(image)
        
        return image, label


def get_data_transforms(augment=True):
    """Get data transformation pipelines"""
    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((256, 256)),
            transforms.RandomCrop((224, 224)),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return train_transform, val_transform


def create_data_loaders(data_dir: str, batch_size: int = 32, test_split: float = 0.2):
    """Create train and validation data loaders"""
    train_transform, val_transform = get_data_transforms(augment=True)
    
    # Load full dataset
    full_dataset = DishDataset(data_dir, transform=None)
    
    # Split into train and validation
    dataset_size = len(full_dataset)
    indices = list(range(dataset_size))
    split = int(np.floor(test_split * dataset_size))
    
    np.random.shuffle(indices)
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Create datasets with transforms
    train_dataset = DishDataset(data_dir, transform=train_transform)
    val_dataset = DishDataset(data_dir, transform=val_transform)
    
    # Subset datasets
    train_subset = torch.utils.data.Subset(train_dataset, train_indices)
    val_subset = torch.utils.data.Subset(val_dataset, val_indices)
    
    # Create data loaders
    train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    return train_loader, val_loader


def download_kaggle_dataset(dataset_name: str, download_dir: str):
    """Download dataset from Kaggle using kaggle API"""
    try:
        import kaggle
        kaggle.api.authenticate()
        kaggle.api.dataset_download_files(dataset_name, path=download_dir, unzip=True)
        print(f"Downloaded {dataset_name} to {download_dir}")
    except ImportError:
        print("Kaggle API not installed. Install with: pip install kaggle")
    except Exception as e:
        print(f"Error downloading dataset: {e}")


def prepare_dataset_structure(base_dir: str):
    """Create directory structure for training data"""
    dirs_to_create = [
        os.path.join(base_dir, 'clean'),
        os.path.join(base_dir, 'dirty'),
        os.path.join(base_dir, 'synthetic'),
        os.path.join(base_dir, 'raw_downloads')
    ]
    
    for dir_path in dirs_to_create:
        os.makedirs(dir_path, exist_ok=True)
    
    print(f"Created dataset structure in {base_dir}")


def resize_and_normalize_images(input_dir: str, output_dir: str, target_size: Tuple[int, int] = (224, 224)):
    """Resize and normalize images for consistent preprocessing"""
    os.makedirs(output_dir, exist_ok=True)
    
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, filename)
            
            # Load, resize and save image
            image = cv2.imread(input_path)
            if image is not None:
                resized = cv2.resize(image, target_size)
                cv2.imwrite(output_path, resized)
    
    print(f"Processed images from {input_dir} to {output_dir}")