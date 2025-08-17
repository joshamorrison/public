import unittest
import torch
import tempfile
import os
import sys
import cv2
import numpy as np
from torch.utils.data import DataLoader

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_classifier.data_utils import (
    DishDataset, 
    get_data_transforms, 
    create_data_loaders, 
    prepare_dataset_structure,
    resize_and_normalize_images
)


class TestDishDataset(unittest.TestCase):
    
    def setUp(self):
        """Setup test data directory with sample images"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        self.clean_dir = os.path.join(self.data_dir, 'clean')
        self.dirty_dir = os.path.join(self.data_dir, 'dirty')
        
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.dirty_dir, exist_ok=True)
        
        # Create sample clean images
        for i in range(3):
            img = np.ones((224, 224, 3), dtype=np.uint8) * 255
            cv2.circle(img, (112, 112), 80, (240, 240, 240), -1)
            cv2.imwrite(os.path.join(self.clean_dir, f'clean_{i}.jpg'), img)
        
        # Create sample dirty images
        for i in range(3):
            img = np.ones((224, 224, 3), dtype=np.uint8) * 200
            cv2.circle(img, (112, 112), 80, (180, 150, 120), -1)
            cv2.imwrite(os.path.join(self.dirty_dir, f'dirty_{i}.jpg'), img)
    
    def tearDown(self):
        """Cleanup temporary files"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_dataset_creation(self):
        """Test DishDataset creation and basic functionality"""
        dataset = DishDataset(self.data_dir)
        
        # Should have 6 total images (3 clean + 3 dirty)
        self.assertEqual(len(dataset), 6)
        
        # Check labels are correct
        clean_count = sum(1 for i in range(len(dataset)) if dataset.labels[i] == 0)
        dirty_count = sum(1 for i in range(len(dataset)) if dataset.labels[i] == 1)
        
        self.assertEqual(clean_count, 3)
        self.assertEqual(dirty_count, 3)
    
    def test_dataset_getitem(self):
        """Test dataset __getitem__ method"""
        dataset = DishDataset(self.data_dir)
        
        # Get first item
        image, label = dataset[0]
        
        # Check types and shapes
        self.assertIsInstance(image, torch.Tensor)
        self.assertIsInstance(label, int)
        self.assertIn(label, [0, 1])  # Binary classification
        
        # Image should be PIL converted to tensor (3, H, W)
        self.assertEqual(len(image.shape), 3)
        self.assertEqual(image.shape[0], 3)  # RGB channels
    
    def test_dataset_with_transforms(self):
        """Test dataset with custom transforms"""
        from torchvision import transforms
        
        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor()
        ])
        
        dataset = DishDataset(self.data_dir, transform=transform)
        image, label = dataset[0]
        
        # Check that transform was applied
        self.assertEqual(image.shape, (3, 128, 128))
    
    def test_dataset_target_size(self):
        """Test dataset with custom target size"""
        dataset = DishDataset(self.data_dir, target_size=(256, 256))
        image, label = dataset[0]
        
        # Should still be PIL image converted to tensor without transforms
        self.assertEqual(len(image.shape), 3)
    
    def test_empty_dataset_directories(self):
        """Test dataset with empty directories"""
        empty_dir = os.path.join(self.temp_dir, 'empty')
        os.makedirs(empty_dir, exist_ok=True)
        
        dataset = DishDataset(empty_dir)
        self.assertEqual(len(dataset), 0)


class TestDataTransforms(unittest.TestCase):
    
    def test_get_data_transforms_with_augmentation(self):
        """Test data transforms with augmentation enabled"""
        train_transform, val_transform = get_data_transforms(augment=True)
        
        self.assertIsNotNone(train_transform)
        self.assertIsNotNone(val_transform)
        
        # Train transform should have more operations (augmentation)
        train_ops = len(train_transform.transforms)
        val_ops = len(val_transform.transforms)
        
        self.assertGreater(train_ops, val_ops)
    
    def test_get_data_transforms_without_augmentation(self):
        """Test data transforms without augmentation"""
        train_transform, val_transform = get_data_transforms(augment=False)
        
        self.assertIsNotNone(train_transform)
        self.assertIsNotNone(val_transform)
        
        # Without augmentation, train and val should have similar operations
        train_ops = len(train_transform.transforms)
        val_ops = len(val_transform.transforms)
        
        self.assertEqual(train_ops, val_ops)
    
    def test_transform_application(self):
        """Test that transforms can be applied to sample image"""
        from PIL import Image
        
        # Create test image
        test_image = Image.new('RGB', (300, 300), color='white')
        
        train_transform, val_transform = get_data_transforms(augment=True)
        
        # Apply transforms
        train_result = train_transform(test_image)
        val_result = val_transform(test_image)
        
        # Check output is tensor with correct shape
        self.assertIsInstance(train_result, torch.Tensor)
        self.assertIsInstance(val_result, torch.Tensor)
        
        self.assertEqual(train_result.shape, (3, 224, 224))
        self.assertEqual(val_result.shape, (3, 224, 224))


class TestDataLoaders(unittest.TestCase):
    
    def setUp(self):
        """Setup test data for data loader tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        self.clean_dir = os.path.join(self.data_dir, 'clean')
        self.dirty_dir = os.path.join(self.data_dir, 'dirty')
        
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.dirty_dir, exist_ok=True)
        
        # Create more sample images for meaningful split
        for i in range(10):
            # Clean images
            clean_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
            cv2.circle(clean_img, (112, 112), 80, (240, 240, 240), -1)
            cv2.imwrite(os.path.join(self.clean_dir, f'clean_{i}.jpg'), clean_img)
            
            # Dirty images
            dirty_img = np.ones((224, 224, 3), dtype=np.uint8) * 200
            cv2.circle(dirty_img, (112, 112), 80, (180, 150, 120), -1)
            cv2.imwrite(os.path.join(self.dirty_dir, f'dirty_{i}.jpg'), dirty_img)
    
    def tearDown(self):
        """Cleanup test data"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_create_data_loaders(self):
        """Test data loader creation"""
        train_loader, val_loader = create_data_loaders(
            self.data_dir, batch_size=4, test_split=0.2
        )
        
        self.assertIsInstance(train_loader, DataLoader)
        self.assertIsInstance(val_loader, DataLoader)
        
        # Check that we have data
        self.assertGreater(len(train_loader.dataset), 0)
        self.assertGreater(len(val_loader.dataset), 0)
        
        # Total should be 20 images (10 clean + 10 dirty)
        total_samples = len(train_loader.dataset) + len(val_loader.dataset)
        self.assertEqual(total_samples, 20)
    
    def test_data_loader_batch_size(self):
        """Test data loader respects batch size"""
        batch_size = 3
        train_loader, val_loader = create_data_loaders(
            self.data_dir, batch_size=batch_size, test_split=0.2
        )
        
        # Get first batch
        train_batch = next(iter(train_loader))
        images, labels = train_batch
        
        # Batch size should be <= specified batch size (last batch might be smaller)
        self.assertLessEqual(images.shape[0], batch_size)
        self.assertLessEqual(labels.shape[0], batch_size)
        
        # Check tensor dimensions
        self.assertEqual(images.shape[1:], (3, 224, 224))  # (batch, channels, height, width)
    
    def test_data_loader_test_split(self):
        """Test data loader respects test split ratio"""
        test_split = 0.3
        train_loader, val_loader = create_data_loaders(
            self.data_dir, batch_size=4, test_split=test_split
        )
        
        total_samples = len(train_loader.dataset) + len(val_loader.dataset)
        val_ratio = len(val_loader.dataset) / total_samples
        
        # Should be approximately the test split ratio (within 10% tolerance)
        self.assertAlmostEqual(val_ratio, test_split, delta=0.1)


class TestDatasetStructure(unittest.TestCase):
    
    def setUp(self):
        """Setup temporary directory for structure tests"""
        self.temp_dir = tempfile.mkdtemp()
    
    def tearDown(self):
        """Cleanup temporary directory"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_prepare_dataset_structure(self):
        """Test dataset structure preparation"""
        prepare_dataset_structure(self.temp_dir)
        
        # Check that required directories were created
        expected_dirs = ['clean', 'dirty', 'synthetic', 'raw_downloads']
        
        for dir_name in expected_dirs:
            dir_path = os.path.join(self.temp_dir, dir_name)
            self.assertTrue(os.path.exists(dir_path))
            self.assertTrue(os.path.isdir(dir_path))


class TestImageProcessing(unittest.TestCase):
    
    def setUp(self):
        """Setup test directories and images"""
        self.temp_dir = tempfile.mkdtemp()
        self.input_dir = os.path.join(self.temp_dir, 'input')
        self.output_dir = os.path.join(self.temp_dir, 'output')
        
        os.makedirs(self.input_dir, exist_ok=True)
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Create test images with different sizes
        for i, size in enumerate([(100, 100), (300, 200), (150, 150)]):
            img = np.random.randint(0, 255, (*size, 3), dtype=np.uint8)
            cv2.imwrite(os.path.join(self.input_dir, f'test_{i}.jpg'), img)
    
    def tearDown(self):
        """Cleanup test directories"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_resize_and_normalize_images(self):
        """Test image resizing and normalization"""
        target_size = (224, 224)
        
        resize_and_normalize_images(self.input_dir, self.output_dir, target_size)
        
        # Check that output files were created
        input_files = [f for f in os.listdir(self.input_dir) if f.endswith('.jpg')]
        output_files = [f for f in os.listdir(self.output_dir) if f.endswith('.jpg')]
        
        self.assertEqual(len(input_files), len(output_files))
        
        # Check that output images have correct size
        for filename in output_files:
            img_path = os.path.join(self.output_dir, filename)
            img = cv2.imread(img_path)
            
            self.assertIsNotNone(img)
            self.assertEqual(img.shape[:2], target_size)  # (height, width)


class TestDataUtilsIntegration(unittest.TestCase):
    """Integration tests for data utilities"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.data_dir = os.path.join(self.temp_dir, 'data')
        
        # Prepare dataset structure
        prepare_dataset_structure(self.data_dir)
        
        # Create sample data
        clean_dir = os.path.join(self.data_dir, 'clean')
        dirty_dir = os.path.join(self.data_dir, 'dirty')
        
        # Create sample images
        for i in range(5):
            # Clean images
            clean_img = np.ones((200, 200, 3), dtype=np.uint8) * 255
            cv2.circle(clean_img, (100, 100), 80, (240, 240, 240), -1)
            cv2.imwrite(os.path.join(clean_dir, f'clean_{i}.jpg'), clean_img)
            
            # Dirty images with variations
            dirty_img = np.ones((180, 180, 3), dtype=np.uint8) * 200
            cv2.circle(dirty_img, (90, 90), 70, (180, 150, 120), -1)
            # Add some "dirt" spots
            cv2.circle(dirty_img, (60, 60), 10, (100, 80, 60), -1)
            cv2.circle(dirty_img, (120, 120), 8, (90, 70, 50), -1)
            cv2.imwrite(os.path.join(dirty_dir, f'dirty_{i}.jpg'), dirty_img)
    
    def tearDown(self):
        """Cleanup integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_data_pipeline(self):
        """Test complete data processing pipeline"""
        # Step 1: Create dataset
        dataset = DishDataset(self.data_dir)
        self.assertEqual(len(dataset), 10)  # 5 clean + 5 dirty
        
        # Step 2: Create data loaders
        train_loader, val_loader = create_data_loaders(
            self.data_dir, batch_size=2, test_split=0.2
        )
        
        # Step 3: Verify data can be loaded and processed
        train_batch = next(iter(train_loader))
        images, labels = train_batch
        
        self.assertEqual(images.shape[1:], (3, 224, 224))
        self.assertTrue(torch.all((labels == 0) | (labels == 1)))  # Binary labels
        
        # Step 4: Verify both classes are present in dataset
        all_labels = []
        for _, label in dataset:
            all_labels.append(label)
        
        self.assertIn(0, all_labels)  # Clean class
        self.assertIn(1, all_labels)  # Dirty class


if __name__ == '__main__':
    unittest.main()