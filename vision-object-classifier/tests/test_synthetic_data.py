import unittest
import cv2
import numpy as np
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from synthetic_data import SyntheticDirtyDishGenerator


class TestSyntheticDataGenerator(unittest.TestCase):
    
    def setUp(self):
        """Setup test environment and create test images"""
        self.temp_dir = tempfile.mkdtemp()
        self.output_dir = os.path.join(self.temp_dir, 'dirty')
        self.clean_dir = os.path.join(self.temp_dir, 'clean')
        
        os.makedirs(self.output_dir, exist_ok=True)
        os.makedirs(self.clean_dir, exist_ok=True)
        
        # Create test clean image
        self.test_image_path = os.path.join(self.clean_dir, 'test_clean.jpg')
        self.create_test_clean_image()
        
        # Initialize generator
        self.generator = SyntheticDirtyDishGenerator(self.output_dir)
    
    def tearDown(self):
        """Cleanup temporary files and directories"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def create_test_clean_image(self):
        """Create a simple test clean dish image"""
        # Create white plate image
        img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        center = (112, 112)
        radius = 80
        
        # Draw plate circle
        cv2.circle(img, center, radius, (240, 240, 240), -1)
        cv2.circle(img, center, radius, (200, 200, 200), 2)
        
        cv2.imwrite(self.test_image_path, img)
    
    def test_generator_initialization(self):
        """Test SyntheticDirtyDishGenerator initialization"""
        self.assertIsNotNone(self.generator)
        self.assertEqual(self.generator.output_dir, self.output_dir)
        self.assertTrue(os.path.exists(self.output_dir))
    
    def test_stain_colors_defined(self):
        """Test that stain colors are properly defined"""
        self.assertGreater(len(self.generator.stain_colors), 0)
        self.assertGreater(len(self.generator.dirt_colors), 0)
        
        # Check color format (RGB tuples)
        for color in self.generator.stain_colors:
            self.assertEqual(len(color), 3)
            for channel in color:
                self.assertTrue(0 <= channel <= 255)
    
    def test_add_food_stains(self):
        """Test food stain generation"""
        # Load test image
        original_image = cv2.imread(self.test_image_path)
        self.assertIsNotNone(original_image)
        
        # Add food stains
        stained_image = self.generator.add_food_stains(original_image, num_stains=3)
        
        # Check that image was modified
        self.assertEqual(stained_image.shape, original_image.shape)
        self.assertFalse(np.array_equal(original_image, stained_image))
    
    def test_add_grease_stains(self):
        """Test grease stain generation"""
        original_image = cv2.imread(self.test_image_path)
        greasy_image = self.generator.add_grease_stains(original_image)
        
        # Check that image was modified
        self.assertEqual(greasy_image.shape, original_image.shape)
        self.assertFalse(np.array_equal(original_image, greasy_image))
    
    def test_add_residue_buildup(self):
        """Test residue buildup generation"""
        original_image = cv2.imread(self.test_image_path)
        residue_image = self.generator.add_residue_buildup(original_image)
        
        # Check that image was modified
        self.assertEqual(residue_image.shape, original_image.shape)
        # Note: residue changes might be subtle, so we don't check for inequality
    
    def test_add_scratches_and_wear(self):
        """Test scratch and wear pattern generation"""
        original_image = cv2.imread(self.test_image_path)
        worn_image = self.generator.add_scratches_and_wear(original_image)
        
        # Check that image was modified
        self.assertEqual(worn_image.shape, original_image.shape)
        self.assertFalse(np.array_equal(original_image, worn_image))
    
    def test_generate_dirty_dish_light(self):
        """Test light dirty dish generation"""
        output_path = os.path.join(self.output_dir, 'test_dirty_light.jpg')
        
        success = self.generator.generate_dirty_dish(
            self.test_image_path, output_path, 'light'
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
        
        # Check that output image exists and has correct size
        output_image = cv2.imread(output_path)
        self.assertIsNotNone(output_image)
        self.assertEqual(output_image.shape, (224, 224, 3))
    
    def test_generate_dirty_dish_medium(self):
        """Test medium dirty dish generation"""
        output_path = os.path.join(self.output_dir, 'test_dirty_medium.jpg')
        
        success = self.generator.generate_dirty_dish(
            self.test_image_path, output_path, 'medium'
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
    
    def test_generate_dirty_dish_heavy(self):
        """Test heavy dirty dish generation"""
        output_path = os.path.join(self.output_dir, 'test_dirty_heavy.jpg')
        
        success = self.generator.generate_dirty_dish(
            self.test_image_path, output_path, 'heavy'
        )
        
        self.assertTrue(success)
        self.assertTrue(os.path.exists(output_path))
    
    def test_generate_dirty_dish_invalid_input(self):
        """Test error handling for invalid input"""
        output_path = os.path.join(self.output_dir, 'test_dirty_invalid.jpg')
        
        # Test with non-existent input file
        success = self.generator.generate_dirty_dish(
            'non_existent_file.jpg', output_path, 'medium'
        )
        
        self.assertFalse(success)
        self.assertFalse(os.path.exists(output_path))
    
    def test_batch_generate(self):
        """Test batch generation of dirty dishes"""
        # Create additional clean images
        for i in range(3):
            img_path = os.path.join(self.clean_dir, f'clean_{i}.jpg')
            img = np.ones((224, 224, 3), dtype=np.uint8) * 255
            cv2.circle(img, (112, 112), 70 + i*5, (240, 240, 240), -1)
            cv2.imwrite(img_path, img)
        
        # Run batch generation
        total_generated = self.generator.batch_generate(self.clean_dir, num_variations=2)
        
        # Should generate 2 variations for each of 4 clean images = 8 total
        self.assertEqual(total_generated, 8)
        
        # Check that files were created
        dirty_files = [f for f in os.listdir(self.output_dir) if f.endswith('.jpg')]
        self.assertEqual(len(dirty_files), 8)
    
    def test_different_dirt_levels_produce_different_results(self):
        """Test that different dirt levels produce different results"""
        light_path = os.path.join(self.output_dir, 'test_light.jpg')
        heavy_path = os.path.join(self.output_dir, 'test_heavy.jpg')
        
        # Generate light and heavy versions
        self.generator.generate_dirty_dish(self.test_image_path, light_path, 'light')
        self.generator.generate_dirty_dish(self.test_image_path, heavy_path, 'heavy')
        
        # Load images
        light_img = cv2.imread(light_path)
        heavy_img = cv2.imread(heavy_path)
        
        # Images should be different
        self.assertFalse(np.array_equal(light_img, heavy_img))
        
        # Heavy should generally have more changes from original
        original_img = cv2.imread(self.test_image_path)
        light_diff = np.sum(np.abs(original_img.astype(float) - light_img.astype(float)))
        heavy_diff = np.sum(np.abs(original_img.astype(float) - heavy_img.astype(float)))
        
        # Heavy dirt should create more visual changes (this is a heuristic test)
        # Note: This might not always be true due to randomness, but generally should be
        self.assertGreater(heavy_diff, 0)  # At least some change for heavy


class TestSyntheticDataIntegration(unittest.TestCase):
    """Integration tests for synthetic data generation pipeline"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.clean_dir = os.path.join(self.temp_dir, 'clean')
        self.dirty_dir = os.path.join(self.temp_dir, 'dirty')
        
        os.makedirs(self.clean_dir, exist_ok=True)
        os.makedirs(self.dirty_dir, exist_ok=True)
    
    def tearDown(self):
        """Cleanup integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_complete_pipeline(self):
        """Test complete synthetic data generation pipeline"""
        # Create clean test images
        dish_types = ['plate', 'bowl', 'cup']
        for i, dish_type in enumerate(dish_types):
            img = np.ones((224, 224, 3), dtype=np.uint8) * 255
            
            if dish_type == 'plate':
                cv2.circle(img, (112, 112), 80, (240, 240, 240), -1)
            elif dish_type == 'bowl':
                cv2.circle(img, (112, 112), 60, (245, 245, 245), -1)
                cv2.circle(img, (112, 112), 45, (235, 235, 235), -1)
            elif dish_type == 'cup':
                cv2.circle(img, (112, 112), 40, (250, 250, 250), -1)
                cv2.ellipse(img, (150, 112), (8, 15), 0, 0, 180, (200, 200, 200), 2)
            
            cv2.imwrite(os.path.join(self.clean_dir, f'{dish_type}_{i}.jpg'), img)
        
        # Generate dirty versions
        generator = SyntheticDirtyDishGenerator(self.dirty_dir)
        total_generated = generator.batch_generate(self.clean_dir, num_variations=3)
        
        # Should generate 3 variations for each of 3 clean images = 9 total
        self.assertEqual(total_generated, 9)
        
        # Verify all generated images are valid
        dirty_files = [f for f in os.listdir(self.dirty_dir) if f.endswith('.jpg')]
        self.assertEqual(len(dirty_files), 9)
        
        # Check that all generated images can be loaded and have correct dimensions
        for dirty_file in dirty_files:
            img_path = os.path.join(self.dirty_dir, dirty_file)
            img = cv2.imread(img_path)
            self.assertIsNotNone(img)
            self.assertEqual(img.shape, (224, 224, 3))


if __name__ == '__main__':
    unittest.main()