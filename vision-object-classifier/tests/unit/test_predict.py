import unittest
import torch
import tempfile
import os
import sys
import cv2
import numpy as np
import json

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from vision_classifier.predict import DishCleanlinessPredictor
from vision_classifier.model import create_model, save_model


class TestDishCleanlinessPredictor(unittest.TestCase):
    
    def setUp(self):
        """Setup test environment with model and test images"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model.pth')
        self.config_path = os.path.join(self.temp_dir, 'test_config.json')
        self.test_image_path = os.path.join(self.temp_dir, 'test_image.jpg')
        
        # Create and save test model
        model = create_model('custom_cnn', num_classes=2, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        save_model(model, optimizer, 0, 0.5, 0.8, self.model_path)
        
        # Create test config
        config = {
            'model_name': 'custom_cnn',
            'num_classes': 2,
            'pretrained': False
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
        
        # Create test image
        test_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        cv2.circle(test_img, (112, 112), 80, (240, 240, 240), -1)
        cv2.imwrite(self.test_image_path, test_img)
    
    def tearDown(self):
        """Cleanup test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_predictor_initialization_with_config(self):
        """Test predictor initialization with config file"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        self.assertIsNotNone(predictor)
        self.assertIsNotNone(predictor.model)
        self.assertEqual(predictor.device.type, 'cpu')
        self.assertEqual(len(predictor.class_names), 2)
    
    def test_predictor_initialization_without_config(self):
        """Test predictor initialization without config file"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            device='cpu'
        )
        
        self.assertIsNotNone(predictor)
        self.assertIsNotNone(predictor.model)
    
    def test_preprocess_image_from_file(self):
        """Test image preprocessing from file path"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        tensor = predictor.preprocess_image(self.test_image_path)
        
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))
    
    def test_preprocess_image_from_numpy(self):
        """Test image preprocessing from numpy array"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        # Create numpy array (BGR format as OpenCV loads)
        img_array = np.ones((224, 224, 3), dtype=np.uint8) * 255
        
        tensor = predictor.preprocess_image(img_array)
        
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))
    
    def test_preprocess_image_from_pil(self):
        """Test image preprocessing from PIL Image"""
        from PIL import Image
        
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        pil_img = Image.new('RGB', (224, 224), color='white')
        tensor = predictor.preprocess_image(pil_img)
        
        self.assertIsInstance(tensor, torch.Tensor)
        self.assertEqual(tensor.shape, (3, 224, 224))
    
    def test_preprocess_image_invalid_input(self):
        """Test error handling for invalid image input"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        # Test with non-existent file
        with self.assertRaises(FileNotFoundError):
            predictor.preprocess_image('non_existent_file.jpg')
        
        # Test with invalid input type
        with self.assertRaises(ValueError):
            predictor.preprocess_image(123)  # Invalid type
    
    def test_predict_single_basic(self):
        """Test basic single image prediction"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        result = predictor.predict_single(self.test_image_path)
        
        # Check result structure
        expected_keys = ['prediction', 'class_name', 'confidence', 'clean_prob', 'dirty_prob', 'probabilities']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check data types and ranges
        self.assertIsInstance(result['prediction'], int)
        self.assertIn(result['prediction'], [0, 1])
        self.assertIsInstance(result['class_name'], str)
        self.assertIn(result['class_name'], ['Clean', 'Dirty'])
        
        self.assertTrue(0 <= result['confidence'] <= 1)
        self.assertTrue(0 <= result['clean_prob'] <= 1)
        self.assertTrue(0 <= result['dirty_prob'] <= 1)
        
        # Probabilities should sum to approximately 1
        prob_sum = result['clean_prob'] + result['dirty_prob']
        self.assertAlmostEqual(prob_sum, 1.0, places=5)
    
    def test_predict_single_with_return_image(self):
        """Test single image prediction with image return"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        result = predictor.predict_single(self.test_image_path, return_image=True)
        
        self.assertIn('original_image', result)
        self.assertIsInstance(result['original_image'], np.ndarray)
        self.assertEqual(len(result['original_image'].shape), 3)  # Should be 3D array
    
    def test_predict_batch(self):
        """Test batch prediction functionality"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        # Create multiple test images
        test_images = []
        for i in range(3):
            img_path = os.path.join(self.temp_dir, f'test_img_{i}.jpg')
            test_img = np.ones((224, 224, 3), dtype=np.uint8) * (200 + i * 10)
            cv2.circle(test_img, (112, 112), 80, (180 + i * 20, 180 + i * 20, 180 + i * 20), -1)
            cv2.imwrite(img_path, test_img)
            test_images.append(img_path)
        
        results = predictor.predict_batch(test_images, batch_size=2)
        
        # Should have results for all images
        self.assertEqual(len(results), 3)
        
        # Check that all results have required structure
        for result in results:
            self.assertIn('image_path', result)
            self.assertIn('success', result)
            
            if result['success']:
                expected_keys = ['prediction', 'class_name', 'confidence', 'clean_prob', 'dirty_prob']
                for key in expected_keys:
                    self.assertIn(key, result)
    
    def test_predict_batch_with_invalid_images(self):
        """Test batch prediction with some invalid images"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        # Mix valid and invalid image paths
        test_images = [
            self.test_image_path,  # Valid
            'non_existent_file.jpg',  # Invalid
            self.test_image_path,  # Valid
        ]
        
        results = predictor.predict_batch(test_images)
        
        self.assertEqual(len(results), 3)
        
        # Check that invalid images are marked as failed
        self.assertTrue(results[0]['success'])  # Valid
        self.assertFalse(results[1]['success'])  # Invalid
        self.assertTrue(results[2]['success'])  # Valid
        
        # Failed result should have error information
        self.assertIn('error', results[1])
    
    def test_visualize_prediction(self):
        """Test prediction visualization (basic functionality)"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        # Test without saving (just check it doesn't crash)
        try:
            result = predictor.visualize_prediction(self.test_image_path)
            self.assertIsNotNone(result)
        except Exception as e:
            # Visualization might fail due to matplotlib backend issues in testing
            # We'll just check that the method exists and can be called
            self.fail(f"Visualization failed: {e}")


class TestPredictionIntegration(unittest.TestCase):
    """Integration tests for the complete prediction pipeline"""
    
    def setUp(self):
        """Setup integration test environment"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'integration_model.pth')
        self.config_path = os.path.join(self.temp_dir, 'integration_config.json')
        
        # Create and train a simple model for integration testing
        model = create_model('custom_cnn', num_classes=2, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        
        # Simple training on dummy data to make model more realistic
        model.train()
        dummy_input = torch.randn(4, 3, 224, 224)
        dummy_target = torch.tensor([0, 1, 0, 1])
        criterion = torch.nn.CrossEntropyLoss()
        
        for _ in range(5):  # Quick training
            optimizer.zero_grad()
            output = model(dummy_input)
            loss = criterion(output, dummy_target)
            loss.backward()
            optimizer.step()
        
        save_model(model, optimizer, 5, loss.item(), 0.75, self.model_path)
        
        # Create config
        config = {
            'model_name': 'custom_cnn',
            'num_classes': 2,
            'pretrained': False
        }
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
    
    def tearDown(self):
        """Cleanup integration test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_end_to_end_prediction_pipeline(self):
        """Test complete end-to-end prediction pipeline"""
        # Create realistic test images
        clean_img_path = os.path.join(self.temp_dir, 'clean_dish.jpg')
        dirty_img_path = os.path.join(self.temp_dir, 'dirty_dish.jpg')
        
        # Clean dish (bright white with clean circle)
        clean_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        cv2.circle(clean_img, (112, 112), 80, (245, 245, 245), -1)
        cv2.circle(clean_img, (112, 112), 80, (200, 200, 200), 2)
        cv2.imwrite(clean_img_path, clean_img)
        
        # Dirty dish (darker with stains)
        dirty_img = np.ones((224, 224, 3), dtype=np.uint8) * 200
        cv2.circle(dirty_img, (112, 112), 80, (180, 160, 140), -1)
        # Add "stains"
        cv2.circle(dirty_img, (90, 90), 15, (120, 100, 80), -1)
        cv2.circle(dirty_img, (130, 130), 10, (100, 80, 60), -1)
        cv2.circle(dirty_img, (100, 130), 8, (90, 70, 50), -1)
        cv2.imwrite(dirty_img_path, dirty_img)
        
        # Initialize predictor
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        # Test single predictions
        clean_result = predictor.predict_single(clean_img_path)
        dirty_result = predictor.predict_single(dirty_img_path)
        
        # Verify results structure
        for result in [clean_result, dirty_result]:
            self.assertIn('class_name', result)
            self.assertIn('confidence', result)
            self.assertIn('clean_prob', result)
            self.assertIn('dirty_prob', result)
            self.assertTrue(0 <= result['confidence'] <= 1)
        
        # Test batch prediction
        batch_results = predictor.predict_batch([clean_img_path, dirty_img_path])
        
        self.assertEqual(len(batch_results), 2)
        
        for result in batch_results:
            self.assertTrue(result['success'])
            self.assertIn('class_name', result)
            self.assertIn('confidence', result)
    
    def test_prediction_consistency(self):
        """Test that predictions are consistent across multiple calls"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        # Create test image
        test_img_path = os.path.join(self.temp_dir, 'consistency_test.jpg')
        test_img = np.ones((224, 224, 3), dtype=np.uint8) * 255
        cv2.circle(test_img, (112, 112), 80, (240, 240, 240), -1)
        cv2.imwrite(test_img_path, test_img)
        
        # Run prediction multiple times
        results = []
        for _ in range(3):
            result = predictor.predict_single(test_img_path)
            results.append(result)
        
        # All predictions should be identical (deterministic model)
        first_result = results[0]
        for result in results[1:]:
            self.assertEqual(result['prediction'], first_result['prediction'])
            self.assertAlmostEqual(result['confidence'], first_result['confidence'], places=5)
            self.assertAlmostEqual(result['clean_prob'], first_result['clean_prob'], places=5)
            self.assertAlmostEqual(result['dirty_prob'], first_result['dirty_prob'], places=5)


class TestPredictionErrorHandling(unittest.TestCase):
    """Test error handling in prediction pipeline"""
    
    def setUp(self):
        """Setup test environment for error handling tests"""
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'error_test_model.pth')
        self.config_path = os.path.join(self.temp_dir, 'error_test_config.json')
        
        # Create minimal model and config
        model = create_model('custom_cnn', num_classes=2, pretrained=False)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        save_model(model, optimizer, 0, 0.5, 0.8, self.model_path)
        
        config = {'model_name': 'custom_cnn', 'num_classes': 2, 'pretrained': False}
        with open(self.config_path, 'w') as f:
            json.dump(config, f)
    
    def tearDown(self):
        """Cleanup error test environment"""
        import shutil
        shutil.rmtree(self.temp_dir)
    
    def test_invalid_model_path(self):
        """Test error handling for invalid model path"""
        with self.assertRaises(Exception):
            DishCleanlinessPredictor(
                model_path='non_existent_model.pth',
                device='cpu'
            )
    
    def test_invalid_config_path(self):
        """Test handling of invalid config path (should not crash)"""
        # Should work without config (uses defaults)
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path='non_existent_config.json',
            device='cpu'
        )
        self.assertIsNotNone(predictor)
    
    def test_corrupted_image_file(self):
        """Test error handling for corrupted image files"""
        predictor = DishCleanlinessPredictor(
            model_path=self.model_path,
            config_path=self.config_path,
            device='cpu'
        )
        
        # Create corrupted image file
        corrupted_img_path = os.path.join(self.temp_dir, 'corrupted.jpg')
        with open(corrupted_img_path, 'w') as f:
            f.write('This is not an image file')
        
        # Should handle gracefully
        with self.assertRaises(Exception):
            predictor.predict_single(corrupted_img_path)


if __name__ == '__main__':
    unittest.main()