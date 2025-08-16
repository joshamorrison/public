import unittest
import torch
import tempfile
import os
import sys

# Add src to path for imports
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from model import create_model, save_model, load_model, ModelEvaluator, FocalLoss


class TestModelCreation(unittest.TestCase):
    
    def test_custom_cnn_creation(self):
        """Test custom CNN model creation"""
        model = create_model('custom_cnn', num_classes=2, pretrained=False)
        self.assertIsNotNone(model)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 2))
    
    def test_resnet50_creation(self):
        """Test ResNet50 model creation"""
        model = create_model('resnet50', num_classes=2, pretrained=False)
        self.assertIsNotNone(model)
        
        # Test forward pass
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 2))
    
    def test_invalid_model_name(self):
        """Test error handling for invalid model names"""
        with self.assertRaises(ValueError):
            create_model('invalid_model', num_classes=2)
    
    def test_different_num_classes(self):
        """Test model creation with different number of classes"""
        model = create_model('custom_cnn', num_classes=5, pretrained=False)
        dummy_input = torch.randn(1, 3, 224, 224)
        output = model(dummy_input)
        self.assertEqual(output.shape, (1, 5))


class TestModelSaveLoad(unittest.TestCase):
    
    def setUp(self):
        """Setup test model and temporary directory"""
        self.model = create_model('custom_cnn', num_classes=2, pretrained=False)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        self.temp_dir = tempfile.mkdtemp()
        self.model_path = os.path.join(self.temp_dir, 'test_model.pth')
    
    def tearDown(self):
        """Cleanup temporary files"""
        if os.path.exists(self.model_path):
            os.remove(self.model_path)
        os.rmdir(self.temp_dir)
    
    def test_save_model(self):
        """Test model saving functionality"""
        save_model(self.model, self.optimizer, epoch=1, loss=0.5, accuracy=0.8, filepath=self.model_path)
        self.assertTrue(os.path.exists(self.model_path))
    
    def test_load_model(self):
        """Test model loading functionality"""
        # Save model first
        save_model(self.model, self.optimizer, epoch=1, loss=0.5, accuracy=0.8, filepath=self.model_path)
        
        # Load model
        loaded_model = load_model(self.model_path, model_name='custom_cnn', num_classes=2, device='cpu')
        self.assertIsNotNone(loaded_model)
        
        # Test that loaded model produces same output
        dummy_input = torch.randn(1, 3, 224, 224)
        original_output = self.model(dummy_input)
        loaded_output = loaded_model(dummy_input)
        
        # Should have same shape
        self.assertEqual(original_output.shape, loaded_output.shape)


class TestFocalLoss(unittest.TestCase):
    
    def test_focal_loss_creation(self):
        """Test FocalLoss creation and forward pass"""
        focal_loss = FocalLoss(alpha=1, gamma=2)
        
        # Create dummy predictions and targets
        predictions = torch.randn(4, 2)  # 4 samples, 2 classes
        targets = torch.tensor([0, 1, 0, 1])
        
        loss = focal_loss(predictions, targets)
        self.assertTrue(torch.is_tensor(loss))
        self.assertEqual(loss.shape, ())  # Scalar loss
    
    def test_focal_loss_reduction(self):
        """Test different reduction modes"""
        predictions = torch.randn(4, 2)
        targets = torch.tensor([0, 1, 0, 1])
        
        # Test mean reduction
        focal_loss_mean = FocalLoss(alpha=1, gamma=2, reduction='mean')
        loss_mean = focal_loss_mean(predictions, targets)
        self.assertEqual(loss_mean.shape, ())
        
        # Test sum reduction
        focal_loss_sum = FocalLoss(alpha=1, gamma=2, reduction='sum')
        loss_sum = focal_loss_sum(predictions, targets)
        self.assertEqual(loss_sum.shape, ())
        
        # Test none reduction
        focal_loss_none = FocalLoss(alpha=1, gamma=2, reduction='none')
        loss_none = focal_loss_none(predictions, targets)
        self.assertEqual(loss_none.shape, (4,))


class TestModelEvaluator(unittest.TestCase):
    
    def setUp(self):
        """Setup test model and evaluator"""
        self.model = create_model('custom_cnn', num_classes=2, pretrained=False)
        self.evaluator = ModelEvaluator(self.model, 'cpu')
        
        # Create dummy data loader
        from torch.utils.data import TensorDataset, DataLoader
        dummy_images = torch.randn(8, 3, 224, 224)
        dummy_labels = torch.randint(0, 2, (8,))
        dataset = TensorDataset(dummy_images, dummy_labels)
        self.data_loader = DataLoader(dataset, batch_size=4)
    
    def test_evaluator_creation(self):
        """Test ModelEvaluator creation"""
        self.assertIsNotNone(self.evaluator)
        self.assertEqual(self.evaluator.device, 'cpu')
    
    def test_evaluate_function(self):
        """Test evaluation on data loader"""
        results = self.evaluator.evaluate(self.data_loader)
        
        # Check that results contain expected keys
        expected_keys = ['accuracy', 'loss', 'predictions', 'targets']
        for key in expected_keys:
            self.assertIn(key, results)
        
        # Check data types
        self.assertIsInstance(results['accuracy'], float)
        self.assertIsInstance(results['loss'], float)
        self.assertIsInstance(results['predictions'], list)
        self.assertIsInstance(results['targets'], list)
        
        # Check data shapes
        self.assertEqual(len(results['predictions']), 8)  # 8 samples
        self.assertEqual(len(results['targets']), 8)
    
    def test_predict_single_image(self):
        """Test single image prediction"""
        dummy_image = torch.randn(3, 224, 224)
        result = self.evaluator.predict_single_image(dummy_image)
        
        # Check result structure
        expected_keys = ['prediction', 'probabilities', 'confidence']
        for key in expected_keys:
            self.assertIn(key, result)
        
        # Check data types and ranges
        self.assertIsInstance(result['prediction'], int)
        self.assertIn(result['prediction'], [0, 1])  # Binary classification
        
        self.assertEqual(len(result['probabilities']), 2)  # 2 classes
        self.assertTrue(0 <= result['confidence'] <= 1)  # Confidence in [0,1]


if __name__ == '__main__':
    unittest.main()