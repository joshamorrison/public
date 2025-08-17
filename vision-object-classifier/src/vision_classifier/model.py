import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
import timm


class DishCleanlinessClassifier(nn.Module):
    def __init__(self, model_name='resnet50', num_classes=2, pretrained=True):
        super(DishCleanlinessClassifier, self).__init__()
        
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'resnet50':
            self.backbone = models.resnet50(pretrained=pretrained)
            # Replace final layer
            num_features = self.backbone.fc.in_features
            self.backbone.fc = nn.Linear(num_features, num_classes)
            
        elif model_name == 'efficientnet_b0':
            self.backbone = timm.create_model('efficientnet_b0', pretrained=pretrained, num_classes=num_classes)
            
        elif model_name == 'mobilenet_v3':
            self.backbone = models.mobilenet_v3_large(pretrained=pretrained)
            num_features = self.backbone.classifier[3].in_features
            self.backbone.classifier[3] = nn.Linear(num_features, num_classes)
            
        elif model_name == 'custom_cnn':
            # Custom CNN for dishes
            self.backbone = self._create_custom_cnn()
        
        else:
            raise ValueError(f"Unsupported model: {model_name}")
    
    def _create_custom_cnn(self):
        """Create custom CNN optimized for dish classification"""
        return nn.Sequential(
            # First block
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Second block
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Third block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Fourth block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.25),
            
            # Global average pooling and classifier
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, self.num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)


class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class ModelEvaluator:
    def __init__(self, model, device):
        self.model = model
        self.device = device
        
    def evaluate(self, data_loader):
        """Evaluate model on validation/test set"""
        self.model.eval()
        total_correct = 0
        total_samples = 0
        total_loss = 0
        
        # For confusion matrix
        all_predictions = []
        all_targets = []
        
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for inputs, targets in data_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                outputs = self.model(inputs)
                loss = criterion(outputs, targets)
                
                _, predicted = torch.max(outputs.data, 1)
                total_samples += targets.size(0)
                total_correct += (predicted == targets).sum().item()
                total_loss += loss.item()
                
                all_predictions.extend(predicted.cpu().numpy())
                all_targets.extend(targets.cpu().numpy())
        
        accuracy = total_correct / total_samples
        avg_loss = total_loss / len(data_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'predictions': all_predictions,
            'targets': all_targets
        }
    
    def predict_single_image(self, image_tensor):
        """Predict on a single image"""
        self.model.eval()
        with torch.no_grad():
            image_tensor = image_tensor.unsqueeze(0).to(self.device)
            outputs = self.model(image_tensor)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            return {
                'prediction': predicted.item(),
                'probabilities': probabilities.cpu().numpy()[0],
                'confidence': probabilities.max().item()
            }


def create_model(model_name='resnet50', num_classes=2, pretrained=True):
    """Factory function to create models"""
    return DishCleanlinessClassifier(model_name=model_name, 
                                   num_classes=num_classes, 
                                   pretrained=pretrained)


def load_model(checkpoint_path, model_name='resnet50', num_classes=2, device='cpu'):
    """Load a saved model"""
    model = create_model(model_name=model_name, num_classes=num_classes, pretrained=False)
    
    # Load state dict
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.to(device)
    model.eval()
    
    return model


def save_model(model, optimizer, epoch, loss, accuracy, filepath):
    """Save model checkpoint"""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'accuracy': accuracy,
    }, filepath)


if __name__ == "__main__":
    # Test model creation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test different models
    models_to_test = ['resnet50', 'efficientnet_b0', 'mobilenet_v3', 'custom_cnn']
    
    for model_name in models_to_test:
        try:
            model = create_model(model_name=model_name)
            model.to(device)
            
            # Test forward pass
            dummy_input = torch.randn(1, 3, 224, 224).to(device)
            output = model(dummy_input)
            
            print(f"{model_name}: Output shape {output.shape}")
            
        except Exception as e:
            print(f"Error with {model_name}: {e}")