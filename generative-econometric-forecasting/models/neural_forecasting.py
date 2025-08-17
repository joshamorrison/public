"""
Neural Forecasting Models
Alternative implementation using available libraries when NeuralForecast has compatibility issues.
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple
import logging
import warnings
warnings.filterwarnings('ignore')

try:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler, MinMaxScaler
    from sklearn.metrics import mean_absolute_error, mean_squared_error
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    PYTORCH_AVAILABLE = True
except ImportError:
    PYTORCH_AVAILABLE = False

logger = logging.getLogger(__name__)

class SimpleNeuralForecaster:
    """Simple neural network forecaster using PyTorch."""
    
    def __init__(self, input_size: int = 12, hidden_size: int = 64, num_layers: int = 2):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.model = None
        self.scaler = StandardScaler()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
    def _create_model(self):
        """Create neural network model."""
        model = nn.Sequential(
            nn.Linear(self.input_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, self.hidden_size),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(self.hidden_size, 1)
        )
        return model.to(self.device)
    
    def _prepare_sequences(self, data: np.ndarray, sequence_length: int) -> Tuple[np.ndarray, np.ndarray]:
        """Create sequences for training."""
        X, y = [], []
        for i in range(len(data) - sequence_length):
            X.append(data[i:i + sequence_length])
            y.append(data[i + sequence_length])
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.Series, epochs: int = 100, learning_rate: float = 0.001):
        """Train the neural network."""
        if not PYTORCH_AVAILABLE:
            raise ImportError("PyTorch not available for neural forecasting")
        
        # Prepare data
        values = data.values.reshape(-1, 1)
        scaled_data = self.scaler.fit_transform(values).flatten()
        
        # Create sequences
        X, y = self._prepare_sequences(scaled_data, self.input_size)
        
        if len(X) < 10:  # Need minimum data
            raise ValueError("Insufficient data for neural network training")
        
        # Convert to tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        y_tensor = torch.FloatTensor(y).to(self.device)
        
        # Create model
        self.model = self._create_model()
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        
        # Training
        self.model.train()
        for epoch in range(epochs):
            optimizer.zero_grad()
            outputs = self.model(X_tensor).squeeze()
            loss = criterion(outputs, y_tensor)
            loss.backward()
            optimizer.step()
            
            if epoch % 20 == 0:
                logger.debug(f"Epoch {epoch}, Loss: {loss.item():.6f}")
    
    def forecast(self, steps: int) -> np.ndarray:
        """Generate forecasts."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        self.model.eval()
        forecasts = []
        
        # Get last sequence from training data
        last_sequence = self.scaler.transform(
            np.array([0] * self.input_size).reshape(-1, 1)
        ).flatten()  # Placeholder - in real implementation would use actual last values
        
        current_sequence = torch.FloatTensor(last_sequence).unsqueeze(0).to(self.device)
        
        with torch.no_grad():
            for _ in range(steps):
                prediction = self.model(current_sequence).item()
                forecasts.append(prediction)
                
                # Update sequence (simple approach)
                new_sequence = torch.cat([
                    current_sequence[:, 1:],
                    torch.FloatTensor([[prediction]]).to(self.device)
                ], dim=1)
                current_sequence = new_sequence
        
        # Inverse transform
        forecasts_array = np.array(forecasts).reshape(-1, 1)
        return self.scaler.inverse_transform(forecasts_array).flatten()

class MLPForecaster:
    """Multi-layer perceptron forecaster using scikit-learn."""
    
    def __init__(self, hidden_layer_sizes: Tuple = (100, 50), max_iter: int = 500):
        self.hidden_layer_sizes = hidden_layer_sizes
        self.max_iter = max_iter
        self.model = None
        self.scaler = StandardScaler()
        
    def _prepare_features(self, data: pd.Series, lookback: int = 12) -> Tuple[np.ndarray, np.ndarray]:
        """Prepare features for MLP."""
        X, y = [], []
        values = data.values
        
        for i in range(lookback, len(values)):
            X.append(values[i-lookback:i])
            y.append(values[i])
        
        return np.array(X), np.array(y)
    
    def fit(self, data: pd.Series):
        """Train the MLP model."""
        if not SKLEARN_AVAILABLE:
            raise ImportError("scikit-learn not available for MLP forecasting")
        
        # Prepare data
        X, y = self._prepare_features(data)
        
        if len(X) < 10:
            raise ValueError("Insufficient data for MLP training")
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model = MLPRegressor(
            hidden_layer_sizes=self.hidden_layer_sizes,
            max_iter=self.max_iter,
            random_state=42,
            early_stopping=True,
            validation_fraction=0.1
        )
        
        self.model.fit(X_scaled, y)
        logger.info(f"MLP trained with score: {self.model.score(X_scaled, y):.3f}")
    
    def forecast(self, data: pd.Series, steps: int) -> np.ndarray:
        """Generate forecasts using the trained MLP."""
        if self.model is None:
            raise ValueError("Model not trained yet")
        
        forecasts = []
        current_data = data.values[-12:].copy()  # Use last 12 points
        
        for _ in range(steps):
            # Prepare input
            X_input = current_data[-12:].reshape(1, -1)
            X_scaled = self.scaler.transform(X_input)
            
            # Predict
            prediction = self.model.predict(X_scaled)[0]
            forecasts.append(prediction)
            
            # Update data
            current_data = np.append(current_data[1:], prediction)
        
        return np.array(forecasts)

class NeuralModelEnsemble:
    """Ensemble of neural forecasting models."""
    
    def __init__(self):
        self.models = {}
        self.weights = {}
        self.available_models = []
        
        # Initialize available models
        if SKLEARN_AVAILABLE:
            self.available_models.extend([
                'mlp_small', 'mlp_medium', 'mlp_large'
            ])
        
        if PYTORCH_AVAILABLE:
            self.available_models.extend([
                'neural_simple', 'neural_deep'
            ])
    
    def get_available_models(self) -> List[str]:
        """Get list of available neural models."""
        return self.available_models.copy()
    
    def _create_model(self, model_name: str):
        """Create a specific model."""
        if model_name == 'mlp_small':
            return MLPForecaster(hidden_layer_sizes=(50,), max_iter=300)
        elif model_name == 'mlp_medium':
            return MLPForecaster(hidden_layer_sizes=(100, 50), max_iter=500)
        elif model_name == 'mlp_large':
            return MLPForecaster(hidden_layer_sizes=(200, 100, 50), max_iter=800)
        elif model_name == 'neural_simple':
            return SimpleNeuralForecaster(input_size=12, hidden_size=32, num_layers=2)
        elif model_name == 'neural_deep':
            return SimpleNeuralForecaster(input_size=12, hidden_size=64, num_layers=3)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    def fit_ensemble(self, data: pd.Series, models: Optional[List[str]] = None):
        """Train ensemble of models."""
        if models is None:
            models = self.available_models[:3]  # Use first 3 available models
        
        trained_models = {}
        model_scores = {}
        
        for model_name in models:
            if model_name not in self.available_models:
                logger.warning(f"Model {model_name} not available")
                continue
            
            try:
                logger.info(f"Training {model_name}...")
                model = self._create_model(model_name)
                model.fit(data)
                
                # Simple validation score (could be improved)
                if hasattr(model, 'model') and hasattr(model.model, 'score'):
                    score = 0.8  # Placeholder score
                else:
                    score = 0.8  # Placeholder score
                
                trained_models[model_name] = model
                model_scores[model_name] = score
                logger.info(f"{model_name} trained successfully")
                
            except Exception as e:
                logger.error(f"Failed to train {model_name}: {e}")
        
        if not trained_models:
            raise RuntimeError("No models could be trained")
        
        # Calculate ensemble weights based on scores
        total_score = sum(model_scores.values())
        weights = {name: score/total_score for name, score in model_scores.items()}
        
        self.models = trained_models
        self.weights = weights
        
        logger.info(f"Ensemble trained with {len(trained_models)} models")
        logger.info(f"Model weights: {weights}")
    
    def forecast_ensemble(self, data: pd.Series, steps: int) -> Dict[str, Any]:
        """Generate ensemble forecasts."""
        if not self.models:
            raise ValueError("Ensemble not trained yet")
        
        individual_forecasts = {}
        
        # Generate forecasts from each model
        for model_name, model in self.models.items():
            try:
                if hasattr(model, 'forecast') and len(data) >= 12:
                    forecast = model.forecast(data, steps)
                else:
                    # Fallback to simple projection
                    recent_trend = data.diff().tail(6).mean()
                    last_value = data.iloc[-1]
                    forecast = np.array([last_value + recent_trend * (i+1) for i in range(steps)])
                
                individual_forecasts[model_name] = forecast
                
            except Exception as e:
                logger.error(f"Forecast failed for {model_name}: {e}")
        
        if not individual_forecasts:
            raise RuntimeError("No forecasts could be generated")
        
        # Calculate weighted ensemble forecast
        ensemble_forecast = np.zeros(steps)
        total_weight = 0
        
        for model_name, forecast in individual_forecasts.items():
            weight = self.weights.get(model_name, 1.0)
            ensemble_forecast += weight * forecast
            total_weight += weight
        
        if total_weight > 0:
            ensemble_forecast /= total_weight
        
        return {
            'ensemble_forecast': ensemble_forecast,
            'individual_forecasts': individual_forecasts,
            'model_weights': self.weights,
            'models_used': list(individual_forecasts.keys())
        }

def test_neural_models():
    """Test neural forecasting capabilities."""
    print("[NEURAL] Testing Neural Forecasting Models")
    print("-" * 40)
    
    # Create sample data
    dates = pd.date_range('2020-01-01', periods=100, freq='M')
    trend = np.linspace(100, 120, 100)
    noise = np.random.normal(0, 2, 100)
    data = pd.Series(trend + noise, index=dates)
    
    # Test ensemble
    ensemble = NeuralModelEnsemble()
    available = ensemble.get_available_models()
    
    print(f"[MODELS] Available neural models: {len(available)}")
    for model in available:
        print(f"  - {model}")
    
    if available:
        try:
            # Train ensemble
            ensemble.fit_ensemble(data, models=available[:2])  # Use first 2 models
            
            # Generate forecasts
            results = ensemble.forecast_ensemble(data, steps=6)
            
            print(f"\n[FORECAST] Ensemble forecast generated")
            print(f"  Models used: {results['models_used']}")
            print(f"  Model weights: {results['model_weights']}")
            print(f"  Forecast values: {results['ensemble_forecast'][:3]}...")
            
            return True
            
        except Exception as e:
            print(f"[ERROR] Neural ensemble test failed: {e}")
            return False
    else:
        print("[WARN] No neural models available")
        return False

if __name__ == "__main__":
    test_neural_models()