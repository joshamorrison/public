"""
Economic GAN
Generative Adversarial Network for synthetic economic time series generation.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Optional, Any, Union, Tuple
import logging
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    import torch.nn.functional as F
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available for GAN models")

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import mean_absolute_error
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EconomicGenerator(nn.Module):
    """Generator network for economic time series."""
    
    def __init__(self, noise_dim: int = 100, output_dim: int = 1, 
                 hidden_dims: List[int] = [128, 256, 512, 256, 128]):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for GAN models")
        
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        
        # Build generator layers
        layers = []
        prev_dim = noise_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        layers.append(nn.Tanh())  # Output between -1 and 1
        
        self.generator = nn.Sequential(*layers)
        
    def forward(self, noise):
        return self.generator(noise)


class EconomicDiscriminator(nn.Module):
    """Discriminator network for economic time series."""
    
    def __init__(self, input_dim: int = 1, 
                 hidden_dims: List[int] = [128, 256, 512, 256, 128]):
        super().__init__()
        
        self.input_dim = input_dim
        
        # Build discriminator layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.LeakyReLU(0.2, inplace=True),
                nn.Dropout(0.3)
            ])
            prev_dim = hidden_dim
        
        # Output layer (probability)
        layers.append(nn.Linear(prev_dim, 1))
        layers.append(nn.Sigmoid())
        
        self.discriminator = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.discriminator(x)


class TimeSeriesGenerator(nn.Module):
    """Generator for sequential time series data."""
    
    def __init__(self, noise_dim: int = 100, output_dim: int = 1, 
                 sequence_length: int = 50, hidden_dim: int = 128):
        super().__init__()
        
        self.noise_dim = noise_dim
        self.output_dim = output_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # LSTM-based generator
        self.lstm = nn.LSTM(noise_dim, hidden_dim, batch_first=True, num_layers=2)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.activation = nn.Tanh()
        
    def forward(self, noise):
        # noise shape: (batch_size, sequence_length, noise_dim)
        lstm_out, _ = self.lstm(noise)
        output = self.output_layer(lstm_out)
        return self.activation(output)


class TimeSeriesDiscriminator(nn.Module):
    """Discriminator for sequential time series data."""
    
    def __init__(self, input_dim: int = 1, sequence_length: int = 50, 
                 hidden_dim: int = 128):
        super().__init__()
        
        self.input_dim = input_dim
        self.sequence_length = sequence_length
        self.hidden_dim = hidden_dim
        
        # LSTM-based discriminator
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True, num_layers=2)
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        lstm_out, (hidden, _) = self.lstm(x)
        # Use last hidden state
        output = self.classifier(hidden[-1])
        return output


class EconomicGAN:
    """Generative Adversarial Network for economic data synthesis."""
    
    def __init__(self, 
                 sequence_length: int = 50,
                 noise_dim: int = 100,
                 learning_rate: float = 0.0002,
                 beta1: float = 0.5,
                 device: str = None):
        """
        Initialize Economic GAN.
        
        Args:
            sequence_length: Length of generated sequences
            noise_dim: Dimension of noise vector
            learning_rate: Learning rate for training
            beta1: Beta1 parameter for Adam optimizer
            device: Device to run on ('cpu', 'cuda')
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for GAN models")
        
        self.sequence_length = sequence_length
        self.noise_dim = noise_dim
        self.learning_rate = learning_rate
        self.beta1 = beta1
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.generator = None
        self.discriminator = None
        self.scaler = StandardScaler()
        
        # Training history
        self.training_history = {
            'g_losses': [],
            'd_losses': [],
            'epochs': []
        }
        
        logger.info(f"Economic GAN initialized on {self.device}")
    
    def prepare_data(self, 
                    economic_data: Union[pd.Series, pd.DataFrame],
                    indicators: List[str] = None) -> torch.Tensor:
        """
        Prepare economic data for GAN training.
        
        Args:
            economic_data: Economic time series data
            indicators: List of indicators if DataFrame
        
        Returns:
            Prepared tensor data
        """
        if isinstance(economic_data, pd.Series):
            data = economic_data.values.reshape(-1, 1)
        elif isinstance(economic_data, pd.DataFrame):
            if indicators:
                data = economic_data[indicators].values
            else:
                data = economic_data.values
        else:
            data = np.array(economic_data).reshape(-1, 1)
        
        # Remove NaN values
        data = data[~np.isnan(data).any(axis=1)]
        
        # Scale data to [-1, 1] for tanh activation
        data_scaled = self.scaler.fit_transform(data)
        
        # Create sequences
        sequences = []
        for i in range(len(data_scaled) - self.sequence_length + 1):
            sequences.append(data_scaled[i:i + self.sequence_length])
        
        sequences = np.array(sequences)
        
        # Convert to tensor
        tensor_data = torch.FloatTensor(sequences).to(self.device)
        
        logger.info(f"Prepared {len(sequences)} sequences of length {self.sequence_length}")
        return tensor_data
    
    def initialize_models(self, input_dim: int = 1):
        """Initialize generator and discriminator models."""
        # Time series models
        self.generator = TimeSeriesGenerator(
            noise_dim=self.noise_dim,
            output_dim=input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=128
        ).to(self.device)
        
        self.discriminator = TimeSeriesDiscriminator(
            input_dim=input_dim,
            sequence_length=self.sequence_length,
            hidden_dim=128
        ).to(self.device)
        
        # Initialize optimizers
        self.g_optimizer = optim.Adam(
            self.generator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, 0.999)
        )
        
        self.d_optimizer = optim.Adam(
            self.discriminator.parameters(),
            lr=self.learning_rate,
            betas=(self.beta1, 0.999)
        )
        
        # Loss function
        self.criterion = nn.BCELoss()
        
        logger.info("GAN models initialized")
    
    def train(self, 
              data: torch.Tensor,
              epochs: int = 1000,
              batch_size: int = 32,
              d_steps: int = 1,
              g_steps: int = 1,
              save_interval: int = 100) -> Dict[str, Any]:
        """
        Train the GAN.
        
        Args:
            data: Training data tensor
            epochs: Number of training epochs
            batch_size: Batch size
            d_steps: Discriminator training steps per iteration
            g_steps: Generator training steps per iteration
            save_interval: Interval to save progress
        
        Returns:
            Training results
        """
        if self.generator is None:
            self.initialize_models(input_dim=data.shape[-1])
        
        # Create data loader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Training loop
        for epoch in range(epochs):
            epoch_g_loss = 0.0
            epoch_d_loss = 0.0
            batch_count = 0
            
            for batch_data, in dataloader:
                batch_size_actual = batch_data.size(0)
                
                # Train Discriminator
                for _ in range(d_steps):
                    self.d_optimizer.zero_grad()
                    
                    # Real data
                    real_data = batch_data
                    real_labels = torch.ones(batch_size_actual, 1).to(self.device)
                    
                    # Fake data
                    noise = torch.randn(
                        batch_size_actual, self.sequence_length, self.noise_dim
                    ).to(self.device)
                    fake_data = self.generator(noise)
                    fake_labels = torch.zeros(batch_size_actual, 1).to(self.device)
                    
                    # Discriminator losses
                    real_loss = self.criterion(self.discriminator(real_data), real_labels)
                    fake_loss = self.criterion(self.discriminator(fake_data.detach()), fake_labels)
                    d_loss = (real_loss + fake_loss) / 2
                    
                    d_loss.backward()
                    self.d_optimizer.step()
                    
                    epoch_d_loss += d_loss.item()
                
                # Train Generator
                for _ in range(g_steps):
                    self.g_optimizer.zero_grad()
                    
                    # Generate fake data
                    noise = torch.randn(
                        batch_size_actual, self.sequence_length, self.noise_dim
                    ).to(self.device)
                    fake_data = self.generator(noise)
                    
                    # Generator loss (try to fool discriminator)
                    g_loss = self.criterion(
                        self.discriminator(fake_data), 
                        torch.ones(batch_size_actual, 1).to(self.device)
                    )
                    
                    g_loss.backward()
                    self.g_optimizer.step()
                    
                    epoch_g_loss += g_loss.item()
                
                batch_count += 1
            
            # Average losses
            avg_g_loss = epoch_g_loss / (batch_count * g_steps)
            avg_d_loss = epoch_d_loss / (batch_count * d_steps)
            
            # Store history
            self.training_history['g_losses'].append(avg_g_loss)
            self.training_history['d_losses'].append(avg_d_loss)
            self.training_history['epochs'].append(epoch)
            
            # Log progress
            if (epoch + 1) % save_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] | "
                           f"G Loss: {avg_g_loss:.4f} | D Loss: {avg_d_loss:.4f}")
        
        logger.info("GAN training completed")
        
        return {
            'training_history': self.training_history,
            'final_g_loss': avg_g_loss,
            'final_d_loss': avg_d_loss,
            'epochs_trained': epochs
        }
    
    def generate_synthetic_data(self, 
                              n_samples: int = 100,
                              return_scaled: bool = False) -> np.ndarray:
        """
        Generate synthetic economic data.
        
        Args:
            n_samples: Number of samples to generate
            return_scaled: Whether to return scaled data
        
        Returns:
            Synthetic data array
        """
        if self.generator is None:
            raise ValueError("Generator not trained. Call train() first.")
        
        self.generator.eval()
        
        with torch.no_grad():
            # Generate noise
            noise = torch.randn(n_samples, self.sequence_length, self.noise_dim).to(self.device)
            
            # Generate synthetic data
            synthetic_data = self.generator(noise)
            synthetic_data = synthetic_data.cpu().numpy()
        
        if not return_scaled:
            # Inverse transform to original scale
            original_shape = synthetic_data.shape
            synthetic_data_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
            synthetic_data_unscaled = self.scaler.inverse_transform(synthetic_data_flat)
            synthetic_data = synthetic_data_unscaled.reshape(original_shape)
        
        logger.info(f"Generated {n_samples} synthetic sequences")
        return synthetic_data
    
    def generate_economic_scenarios(self, 
                                   scenario_types: List[str] = None,
                                   samples_per_scenario: int = 50) -> Dict[str, np.ndarray]:
        """
        Generate different economic scenarios.
        
        Args:
            scenario_types: Types of scenarios to generate
            samples_per_scenario: Number of samples per scenario
        
        Returns:
            Dictionary of scenario data
        """
        if scenario_types is None:
            scenario_types = ['normal', 'recession', 'boom', 'volatile']
        
        scenarios = {}
        
        for scenario_type in scenario_types:
            if scenario_type == 'normal':
                # Standard generation
                scenario_data = self.generate_synthetic_data(samples_per_scenario)
                
            elif scenario_type == 'recession':
                # Generate with modified noise for recession patterns
                scenario_data = self._generate_scenario_data(
                    samples_per_scenario, 
                    noise_modifier=lambda x: x - 0.5  # Shift towards negative
                )
                
            elif scenario_type == 'boom':
                # Generate with modified noise for boom patterns
                scenario_data = self._generate_scenario_data(
                    samples_per_scenario,
                    noise_modifier=lambda x: x + 0.3  # Shift towards positive
                )
                
            elif scenario_type == 'volatile':
                # Generate with higher variance noise
                scenario_data = self._generate_scenario_data(
                    samples_per_scenario,
                    noise_modifier=lambda x: x * 2.0  # Increase volatility
                )
            else:
                logger.warning(f"Unknown scenario type: {scenario_type}")
                continue
            
            scenarios[scenario_type] = scenario_data
            logger.info(f"Generated {samples_per_scenario} samples for {scenario_type} scenario")
        
        return scenarios
    
    def _generate_scenario_data(self, 
                              n_samples: int,
                              noise_modifier: callable) -> np.ndarray:
        """Generate data with modified noise patterns."""
        if self.generator is None:
            raise ValueError("Generator not trained")
        
        self.generator.eval()
        
        with torch.no_grad():
            # Generate base noise
            noise = torch.randn(n_samples, self.sequence_length, self.noise_dim).to(self.device)
            
            # Apply modifier
            noise = noise_modifier(noise)
            
            # Generate synthetic data
            synthetic_data = self.generator(noise)
            synthetic_data = synthetic_data.cpu().numpy()
            
            # Inverse transform
            original_shape = synthetic_data.shape
            synthetic_data_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
            synthetic_data_unscaled = self.scaler.inverse_transform(synthetic_data_flat)
            synthetic_data = synthetic_data_unscaled.reshape(original_shape)
        
        return synthetic_data
    
    def evaluate_synthetic_quality(self, 
                                  real_data: np.ndarray,
                                  synthetic_data: np.ndarray) -> Dict[str, float]:
        """
        Evaluate quality of synthetic data.
        
        Args:
            real_data: Real economic data
            synthetic_data: Generated synthetic data
        
        Returns:
            Quality metrics
        """
        # Statistical similarity metrics
        real_flat = real_data.reshape(-1, real_data.shape[-1])
        synth_flat = synthetic_data.reshape(-1, synthetic_data.shape[-1])
        
        metrics = {}
        
        for i in range(real_flat.shape[1]):
            real_col = real_flat[:, i]
            synth_col = synth_flat[:, i]
            
            # Distribution comparison
            metrics[f'mean_diff_dim_{i}'] = abs(np.mean(real_col) - np.mean(synth_col))
            metrics[f'std_diff_dim_{i}'] = abs(np.std(real_col) - np.std(synth_col))
            
            # Statistical tests
            from scipy.stats import ks_2samp
            ks_stat, ks_pvalue = ks_2samp(real_col, synth_col)
            metrics[f'ks_statistic_dim_{i}'] = ks_stat
            metrics[f'ks_pvalue_dim_{i}'] = ks_pvalue
        
        # Overall metrics
        metrics['overall_mean_diff'] = np.mean([v for k, v in metrics.items() if 'mean_diff' in k])
        metrics['overall_std_diff'] = np.mean([v for k, v in metrics.items() if 'std_diff' in k])
        
        return metrics
    
    def save_model(self, filepath: str):
        """Save trained GAN models."""
        if self.generator is None or self.discriminator is None:
            raise ValueError("Models not trained")
        
        torch.save({
            'generator_state': self.generator.state_dict(),
            'discriminator_state': self.discriminator.state_dict(),
            'g_optimizer_state': self.g_optimizer.state_dict(),
            'd_optimizer_state': self.d_optimizer.state_dict(),
            'scaler': self.scaler,
            'training_history': self.training_history,
            'config': {
                'sequence_length': self.sequence_length,
                'noise_dim': self.noise_dim,
                'learning_rate': self.learning_rate,
                'beta1': self.beta1
            }
        }, filepath)
        
        logger.info(f"GAN models saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained GAN models."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore config
        config = checkpoint['config']
        self.sequence_length = config['sequence_length']
        self.noise_dim = config['noise_dim']
        self.learning_rate = config['learning_rate']
        self.beta1 = config['beta1']
        
        # Initialize models with proper dimensions
        # Note: input_dim needs to be inferred from saved states
        generator_state = checkpoint['generator_state']
        output_dim = list(generator_state.values())[-2].shape[0]  # From output layer
        
        self.initialize_models(input_dim=output_dim)
        
        # Load states
        self.generator.load_state_dict(checkpoint['generator_state'])
        self.discriminator.load_state_dict(checkpoint['discriminator_state'])
        self.g_optimizer.load_state_dict(checkpoint['g_optimizer_state'])
        self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state'])
        
        # Load scaler and history
        self.scaler = checkpoint['scaler']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"GAN models loaded from {filepath}")
    
    def plot_training_history(self):
        """Plot training loss history."""
        plt.figure(figsize=(12, 4))
        
        plt.subplot(1, 2, 1)
        plt.plot(self.training_history['epochs'], self.training_history['g_losses'], label='Generator')
        plt.plot(self.training_history['epochs'], self.training_history['d_losses'], label='Discriminator')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Losses')
        plt.legend()
        plt.grid(True)
        
        plt.subplot(1, 2, 2)
        plt.plot(self.training_history['g_losses'], label='Generator')
        plt.plot(self.training_history['d_losses'], label='Discriminator')
        plt.xlabel('Iteration')
        plt.ylabel('Loss')
        plt.title('Loss Progression')
        plt.legend()
        plt.grid(True)
        
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    # Example usage
    if TORCH_AVAILABLE:
        # Create sample economic data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
        
        # Simulate GDP growth with trend and cycles
        trend = np.linspace(0.02, 0.025, len(dates))
        cycle = 0.01 * np.sin(2 * np.pi * np.arange(len(dates)) / 12)
        noise = np.random.normal(0, 0.005, len(dates))
        gdp_growth = trend + cycle + noise
        
        economic_data = pd.Series(gdp_growth, index=dates, name='gdp_growth')
        
        print("Training Economic GAN...")
        
        # Initialize and train GAN
        gan = EconomicGAN(sequence_length=24, noise_dim=50)
        
        # Prepare data
        training_data = gan.prepare_data(economic_data)
        
        # Train GAN
        training_results = gan.train(
            training_data, 
            epochs=500, 
            batch_size=16, 
            save_interval=100
        )
        
        print(f"Training completed. Final losses - G: {training_results['final_g_loss']:.4f}, D: {training_results['final_d_loss']:.4f}")
        
        # Generate synthetic data
        synthetic_samples = gan.generate_synthetic_data(n_samples=50)
        print(f"Generated synthetic data shape: {synthetic_samples.shape}")
        
        # Generate economic scenarios
        scenarios = gan.generate_economic_scenarios(
            scenario_types=['normal', 'recession', 'boom'],
            samples_per_scenario=20
        )
        
        print("Generated scenarios:")
        for scenario_type, data in scenarios.items():
            mean_value = np.mean(data)
            std_value = np.std(data)
            print(f"  {scenario_type}: mean={mean_value:.4f}, std={std_value:.4f}")
        
        # Evaluate quality
        real_sequences = training_data.cpu().numpy()
        quality_metrics = gan.evaluate_synthetic_quality(
            real_sequences[:50], synthetic_samples
        )
        
        print("Quality metrics:")
        for metric, value in quality_metrics.items():
            if 'overall' in metric:
                print(f"  {metric}: {value:.4f}")
        
        print("Economic GAN example completed")
    else:
        print("PyTorch not available - skipping GAN example")