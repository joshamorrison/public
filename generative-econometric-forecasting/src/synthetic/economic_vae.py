"""
Economic VAE
Variational Autoencoder for probabilistic economic data generation and uncertainty modeling.
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
    from torch.distributions import Normal, kl_divergence
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logging.warning("PyTorch not available for VAE models")

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


class EconomicEncoder(nn.Module):
    """Encoder network for Economic VAE."""
    
    def __init__(self, input_dim: int, latent_dim: int = 32, 
                 hidden_dims: List[int] = [128, 64]):
        super().__init__()
        
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for VAE models")
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        # Build encoder layers
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        self.encoder = nn.Sequential(*layers)
        
        # Latent space parameters
        self.mu_layer = nn.Linear(prev_dim, latent_dim)
        self.logvar_layer = nn.Linear(prev_dim, latent_dim)
        
    def forward(self, x):
        encoded = self.encoder(x)
        mu = self.mu_layer(encoded)
        logvar = self.logvar_layer(encoded)
        return mu, logvar


class EconomicDecoder(nn.Module):
    """Decoder network for Economic VAE."""
    
    def __init__(self, latent_dim: int, output_dim: int,
                 hidden_dims: List[int] = [64, 128]):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        
        # Build decoder layers
        layers = []
        prev_dim = latent_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.2)
            ])
            prev_dim = hidden_dim
        
        # Output layer
        layers.append(nn.Linear(prev_dim, output_dim))
        
        self.decoder = nn.Sequential(*layers)
        
    def forward(self, z):
        return self.decoder(z)


class TimeSeriesVAE(nn.Module):
    """VAE with LSTM for time series data."""
    
    def __init__(self, input_dim: int, latent_dim: int = 32,
                 hidden_dim: int = 64, num_layers: int = 2):
        super().__init__()
        
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        
        # Encoder LSTM
        self.encoder_lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
            batch_first=True, dropout=0.2
        )
        self.encoder_mu = nn.Linear(hidden_dim, latent_dim)
        self.encoder_logvar = nn.Linear(hidden_dim, latent_dim)
        
        # Decoder
        self.decoder_input = nn.Linear(latent_dim, hidden_dim)
        self.decoder_lstm = nn.LSTM(
            hidden_dim, hidden_dim, num_layers,
            batch_first=True, dropout=0.2
        )
        self.decoder_output = nn.Linear(hidden_dim, input_dim)
        
    def encode(self, x):
        # x shape: (batch_size, seq_len, input_dim)
        lstm_out, (hidden, _) = self.encoder_lstm(x)
        # Use last hidden state
        h_last = hidden[-1]  # Shape: (batch_size, hidden_dim)
        
        mu = self.encoder_mu(h_last)
        logvar = self.encoder_logvar(h_last)
        return mu, logvar
    
    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def decode(self, z, seq_len):
        # z shape: (batch_size, latent_dim)
        batch_size = z.size(0)
        
        # Initialize decoder input
        decoder_input = self.decoder_input(z)  # (batch_size, hidden_dim)
        
        # Repeat for sequence length
        decoder_input = decoder_input.unsqueeze(1).repeat(1, seq_len, 1)
        
        # LSTM decode
        lstm_out, _ = self.decoder_lstm(decoder_input)
        output = self.decoder_output(lstm_out)
        
        return output
    
    def forward(self, x):
        seq_len = x.size(1)
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z, seq_len)
        return recon, mu, logvar


class EconomicVAE:
    """Variational Autoencoder for economic data generation and modeling."""
    
    def __init__(self, 
                 latent_dim: int = 32,
                 learning_rate: float = 0.001,
                 beta: float = 1.0,
                 use_time_series: bool = True,
                 device: str = None):
        """
        Initialize Economic VAE.
        
        Args:
            latent_dim: Dimension of latent space
            learning_rate: Learning rate for training
            beta: Beta parameter for KL divergence weighting
            use_time_series: Whether to use time series VAE architecture
            device: Device to run on
        """
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch required for VAE models")
        
        self.latent_dim = latent_dim
        self.learning_rate = learning_rate
        self.beta = beta
        self.use_time_series = use_time_series
        
        # Set device
        self.device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Initialize models
        self.vae = None
        self.optimizer = None
        self.scaler = StandardScaler()
        
        # Training history
        self.training_history = {
            'total_losses': [],
            'recon_losses': [],
            'kl_losses': [],
            'epochs': []
        }
        
        logger.info(f"Economic VAE initialized on {self.device}")
    
    def prepare_data(self, 
                    economic_data: Union[pd.Series, pd.DataFrame],
                    sequence_length: int = 50) -> torch.Tensor:
        """
        Prepare economic data for VAE training.
        
        Args:
            economic_data: Economic time series data
            sequence_length: Length of sequences for time series VAE
        
        Returns:
            Prepared tensor data
        """
        if isinstance(economic_data, pd.Series):
            data = economic_data.values.reshape(-1, 1)
        elif isinstance(economic_data, pd.DataFrame):
            data = economic_data.values
        else:
            data = np.array(economic_data)
            if data.ndim == 1:
                data = data.reshape(-1, 1)
        
        # Remove NaN values
        data = data[~np.isnan(data).any(axis=1)]
        
        # Scale data
        data_scaled = self.scaler.fit_transform(data)
        
        if self.use_time_series:
            # Create sequences
            sequences = []
            for i in range(len(data_scaled) - sequence_length + 1):
                sequences.append(data_scaled[i:i + sequence_length])
            
            sequences = np.array(sequences)
            tensor_data = torch.FloatTensor(sequences).to(self.device)
            logger.info(f"Prepared {len(sequences)} sequences of length {sequence_length}")
        else:
            # Use individual data points
            tensor_data = torch.FloatTensor(data_scaled).to(self.device)
            logger.info(f"Prepared {len(data_scaled)} data points")
        
        return tensor_data
    
    def initialize_model(self, input_dim: int, sequence_length: int = None):
        """Initialize VAE model."""
        if self.use_time_series:
            if sequence_length is None:
                raise ValueError("sequence_length required for time series VAE")
            
            self.vae = TimeSeriesVAE(
                input_dim=input_dim,
                latent_dim=self.latent_dim,
                hidden_dim=64,
                num_layers=2
            ).to(self.device)
        else:
            # Standard VAE
            self.encoder = EconomicEncoder(
                input_dim=input_dim,
                latent_dim=self.latent_dim,
                hidden_dims=[128, 64]
            ).to(self.device)
            
            self.decoder = EconomicDecoder(
                latent_dim=self.latent_dim,
                output_dim=input_dim,
                hidden_dims=[64, 128]
            ).to(self.device)
            
            # Combine encoder and decoder
            class StandardVAE(nn.Module):
                def __init__(self, encoder, decoder):
                    super().__init__()
                    self.encoder = encoder
                    self.decoder = decoder
                
                def forward(self, x):
                    mu, logvar = self.encoder(x)
                    std = torch.exp(0.5 * logvar)
                    eps = torch.randn_like(std)
                    z = mu + eps * std
                    recon = self.decoder(z)
                    return recon, mu, logvar
            
            self.vae = StandardVAE(self.encoder, self.decoder).to(self.device)
        
        # Initialize optimizer
        self.optimizer = optim.Adam(self.vae.parameters(), lr=self.learning_rate)
        
        logger.info("VAE model initialized")
    
    def vae_loss(self, recon_x, x, mu, logvar):
        """Calculate VAE loss (reconstruction + KL divergence)."""
        # Reconstruction loss
        recon_loss = F.mse_loss(recon_x, x, reduction='sum')
        
        # KL divergence loss
        kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        
        # Total loss
        total_loss = recon_loss + self.beta * kl_loss
        
        return total_loss, recon_loss, kl_loss
    
    def train(self, 
              data: torch.Tensor,
              epochs: int = 500,
              batch_size: int = 32,
              save_interval: int = 50) -> Dict[str, Any]:
        """
        Train the VAE.
        
        Args:
            data: Training data tensor
            epochs: Number of training epochs
            batch_size: Batch size
            save_interval: Interval to log progress
        
        Returns:
            Training results
        """
        if self.vae is None:
            if self.use_time_series:
                input_dim = data.shape[-1]
                sequence_length = data.shape[1]
                self.initialize_model(input_dim, sequence_length)
            else:
                input_dim = data.shape[-1]
                self.initialize_model(input_dim)
        
        # Create data loader
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        self.vae.train()
        
        for epoch in range(epochs):
            epoch_total_loss = 0.0
            epoch_recon_loss = 0.0
            epoch_kl_loss = 0.0
            batch_count = 0
            
            for batch_data, in dataloader:
                self.optimizer.zero_grad()
                
                # Forward pass
                recon_batch, mu, logvar = self.vae(batch_data)
                
                # Calculate loss
                total_loss, recon_loss, kl_loss = self.vae_loss(
                    recon_batch, batch_data, mu, logvar
                )
                
                # Backward pass
                total_loss.backward()
                self.optimizer.step()
                
                # Accumulate losses
                epoch_total_loss += total_loss.item()
                epoch_recon_loss += recon_loss.item()
                epoch_kl_loss += kl_loss.item()
                batch_count += 1
            
            # Average losses
            avg_total_loss = epoch_total_loss / batch_count
            avg_recon_loss = epoch_recon_loss / batch_count
            avg_kl_loss = epoch_kl_loss / batch_count
            
            # Store history
            self.training_history['total_losses'].append(avg_total_loss)
            self.training_history['recon_losses'].append(avg_recon_loss)
            self.training_history['kl_losses'].append(avg_kl_loss)
            self.training_history['epochs'].append(epoch)
            
            # Log progress
            if (epoch + 1) % save_interval == 0:
                logger.info(f"Epoch [{epoch+1}/{epochs}] | "
                           f"Total Loss: {avg_total_loss:.4f} | "
                           f"Recon Loss: {avg_recon_loss:.4f} | "
                           f"KL Loss: {avg_kl_loss:.4f}")
        
        logger.info("VAE training completed")
        
        return {
            'training_history': self.training_history,
            'final_total_loss': avg_total_loss,
            'final_recon_loss': avg_recon_loss,
            'final_kl_loss': avg_kl_loss,
            'epochs_trained': epochs
        }
    
    def generate_samples(self, 
                        n_samples: int = 100,
                        sequence_length: int = None) -> np.ndarray:
        """
        Generate synthetic samples from learned distribution.
        
        Args:
            n_samples: Number of samples to generate
            sequence_length: Sequence length for time series VAE
        
        Returns:
            Generated samples
        """
        if self.vae is None:
            raise ValueError("VAE not trained. Call train() first.")
        
        self.vae.eval()
        
        with torch.no_grad():
            # Sample from latent space
            z = torch.randn(n_samples, self.latent_dim).to(self.device)
            
            if self.use_time_series:
                if sequence_length is None:
                    raise ValueError("sequence_length required for time series generation")
                generated = self.vae.decode(z, sequence_length)
            else:
                generated = self.vae.decoder(z)
            
            generated = generated.cpu().numpy()
        
        # Inverse transform to original scale
        if self.use_time_series:
            original_shape = generated.shape
            generated_flat = generated.reshape(-1, generated.shape[-1])
            generated_unscaled = self.scaler.inverse_transform(generated_flat)
            generated = generated_unscaled.reshape(original_shape)
        else:
            generated = self.scaler.inverse_transform(generated)
        
        logger.info(f"Generated {n_samples} synthetic samples")
        return generated
    
    def interpolate_in_latent_space(self, 
                                  sample1: np.ndarray,
                                  sample2: np.ndarray,
                                  n_steps: int = 10) -> np.ndarray:
        """
        Interpolate between two samples in latent space.
        
        Args:
            sample1: First sample
            sample2: Second sample
            n_steps: Number of interpolation steps
        
        Returns:
            Interpolated samples
        """
        if self.vae is None:
            raise ValueError("VAE not trained")
        
        # Scale samples
        sample1_scaled = self.scaler.transform(sample1.reshape(1, -1))
        sample2_scaled = self.scaler.transform(sample2.reshape(1, -1))
        
        # Convert to tensors
        s1_tensor = torch.FloatTensor(sample1_scaled).to(self.device)
        s2_tensor = torch.FloatTensor(sample2_scaled).to(self.device)
        
        self.vae.eval()
        
        with torch.no_grad():
            # Encode to latent space
            if self.use_time_series:
                mu1, _ = self.vae.encode(s1_tensor.unsqueeze(0))
                mu2, _ = self.vae.encode(s2_tensor.unsqueeze(0))
            else:
                mu1, _ = self.vae.encoder(s1_tensor)
                mu2, _ = self.vae.encoder(s2_tensor)
            
            # Interpolate in latent space
            interpolated_samples = []
            for i in range(n_steps):
                alpha = i / (n_steps - 1)
                z_interp = (1 - alpha) * mu1 + alpha * mu2
                
                # Decode
                if self.use_time_series:
                    seq_len = sample1.shape[0] if sample1.ndim > 1 else len(sample1)
                    decoded = self.vae.decode(z_interp, seq_len)
                else:
                    decoded = self.vae.decoder(z_interp)
                
                decoded = decoded.cpu().numpy()
                interpolated_samples.append(decoded)
        
        interpolated_samples = np.array(interpolated_samples)
        
        # Inverse transform
        if self.use_time_series:
            original_shape = interpolated_samples.shape
            flat_samples = interpolated_samples.reshape(-1, interpolated_samples.shape[-1])
            unscaled_samples = self.scaler.inverse_transform(flat_samples)
            interpolated_samples = unscaled_samples.reshape(original_shape)
        else:
            interpolated_samples = self.scaler.inverse_transform(
                interpolated_samples.reshape(-1, interpolated_samples.shape[-1])
            )
        
        return interpolated_samples
    
    def get_latent_representation(self, data: np.ndarray) -> np.ndarray:
        """
        Get latent representation of data.
        
        Args:
            data: Input data
        
        Returns:
            Latent representation
        """
        if self.vae is None:
            raise ValueError("VAE not trained")
        
        # Scale data
        data_scaled = self.scaler.transform(data.reshape(-1, data.shape[-1]))
        data_tensor = torch.FloatTensor(data_scaled).to(self.device)
        
        self.vae.eval()
        
        with torch.no_grad():
            if self.use_time_series:
                mu, logvar = self.vae.encode(data_tensor.unsqueeze(0))
            else:
                mu, logvar = self.vae.encoder(data_tensor)
            
            # Use mean of latent distribution
            latent_repr = mu.cpu().numpy()
        
        return latent_repr
    
    def generate_economic_scenarios(self, 
                                   base_conditions: Dict[str, float],
                                   n_scenarios: int = 100,
                                   uncertainty_scale: float = 1.0) -> Dict[str, np.ndarray]:
        """
        Generate economic scenarios with controlled uncertainty.
        
        Args:
            base_conditions: Base economic conditions
            n_scenarios: Number of scenarios to generate
            uncertainty_scale: Scale factor for uncertainty
        
        Returns:
            Generated economic scenarios
        """
        # Sample from latent space with different uncertainty levels
        scenarios = {}
        
        # Base scenario (low uncertainty)
        z_base = torch.randn(n_scenarios // 3, self.latent_dim).to(self.device) * 0.5
        
        # Moderate uncertainty scenario
        z_moderate = torch.randn(n_scenarios // 3, self.latent_dim).to(self.device) * 1.0
        
        # High uncertainty scenario
        z_high = torch.randn(n_scenarios // 3, self.latent_dim).to(self.device) * 2.0 * uncertainty_scale
        
        scenario_types = {
            'low_uncertainty': z_base,
            'moderate_uncertainty': z_moderate,
            'high_uncertainty': z_high
        }
        
        self.vae.eval()
        
        for scenario_type, z_samples in scenario_types.items():
            with torch.no_grad():
                if self.use_time_series:
                    # Need sequence length - use a default
                    seq_len = 12  # Default 12 periods
                    generated = self.vae.decode(z_samples, seq_len)
                else:
                    generated = self.vae.decoder(z_samples)
                
                generated = generated.cpu().numpy()
                
                # Inverse transform
                if self.use_time_series:
                    original_shape = generated.shape
                    generated_flat = generated.reshape(-1, generated.shape[-1])
                    generated_unscaled = self.scaler.inverse_transform(generated_flat)
                    generated = generated_unscaled.reshape(original_shape)
                else:
                    generated = self.scaler.inverse_transform(generated)
                
                scenarios[scenario_type] = generated
        
        return scenarios
    
    def save_model(self, filepath: str):
        """Save trained VAE model."""
        if self.vae is None:
            raise ValueError("Model not trained")
        
        torch.save({
            'vae_state': self.vae.state_dict(),
            'optimizer_state': self.optimizer.state_dict(),
            'scaler': self.scaler,
            'training_history': self.training_history,
            'config': {
                'latent_dim': self.latent_dim,
                'learning_rate': self.learning_rate,
                'beta': self.beta,
                'use_time_series': self.use_time_series
            }
        }, filepath)
        
        logger.info(f"VAE model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load trained VAE model."""
        checkpoint = torch.load(filepath, map_location=self.device)
        
        # Restore config
        config = checkpoint['config']
        self.latent_dim = config['latent_dim']
        self.learning_rate = config['learning_rate']
        self.beta = config['beta']
        self.use_time_series = config['use_time_series']
        
        # Load states
        self.scaler = checkpoint['scaler']
        self.training_history = checkpoint['training_history']
        
        logger.info(f"VAE model loaded from {filepath}")
    
    def plot_latent_space(self, data: np.ndarray, labels: List[str] = None):
        """Plot 2D visualization of latent space."""
        if self.latent_dim < 2:
            logger.warning("Latent dimension < 2, cannot plot")
            return
        
        # Get latent representations
        latent_repr = self.get_latent_representation(data)
        
        if self.latent_dim > 2:
            # Use PCA to reduce to 2D
            pca = PCA(n_components=2)
            latent_2d = pca.fit_transform(latent_repr)
        else:
            latent_2d = latent_repr
        
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            unique_labels = list(set(labels))
            colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
            
            for i, label in enumerate(unique_labels):
                mask = np.array(labels) == label
                plt.scatter(latent_2d[mask, 0], latent_2d[mask, 1], 
                           c=[colors[i]], label=label, alpha=0.7)
            plt.legend()
        else:
            plt.scatter(latent_2d[:, 0], latent_2d[:, 1], alpha=0.7)
        
        plt.xlabel('Latent Dimension 1')
        plt.ylabel('Latent Dimension 2')
        plt.title('Latent Space Visualization')
        plt.grid(True)
        plt.show()


if __name__ == "__main__":
    # Example usage
    if TORCH_AVAILABLE:
        # Create sample economic data
        np.random.seed(42)
        dates = pd.date_range('2020-01-01', '2023-12-01', freq='M')
        
        # Multi-dimensional economic data
        gdp_growth = 0.02 + 0.01 * np.sin(2 * np.pi * np.arange(len(dates)) / 12) + np.random.normal(0, 0.005, len(dates))
        inflation = 0.025 + 0.008 * np.sin(2 * np.pi * np.arange(len(dates)) / 12 + np.pi/4) + np.random.normal(0, 0.003, len(dates))
        unemployment = 0.05 + 0.02 * np.sin(2 * np.pi * np.arange(len(dates)) / 12 + np.pi) + np.random.normal(0, 0.002, len(dates))
        
        economic_data = pd.DataFrame({
            'gdp_growth': gdp_growth,
            'inflation': inflation,
            'unemployment': unemployment
        }, index=dates)
        
        print("Training Economic VAE...")
        
        # Initialize and train VAE
        vae = EconomicVAE(latent_dim=16, use_time_series=True)
        
        # Prepare data
        training_data = vae.prepare_data(economic_data, sequence_length=24)
        
        # Train VAE
        training_results = vae.train(
            training_data,
            epochs=300,
            batch_size=16,
            save_interval=50
        )
        
        print(f"Training completed. Final total loss: {training_results['final_total_loss']:.4f}")
        
        # Generate synthetic samples
        synthetic_samples = vae.generate_samples(n_samples=50, sequence_length=24)
        print(f"Generated synthetic data shape: {synthetic_samples.shape}")
        
        # Generate economic scenarios
        base_conditions = {
            'gdp_growth': 0.02,
            'inflation': 0.025,
            'unemployment': 0.05
        }
        
        scenarios = vae.generate_economic_scenarios(
            base_conditions=base_conditions,
            n_scenarios=90,
            uncertainty_scale=1.5
        )
        
        print("Generated scenarios:")
        for scenario_type, data in scenarios.items():
            mean_values = np.mean(data, axis=(0, 1))
            print(f"  {scenario_type}: mean GDP={mean_values[0]:.4f}, inflation={mean_values[1]:.4f}, unemployment={mean_values[2]:.4f}")
        
        # Test interpolation
        sample1 = economic_data.iloc[:24].values
        sample2 = economic_data.iloc[12:36].values
        
        interpolated = vae.interpolate_in_latent_space(sample1, sample2, n_steps=5)
        print(f"Interpolated samples shape: {interpolated.shape}")
        
        print("Economic VAE example completed")
    else:
        print("PyTorch not available - skipping VAE example")