"""
Synthetic data generation for economic forecasting.
Implements GANs, VAEs, and other generative models for economic data synthesis.
"""

from .economic_gan import EconomicGAN
from .economic_vae import EconomicVAE
from .scenario_simulator import EconomicScenarioSimulator
from .data_augmentation import EconomicDataAugmentor

__all__ = [
    'EconomicGAN',
    'EconomicVAE',
    'EconomicScenarioSimulator',
    'EconomicDataAugmentor'
]