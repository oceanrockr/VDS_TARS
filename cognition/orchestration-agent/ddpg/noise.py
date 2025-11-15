"""
Ornstein-Uhlenbeck Noise Process

Implements temporally correlated noise for exploration in continuous action spaces.

The OU process is defined by:
    dx = θ(μ - x)dt + σdW

Where:
    - θ: Mean reversion rate
    - μ: Long-term mean
    - σ: Volatility
    - dW: Wiener process

Author: T.A.R.S. Cognitive Team
Version: v0.9.2-alpha
"""

import numpy as np
from typing import Optional


class OUNoise:
    """
    Ornstein-Uhlenbeck process for temporally correlated exploration noise.

    This noise process is useful for physical control tasks where actions
    should have temporal smoothness.

    Args:
        size (int): Dimension of action space
        mu (float): Long-term mean
        theta (float): Mean reversion rate
        sigma (float): Volatility
        dt (float): Time step
        x0 (Optional[np.ndarray]): Initial state
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma: float = 0.2,
        dt: float = 1e-2,
        x0: Optional[np.ndarray] = None
    ):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def reset(self):
        """Reset the internal state to the initial state."""
        self.x = (
            self.x0
            if self.x0 is not None
            else np.ones(self.size) * self.mu
        )

    def sample(self) -> np.ndarray:
        """
        Generate a sample from the OU process.

        Returns:
            np.ndarray: Noise sample of shape (size,)
        """
        dx = (
            self.theta * (self.mu - self.x) * self.dt +
            self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        )
        self.x = self.x + dx
        return self.x

    def __repr__(self) -> str:
        return (
            f"OUNoise(mu={self.mu}, theta={self.theta}, "
            f"sigma={self.sigma}, dt={self.dt})"
        )


class AdaptiveOUNoise:
    """
    Adaptive Ornstein-Uhlenbeck noise with decaying sigma.

    The noise level decreases over time to reduce exploration
    as the policy improves.

    Args:
        size (int): Dimension of action space
        mu (float): Long-term mean
        theta (float): Mean reversion rate
        sigma_start (float): Initial volatility
        sigma_end (float): Final volatility
        sigma_decay (float): Decay rate for sigma
        dt (float): Time step
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        theta: float = 0.15,
        sigma_start: float = 0.3,
        sigma_end: float = 0.05,
        sigma_decay: float = 0.9995,
        dt: float = 1e-2
    ):
        self.size = size
        self.mu = mu
        self.theta = theta
        self.sigma = sigma_start
        self.sigma_start = sigma_start
        self.sigma_end = sigma_end
        self.sigma_decay = sigma_decay
        self.dt = dt
        self.reset()

    def reset(self):
        """Reset the internal state."""
        self.x = np.ones(self.size) * self.mu
        self.sigma = self.sigma_start

    def sample(self) -> np.ndarray:
        """
        Generate a sample with decaying noise.

        Returns:
            np.ndarray: Noise sample
        """
        dx = (
            self.theta * (self.mu - self.x) * self.dt +
            self.sigma * np.sqrt(self.dt) * np.random.randn(self.size)
        )
        self.x = self.x + dx

        # Decay sigma
        self.sigma = max(self.sigma_end, self.sigma * self.sigma_decay)

        return self.x

    def get_sigma(self) -> float:
        """Get current sigma value."""
        return self.sigma


class GaussianNoise:
    """
    Simple Gaussian noise for exploration.

    Alternative to OU noise that's simpler but lacks temporal correlation.

    Args:
        size (int): Dimension of action space
        mu (float): Mean
        sigma (float): Standard deviation
    """

    def __init__(
        self,
        size: int,
        mu: float = 0.0,
        sigma: float = 0.1
    ):
        self.size = size
        self.mu = mu
        self.sigma = sigma

    def sample(self) -> np.ndarray:
        """
        Generate a sample from Gaussian distribution.

        Returns:
            np.ndarray: Noise sample
        """
        return np.random.normal(self.mu, self.sigma, self.size)

    def reset(self):
        """No-op for compatibility with OU noise."""
        pass
