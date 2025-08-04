"""
Trading Strategies Module

This module contains various trading strategies with consistent interfaces.
All strategies implement the get_action method for compatibility with the portfolio environment.
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any


def softmax(x: np.ndarray) -> np.ndarray:
    """Compute softmax values for numerical stability"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


class BaseStrategy(ABC):
    """
    Abstract base class for all trading strategies.
    
    All strategies must implement get_action method that takes an observation
    and returns portfolio weights that sum to 1.0.
    """
    
    def __init__(self, name: str):
        self.name = name
        self.env = None
    
    @abstractmethod
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """
        Generate portfolio weights from observation.
        
        Args:
            observation: Environment observation array
            
        Returns:
            Portfolio weights that sum to 1.0
        """
        pass
    
    def set_env(self, env) -> None:
        """Set environment reference for feature access"""
        self.env = env
    
    def get_config(self) -> Dict[str, Any]:
        """Return strategy configuration for serialization"""
        return {"name": self.name, "type": self.__class__.__name__}


class EqualWeightStrategy(BaseStrategy):
    """
    Simple equal weight strategy.
    Allocates portfolio equally across all assets.
    """
    
    def __init__(self):
        super().__init__("Equal Weight")
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Return equal weights for all assets"""
        n_assets = observation.shape[0]
        return np.ones(n_assets) / n_assets


class GreedyStrategy(BaseStrategy):
    """
    Momentum-based greedy strategy.
    Allocates more weight to assets with better recent performance.
    """
    
    def __init__(self, lookback: int = 5, temperature: float = 1.0):
        super().__init__(f"Greedy (lookback={lookback})")
        self.lookback = lookback
        self.temperature = temperature
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Calculate returns and allocate using softmax"""
        returns = []
        n_assets = observation.shape[0]
        
        # Get feature index for close price
        close_idx = 3 if self.env is None else self.env.features.index('close')
        
        # Calculate returns for each asset
        for i in range(n_assets):
            asset_data = observation[i, -self.lookback:, close_idx]
            
            if len(asset_data) < 2 or asset_data[0] == 0 or np.isnan(asset_data[0]):
                returns.append(0.0)
            else:
                asset_return = (asset_data[-1] - asset_data[0]) / (asset_data[0] + 1e-8)
                returns.append(asset_return)
        
        # Use softmax with temperature for allocation
        scores = softmax(np.array(returns) / self.temperature)
        return scores
    
    def get_config(self) -> Dict[str, Any]:
        """Return strategy configuration"""
        config = super().get_config()
        config.update({
            "lookback": self.lookback,
            "temperature": self.temperature
        })
        return config


class MomentumStrategy(BaseStrategy):
    """
    Advanced momentum strategy using multiple indicators.
    Combines price momentum with RSI and MACD signals.
    """
    
    def __init__(self, lookback: int = 10, rsi_weight: float = 0.3, 
                 macd_weight: float = 0.3, price_weight: float = 0.4):
        super().__init__(f"Momentum (lookback={lookback})")
        self.lookback = lookback
        self.rsi_weight = rsi_weight
        self.macd_weight = macd_weight
        self.price_weight = price_weight
        
        # Normalize weights
        total_weight = rsi_weight + macd_weight + price_weight
        self.rsi_weight /= total_weight
        self.macd_weight /= total_weight
        self.price_weight /= total_weight
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Generate weights using multi-indicator momentum"""
        n_assets = observation.shape[0]
        scores = np.zeros(n_assets)
        
        # Get feature indices
        if self.env is not None:
            close_idx = self.env.features.index('close')
            rsi_idx = self.env.features.index('rsi')
            macd_idx = self.env.features.index('macd_line')
        else:
            close_idx, rsi_idx, macd_idx = 3, 5, 6  # Default indices
        
        for i in range(n_assets):
            asset_data = observation[i, -self.lookback:]
            
            # Price momentum score
            close_prices = asset_data[:, close_idx]
            if len(close_prices) >= 2 and close_prices[0] != 0:
                price_momentum = (close_prices[-1] - close_prices[0]) / close_prices[0]
            else:
                price_momentum = 0.0
            
            # RSI momentum score (normalized)
            rsi_values = asset_data[:, rsi_idx]
            rsi_momentum = np.mean(rsi_values[~np.isnan(rsi_values)]) / 100.0 if len(rsi_values) > 0 else 0.5
            
            # MACD momentum score
            macd_values = asset_data[:, macd_idx]
            macd_momentum = np.mean(macd_values[~np.isnan(macd_values)]) if len(macd_values) > 0 else 0.0
            
            # Combine scores
            combined_score = (
                self.price_weight * price_momentum +
                self.rsi_weight * (rsi_momentum - 0.5) +  # Center RSI around 0
                self.macd_weight * macd_momentum
            )
            scores[i] = combined_score
        
        # Convert to weights using softmax
        weights = softmax(scores)
        return weights
    
    def get_config(self) -> Dict[str, Any]:
        """Return strategy configuration"""
        config = super().get_config()
        config.update({
            "lookback": self.lookback,
            "rsi_weight": self.rsi_weight,
            "macd_weight": self.macd_weight,
            "price_weight": self.price_weight
        })
        return config


class RandomStrategy(BaseStrategy):
    """
    Random portfolio allocation strategy.
    Useful as a baseline for comparison.
    """
    
    def __init__(self, seed: Optional[int] = None):
        super().__init__(f"Random (seed={seed})")
        self.rng = np.random.RandomState(seed)
    
    def get_action(self, observation: np.ndarray) -> np.ndarray:
        """Generate random portfolio weights"""
        n_assets = observation.shape[0]
        weights = self.rng.exponential(1.0, n_assets)
        return weights / weights.sum()
    
    def get_config(self) -> Dict[str, Any]:
        """Return strategy configuration"""
        config = super().get_config()
        config.update({
            "seed": self.rng.get_state()[1][0] if hasattr(self.rng, 'get_state') else None
        })
        return config


# Strategy factory for easy creation
STRATEGY_REGISTRY = {
    'equal_weight': EqualWeightStrategy,
    'greedy': GreedyStrategy,
    'momentum': MomentumStrategy,
    'random': RandomStrategy
}


def create_strategy(strategy_type: str, **kwargs) -> BaseStrategy:
    """
    Factory function to create strategies.
    
    Args:
        strategy_type: Type of strategy to create
        **kwargs: Strategy-specific parameters
        
    Returns:
        Initialized strategy instance
    """
    if strategy_type not in STRATEGY_REGISTRY:
        raise ValueError(f"Unknown strategy: {strategy_type}. Available: {list(STRATEGY_REGISTRY.keys())}")
    
    strategy_class = STRATEGY_REGISTRY[strategy_type]
    return strategy_class(**kwargs)