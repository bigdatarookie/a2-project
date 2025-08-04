"""
Portfolio Environment Module

This module contains the core portfolio management environment for reinforcement learning.
Provides memory-efficient, vectorized operations with optional rendering capabilities.
"""

import gymnasium as gym
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
import gc
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional


@dataclass
class EnvState:
    """Immutable state container for better memory management"""
    current_step: int
    portfolio_value: float
    weights: np.ndarray
    returns_history: List[float]
    daily_returns: List[float]
    weights_history: List[np.ndarray]
    current_prices: np.ndarray


class OptimizedPortfolioEnv(gym.Env):
    """
    Memory-efficient portfolio management environment with optional rendering.
    
    This environment supports:
    - Vectorized operations for performance
    - Optional rendering for development and monitoring
    - Configurable transaction costs and parameters
    - Multi-asset portfolio management
    
    Args:
        df: DataFrame with OHLCV data and indicators
        window_size: Size of observation window (default: 10)
        n_assets: Number of assets to trade (None for all available)
        transaction_cost: Transaction cost as fraction (default: 0.001)
        use_vectorized: Use vectorized operations (default: True)
        enable_rendering: Enable visualization (default: False)
        render_interval: Steps between renders (default: 10)
    """
    
    def __init__(self, df: pd.DataFrame, window_size: int = 10, n_assets: Optional[int] = None, 
                 transaction_cost: float = 0.001, use_vectorized: bool = True, 
                 enable_rendering: bool = False, render_interval: int = 10):
        super().__init__()
        
        # Core parameters
        self.window_size = window_size
        self.transaction_cost = transaction_cost
        self.use_vectorized = use_vectorized
        
        # Rendering parameters
        self.enable_rendering = enable_rendering
        self.render_interval = render_interval
        self.render_counter = 0
        
        # Get unique tickers
        self.tickers = df['tic'].unique()
        self.n_assets = len(self.tickers) if n_assets is None else min(n_assets, len(self.tickers))
        self.tickers = self.tickers[:self.n_assets]
        
        # Define features
        self.features = ['open', 'high', 'low', 'close', 'volume', 
                        'rsi', 'macd_line', 'bb_middle']
        self.n_features = len(self.features)

        # Prepare data efficiently
        self._prepare_data(df)

        # Define spaces
        self.action_space = spaces.Box(low=0, high=1, shape=(self.n_assets,))
        self.observation_space = spaces.Box(
            low=-np.inf, 
            high=np.inf, 
            shape=(self.n_assets, self.window_size, self.n_features)
        )
        
        # Initialize state
        self.state = None
        
        # Setup date mapping for charts
        self._setup_date_mapping(df)
        
    def _setup_date_mapping(self, df: pd.DataFrame) -> None:
        """Create mapping between steps and actual dates for chart readability"""
        try:
            unique_dates = pd.to_datetime(df.index).unique()
            unique_dates = sorted(unique_dates)
            self.dates = unique_dates[self.window_size:self.window_size + self.max_step + 1]
        except Exception as e:
            self.dates = list(range(self.window_size, self.window_size + self.max_step + 1))
            print(f"Date mapping failed, using step numbers: {e}")
        
    def _prepare_data(self, df: pd.DataFrame) -> None:
        """Prepare data with vectorized operations"""
        # Filter relevant columns only
        features = ['tic'] + self.features
        df = df[features].copy()
        
        # Convert to structured numpy array for memory efficiency
        data_list = []
        min_length = float('inf')
        
        for ticker in self.tickers:
            ticker_data = df[df['tic'] == ticker].copy()
            min_length = min(min_length, len(ticker_data))
            data_list.append(ticker_data.iloc[:min_length])
        
        # Create 3D array [n_assets, time_steps, features]
        self.data = np.stack([
            df[df['tic'] == ticker][self.features].values[:min_length]
            for ticker in self.tickers
        ])
        
        # Pre-calculate normalization factors for efficiency
        self.data_mean = np.mean(self.data, axis=1, keepdims=True)
        self.data_std = np.std(self.data, axis=1, keepdims=True) + 1e-8
        
        # Set max steps
        self.max_step = min_length - self.window_size - 1
        
        # Clear memory
        del data_list
        gc.collect()
    
    def _get_observation(self) -> np.ndarray:
        """Get normalized observation window efficiently"""
        if self.use_vectorized:
            obs = self.data[:, self.state.current_step-self.window_size:self.state.current_step]
            return (obs - self.data_mean) / self.data_std
        else:
            obs = np.zeros((self.n_assets, self.window_size, self.n_features))
            for i in range(self.n_assets):
                asset_data = self.data[i, self.state.current_step-self.window_size:self.state.current_step]
                mean = np.mean(asset_data, axis=0, keepdims=True)
                std = np.std(asset_data, axis=0, keepdims=True) + 1e-8
                obs[i] = (asset_data - mean) / std
            return obs
    
    def _calculate_returns_vectorized(self) -> Tuple[np.ndarray, float]:
        """Calculate returns using vectorized operations"""
        current_prices = self.data[:, self.state.current_step, self.features.index('close')]
        next_prices = self.data[:, self.state.current_step + 1, self.features.index('close')]
        asset_returns = (next_prices / current_prices) - 1
        portfolio_return = np.sum(self.state.weights * asset_returns)
        return asset_returns, portfolio_return
    
    def reset(self, seed=None):
        """Reset environment state"""
        super().reset(seed=seed)
        
        initial_prices = self.data[:, self.window_size, self.features.index('close')]
        
        self.state = EnvState(
            current_step=self.window_size,
            portfolio_value=1.0,
            weights=np.ones(self.n_assets) / self.n_assets,
            returns_history=[1.0],
            daily_returns=[],
            weights_history=[],
            current_prices=initial_prices
        )
        
        self.render_counter = 0
        return self._get_observation(), {}
    
    def step(self, action):
        """Execute environment step with optional rendering"""
        if self.state.current_step >= self.max_step:
            return self._get_observation(), 0.0, True, False, {
                'portfolio_value': self.state.portfolio_value,
                'weights': self.state.weights
            }
        
        # Update weights
        new_weights = action / np.sum(action)
        
        # Calculate transaction costs
        if len(self.state.weights_history) > 0:
            transaction_costs = np.sum(np.abs(new_weights - self.state.weights)) * self.transaction_cost
        else:
            transaction_costs = 0
        
        # Calculate returns
        if self.use_vectorized:
            _, portfolio_return = self._calculate_returns_vectorized()
        else:
            current_prices = self.data[:, self.state.current_step, self.features.index('close')]
            next_prices = self.data[:, self.state.current_step + 1, self.features.index('close')]
            asset_returns = (next_prices / current_prices) - 1
            portfolio_return = np.sum(new_weights * asset_returns)
        
        # Apply transaction costs
        portfolio_return -= transaction_costs
        
        # Get current prices for visualization
        current_prices = self.data[:, self.state.current_step, self.features.index('close')]
        
        # Update state
        self.state = EnvState(
            current_step=self.state.current_step + 1,
            portfolio_value=self.state.portfolio_value * (1 + portfolio_return),
            weights=new_weights,
            returns_history=self.state.returns_history + [self.state.portfolio_value * (1 + portfolio_return)],
            daily_returns=self.state.daily_returns + [portfolio_return],
            weights_history=self.state.weights_history + [new_weights.copy()],
            current_prices=current_prices
        )
        
        # Optional rendering at specified intervals
        if self.enable_rendering:
            self.render_counter += 1
            if self.render_counter % self.render_interval == 0:
                self.render()
        
        done = self.state.current_step >= self.max_step
        
        return self._get_observation(), portfolio_return, done, False, {
            'portfolio_value': self.state.portfolio_value,
            'weights': self.state.weights,
            'transaction_costs': transaction_costs,
            'returns': portfolio_return
        }
    
    def render(self, mode='inline'):
        """Enhanced rendering with date-indexed charts for Jupyter notebook"""
        if not self.enable_rendering:
            return
            
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        try:
            current_date = self.dates[min(self.state.current_step - self.window_size, len(self.dates)-1)]
            if isinstance(current_date, pd.Timestamp):
                date_str = current_date.strftime("%Y-%m-%d %H:%M")
            else:
                date_str = f"Step {current_date}"
        except:
            date_str = f"Step {self.state.current_step}"
            
        fig.suptitle(f'Portfolio Analysis - {date_str}', fontsize=14)
        
        # Portfolio value over time with dates
        steps = list(range(len(self.state.returns_history)))
        
        try:
            if isinstance(self.dates[0], pd.Timestamp):
                step_dates = [self.dates[min(i, len(self.dates)-1)] for i in steps]
                axes[0, 0].plot(step_dates, self.state.returns_history, 'b-', linewidth=2)
                axes[0, 0].xaxis.set_major_formatter(mdates.DateFormatter('%m-%d %H:%M'))
                plt.setp(axes[0, 0].xaxis.get_majorticklabels(), rotation=45)
            else:
                axes[0, 0].plot(steps, self.state.returns_history, 'b-', linewidth=2)
        except:
            axes[0, 0].plot(steps, self.state.returns_history, 'b-', linewidth=2)
            
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].grid(True, alpha=0.3)
        axes[0, 0].set_ylabel('Value ($)')
        
        # Daily returns
        if len(self.state.daily_returns) > 0:
            axes[0, 1].plot(self.state.daily_returns, 'g-', linewidth=1)
            axes[0, 1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
            axes[0, 1].set_title('Daily Returns')
            axes[0, 1].grid(True, alpha=0.3)
            axes[0, 1].set_ylabel('Return (%)')
        
        # Current allocations
        axes[1, 0].bar(range(len(self.tickers)), self.state.weights, color='orange', alpha=0.7)
        axes[1, 0].set_title('Current Allocations')
        axes[1, 0].set_xticks(range(len(self.tickers)))
        axes[1, 0].set_xticklabels(self.tickers, rotation=45)
        axes[1, 0].set_ylabel('Weight')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Current prices
        axes[1, 1].bar(range(len(self.tickers)), self.state.current_prices, color='skyblue', alpha=0.7)
        axes[1, 1].set_title('Current Prices')
        axes[1, 1].set_xticks(range(len(self.tickers)))
        axes[1, 1].set_xticklabels(self.tickers, rotation=45)
        axes[1, 1].set_ylabel('Price ($)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
        gc.collect()
    
    def set_render_interval(self, interval: int) -> None:
        """Update rendering interval during runtime"""
        self.render_interval = max(1, interval)
        
    def toggle_rendering(self, enable: bool) -> None:
        """Enable or disable rendering during runtime"""
        self.enable_rendering = enable
        if enable:
            self.render_counter = 0

    def cleanup(self) -> None:
        """Clean up environment resources"""
        self.state = None
        self.data = None
        self.data_mean = None
        self.data_std = None
        gc.collect()