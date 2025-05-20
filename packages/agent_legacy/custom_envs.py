"""
Custom enviroments for portfolio management.
"""
# Import libraries
import logging
from datetime import datetime, timedelta
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import pandas as pd
import numpy as np
from typing import List, Dict, Optional, Tuple, Union

# Configure logger
logger = logging.getLogger(__name__)


class MultiTickerPortfolioEnv(Env):
    """
    A multi-ticker portfolio environment using real historical stock data.
    - Uses price data from multiple tickers in a pandas DataFrame
    - Actions: Continuous values representing target allocation percentages for each ticker
    - Observations: Price history for all tickers + technical indicators + current allocation percentages
    - Reward: Portfolio return based on Sharpe ratio, diversification, and drawdown penalty
    """

    def __init__(
        self,
        df: pd.DataFrame,
        history_length: int = 10,
        initial_cash: float = 10000,
        risk_aversion: float = 2.0,
        price_column: str = 'close',
        date_column: str = 'date',
        ticker_column: str = 'tic',
        feature_columns: Optional[List[str]] = None,
        transaction_cost: float = 0.0005,           # 0.05% transaction cost
        reward_weights: Optional[Dict[str, float]] = None
    ):
        """
        Initialize a multi-ticker portfolio environment using real stock data.

        Args:
            df (pandas.DataFrame): DataFrame containing stock price data
            history_length (int): Number of time steps of price history to include in observations
            initial_cash (float): Starting cash amount
            risk_aversion (float): Risk aversion parameter that penalizes drawdowns (higher = more risk averse)
            price_column (str): Column name in df that contains price data
            date_column (str): Column name in df that contains date information
            ticker_column (str): Column name in df that contains ticker symbols
            feature_columns (list): Optional list of additional feature columns to include in observations
            transaction_cost (float): Cost of trading as percentage of trade value (0.001 = 0.1%)
            reward_weights (dict): Custom weights for reward components

        Reward Function Components (default weights):
            - Sharpe ratio (65%): Risk-adjusted returns using rolling window
            - Diversification (15%): Encourages balanced asset allocation
            - Drawdown penalty (20%): Non-linear penalty for maximum drawdowns
        """
        super(MultiTickerPortfolioEnv, self).__init__()

        # Store parameters
        self.risk_aversion = risk_aversion
        self.price_column = price_column
        self.date_column = date_column
        self.ticker_column = ticker_column
        self.feature_columns = feature_columns or []
        self.transaction_cost = transaction_cost

        # Set reward weights with defaults or custom values
        default_weights = {
            'sharpe': 0.65,
            'diversification': 0.15,
            'drawdown': 0.20
        }
        self.reward_weights = reward_weights or default_weights

        # Preprocess the data
        self.raw_df = df.copy()

        # Get unique tickers and dates in the dataset
        self.tickers = sorted(df[ticker_column].unique())
        self.n_stocks = len(self.tickers)
        logger.info(f"Environment created with {self.n_stocks} tickers: {self.tickers}")

        # Convert date strings to pandas datetime format with timezone handling
        if not pd.api.types.is_datetime64_any_dtype(df[date_column]):
            df[date_column] = pd.to_datetime(df[date_column], utc=True)

        # Create a date-sorted, pivot table format with tickers as columns and prices as values
        pivot_df = df.pivot(index=date_column, columns=ticker_column, values=price_column)

        # Sort by date to ensure time series is in correct order
        pivot_df = pivot_df.sort_index()

        # Forward fill then backward fill to handle any missing values
        self.price_data = pivot_df.ffill().bfill()

        # Store the original dates with timezone information for reference
        self.dates_with_tz = self.price_data.index.copy()

        # Automatically determine annualization factor based on time steps
        self.annualization_factor = self._determine_annualization_factor()
        logger.info(f"Auto-detected annualization factor: {self.annualization_factor}")

        # Convert to numpy array for faster access during training
        self.prices = self.price_data.values

        # Process additional features if provided
        self.feature_data = {}
        if self.feature_columns:
            for feature in self.feature_columns:
                if feature not in df.columns:
                    logger.warning(f"Feature '{feature}' not found in data, skipping")
                    continue

                feature_pivot = df.pivot(index=date_column, columns=ticker_column, values=feature)
                feature_pivot = feature_pivot.reindex(self.price_data.index)
                self.feature_data[feature] = feature_pivot.ffill().bfill().values

        # Store dates in a standardized format that preserves minute-level precision
        self.price_data.index = self.price_data.index.strftime('%Y-%m-%d %H:%M:%S')
        self.dates = self.price_data.index.values
        self.n_steps = len(self.dates)

        # Print sample dates to verify format
        logger.info(f"Sample dates in environment: {self.dates[:3]} (preserved down to minute resolution)")

        # Store parameters
        self.history_length = history_length
        self.initial_cash = initial_cash

        # Define action and observation spaces
        # Continuous action space: target allocation for each stock [0, 1]
        self.action_space = Box(
            low=0,
            high=1,
            shape=(self.n_stocks,),
            dtype=np.float32
        )

        # Calculate observation dimension
        # - Normalized price history: n_stocks * history_length values
        # - Momentum indicators: n_stocks values
        # - Current portfolio allocation: n_stocks values for stocks + 1 for cash
        # - Additional features: len(feature_columns) * n_stocks * history_length
        feature_dim = len(self.feature_data) * self.n_stocks * history_length if self.feature_data else 0

        obs_dim = (
            history_length * self.n_stocks +
            1 * self.n_stocks +  # Momentum indicators
            self.n_stocks + 1 +  # Stock allocations + cash allocation
            feature_dim
        )

        self.observation_space = Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

        # Tracking variables
        self.total_transaction_costs = 0

    def _determine_annualization_factor(self) -> int:
        """
        Automatically determine the annualization factor based on the time step frequency.

        Returns:
            int: Annualization factor (252 for daily, 252*24 for hourly, etc.)
        """
        if len(self.dates_with_tz) < 2:
            logger.warning("Not enough data points to determine frequency, using default (252)")
            return 252

        # Get the first few time deltas to determine the most common frequency
        # Skip potential irregular first few rows
        start_idx = min(5, len(self.dates_with_tz) - 10)
        end_idx = min(start_idx + 10, len(self.dates_with_tz) - 1)

        time_deltas = []
        for i in range(start_idx, end_idx):
            delta = self.dates_with_tz[i+1] - self.dates_with_tz[i]
            # Skip zero deltas (can happen with irregular data)
            if delta.total_seconds() > 0:
                time_deltas.append(delta.total_seconds())

        if not time_deltas:
            logger.warning("Could not determine time frequency, using default (252)")
            return 252

        # Get the most common time delta in seconds
        avg_seconds = sum(time_deltas) / len(time_deltas)

        # Map seconds to annualization factor
        if avg_seconds <= 90:  # ~1 minute or less
            return 252 * 24 * 60  # ~Trading days * hours * minutes
        elif avg_seconds <= 5400:  # ~90 minutes or less
            return 252 * 24  # ~Trading days * hours
        elif avg_seconds <= 129600:  # ~36 hours or less
            return 252  # ~Trading days per year
        elif avg_seconds <= 345600:  # ~4 days or less
            return 52  # ~Weeks per year
        else:
            return 12  # ~Months per year

    def reset(self, seed=None, options=None):
        """Reset the environment to initial state."""
        super().reset(seed=seed)

        # Start at a point where enough history is available
        self.current_step = self.history_length

        # Initialize portfolio: all cash, no stocks
        self.cash = self.initial_cash
        self.shares = np.zeros(self.n_stocks)

        # Track portfolio value over time
        self.portfolio_value_history = [self.initial_cash] * self.history_length
        self.max_portfolio_value = self.initial_cash

        # Reset transaction cost tracker
        self.total_transaction_costs = 0

        # Track trades for analysis
        self.trade_history = []

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        """
        Execute one step in the environment.

        Args:
            action: Array of target allocation percentages for each asset

        Returns:
            observation, reward, terminated, truncated, info
        """
        # Execute trades based on action (continuous allocation targets)
        transaction_cost = self._execute_trades(action)

        # Move to next time step
        self.current_step += 1

        # Calculate portfolio value and update history
        current_portfolio_value = self._get_portfolio_value()
        self.portfolio_value_history.append(current_portfolio_value)

        # Update maximum portfolio value for drawdown calculation
        if current_portfolio_value > self.max_portfolio_value:
            self.max_portfolio_value = current_portfolio_value

        # Calculate drawdown
        drawdown = (self.max_portfolio_value - current_portfolio_value) / self.max_portfolio_value if self.max_portfolio_value > 0 else 0

        # Calculate immediate return (for info only, not used in reward)
        prev_value = self.portfolio_value_history[-2]
        immediate_return = (current_portfolio_value / prev_value) - 1 if prev_value > 0 else 0

        # Initialize reward components
        sharpe_component = 0.0
        diversification_score = 0.0

        # 1. Sharpe Ratio Component
        window_size = min(20, len(self.portfolio_value_history) - 1)
        if window_size >= 5:  # Need at least a few points for meaningful calculation
            # Calculate rolling returns over the window
            values = np.array(self.portfolio_value_history[-window_size-1:])
            rolling_returns = (values[1:] / values[:-1]) - 1

            # Calculate mean and std of returns for Sharpe ratio
            mean_return = np.mean(rolling_returns)
            std_return = np.std(rolling_returns) + 1e-6  # Small epsilon to prevent division by zero

            # Calculate Sharpe-like ratio (without risk-free rate for simplicity)
            sharpe_component = (mean_return / std_return) * 10.0  # Scale appropriately

        # 2. Diversification Incentive
        # Calculate current portfolio allocations
        stock_values = self.shares * self.prices[self.current_step, :]
        total_value = self.cash + np.sum(stock_values)
        allocations = stock_values / total_value if total_value > 0 else np.zeros(self.n_stocks)

        # Calculate Herfindahl-Hirschman Index (HHI) - measure of concentration
        hhi = np.sum(allocations**2)

        # Calculate diversification score (1 = perfectly diversified, 0 = completely concentrated)
        ideal_hhi = 1.0 / self.n_stocks  # Perfect diversification
        max_penalty = 1.0
        # Penalize concentration that exceeds ideal by more than 50%
        concentration_ratio = min(max_penalty, max(0, (hhi - ideal_hhi) / (1.0 - ideal_hhi)))
        diversification_score = 1.0 - concentration_ratio

        # 3. Non-linear Drawdown Penalty
        # Penalize drawdowns more severely as they increase (squared term)
        drawdown_penalty = self.risk_aversion * (drawdown**2) * 20.0

        # Combine all components with weights from reward_weights
        reward = (
            sharpe_component * self.reward_weights['sharpe'] +
            diversification_score * self.reward_weights['diversification'] -
            drawdown_penalty * self.reward_weights['drawdown']
        )

        # Check if episode is done
        terminated = self.current_step >= self.n_steps - 1
        truncated = False

        # Get observation and info
        observation = self._get_observation()

        # Add detailed metrics to info for analysis
        info = self._get_info()
        info['drawdown'] = drawdown
        info['returns'] = immediate_return
        info['sharpe_component'] = sharpe_component
        info['diversification_score'] = diversification_score
        info['drawdown_penalty'] = drawdown_penalty
        info['allocations'] = allocations
        info['transaction_cost'] = transaction_cost
        info['total_transaction_costs'] = self.total_transaction_costs

        return observation, float(reward), terminated, truncated, info

    def _get_observation(self):
        """
        Get observation that includes price history, features, momentum indicators and portfolio state.
        Uses vectorized operations for better performance.
        """
        # Get price history ending at current step for all stocks
        end_idx = self.current_step
        start_idx = max(0, end_idx - self.history_length)
        price_history = self.prices[start_idx:end_idx, :]

        # Pad if not enough history
        if len(price_history) < self.history_length:
            padding_shape = (self.history_length - len(price_history), self.n_stocks)
            padding = np.ones(padding_shape) * price_history[0, :]
            price_history = np.vstack([padding, price_history])

        # Normalize price history (vectorized)
        last_prices = price_history[-1, :]
        last_prices_nonzero = np.where(last_prices > 0, last_prices, 1.0)
        normalized_history = price_history / last_prices_nonzero

        # Process additional features if available
        feature_history_flat = np.array([])
        if self.feature_data:
            feature_history_list = []
            for feature_name, feature_data in self.feature_data.items():
                feature_history = feature_data[start_idx:end_idx, :]

                # Pad if not enough history
                if len(feature_history) < self.history_length:
                    padding_shape = (self.history_length - len(feature_history), self.n_stocks)
                    padding = np.ones(padding_shape) * feature_history[0, :]
                    feature_history = np.vstack([padding, feature_history])

                # Normalize feature data (vectorized)
                last_values = feature_history[-1, :]
                # Replace zeros with ones to avoid division by zero
                last_values_nonzero = np.where(np.abs(last_values) > 1e-6, last_values, 1.0)
                normalized_feature = feature_history / last_values_nonzero

                feature_history_list.append(normalized_feature.flatten())

            if feature_history_list:
                feature_history_flat = np.concatenate(feature_history_list)

        # Calculate momentum (vectorized where possible)
        short_lookback = min(3, len(price_history) - 1)
        if short_lookback > 0:
            # Get current and past prices
            current_prices = price_history[-1, :]
            past_prices = price_history[-short_lookback-1, :]

            # Calculate momentum (% change)
            # Avoid division by zero
            past_prices_nonzero = np.where(past_prices > 0, past_prices, 1.0)
            momentum_indicators = (current_prices / past_prices_nonzero) - 1
        else:
            momentum_indicators = np.zeros(self.n_stocks)

        # Flatten the normalized price history
        price_history_flat = normalized_history.flatten()

        # Get current portfolio allocation as percentages
        total_value = self._get_portfolio_value()
        stock_values = self.shares * self.prices[self.current_step, :]
        portfolio_allocations = stock_values / total_value if total_value > 0 else np.zeros(self.n_stocks)

        # Add cash allocation as a feature
        cash_allocation = self.cash / total_value if total_value > 0 else 1.0

        # Combine everything into the observation
        observation_parts = [
            price_history_flat,          # Normalized price history
            momentum_indicators,         # Simple momentum indicators
            portfolio_allocations,       # Current stock allocations
            [cash_allocation]            # Cash allocation
        ]

        # Add feature history if available
        if feature_history_flat.size > 0:
            observation_parts.append(feature_history_flat)

        observation = np.concatenate(observation_parts).astype(np.float32)

        return observation

    def _execute_trades(self, action):
        """
        Execute trades based on target allocation action values.
        Returns the transaction cost incurred.
        """
        # Ensure action values are valid
        action = np.clip(action, 0, 1)

        # Normalize action to ensure sum <= 1 (remaining is cash)
        action_sum = np.sum(action)
        if action_sum > 1.0:
            action = action / action_sum

        # Calculate current portfolio value and allocations
        current_prices = self.prices[self.current_step, :]
        total_value = self._get_portfolio_value()
        current_stock_values = self.shares * current_prices

        # Calculate target values for each stock
        target_stock_values = action * total_value

        # Calculate value difference (positive = buy, negative = sell)
        value_difference = target_stock_values - current_stock_values

        # Track total transaction value for cost calculation
        total_transaction_value = 0

        # Execute trades (sells first to ensure liquidity)
        # Sell orders - stocks where target allocation is lower than current
        for i, diff in enumerate(value_difference):
            if diff < 0 and current_prices[i] > 0:
                # Convert value to shares
                shares_to_sell = abs(diff) / current_prices[i]
                shares_to_sell = min(shares_to_sell, self.shares[i])  # Can't sell more than we have

                transaction_value = shares_to_sell * current_prices[i]
                total_transaction_value += transaction_value

                # Execute sell (account for transaction costs)
                self.cash += transaction_value * (1 - self.transaction_cost)
                self.shares[i] -= shares_to_sell

                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'date': self.dates[self.current_step],
                    'ticker': self.tickers[i],
                    'action': 'sell',
                    'shares': shares_to_sell,
                    'price': current_prices[i],
                    'value': transaction_value
                })

        # Buy orders - stocks where target allocation is higher than current
        for i, diff in enumerate(value_difference):
            if diff > 0 and current_prices[i] > 0:
                # Account for transaction costs in buy calculation
                available_cash_with_cost = self.cash / (1 + self.transaction_cost)

                # Calculate maximum shares we can buy with available cash
                max_shares = available_cash_with_cost / current_prices[i]
                shares_to_buy = min(diff / current_prices[i], max_shares)

                transaction_value = shares_to_buy * current_prices[i]
                total_transaction_value += transaction_value

                # Execute buy (including transaction costs)
                self.cash -= transaction_value * (1 + self.transaction_cost)
                self.shares[i] += shares_to_buy

                # Record trade
                self.trade_history.append({
                    'step': self.current_step,
                    'date': self.dates[self.current_step],
                    'ticker': self.tickers[i],
                    'action': 'buy',
                    'shares': shares_to_buy,
                    'price': current_prices[i],
                    'value': transaction_value
                })

        # Calculate transaction cost for this step
        transaction_cost = total_transaction_value * self.transaction_cost
        self.total_transaction_costs += transaction_cost

        return transaction_cost

    def _get_portfolio_value(self):
        """Calculate total portfolio value (cash + all stock holdings)"""
        current_prices = self.prices[self.current_step, :]
        stock_values = self.shares * current_prices
        return self.cash + np.sum(stock_values)

    def _get_info(self):
        """Return dictionary with current state information"""
        return {
            'date': self.dates[self.current_step],
            'portfolio_value': self._get_portfolio_value(),
            'cash': self.cash,
            'shares': self.shares.copy(),
            'max_portfolio_value': self.max_portfolio_value,
            'annualization_factor': self.annualization_factor,
            'portfolio_history': self.portfolio_value_history.copy(),
            'trade_history': self.trade_history.copy()
        }
