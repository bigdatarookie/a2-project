# %% [markdown]
# # Reinforcement Learning Portfolio Strategy Evaluation
#
# This notebook implements a rigorous backtesting framework for evaluating RL-based portfolio optimization models
# against traditional benchmark strategies, with transaction costs factored in.

# %% [markdown]
# ## 1. Setup and Imports
# %%
# Standard libraries
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional, Union, Callable
import warnings

# Gymnasium
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box

# Stable Baselines for RL
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# Silence warnings
warnings.filterwarnings('ignore')


# %%
# Global configuration parameters
CONFIG = {
    'data_path': 'data/rts_2h_2024.csv',    # Path to price data
    'model_path': 'data/models/final_model.zip',  # Path to trained model
    'initial_cash': 10000,                    # Initial investment amount
    'transaction_cost': 0.0005,               # 0.05% transaction cost
    'test_size': 0.2,                         # Proportion of data for testing
    'history_length': 24,                     # Price history length
    'risk_aversion': 2.0,                     # Risk aversion parameter
    'random_seed': 42,                        # Random seed for reproducibility
    'results_path': 'data/models/results/',   # Path to save results
}

# Create results directory if it doesn't exist
os.makedirs(CONFIG['results_path'], exist_ok=True)


# %% [markdown]
# ## 2. Custom Environment Definition

# %%
class MultiTickerPortfolioEnv(Env):
    """
    A multi-ticker portfolio environment using real historical stock data.
    - Uses 'close' prices from multiple tickers in a pandas DataFrame
    - Actions: Buy/Sell/Hold decisions for each ticker
    - Observations: Price history for all tickers + technical indicators + current allocation percentages
    - Reward: Portfolio return penalized by drawdown
    """

    def __init__(self, df, history_length=10, initial_cash=10000, risk_aversion=2.0):
        """
        Initialize a multi-ticker portfolio environment using real stock data.

        Args:
            df (pandas.DataFrame): DataFrame containing stock price data with 'tic', 'date', and 'close' columns
            history_length (int): Number of time steps of price history to include in observations
            initial_cash (float): Starting cash amount
            risk_aversion (float): Risk aversion parameter that penalizes drawdowns (higher = more risk averse)
                                  Used in the reward function to balance returns vs drawdowns

        Reward Function Components:
            - Short-term returns (30%): Immediate portfolio returns scaled by 100
            - Sharpe ratio (40%): Risk-adjusted returns using rolling window
            - Consistency (10%): Bonus for maintaining positive returns
            - Diversification (10%): Encourages balanced asset allocation
            - Drawdown penalty (10%): Non-linear penalty for maximum drawdowns
        """
        super(MultiTickerPortfolioEnv, self).__init__()

        # Store risk aversion parameter
        self.risk_aversion = risk_aversion

        # Preprocess the data
        self.raw_df = df.copy()

        # Get unique tickers and dates in the dataset
        self.tickers = sorted(df['tic'].unique())
        self.n_stocks = len(self.tickers)
        print(f"Environment created with {self.n_stocks} tickers: {self.tickers}")

        # Convert date strings to pandas datetime format with timezone handling
        if not pd.api.types.is_datetime64_any_dtype(df['date']):
            df['date'] = pd.to_datetime(df['date'], utc=True)

        # Create a date-sorted, pivot table format with tickers as columns and close prices as values
        pivot_df = df.pivot(index='date', columns='tic', values='close')

        # Sort by date to ensure time series is in correct order
        pivot_df = pivot_df.sort_index()

        # Forward fill then backward fill to handle any missing values
        self.price_data = pivot_df.ffill().bfill()

        # Store the original dates with timezone information for reference
        self.dates_with_tz = self.price_data.index.copy()

        # Convert to numpy array for faster access during training
        self.prices = self.price_data.values

        # Store dates in a standardized format (YYYY-MM-DD HH:00:00) to preserve hourly data
        # This still removes timezone info which causes lookup issues
        self.price_data.index = self.price_data.index.strftime('%Y-%m-%d %H:00:00')
        self.dates = self.price_data.index.values
        self.n_steps = len(self.dates)

        # Print sample dates to verify format
        print(f"Sample dates in environment: {self.dates[:3]} (preserved hourly data)")

        # Store parameters
        self.history_length = history_length
        self.initial_cash = initial_cash

        # Define action and observation spaces
        # Each stock can have 3 actions: 0=Buy, 1=Hold, 2=Sell
        # We encode this as a flat action space with 3^n_stocks possibilities
        self.action_space = Discrete(3**self.n_stocks)

        # Observation: history_length prices for each stock + momentum indicators + current allocation percentages
        # Each observation contains:
        # - Normalized price history: n_stocks * history_length values
        # - Momentum indicators: n_stocks values (simple % change)
        # - Current portfolio allocation: n_stocks values for stocks + 1 for cash
        obs_dim = (history_length * self.n_stocks +
                   1 * self.n_stocks +  # Simple momentum indicators
                   self.n_stocks + 1)   # Stock allocations + cash allocation

        self.observation_space = Box(
            low=-np.inf,  # Some indicators like momentum can be negative
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        # Start at a point where enough history is available
        self.current_step = self.history_length

        # Initialize portfolio: all cash, no stocks
        self.cash = self.initial_cash
        self.shares = np.zeros(self.n_stocks)

        # Track portfolio value over time
        self.portfolio_value_history = [self.initial_cash] * self.history_length
        self.max_portfolio_value = self.initial_cash

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Execute trades based on action
        self._execute_trades(action)

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

        # Calculate immediate return (short-term component)
        prev_value = self.portfolio_value_history[-2]
        immediate_return = (current_portfolio_value / prev_value) - 1 if prev_value > 0 else 0

        # Amplify immediate returns to make small changes more visible
        scaled_immediate_return = immediate_return * 100.0

        # Initialize reward components
        sharpe_component = 0.0
        consistency_bonus = 0.0
        diversification_score = 0.0

        # 1. Sharpe Ratio Component (addresses time scale misalignment)
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

        # 2. Consistency Bonus (rewards sustained performance)
        lookback = min(5, len(self.portfolio_value_history) - 1)
        if lookback >= 3:
            # Count number of positive days in recent history
            positive_days = np.sum(np.array(self.portfolio_value_history[-lookback:]) >
                                  np.array(self.portfolio_value_history[-lookback-1:-1]))
            consistency_bonus = 0.5 * (positive_days / lookback)  # Normalize to [0, 0.5]

        # 3. Diversification Incentive
        # Calculate current portfolio allocations
        allocations = self.shares * self.prices[self.current_step, :] / current_portfolio_value if current_portfolio_value > 0 else np.zeros(self.n_stocks)

        # Calculate Herfindahl-Hirschman Index (HHI) - measure of concentration
        # HHI ranges from 1/n (perfectly diversified) to 1 (completely concentrated)
        hhi = np.sum(allocations**2)

        # Calculate diversification score (1 = perfectly diversified, 0 = completely concentrated)
        ideal_hhi = 1.0 / self.n_stocks  # Perfect diversification
        max_penalty = 1.0
        # Penalize concentration that exceeds ideal by more than 50%
        concentration_ratio = min(max_penalty, max(0, (hhi - ideal_hhi) / (1.0 - ideal_hhi)))
        diversification_score = 1.0 - concentration_ratio

        # 4. Non-linear Drawdown Penalty
        # Penalize drawdowns more severely as they increase (squared term)
        drawdown_penalty = self.risk_aversion * (drawdown**2) * 20.0

        # 5. Combine all components with appropriate weights
        reward = (
            scaled_immediate_return * 0.3 +       # Short-term return (30% weight)
            sharpe_component * 0.4 +              # Risk-adjusted return (40% weight)
            consistency_bonus * 0.1 +             # Consistency bonus (10% weight)
            diversification_score * 0.1 -         # Diversification incentive (10% weight)
            drawdown_penalty * 0.1                # Drawdown penalty (10% weight)
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
        info['portfolio_value'] = current_portfolio_value
        info['sharpe_component'] = sharpe_component
        info['consistency_bonus'] = consistency_bonus
        info['diversification_score'] = diversification_score
        info['drawdown_penalty'] = drawdown_penalty
        info['allocations'] = allocations

        return observation, float(reward), terminated, truncated, info

    def _get_observation(self):
        """
        Get a simplified observation that's easier for the agent to learn from.
        Focuses on recent price changes and current portfolio state.
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

        # Normalize price history by dividing by the last price
        # This makes the features relative rather than absolute
        last_prices = price_history[-1, :]
        normalized_history = np.zeros_like(price_history)
        for i in range(self.n_stocks):
            if last_prices[i] > 0:
                normalized_history[:, i] = price_history[:, i] / last_prices[i]
            else:
                normalized_history[:, i] = price_history[:, i]

        # Calculate simple momentum indicators (% change over different periods)
        momentum_indicators = []
        for i in range(self.n_stocks):
            # Get most recent price for this stock
            current_price = price_history[-1, i] if len(price_history) > 0 else 1.0

            # Short-term momentum (last 3 periods if available)
            short_lookback = min(3, len(price_history) - 1)
            if short_lookback > 0 and price_history[-short_lookback-1, i] > 0:
                short_momentum = (current_price / price_history[-short_lookback-1, i]) - 1
            else:
                short_momentum = 0

            momentum_indicators.append(short_momentum)

        # Flatten the normalized price history
        price_history_flat = normalized_history.flatten()

        # Get current portfolio allocation as percentages
        total_value = self._get_portfolio_value()
        stock_values = self.shares * self.prices[self.current_step, :]
        portfolio_allocations = stock_values / total_value if total_value > 0 else np.zeros(self.n_stocks)

        # Add cash allocation as a feature
        cash_allocation = self.cash / total_value if total_value > 0 else 1.0

        # Combine everything into the observation
        observation = np.concatenate([
            price_history_flat,          # Normalized price history
            momentum_indicators,         # Simple momentum indicators
            portfolio_allocations,       # Current stock allocations
            [cash_allocation]            # Cash allocation
        ]).astype(np.float32)

        return observation

    def _execute_trades(self, action):
        # Decode the flat action into per-stock actions
        stock_actions = self._decode_action(action)

        # Get current prices
        current_prices = self.prices[self.current_step, :]

        # Calculate volatility for each stock to determine position sizing
        volatilities = []
        min_volatility = 1e-4  # Define a minimum volatility to avoid division by zero
        for i in range(self.n_stocks):
            # Get recent price history for this stock
            history_window = min(20, self.current_step)  # Use up to 20 periods
            if history_window > 1:
                stock_prices = self.prices[self.current_step - history_window:self.current_step, i]
                # Calculate log returns
                returns = np.diff(np.log(stock_prices))
                # Calculate volatility (standard deviation of returns)
                volatility = np.std(returns) if len(returns) > 0 else min_volatility
            else:
                volatility = min_volatility  # Default volatility if not enough history

            # Ensure volatility is not below the minimum threshold
            volatility = max(volatility, min_volatility)
            volatilities.append(volatility)

        # Normalize volatilities for inverse weighting (lower volatility = higher allocation)
        total_vol = sum(volatilities) if sum(volatilities) > 0 else len(volatilities)
        inv_vol_weights = [(1 / vol) / (len(volatilities) / total_vol) for vol in volatilities]

        # Safeguard against sum(inv_vol_weights) being zero
        sum_inv_vol_weights = sum(inv_vol_weights)
        if sum_inv_vol_weights == 0:
            sum_inv_vol_weights = 1  # Prevent division by zero

        # Execute trades for each stock
        for i, act in enumerate(stock_actions):
            price = current_prices[i]

            if price <= 0:  # Skip invalid prices
                continue

            if act == 0:  # Buy
                # Use adaptive position sizing based on inverse volatility
                # Higher volatility = smaller position size
                position_size = 0.1 * inv_vol_weights[i] / sum(inv_vol_weights)
                # Cap at 20% of cash
                position_size = min(position_size, 0.2)

                buy_amount = self.cash * position_size
                if buy_amount > 0:
                    self.shares[i] += buy_amount / price
                    self.cash -= buy_amount

            elif act == 2:  # Sell
                # Sell based on volatility - sell more of higher volatility stocks
                # Normalize to maximum 20% of holdings
                sell_fraction = min(0.2, 0.1 * volatilities[i] / max(volatilities) if max(volatilities) > 0 else 0.1)

                if self.shares[i] > 0:
                    sell_shares = self.shares[i] * sell_fraction
                    self.cash += sell_shares * price
                    self.shares[i] -= sell_shares

            # Action 1: Hold - do nothing

    def _decode_action(self, action_idx):
        """Convert flat action index to per-stock actions"""
        actions = []
        remaining = action_idx

        for _ in range(self.n_stocks):
            actions.append(remaining % 3)  # 0: Buy, 1: Hold, 2: Sell
            remaining //= 3

        return actions

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
            'max_portfolio_value': self.max_portfolio_value
        }

# %% [markdown]
# ## 3. Data Loading and Splitting

# %%
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load price data from CSV file

    Args:
        file_path: Path to CSV file with price data

    Returns:
        DataFrame with price data
    """
    try:
        df = pd.read_csv(file_path)
        print(f"Loaded data from {file_path} with shape {df.shape}")
        print("Sample data:")
        print(df.head())
        return df
    except Exception as e:
        print(f"Error loading data: {e}")
        return None

# %%
def train_test_split(
    df: pd.DataFrame,
    test_size: float = 0.2,
    split_date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split data into training and test sets, either by proportion or specific date

    Args:
        df: DataFrame with price data
        test_size: Proportion of data to use for testing (if split_date is None)
        split_date: Specific date to split on (format: 'YYYY-MM-DD')

    Returns:
        Tuple of (train_df, test_df)
    """
    if df is None:
        return None, None

    # Make a copy to avoid modifying the original
    df_copy = df.copy()

    # Ensure date column is datetime
    if 'date' in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy['date'])

    if split_date:
        # Split by specific date
        split_date = pd.to_datetime(split_date)
        train_df = df_copy[df_copy['date'] < split_date]
        test_df = df_copy[df_copy['date'] >= split_date]
    else:
        # Split by proportion
        dates = df_copy['date'].unique()
        split_idx = int(len(dates) * (1 - test_size))
        split_date = dates[split_idx]

        train_df = df_copy[df_copy['date'] < split_date]
        test_df = df_copy[df_copy['date'] >= split_date]

    print(f"Training data: {len(train_df)} samples from {train_df['date'].min()} to {train_df['date'].max()}")
    print(f"Testing data: {len(test_df)} samples from {test_df['date'].min()} to {test_df['date'].max()}")

    return train_df, test_df

# %%
# Checking data integrity

train_data, test_data = train_test_split(data, test_size=0.2)
test_data


# %% [markdown]
# ## 4. Model Loading

# %%
def load_model(model_path: str) -> A2C:
    """
    Load a trained model from file

    Args:
        model_path: Path to the saved model file

    Returns:
        Loaded model
    """
    try:
        model = A2C.load(model_path)
        print(f"Successfully loaded model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# %% [markdown]
# ## 5. Strategy Implementation

# %%
def a2c_strategy(model: A2C) -> Callable:
    """
    Create a strategy function from a trained A2C model

    Args:
        model: Trained A2C model

    Returns:
        Strategy function that takes observation and returns action
    """
    def strategy(observation):
        action, _ = model.predict(observation, deterministic=True)
        return action
    return strategy

def equal_weight_strategy(observation):
    """
    Equal weight strategy - allocate portfolio equally across all assets

    Args:
        observation: Environment observation

    Returns:
        Action index
    """
    # Extract number of stocks from observation
    obs_size = len(observation)

    # Try to estimate n_stocks from observation length
    # We know obs_dim = history_length * n_stocks + n_stocks + n_stocks + 1
    # Let's try common values for n_stocks (e.g., 3-10)
    for n in range(3, 11):
        # Check if obs_size matches a valid formula with this n_stocks
        if (obs_size - 1) % (n + 2) == 0:
            n_stocks = n
            break
    else:
        # Default fallback
        n_stocks = 3
        print(f"Warning: Could not determine number of stocks from observation size {obs_size}. Using default: {n_stocks}")

    # Current allocations are the last n_stocks values before cash
    portfolio_allocations = observation[-n_stocks-1:-1]

    # If minimal allocation, buy all stocks equally
    if np.sum(portfolio_allocations) < 0.01:
        return 0  # This will buy all stocks with the default allocation
    else:
        # Otherwise hold
        return 1

def momentum_strategy(observation):
    """
    Momentum strategy - buy stocks with positive momentum, sell those with negative

    Args:
        observation: Environment observation

    Returns:
        Action index
    """
    # Extract number of stocks from observation
    obs_size = len(observation)

    # Last n_stocks + 1 elements are allocations (n_stocks) + cash (1)
    # One element before that is the start of allocation data
    # Count backwards to find n_stocks

    # Try to estimate n_stocks from observation length
    # We know obs_dim = history_length * n_stocks + n_stocks + n_stocks + 1
    # Let's try common values for n_stocks (e.g., 3-10)
    for n in range(3, 11):
        # Check if obs_size matches a valid formula with this n_stocks
        if (obs_size - 1) % (n + 2) == 0:
            n_stocks = n
            break
    else:
        # Default fallback
        n_stocks = 3
        print(f"Warning: Could not determine number of stocks from observation size {obs_size}. Using default: {n_stocks}")

    # Calculate history_length based on observation size and n_stocks
    history_length = (obs_size - 2*n_stocks - 1) // n_stocks

    # Extract momentum indicators from observation
    # Momentum indicators come after price history
    momentum_start = history_length * n_stocks
    momentum_indicators = observation[momentum_start:momentum_start+n_stocks]

    # Determine action for each stock based on momentum
    stock_actions = []
    for i in range(n_stocks):
        if momentum_indicators[i] > 0.01:  # Positive momentum
            stock_actions.append(0)  # Buy
        elif momentum_indicators[i] < -0.01:  # Negative momentum
            stock_actions.append(2)  # Sell
        else:
            stock_actions.append(1)  # Hold

    # Encode actions into a single index
    action_index = 0
    for i, action in enumerate(stock_actions):
        action_index += action * (3 ** i)

    return action_index

# %% [markdown]
# ## 6. Backtesting Infrastructure

# %%
def backtest_strategy(
    env,
    strategy_fn,
    initial_cash=10000,
    name="Strategy",
    transaction_cost=0.0005
):
    """
    Backtest a strategy in the given environment with transaction costs

    Args:
        env: The trading environment
        strategy_fn: Strategy function that takes observation and returns action
        initial_cash: Initial investment amount
        name: Name of the strategy
        transaction_cost: Transaction cost as a fraction (0.0005 = 0.05%)

    Returns:
        Dictionary with backtest results
    """
    # Reset the environment
    observation, info = env.reset(seed=CONFIG['random_seed'])

    # Lists to track metrics
    dates = []
    portfolio_values = []
    returns = []
    allocations = []
    actions = []
    cash_values = []
    transaction_costs_history = []
    cumulative_costs = 0
    reward_components = []

    # Initialize previous allocations
    prev_allocations = np.zeros(len(env.tickers))

    # Run simulation
    while True:
        # Get action from strategy
        action = strategy_fn(observation)

        # Store previous values for transaction cost calculation
        prev_portfolio_value = info['portfolio_value'] if len(portfolio_values) > 0 else initial_cash

        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Store date and cash
        dates.append(info['date'])
        cash_values.append(info['cash'])

        # Extract current allocations
        current_allocations = info['allocations'] if 'allocations' in info else np.zeros(len(env.tickers))

        # Calculate transaction costs
        allocation_changes = np.abs(current_allocations - prev_allocations)
        txn_cost = transaction_cost * prev_portfolio_value * np.sum(allocation_changes)

        # Adjust portfolio value for transaction costs
        adjusted_portfolio_value = info['portfolio_value'] - txn_cost
        portfolio_values.append(adjusted_portfolio_value)

        # Update transaction cost metrics
        transaction_costs_history.append(txn_cost)
        cumulative_costs += txn_cost

        # Store other metrics
        allocations.append(current_allocations)
        actions.append(action)

        # Store reward components if available
        if 'sharpe_component' in info:
            reward_components.append({
                'sharpe': info['sharpe_component'],
                'consistency': info['consistency_bonus'],
                'diversification': info['diversification_score'],
                'drawdown_penalty': info['drawdown_penalty'],
                'drawdown': info['drawdown']
            })

        # Update previous allocations for next iteration
        prev_allocations = current_allocations.copy()

        # Calculate return
        if len(portfolio_values) > 1:
            period_return = (portfolio_values[-1] / portfolio_values[-2]) - 1
            returns.append(period_return)

        # Check if done
        if terminated or truncated:
            break

    # Convert to numpy arrays
    portfolio_values = np.array(portfolio_values)
    returns = np.array(returns)
    allocations = np.array(allocations)
    transaction_costs_history = np.array(transaction_costs_history)

    # Calculate performance metrics
    total_return = (portfolio_values[-1] / initial_cash) - 1

    # Calculate period length and annualization factor
    date_objects = [pd.to_datetime(d) for d in dates]
    period_diffs = [(date_objects[i+1] - date_objects[i]).total_seconds() for i in range(len(date_objects)-1)]

    if len(period_diffs) > 0:
        avg_period_seconds = sum(period_diffs) / len(period_diffs)
        seconds_per_year = 365.25 * 24 * 60 * 60
        periods_per_year = seconds_per_year / avg_period_seconds

        # Annualized return
        annualized_return = (1 + total_return) ** (periods_per_year / len(portfolio_values)) - 1

        # Annualized volatility
        if len(returns) > 1:
            volatility = np.std(returns) * np.sqrt(periods_per_year)
        else:
            volatility = 0
    else:
        periods_per_year = 252  # Default to trading days per year
        annualized_return = total_return
        volatility = 0

    # Max drawdown
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Risk metrics
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Downside volatility (for Sortino ratio)
    downside_returns = returns[returns < 0]
    downside_volatility = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 1e-6
    sortino_ratio = annualized_return / downside_volatility

    # Create DataFrame with time series data
    results_df = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'cash': cash_values,
        'transaction_costs': transaction_costs_history
    })

    # Add allocations
    for i, ticker in enumerate(env.tickers):
        if allocations.shape[1] == len(env.tickers):
            results_df[f'allocation_{ticker}'] = allocations[:, i]

    # Add reward components if available
    if reward_components:
        results_df['sharpe_component'] = [rc['sharpe'] for rc in reward_components]
        results_df['consistency_bonus'] = [rc['consistency'] for rc in reward_components]
        results_df['diversification_score'] = [rc['diversification'] for rc in reward_components]
        results_df['drawdown_penalty'] = [rc['drawdown_penalty'] for rc in reward_components]
        results_df['drawdown'] = [rc['drawdown'] for rc in reward_components]

    # Set date as index
    results_df['date'] = pd.to_datetime(results_df['date'])
    results_df.set_index('date', inplace=True)

    # Return comprehensive results dictionary
    return {
        'name': name,
        'data': results_df,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'final_value': portfolio_values[-1],
        'transaction_costs': {
            'total': cumulative_costs,
            'average': cumulative_costs / len(portfolio_values) if len(portfolio_values) > 0 else 0,
            'percentage': transaction_cost * 100
        },
        'periods_per_year': periods_per_year
    }

# %%
def run_backtests(
    test_data,
    model=None,
    transaction_cost=CONFIG['transaction_cost'],
    initial_cash=CONFIG['initial_cash'],
    history_length=CONFIG['history_length'],
    risk_aversion=CONFIG['risk_aversion']
):
    """
    Run backtest for all strategies on the test data

    Args:
        test_data: Test data DataFrame
        model: Trained A2C model (optional)
        transaction_cost: Transaction cost
        initial_cash: Initial investment
        history_length: History length for environment
        risk_aversion: Risk aversion parameter

    Returns:
        Dictionary with backtest results for all strategies
    """
    print(f"Running backtests with transaction cost: {transaction_cost*100:.3f}%")

    # If a model is provided, adjust the history_length parameter to match the model's expected observation space
    if model:
        # Extract model's observation space shape
        model_obs_shape = model.observation_space.shape[0]
        print(f"Model observation space shape: {model_obs_shape}")

        # Get number of stocks
        n_stocks = len(test_data['tic'].unique())

        # Calculate the history_length that would result in the correct observation space
        # obs_dim = (history_length * n_stocks + n_stocks + n_stocks + 1)
        # Solve for history_length:
        # history_length = (obs_dim - 2*n_stocks - 1) / n_stocks
        calculated_history_length = (model_obs_shape - 2*n_stocks - 1) // n_stocks

        if calculated_history_length > 0:
            print(f"Adjusting history_length from {history_length} to {calculated_history_length} to match model's observation space")
            history_length = calculated_history_length
        else:
            print(f"Warning: Could not calculate a valid history_length from model's observation space")
            print(f"Using default history_length={history_length}. This may cause errors.")

    # Create environment
    env = MultiTickerPortfolioEnv(
        test_data,
        history_length=history_length,
        initial_cash=initial_cash,
        risk_aversion=risk_aversion
    )

    results = {}

    # Test A2C model if provided
    if model:
        print(f"Testing A2C strategy...")
        a2c_results = backtest_strategy(
            env,
            a2c_strategy(model),
            initial_cash=initial_cash,
            name="A2C Agent",
            transaction_cost=transaction_cost
        )
        results['a2c'] = a2c_results

    # Reset environment for equal weight strategy
    env.reset(seed=CONFIG['random_seed'])
    print(f"Testing Equal Weight strategy...")
    ew_results = backtest_strategy(
        env,
        equal_weight_strategy,
        initial_cash=initial_cash,
        name="Equal Weight",
        transaction_cost=transaction_cost
    )
    results['equal_weight'] = ew_results

    # Reset environment for momentum strategy
    env.reset(seed=CONFIG['random_seed'])
    print(f"Testing Momentum strategy...")
    mom_results = backtest_strategy(
        env,
        momentum_strategy,
        initial_cash=initial_cash,
        name="Momentum",
        transaction_cost=transaction_cost
    )
    results['momentum'] = mom_results

    return results

# %% [markdown]
# ## 7. Visualization Functions

# %%
def plot_portfolio_comparison(results_dict):
    """
    Plot portfolio value comparison between strategies

    Args:
        results_dict: Dictionary of backtest results
    """
    plt.figure(figsize=(12, 8))

    # Format y-axis as currency
    def currency_formatter(x, pos):
        return f'${x:,.0f}'

    # Upper plot - Portfolio Value
    plt.subplot(2, 1, 1)

    for strategy, results in results_dict.items():
        plt.plot(results['data'].index, results['data']['portfolio_value'], label=results['name'])

    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.title('Portfolio Value Comparison', fontsize=14)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend()

    # Lower plot - Drawdowns
    plt.subplot(2, 1, 2)

    for strategy, results in results_dict.items():
        if 'drawdown' in results['data'].columns:
            plt.plot(results['data'].index, results['data']['drawdown'], label=f"{results['name']} Drawdown")
        else:
            # Calculate drawdown if not already available
            portfolio_values = results['data']['portfolio_value'].values
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            plt.plot(results['data'].index, drawdown, label=f"{results['name']} Drawdown")

    plt.title('Drawdown Comparison', fontsize=14)
    plt.ylabel('Drawdown', fontsize=12)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1%}'))
    plt.grid(True, alpha=0.3)
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"{CONFIG['results_path']}/portfolio_comparison.png", dpi=300)
    plt.show()

# %%
def plot_asset_allocation(results):
    """
    Plot asset allocation over time for a strategy

    Args:
        results: Results dictionary for a single strategy
    """
    plt.figure(figsize=(12, 6))

    # Find allocation columns
    allocation_cols = [col for col in results['data'].columns if col.startswith('allocation_')]

    if allocation_cols:
        # Create a copy of the allocation data
        allocation_data = results['data'][allocation_cols].copy()

        # Rename columns to ticker names
        allocation_data.columns = [col.replace('allocation_', '') for col in allocation_cols]

        # Add cash allocation
        cash_allocation = results['data']['cash'] / results['data']['portfolio_value']
        allocation_data['Cash'] = cash_allocation

        # Plot stacked area chart
        allocation_data.plot.area(figsize=(12, 6), alpha=0.7)
        plt.title(f'Asset Allocation Over Time - {results["name"]}', fontsize=14)
        plt.ylabel('Allocation (%)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

        plt.tight_layout()
        plt.savefig(f"{CONFIG['results_path']}/allocation_{results['name'].replace(' ', '_').lower()}.png", dpi=300)
        plt.show()
    else:
        print("No allocation data available for plotting")

# %%
def plot_reward_components(results):
    """
    Plot reward components for a strategy

    Args:
        results: Results dictionary for a single strategy
    """
    # Check if reward components are available
    component_cols = ['sharpe_component', 'consistency_bonus',
                     'diversification_score', 'drawdown_penalty']

    if all(col in results['data'].columns for col in component_cols):
        plt.figure(figsize=(12, 8))

        # Plot reward components
        plt.subplot(2, 1, 1)
        for col in component_cols:
            # Plot drawdown penalty as negative
            if col == 'drawdown_penalty':
                plt.plot(results['data'].index, -results['data'][col],
                         label=col.replace('_', ' ').title() + ' (negative)')
            else:
                plt.plot(results['data'].index, results['data'][col],
                         label=col.replace('_', ' ').title())

        plt.title(f'Reward Components - {results["name"]}', fontsize=14)
        plt.ylabel('Component Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend()

        # Plot portfolio value alongside
        plt.subplot(2, 1, 2)
        plt.plot(results['data'].index, results['data']['portfolio_value'], color='green')
        plt.title('Portfolio Value', fontsize=14)
        plt.ylabel('Value ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))

        plt.tight_layout()
        plt.savefig(f"{CONFIG['results_path']}/reward_components_{results['name'].replace(' ', '_').lower()}.png", dpi=300)
        plt.show()
    else:
        print("Reward component data not available")

# %%
def plot_transaction_costs(results):
    """
    Plot transaction costs analysis

    Args:
        results: Results dictionary for a single strategy
    """
    if 'transaction_costs' in results['data'].columns:
        plt.figure(figsize=(12, 8))

        # Transaction costs per period
        plt.subplot(2, 1, 1)
        plt.plot(results['data'].index, results['data']['transaction_costs'], color='red')
        plt.title(f'Transaction Costs Per Period - {results["name"]}', fontsize=14)
        plt.ylabel('Cost ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:.2f}'))

        # Cumulative transaction costs
        plt.subplot(2, 1, 2)
        cumulative = results['data']['transaction_costs'].cumsum()
        plt.plot(results['data'].index, cumulative, color='darkred')
        plt.title('Cumulative Transaction Costs', fontsize=14)
        plt.ylabel('Cost ($)', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:.2f}'))

        plt.tight_layout()
        plt.savefig(f"{CONFIG['results_path']}/transaction_costs_{results['name'].replace(' ', '_').lower()}.png", dpi=300)
        plt.show()

        # Print transaction cost summary
        total_costs = results['transaction_costs']['total']
        initial_cash = CONFIG['initial_cash']
        print(f"\nTransaction Cost Summary ({results['name']}):")
        print(f"Total transaction costs: ${total_costs:.2f}")
        print(f"As percentage of initial investment: {total_costs/initial_cash*100:.2f}%")
        print(f"Average cost per period: ${results['transaction_costs']['average']:.2f}")
        print(f"Transaction cost rate: {results['transaction_costs']['percentage']:.3f}%")

        # Impact on returns (without vs with costs)
        gross_final_value = results['final_value'] + total_costs
        gross_return = (gross_final_value / initial_cash) - 1
        net_return = results['total_return']
        impact = gross_return - net_return
        print("\nImpact on Returns:")
        print(f"Gross return (without costs): {gross_return:.2%}")
        print(f"Net return (with costs): {net_return:.2%}")
        print(f"Transaction cost impact: {impact:.2%}")
    else:
        print("Transaction cost data not available")

# %%
def display_performance_metrics(results_dict):
    """
    Display performance metrics table for all strategies

    Args:
        results_dict: Dictionary of backtest results

    Returns:
        DataFrame with performance metrics
    """
    # Define metrics to display
    metrics = [
        ('Total Return', lambda r: f"{r['total_return']:.2%}"),
        ('Annualized Return', lambda r: f"{r['annualized_return']:.2%}"),
        ('Volatility', lambda r: f"{r['volatility']:.2%}"),
        ('Max Drawdown', lambda r: f"{r['max_drawdown']:.2%}"),
        ('Sharpe Ratio', lambda r: f"{r['sharpe_ratio']:.2f}"),
        ('Sortino Ratio', lambda r: f"{r['sortino_ratio']:.2f}"),
        ('Final Value', lambda r: f"${r['final_value']:.2f}"),
        ('Transaction Costs', lambda r: f"${r['transaction_costs']['total']:.2f}"),
        ('Cost % of Initial', lambda r: f"{r['transaction_costs']['total']/CONFIG['initial_cash']*100:.2f}%"),
    ]

    # Create DataFrame
    data = {}
    for strategy, results in results_dict.items():
        data[results['name']] = [fn(results) for _, fn in metrics]

    df = pd.DataFrame(data, index=[name for name, _ in metrics])

    # Find best strategy
    best_strategy = max(results_dict.values(), key=lambda x: x['total_return'])

    print("\n=== Performance Metrics ===")
    print(df)
    print(f"\nBest strategy: {best_strategy['name']} with return of {best_strategy['total_return']:.2%}")

    # Save to CSV
    df.to_csv(f"{CONFIG['results_path']}/performance_metrics.csv")

    return df

# %% [markdown]
# ## 8. Main Execution

# %%
def run_evaluation_pipeline(
    data_path=CONFIG['data_path'],
    model_path=CONFIG['model_path'],
    test_size=CONFIG['test_size'],
    split_date=None,
    transaction_cost=CONFIG['transaction_cost']
):
    """
    Run the complete evaluation pipeline

    Args:
        data_path: Path to the data file
        model_path: Path to the model file
        test_size: Proportion of data for testing
        split_date: Specific date to split on ('YYYY-MM-DD')
        transaction_cost: Transaction cost rate

    Returns:
        Dictionary with all backtest results
    """
    print("=" * 50)
    print("PORTFOLIO OPTIMIZATION STRATEGY EVALUATION")
    print("=" * 50)

    # Step 1: Load data
    print("\n1. Loading data...")
    data = load_data(data_path)
    if data is None:
        print("Failed to load data. Aborting.")
        return None

    # Print data statistics
    print(f"Data statistics:")
    print(f"  - Number of records: {len(data)}")
    print(f"  - Date range: {data['date'].min()} to {data['date'].max()}")
    print(f"  - Tickers: {sorted(data['tic'].unique())}")
    print(f"  - Number of tickers: {len(data['tic'].unique())}")

    # Step 2: Split into train/test
    print("\n2. Splitting data into train/test sets...")
    train_data, test_data = train_test_split(data, test_size=test_size, split_date=split_date)
    if test_data is None:
        print("Failed to split data. Aborting.")
        return None

    # Step 3: Load the model
    print("\n3. Loading A2C model...")
    model = load_model(model_path)

    if model is None:
        print(f"Failed to load model from {model_path}. Aborting.")
        return None

    # Print model information
    print("Model information:")
    print(f"  - Observation space: {model.observation_space}")
    print(f"  - Action space: {model.action_space}")

    try:
        # Validate model by making a simple prediction
        dummy_env = MultiTickerPortfolioEnv(test_data, history_length=1)  # Temporary env just for observation space
        dummy_obs, _ = dummy_env.reset()

        # Calculate expected observation space size for this env
        n_stocks = len(test_data['tic'].unique())
        expected_obs_size = model.observation_space.shape[0]

        # Calculate history_length that will give the correct observation space
        required_history_length = (expected_obs_size - 2*n_stocks - 1) // n_stocks

        print(f"  - Required history_length: {required_history_length}")
        print(f"  - Number of stocks: {n_stocks}")

        # Verify that we can create a correctly sized observation space
        history_length_validation = (required_history_length * n_stocks + 2*n_stocks + 1)
        print(f"  - Calculated observation space size: {history_length_validation} (should match {expected_obs_size})")

    except Exception as e:
        print(f"Warning: Model validation step failed: {e}")
        print("Will attempt to proceed with backtesting anyway.")

    # Step 4: Run backtests
    print("\n4. Running backtests...")
    backtest_results = run_backtests(
        test_data=test_data,
        model=model,
        transaction_cost=transaction_cost
    )

    if not backtest_results:
        print("Failed to run backtests. Aborting.")
        return None

    # Step 5: Visualize results
    print("\n5. Generating visualizations...")

    # Overall portfolio comparison
    print("\nPlotting portfolio value comparison...")
    plot_portfolio_comparison(backtest_results)

    # Performance metrics table
    print("\nDisplaying performance metrics...")
    display_performance_metrics(backtest_results)

    # Detailed analysis for A2C strategy
    if 'a2c' in backtest_results:
        print("\nAnalyzing A2C strategy...")
        print("\nPlotting asset allocation for A2C...")
        plot_asset_allocation(backtest_results['a2c'])

        print("\nPlotting reward components for A2C...")
        plot_reward_components(backtest_results['a2c'])

        print("\nAnalyzing transaction costs for A2C...")
        plot_transaction_costs(backtest_results['a2c'])

    print("\n6. Evaluation complete!")
    print(f"Results saved to {CONFIG['results_path']}")

    return backtest_results

# %%
# Run the entire pipeline if executed directly
if __name__ == "__main__":
    results = run_evaluation_pipeline()


# %% [markdown]
# ## 9. Data Quality Analysis

# %%
def analyze_data_consistency(df, price_col='close'):
    """
    Analyze price data for consistency issues like gaps, missing values, and extreme changes.

    Args:
        df: DataFrame with price data
        price_col: Name of the price column

    Returns:
        DataFrame with analysis results
    """
    print("=== Data Consistency Analysis ===")

    # Make a copy and ensure date is datetime
    df_copy = df.copy()
    if 'date' in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy['date'])

    # Group by ticker
    tickers = df_copy['tic'].unique()
    results = {}

    for ticker in tickers:
        ticker_data = df_copy[df_copy['tic'] == ticker].sort_values('date')

        if len(ticker_data) < 2:
            print(f"Warning: Insufficient data for {ticker}")
            continue

        # Check for missing values
        missing_count = ticker_data[price_col].isna().sum()
        missing_pct = missing_count / len(ticker_data) * 100

        # Check for zeros or negative prices
        zeros = (ticker_data[price_col] <= 0).sum()

        # Calculate returns and check for extreme price movements
        ticker_data['return'] = ticker_data[price_col].pct_change()
        mean_return = ticker_data['return'].mean()
        std_return = ticker_data['return'].std()

        # Define extreme as > 3 standard deviations
        extreme_threshold = 3 * std_return
        extreme_changes = (ticker_data['return'].abs() > extreme_threshold).sum()
        extreme_pct = extreme_changes / len(ticker_data) * 100

        # Analyze time gaps
        ticker_data['time_diff'] = ticker_data['date'].diff()
        time_diffs_seconds = ticker_data['time_diff'].dt.total_seconds()
        expected_interval = time_diffs_seconds.median()
        irregular_intervals = (time_diffs_seconds > expected_interval * 1.5).sum()

        results[ticker] = {
            'count': len(ticker_data),
            'missing_values': missing_count,
            'missing_pct': missing_pct,
            'zeros_or_negative': zeros,
            'mean_return': mean_return,
            'std_return': std_return,
            'extreme_changes': extreme_changes,
            'extreme_pct': extreme_pct,
            'irregular_intervals': irregular_intervals,
            'expected_interval_hours': expected_interval / 3600,  # Convert to hours
            'price_range': f"{ticker_data[price_col].min():.2f} - {ticker_data[price_col].max():.2f}"
        }

    # Convert to DataFrame for easy viewing
    results_df = pd.DataFrame(results).T
    print(results_df[['count', 'missing_pct', 'zeros_or_negative', 'extreme_pct', 'irregular_intervals']])

    # Overall warnings
    print("\nPotential Issues:")
    for ticker, data in results.items():
        warnings = []
        if data['missing_pct'] > 0:
            warnings.append(f"Missing values: {data['missing_pct']:.1f}%")
        if data['zeros_or_negative'] > 0:
            warnings.append(f"Zeros or negative prices: {data['zeros_or_negative']}")
        if data['extreme_pct'] > 1.0:  # More than 1% extreme changes
            warnings.append(f"Extreme price changes: {data['extreme_pct']:.1f}%")
        if data['irregular_intervals'] > len(df_copy[df_copy['tic'] == ticker]) * 0.01:
            warnings.append(f"Irregular time intervals: {data['irregular_intervals']}")

        if warnings:
            print(f"- {ticker}: {', '.join(warnings)}")

    return results_df

# %%
def plot_price_consistency(df, tickers=None, price_col='close'):
    """
    Plot price data to visually inspect for consistency issues.

    Args:
        df: DataFrame with price data
        tickers: List of tickers to plot (None for all)
        price_col: Name of the price column
    """
    df_copy = df.copy()
    if 'date' in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy['date'])

    # Select tickers to plot
    if tickers is None:
        tickers = df_copy['tic'].unique()
        if len(tickers) > 6:  # Limit to 6 tickers for readability
            print(f"Limiting plot to first 6 of {len(tickers)} tickers")
            tickers = tickers[:6]

    # Create plots: prices, returns and volume if available
    fig, axes = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

    # Plot prices
    for ticker in tickers:
        ticker_data = df_copy[df_copy['tic'] == ticker].sort_values('date')
        axes[0].plot(ticker_data['date'], ticker_data[price_col], label=ticker)

    axes[0].set_title('Price Data', fontsize=14)
    axes[0].set_ylabel(f'{price_col} Price', fontsize=12)
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Plot returns
    for ticker in tickers:
        ticker_data = df_copy[df_copy['tic'] == ticker].sort_values('date')
        returns = ticker_data[price_col].pct_change()
        axes[1].plot(ticker_data['date'], returns, label=ticker, alpha=0.7)

    axes[1].set_title('Returns', fontsize=14)
    axes[1].set_ylabel('Return (%)', fontsize=12)
    axes[1].set_xlabel('Date', fontsize=12)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(f"{CONFIG['results_path']}/price_consistency.png", dpi=300)
    plt.show()

# %%
def analyze_transaction_frequency(results):
    """
    Analyze transaction frequency and size patterns in backtest results.

    Args:
        results: Results dictionary for a strategy
    """
    data = results['data']

    # Calculate allocation changes between periods
    allocation_cols = [col for col in data.columns if col.startswith('allocation_')]

    if allocation_cols:
        plt.figure(figsize=(14, 8))

        # Plot transaction costs
        plt.subplot(2, 1, 1)
        plt.plot(data.index, data['transaction_costs'], color='red')
        plt.title('Transaction Costs Over Time', fontsize=14)
        plt.ylabel('Cost ($)', fontsize=12)
        plt.grid(True, alpha=0.3)

        # Calculate and plot allocation changes
        changes = pd.DataFrame(index=data.index)
        changes['total_change'] = 0

        for i in range(1, len(data)):
            change_sum = 0
            for col in allocation_cols:
                # Skip NaN values
                if pd.notna(data[col].iloc[i]) and pd.notna(data[col].iloc[i-1]):
                    change_sum += abs(data[col].iloc[i] - data[col].iloc[i-1])
            changes.loc[data.index[i], 'total_change'] = change_sum

        plt.subplot(2, 1, 2)
        plt.plot(changes.index, changes['total_change'], color='blue')
        plt.title('Total Allocation Changes Between Periods', fontsize=14)
        plt.ylabel('Sum of Absolute Changes', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(f"{CONFIG['results_path']}/transaction_frequency.png", dpi=300)
        plt.show()

        # Analyze correlation between price movements and transactions
        print("\n=== Transaction Analysis ===")
        avg_cost = data['transaction_costs'].mean()
        max_cost = data['transaction_costs'].max()

        print(f"Average transaction cost: ${avg_cost:.2f}")
        print(f"Maximum transaction cost: ${max_cost:.2f}")
        print(f"Transaction cost > 2x average: {(data['transaction_costs'] > 2*avg_cost).sum()} instances")

        # Calculate trading activity by time period
        if len(data) > 20:
            data['month'] = data.index.to_period('M')
            monthly_costs = data.groupby('month')['transaction_costs'].sum()

            print("\nMonthly Trading Activity:")
            print(monthly_costs)

        return changes
    else:
        print("No allocation data available for transaction analysis")
        return None

# %%
def check_data_sampling(df):
    """
    Check the sampling consistency of the data.

    Args:
        df: DataFrame with price data containing date column
    """
    df_copy = df.copy()
    if 'date' in df_copy.columns:
        df_copy['date'] = pd.to_datetime(df_copy['date'])

    # Group by ticker and calculate time differences
    tickers = df_copy['tic'].unique()

    print("=== Data Sampling Analysis ===")
    print(f"Total unique dates: {df_copy['date'].nunique()}")

    all_diffs = []
    sampling_stats = {}

    for ticker in tickers:
        ticker_data = df_copy[df_copy['tic'] == ticker].sort_values('date')
        time_diffs = ticker_data['date'].diff().dropna()

        # Convert to hours for readability
        time_diffs_hours = time_diffs.dt.total_seconds() / 3600

        # Calculate statistics
        if len(time_diffs_hours) > 0:
            min_diff = time_diffs_hours.min()
            max_diff = time_diffs_hours.max()
            mean_diff = time_diffs_hours.mean()
            median_diff = time_diffs_hours.median()

            # Count unique time differences (should be consistent for regular sampling)
            unique_diffs = time_diffs_hours.unique()

            sampling_stats[ticker] = {
                'min_hours': min_diff,
                'max_hours': max_diff,
                'mean_hours': mean_diff,
                'median_hours': median_diff,
                'unique_intervals': len(unique_diffs),
                'sample_count': len(ticker_data)
            }

            all_diffs.extend(time_diffs_hours.tolist())

    # Create summary DataFrame
    stats_df = pd.DataFrame(sampling_stats).T
    print(stats_df)

    # Plot histogram of all time differences
    plt.figure(figsize=(12, 6))
    plt.hist(all_diffs, bins=50, alpha=0.7)
    plt.title('Distribution of Time Intervals Between Data Points', fontsize=14)
    plt.xlabel('Hours', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.savefig(f"{CONFIG['results_path']}/sampling_intervals.png", dpi=300)
    plt.show()

    return stats_df

# %%
def run_data_quality_check(data_path=CONFIG['data_path']):
    """
    Run a comprehensive data quality check on input data.

    Args:
        data_path: Path to data file
    """
    print("=" * 50)
    print("DATA QUALITY ANALYSIS")
    print("=" * 50)

    # Load data
    data = load_data(data_path)
    if data is None:
        print("Failed to load data for analysis.")
        return

    # Check data sampling consistency
    sampling_stats = check_data_sampling(data)

    # Run consistency analysis
    consistency_results = analyze_data_consistency(data, price_col='close')

    # Plot price data
    plot_price_consistency(data, price_col='close')

    # Check time series continuity
    date_counts = data.groupby('date')['tic'].count()

    # Check if all tickers have data for all dates
    expected_count = len(data['tic'].unique())
    missing_dates = date_counts[date_counts < expected_count]

    if len(missing_dates) > 0:
        print("\nDates with incomplete ticker data:")
        print(missing_dates)

        # Check which tickers are missing for a sample date
        if len(missing_dates) > 0:
            sample_date = missing_dates.index[0]
            print(f"\nFor date {sample_date}, missing tickers:")
            available_tickers = data[data['date'] == sample_date]['tic'].unique()
            all_tickers = data['tic'].unique()
            missing_tickers = set(all_tickers) - set(available_tickers)
            print(missing_tickers)
    else:
        print("\nAll dates have data for all tickers.")

    return consistency_results

# %%
# 1. Check your input data quality
data_quality = run_data_quality_check()

# 2. After running your backtest, analyze the transaction patterns
# Assuming 'results' contains your backtest results
analyze_transaction_frequency(results['a2c'])  # For A2C agent
