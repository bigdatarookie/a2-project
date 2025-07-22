# %% [markdown]
# ## 1. Custom Environment Definition

# %%
# # Import libraries
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Box, Discrete
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta

# %%
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
        df,
        history_length=10,
        initial_cash=10000,
        risk_aversion=2.0,
        price_column='close',
        date_column='date',
        ticker_column='tic',
        feature_columns=None,
        log_to_mlflow=False
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
            log_to_mlflow (bool): Whether to log metrics to MLflow during training

        Reward Function Components:
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
        self.log_to_mlflow = log_to_mlflow

        # Preprocess the data
        self.raw_df = df.copy()

        # Get unique tickers and dates in the dataset
        self.tickers = sorted(df[ticker_column].unique())
        self.n_stocks = len(self.tickers)
        print(f"Environment created with {self.n_stocks} tickers: {self.tickers}")

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

        # Convert to numpy array for faster access during training
        self.prices = self.price_data.values

        # Process additional features if provided
        self.feature_data = {}
        if self.feature_columns:
            for feature in self.feature_columns:
                feature_pivot = df.pivot(index=date_column, columns=ticker_column, values=feature)
                feature_pivot = feature_pivot.reindex(self.price_data.index)
                self.feature_data[feature] = feature_pivot.ffill().bfill().values

        # Store dates in a standardized format that preserves minute-level precision
        self.price_data.index = self.price_data.index.strftime('%Y-%m-%d %H:%M:%S')
        self.dates = self.price_data.index.values
        self.n_steps = len(self.dates)

        # Print sample dates to verify format
        print(f"Sample dates in environment: {self.dates[:3]} (preserved down to minute resolution)")

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
        feature_dim = len(self.feature_columns) * self.n_stocks * history_length if self.feature_columns else 0

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

        # MLflow logging setup
        self.mlflow_metrics = {}
        if self.log_to_mlflow:
            try:
                import mlflow
                self.mlflow = mlflow
            except ImportError:
                print("Warning: mlflow not installed. Disabling MLflow logging.")
                self.log_to_mlflow = False

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

        # Reset MLflow metrics
        self.mlflow_metrics = {}

        # Get initial observation
        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Execute trades based on action (continuous allocation targets)
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

        # Calculate immediate return (for info only, not used in reward)
        prev_value = self.portfolio_value_history[-2]
        immediate_return = (current_portfolio_value / prev_value) - 1 if prev_value > 0 else 0

        # Initialize reward components
        sharpe_component = 0.0
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

        # 2. Diversification Incentive
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

        # 3. Non-linear Drawdown Penalty
        # Penalize drawdowns more severely as they increase (squared term)
        drawdown_penalty = self.risk_aversion * (drawdown**2) * 20.0

        # Combine all components with updated weights (removed short-term and consistency components)
        reward = (
            sharpe_component * 0.65 +              # Risk-adjusted return (65% weight)
            diversification_score * 0.15 -         # Diversification incentive (15% weight)
            drawdown_penalty * 0.20                # Drawdown penalty (20% weight)
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
        info['diversification_score'] = diversification_score
        info['drawdown_penalty'] = drawdown_penalty
        info['allocations'] = allocations

        # Log metrics to MLflow if enabled
        if self.log_to_mlflow and terminated:
            self._log_mlflow_metrics()

        return observation, float(reward), terminated, truncated, info

    def _get_observation(self):
        """
        Get observation that includes price history, features, momentum indicators and portfolio state.
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
        last_prices = price_history[-1, :]
        normalized_history = np.zeros_like(price_history)
        for i in range(self.n_stocks):
            if last_prices[i] > 0:
                normalized_history[:, i] = price_history[:, i] / last_prices[i]
            else:
                normalized_history[:, i] = price_history[:, i]

        # Process additional features if available
        feature_history_flat = np.array([])
        if self.feature_columns:
            feature_history_list = []
            for feature in self.feature_columns:
                feature_history = self.feature_data[feature][start_idx:end_idx, :]

                # Pad if not enough history
                if len(feature_history) < self.history_length:
                    padding_shape = (self.history_length - len(feature_history), self.n_stocks)
                    padding = np.ones(padding_shape) * feature_history[0, :]
                    feature_history = np.vstack([padding, feature_history])

                # Normalize feature history (if possible)
                normalized_feature = np.zeros_like(feature_history)
                for i in range(self.n_stocks):
                    if np.abs(feature_history[-1, i]) > 1e-6:
                        normalized_feature[:, i] = feature_history[:, i] / feature_history[-1, i]
                    else:
                        normalized_feature[:, i] = feature_history[:, i]

                feature_history_list.append(normalized_feature.flatten())

            if feature_history_list:
                feature_history_flat = np.concatenate(feature_history_list)

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
        Action values are assumed to be in [0, 1] range representing desired
        allocation percentage for each asset.
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

        # Execute trades (sells first to ensure liquidity)
        # Sell orders - stocks where target allocation is lower than current
        for i, diff in enumerate(value_difference):
            if diff < 0 and current_prices[i] > 0:
                # Convert value to shares
                shares_to_sell = abs(diff) / current_prices[i]
                shares_to_sell = min(shares_to_sell, self.shares[i])  # Can't sell more than we have

                # Execute sell
                self.cash += shares_to_sell * current_prices[i]
                self.shares[i] -= shares_to_sell

        # Buy orders - stocks where target allocation is higher than current
        for i, diff in enumerate(value_difference):
            if diff > 0 and current_prices[i] > 0:
                # Calculate maximum shares we can buy with available cash
                max_shares = self.cash / current_prices[i]
                shares_to_buy = min(diff / current_prices[i], max_shares)

                # Execute buy
                self.cash -= shares_to_buy * current_prices[i]
                self.shares[i] += shares_to_buy

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

    def _log_mlflow_metrics(self):
        """Log episode metrics to MLflow if enabled"""
        if not self.log_to_mlflow:
            return

        try:
            # Calculate final metrics
            portfolio_values = np.array(self.portfolio_value_history)
            returns = np.diff(portfolio_values) / portfolio_values[:-1]

            # Log relevant metrics
            self.mlflow.log_metric("final_portfolio_value", self._get_portfolio_value())
            self.mlflow.log_metric("total_return", (self._get_portfolio_value() / self.initial_cash) - 1)
            self.mlflow.log_metric("sharpe_ratio", np.mean(returns) / (np.std(returns) + 1e-6) * np.sqrt(252))  # Annualized

            # Max drawdown
            peak = np.maximum.accumulate(portfolio_values)
            drawdown = (peak - portfolio_values) / peak
            max_drawdown = np.max(drawdown)
            self.mlflow.log_metric("max_drawdown", max_drawdown)

            # Log custom metrics that were tracked during the episode
            for metric_name, metric_value in self.mlflow_metrics.items():
                self.mlflow.log_metric(metric_name, metric_value)

        except Exception as e:
            print(f"Error logging to MLflow: {e}")

# %% [markdown]
# ## 2. Environment Testing

# %%
# Test the environment with some real data
def test_environment(daily_data, minute_data, feature_list=None):
    """
    Test the MultiTickerPortfolioEnv with real market data

    Args:
        daily_data (pd.DataFrame): DataFrame containing daily price data
        minute_data (pd.DataFrame): DataFrame containing minute price data
        feature_list (list): List of feature columns to include in environment
    """
    # Default feature list if none provided
    if feature_list is None:
        feature_list = ['volume']

    # Create a DataFrame in the format expected by the environment
    # Handle different column name formats depending on input data format
    if 'Close' in daily_data.columns:
        df = daily_data['Close'].reset_index()
        df = df.melt(id_vars=['Date'], var_name='tic', value_name='close')
        df = df.rename(columns={'Date': 'date'})

        # Add additional features if they exist in the data
        for feature in feature_list:
            if feature.title() in daily_data.columns:
                feature_data = daily_data[feature.title()].reset_index()
                feature_data = feature_data.melt(id_vars=['Date'], var_name='tic', value_name=feature.lower())
                df = pd.merge(df, feature_data[['Date', 'tic', feature.lower()]],
                             left_on=['date', 'tic'], right_on=['Date', 'tic'])
                df = df.drop(columns=['Date'])
    else:
        # Assume data is already in the expected format with lowercase column names
        df = daily_data

        # Ensure required columns exist
        required_cols = ['date', 'tic', 'close']
        missing_cols = [col for col in required_cols if col not in df.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")

        # Ensure required feature columns exist
        missing_features = [f for f in feature_list if f not in df.columns]
        if missing_features:
            print(f"Warning: Some requested features are missing: {missing_features}")
            feature_list = [f for f in feature_list if f in df.columns]

    print(f"Test data shape: {df.shape}")
    print(f"Sample of test data:\n{df.head()}")
    print(f"Using features: {feature_list}")

    # Create the environment with default settings
    env = MultiTickerPortfolioEnv(
        df=df,
        history_length=5,
        initial_cash=10000,
        feature_columns=feature_list,  # Use the provided feature list
        price_column='close',  # Use lowercase column names
        date_column='date',
        ticker_column='tic'
    )

    # Reset the environment to initialize
    obs, info = env.reset()

    print(f"\nInitial observation shape: {obs.shape}")
    print(f"Initial info: {info}")

    # Test continuous actions directly controlling position sizing
    # Run a simple test episode
    print("\nRunning test episode...")

    # Number of steps to simulate
    num_steps = 20
    portfolio_values = [info['portfolio_value']]
    rewards = []
    actions_history = []

    for i in range(num_steps):
        # Generate some test actions (normalized allocation targets)
        # Alternate between different allocation strategies to test action impact
        if i % 3 == 0:
            # Equal weight strategy
            action = np.ones(env.n_stocks) / env.n_stocks
        elif i % 3 == 1:
            # Concentrate in first two stocks
            action = np.zeros(env.n_stocks)
            action[0:2] = 0.5
        else:
            # Random allocations
            action = np.random.rand(env.n_stocks)
            action = action / np.sum(action)

        # Step the environment
        obs, reward, terminated, truncated, info = env.step(action)

        # Store results
        portfolio_values.append(info['portfolio_value'])
        rewards.append(reward)
        actions_history.append(action)

        print(f"Step {i+1}: Value=${info['portfolio_value']:.2f}, Reward={reward:.4f}, Date={info['date']}")

        if terminated or truncated:
            break

    # Plot results
    plt.figure(figsize=(10, 6))
    plt.subplot(2, 1, 1)
    plt.plot(portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Step')
    plt.ylabel('Value ($)')
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(rewards)
    plt.title('Reward Over Time')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.grid(True)
    plt.tight_layout()

    # Test minute-level data handling
    print("\nTesting minute-level data handling...")
    try:
        # Check if minute data is already in expected format
        if isinstance(minute_data, pd.DataFrame):
            if 'Close' in minute_data.columns:
                df_minute = minute_data['Close'].reset_index()
                # Handle 'Datetime' or 'Date' column depending on format
                date_col = 'Datetime' if 'Datetime' in df_minute.columns else 'Date'
                df_minute = df_minute.melt(id_vars=[date_col], var_name='tic', value_name='close')
                df_minute = df_minute.rename(columns={date_col: 'date'})
            else:
                # Assume data is already in expected format
                df_minute = minute_data

            # Create environment with minute data
            env_minute = MultiTickerPortfolioEnv(
                df=df_minute,
                history_length=10,
                initial_cash=10000,
                price_column='close',  # Use lowercase column names
                date_column='date',
                ticker_column='tic'
            )

            obs_minute, info_minute = env_minute.reset()
            print(f"Minute-level environment created successfully")
            print(f"First few dates: {env_minute.dates[0:5]}")
        else:
            print("Minute data not provided as DataFrame")
    except Exception as e:
        print(f"Couldn't test minute data: {e}")

    # Test MLflow integration
    print("\nTesting MLflow integration...")
    try:
        import mlflow
        # Create environment with MLflow logging enabled
        env_mlflow = MultiTickerPortfolioEnv(
            df=df,
            history_length=5,
            initial_cash=10000,
            log_to_mlflow=True,
            price_column='close',
            date_column='date',
            ticker_column='tic',
            feature_columns=feature_list
        )
        print("MLflow integration tested successfully")
    except ImportError:
        print("MLflow not installed, skipping MLflow test")

    print("\nTest complete!")
    return env

# Run the test when executed directly
if __name__ == "__main__":
    # Define features to include
    features = ['volume', 'open', 'high', 'low', 'obv', 'rsi', 'atr']

    # Data importing
    processed_data = pd.read_csv('data/rts_1d_2023_2024_indicators.csv')
    processed_minute_data = pd.read_csv('data/rts_1min_2024_dec_indicators.csv')

    # Test the environment
    test_environment(processed_data, processed_minute_data, features)
