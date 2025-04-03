# %% [markdown]
# # RL Portfolio Optimization Environment and Training
#
# This notebook contains the environment definition and agent training with Optuna hyperparameter optimization and MLflow tracking.

# %%
# Import libraries
import gymnasium as gym
from gymnasium import Env
from gymnasium.spaces import Discrete, Box
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
from torch import nn
import optuna
import mlflow
import os

# %% [markdown]
# ## 1. Custom Environment Definition

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
        for i in range(self.n_stocks):
            # Get recent price history for this stock
            history_window = min(20, self.current_step)  # Use up to 20 periods
            if history_window > 1:
                stock_prices = self.prices[self.current_step - history_window:self.current_step, i]
                # Calculate log returns
                returns = np.diff(np.log(stock_prices))
                # Calculate volatility (standard deviation of returns)
                volatility = np.std(returns) if len(returns) > 0 else 0.01
            else:
                volatility = 0.01  # Default volatility if not enough history

            volatilities.append(volatility)

        # Normalize volatilities for inverse weighting (lower volatility = higher allocation)
        total_vol = sum(volatilities) if sum(volatilities) > 0 else len(volatilities)
        inv_vol_weights = [(1 / vol) / (len(volatilities) / total_vol) for vol in volatilities]

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
# ## 2. Load and Prepare Data

# %%
# Load the data
def load_data(file_path='data/rts_daily_jan.csv'):
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}")
    print(df.info())
    return df

# %%
# Load data
df = load_data()
df.head()

# %% [markdown]
# ## 3. Hyperparameter Optimization with Optuna

# %%
def objective(trial):
    """Optuna objective function for hyperparameter optimization"""

    # Define hyperparameter search space
    history_length = trial.suggest_int("history_length", 5, 30)
    risk_aversion = trial.suggest_float("risk_aversion", 1.0, 5.0)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.001, 0.1, log=True)
    n_steps = trial.suggest_int("n_steps", 8, 32)

    # Log trial parameters to MLflow
    mlflow.log_params({
        "history_length": history_length,
        "risk_aversion": risk_aversion,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "n_steps": n_steps,
    })

    # Create environment
    env = MultiTickerPortfolioEnv(
        df,
        history_length=history_length,
        risk_aversion=risk_aversion,
        initial_cash=10000
    )

    # Create vectorized environment for parallel training
    num_cpu = os.cpu_count()
    try:
        vectorized_env = make_vec_env(
            env_id=lambda: MultiTickerPortfolioEnv(
                df,
                history_length=history_length,
                initial_cash=10000,
                risk_aversion=risk_aversion
            ),
            n_envs=num_cpu,
            seed=42
        )

        # Create eval environment
        eval_env = Monitor(MultiTickerPortfolioEnv(
            df,
            history_length=history_length,
            initial_cash=10000,
            risk_aversion=risk_aversion
        ))

        # Configure model
        model = A2C(
            "MlpPolicy",
            vectorized_env,
            learning_rate=learning_rate,
            ent_coef=ent_coef,
            n_steps=n_steps,
            gamma=0.99,
            verbose=0  # Set to 0 to reduce output during optimization
        )

        # Train for a shorter time during hyperparameter optimization
        model.learn(total_timesteps=50000)

        # Evaluate
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=5)

        # Log metrics to MLflow
        mlflow.log_metric("mean_reward", mean_reward)
        mlflow.log_metric("std_reward", std_reward)

        # Clean up
        vectorized_env.close()
        eval_env.close()

        return mean_reward

    except Exception as e:
        print(f"Trial failed with error: {e}")
        return float('-inf')  # Return worst possible score on failure

# %%
def run_optimization(n_trials=20):
    """Run hyperparameter optimization with Optuna and MLflow tracking"""

    # Set up MLflow experiment
    mlflow.set_experiment("a2c_portfolio_optimization")

    # Create Optuna study
    study = optuna.create_study(direction="maximize")

    # Run optimization with MLflow tracking
    for _ in range(n_trials):
        with mlflow.start_run():
            study.optimize(objective, n_trials=1)

    # Get best parameters
    best_params = study.best_params
    best_value = study.best_value

    print(f"Best trial value: {best_value}")
    print("Best parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")

    return best_params

# %% [markdown]
# ## 4. Train Final Model with Best Parameters

# %%
def train_final_model(params, total_timesteps=100000):
    """Train the final model with the best hyperparameters and log with MLflow"""

    # Start a new MLflow run for the final model
    with mlflow.start_run(run_name="final_model"):
        # Log parameters
        mlflow.log_params(params)

        # Create environment with best parameters
        num_cpu = min(12, os.cpu_count() or 1)
        vectorized_env = make_vec_env(
            env_id=lambda: MultiTickerPortfolioEnv(
                df,
                history_length=params.get("history_length", 24),
                initial_cash=10000,
                risk_aversion=params.get("risk_aversion", 3.0)
            ),
            n_envs=num_cpu,
            seed=42
        )

        # Create evaluation environment
        eval_env = Monitor(MultiTickerPortfolioEnv(
            df,
            history_length=params.get("history_length", 24),
            initial_cash=10000,
            risk_aversion=params.get("risk_aversion", 3.0)
        ))

        # Set up callback for saving best model
        os.makedirs("data/models", exist_ok=True)
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path="data/models/",
            log_path="data/logs/",
            eval_freq=2500,
            n_eval_episodes=5,
            deterministic=True,
            verbose=1
        )

        # Configure and train model
        model = A2C(
            "MlpPolicy",
            vectorized_env,
            learning_rate=params.get("learning_rate", 5e-4),
            ent_coef=params.get("ent_coef", 0.01),
            n_steps=params.get("n_steps", 24),
            gamma=0.99,
            verbose=1
        )

        # Train model
        model.learn(
            total_timesteps=total_timesteps,
            callback=eval_callback
        )

        # Evaluate final model
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
        print(f"Final model mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

        # Log final metrics
        mlflow.log_metric("final_mean_reward", mean_reward)
        mlflow.log_metric("final_std_reward", std_reward)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        # Save model locally
        model.save("data/models/final_model")

        # Clean up
        vectorized_env.close()
        eval_env.close()

        return model

# %% [markdown]
# ## 5. Run Complete Workflow

# %%
# Run hyperparameter optimization
if __name__ == "__main__":
    # Set random seed for reproducibility
    np.random.seed(42)

    print("Starting hyperparameter optimization...")
    best_params = run_optimization(n_trials=10)  # Adjust number of trials as needed

    print("\nTraining final model with best parameters...")
    final_model = train_final_model(best_params, total_timesteps=100000)

    print("Training completed.")
