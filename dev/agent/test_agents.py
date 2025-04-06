# %% [markdown]
# # This notebook is dedicated to testing custom agents

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import gymnasium as gym
from gymnasium.wrappers import RecordEpisodeStatistics
import time
from tqdm import tqdm
from datetime import datetime
import os
import random
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.utils import set_random_seed
from packages.agent.custom_envs import MultiTickerPortfolioEnv
from packages.agent.agents import A2CAgent

# %%
# Define a helper function to generate synthetic stock data for testing
def generate_test_data(n_tickers=5, n_days=250, seed=42):
    """Generate synthetic stock price data for testing."""
    np.random.seed(seed)
    random.seed(seed)

    # Generate dates (daily data)
    base_date = datetime(2021, 1, 1)
    dates = [base_date + pd.Timedelta(days=i) for i in range(n_days)]

    # Create ticker names
    tickers = [f"STOCK_{i}" for i in range(n_tickers)]

    # Generate stock prices with random walks
    data = []
    for ticker in tickers:
        # Initial price between 50 and 200
        price = np.random.uniform(50, 200)

        # Generate price series with random walk
        prices = [price]
        for _ in range(1, n_days):
            # Daily return between -2% and +2%
            daily_return = np.random.normal(0.0005, 0.015)  # Slightly positive drift
            price *= (1 + daily_return)
            prices.append(price)

        # Add to data
        for i, date in enumerate(dates):
            data.append({
                'date': date,
                'tic': ticker,
                'open': prices[i] * (1 - np.random.uniform(0, 0.005)),
                'high': prices[i] * (1 + np.random.uniform(0, 0.01)),
                'low': prices[i] * (1 - np.random.uniform(0, 0.01)),
                'close': prices[i],
                'volume': np.random.randint(100000, 10000000),
                # Add some technical indicators as features
                'ma10': np.mean(prices[max(0, i-10):i+1]) if i > 0 else prices[i],
                'ma30': np.mean(prices[max(0, i-30):i+1]) if i > 0 else prices[i],
                'rsi': np.random.uniform(30, 70)  # Simplified RSI
            })

    # Convert to DataFrame
    df = pd.DataFrame(data)

    print(f"Generated synthetic data with {n_tickers} tickers and {n_days} days")
    return df

# %%
# Create a logging callback for training
class TensorboardCallback(BaseCallback):
    """Custom callback for logging metrics during training."""

    def __init__(self, verbose=0):
        super(TensorboardCallback, self).__init__(verbose)
        self.rewards = []
        self.portfolio_values = []

    def _on_step(self):
        # Get info from the last step
        info = self.locals['infos'][0]
        self.portfolio_values.append(info['portfolio_value'])

        # Return True to continue training
        return True

# %%
# Define baseline strategies for comparison
def equal_weight_strategy(n_stocks):
    """Equal weight strategy - allocate equally across all stocks."""
    return np.ones(n_stocks) / n_stocks

def random_strategy(n_stocks):
    """Random allocation strategy."""
    action = np.random.random(n_stocks)
    return action / np.sum(action)  # Normalize

def concentrated_strategy(n_stocks):
    """Concentrated strategy - invest in first stock only."""
    action = np.zeros(n_stocks)
    action[0] = 1.0
    return action

def momentum_strategy(observation, env):
    """Simple momentum strategy - allocate based on recent performance."""
    # Extract portfolio allocations from observation
    # Note: This is a simplified interpretation based on environment structure
    n_stocks = env.n_stocks
    history_length = env.history_length

    # Calculate momentum based on price history in observation
    # This is an approximation - the exact structure depends on observation space design
    price_history_end = n_stocks * history_length
    recent_prices = observation[:price_history_end].reshape(history_length, n_stocks)

    # Calculate returns over the available history
    returns = recent_prices[-1] / recent_prices[0] - 1

    # Allocate to positive momentum stocks
    action = np.zeros(n_stocks)
    positive_returns = returns > 0

    if np.any(positive_returns):
        # Normalize positive returns to sum to 1
        positive_returns_values = returns[positive_returns]
        weights = positive_returns_values / np.sum(positive_returns_values)
        action[positive_returns] = weights
    else:
        # If no positive momentum, equal weight
        action = np.ones(n_stocks) / n_stocks

    return action

# %%
# Test A2CAgent initialization
def test_agent_initialization(df):
    """Test initialization of A2CAgent with different parameters."""
    print("\n==== Testing A2CAgent Initialization ====")

    # Basic environment parameters
    env_kwargs = {
        'df': df,
        'history_length': 10,
        'initial_cash': 10000
    }

    # Test 1: Basic initialization
    print("\nTest 1: Basic initialization")
    start_time = time.time()
    agent1 = A2CAgent(env_kwargs=env_kwargs)
    print(f"Initialize basic agent: {time.time() - start_time:.2f}s")

    # Test 2: With custom model parameters
    print("\nTest 2: Custom model parameters")
    model_kwargs = {
        'learning_rate': 0.001,
        'n_steps': 5,
        'gamma': 0.99,
        'ent_coef': 0.01
    }
    start_time = time.time()
    agent2 = A2CAgent(env_kwargs=env_kwargs, model_kwargs=model_kwargs)
    print(f"Initialize agent with custom parameters: {time.time() - start_time:.2f}s")

    # Test 3: With seed
    print("\nTest 3: With random seed")
    start_time = time.time()
    agent3 = A2CAgent(env_kwargs=env_kwargs, seed=42)
    print(f"Initialize agent with seed: {time.time() - start_time:.2f}s")

    return agent1, agent2, agent3

# %%

if __name__ == "__main__":
    # Generate test data
    df = generate_test_data(n_tickers=5, n_days=500)
    # 1. Test initialization
    agent, _, _ = test_agent_initialization(df)

# %%
# Test A2CAgent training
def test_agent_training(agent, total_timesteps=10000):
    """Test training of A2CAgent."""
    print("\n==== Testing A2CAgent Training ====")

    # Create callback for monitoring
    callback = TensorboardCallback()

    # Train the agent
    print(f"\nTraining agent for {total_timesteps} timesteps...")
    start_time = time.time()
    agent.train(
        total_timesteps=total_timesteps,
        callback=callback,
        log_interval=100
    )
    training_time = time.time() - start_time
    print(f"Training completed in {training_time:.2f} seconds")
    print(f"Training speed: {total_timesteps / training_time:.2f} steps/second")

    # Plot training progress
    plt.figure(figsize=(12, 5))
    plt.plot(callback.portfolio_values)
    plt.title('Portfolio Value During Training')
    plt.xlabel('Training Step')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.show()

    return agent, callback

# %%
# Test agent prediction and behavior
def test_agent_prediction(agent, df, n_steps=50):
    """Test prediction behavior of the trained agent."""
    print("\n==== Testing A2CAgent Prediction ====")

    # Create a test environment
    env = MultiTickerPortfolioEnv(
        df=df,
        history_length=10,
        initial_cash=10000
    )

    # Reset environment
    observation, info = env.reset(seed=123)

    # Lists to store data
    actions = []
    portfolio_values = []
    allocations = []
    dates = []

    # Run the agent for n_steps
    for _ in range(n_steps):
        # Get action from agent
        action = agent.predict(observation)
        actions.append(action)

        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Store info
        portfolio_values.append(info['portfolio_value'])
        allocations.append(info['allocations'] if 'allocations' in info else None)
        dates.append(info['date'])

        if terminated or truncated:
            break

    # Convert to arrays
    actions = np.array(actions)

    # Plot results
    plt.figure(figsize=(15, 10))

    # Portfolio value
    plt.subplot(2, 1, 1)
    plt.plot(dates, portfolio_values)
    plt.title('Portfolio Value')
    plt.xlabel('Date')
    plt.ylabel('Value ($)')
    plt.grid(True)

    # Actions/Allocations over time
    plt.subplot(2, 1, 2)
    # Plot each ticker's allocation as a stacked area
    for i in range(actions.shape[1]):
        plt.plot(dates, actions[:, i], label=f'Stock {i}')

    plt.title('Agent Actions Over Time')
    plt.xlabel('Date')
    plt.ylabel('Allocation')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return actions, portfolio_values, dates

# %%
# Test saving and loading
def test_save_load(agent, df, n_steps=10):
    """Test saving and loading the agent."""
    print("\n==== Testing A2CAgent Save/Load ====")

    # Create a directory for saved models
    os.makedirs("models", exist_ok=True)

    # Save the agent
    model_path = "models/test_a2c_agent"
    print(f"\nSaving agent to {model_path}")
    agent.save(model_path)

    # Create environment kwargs for loading
    env_kwargs = {
        'df': df,
        'history_length': 10,
        'initial_cash': 10000
    }

    # Load the agent
    print(f"Loading agent from {model_path}")
    loaded_agent = A2CAgent.load(
        path=model_path,
        env_kwargs=env_kwargs
    )

    # Compare predictions between original and loaded agents
    print("\nComparing predictions between original and loaded agents...")

    # Create a test environment
    env = MultiTickerPortfolioEnv(
        df=df,
        history_length=10,
        initial_cash=10000
    )

    # Reset environment
    observation, info = env.reset(seed=456)

    # Lists to store actions
    original_actions = []
    loaded_actions = []

    # Run both agents on the same environment states
    for _ in range(n_steps):
        # Get actions from both agents
        original_action = agent.predict(observation)
        loaded_action = loaded_agent.predict(observation)

        original_actions.append(original_action)
        loaded_actions.append(loaded_action)

        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(original_action)

        if terminated or truncated:
            break

    # Convert to arrays
    original_actions = np.array(original_actions)
    loaded_actions = np.array(loaded_actions)

    # Calculate differences
    action_diff = np.abs(original_actions - loaded_actions)
    mean_diff = action_diff.mean()
    max_diff = action_diff.max()

    print(f"Mean action difference: {mean_diff:.6f}")
    print(f"Max action difference: {max_diff:.6f}")

    return loaded_agent

# %%
# Compare agent against baseline strategies
def compare_strategies(agent, df, n_episodes=1):
    """Compare A2CAgent performance against baseline strategies."""
    print("\n==== Comparing A2CAgent Against Baselines ====")

    # Create environment
    env = MultiTickerPortfolioEnv(
        df=df,
        history_length=10,
        initial_cash=10000
    )

    # Define strategies
    strategies = {
        'A2C': lambda obs: agent.predict(obs),
        'Equal Weight': lambda obs: equal_weight_strategy(env.n_stocks),
        'Random': lambda obs: random_strategy(env.n_stocks),
        'Concentrated': lambda obs: concentrated_strategy(env.n_stocks),
        'Momentum': lambda obs: momentum_strategy(obs, env)
    }

    # Results dictionary
    results = {}

    # Test each strategy
    for name, strategy_fn in strategies.items():
        print(f"\nTesting strategy: {name}")

        # Run for multiple episodes and average
        strategy_values = []
        strategy_returns = []

        for episode in range(n_episodes):
            # Reset the environment
            observation, info = env.reset(seed=episode)

            # Run for a number of steps
            portfolio_values = [info['portfolio_value']]
            portfolio_returns = []

            while True:
                # Generate action based on strategy
                action = strategy_fn(observation)

                # Take a step
                observation, reward, terminated, truncated, info = env.step(action)

                # Record values
                portfolio_values.append(info['portfolio_value'])
                if len(portfolio_values) > 1:
                    ret = (portfolio_values[-1] / portfolio_values[-2]) - 1
                    portfolio_returns.append(ret)

                if terminated or truncated:
                    break

            strategy_values.append(portfolio_values)
            strategy_returns.append(portfolio_returns)

        # Average across episodes
        avg_values = np.mean([v for v in strategy_values], axis=0)

        # Calculate metrics
        final_value = avg_values[-1]
        total_return = (final_value / avg_values[0]) - 1

        # Store results
        results[name] = {
            'portfolio_values': avg_values,
            'total_return': total_return,
            'final_value': final_value
        }

        # Print results
        print(f"  Final portfolio value: ${final_value:.2f}")
        print(f"  Total return: {total_return:.2%}")

    # Plot comparison
    plt.figure(figsize=(12, 8))

    # Plot portfolio values
    plt.subplot(2, 1, 1)
    for name, result in results.items():
        plt.plot(result['portfolio_values'], label=name)

    plt.title('Portfolio Value Comparison')
    plt.xlabel('Time Step')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)

    # Plot bar chart of total returns
    plt.subplot(2, 1, 2)
    names = list(results.keys())
    returns = [result['total_return'] for result in results.values()]

    plt.bar(names, returns)
    plt.title('Total Return Comparison')
    plt.ylabel('Total Return')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    return results

# %%
# Test performance benchmarks
def test_performance(agent, df, n_tests=3):
    """Test agent performance (speed)."""
    print("\n==== Testing A2CAgent Performance ====")

    # Create environment
    env = MultiTickerPortfolioEnv(
        df=df,
        history_length=10,
        initial_cash=10000
    )

    # Reset environment
    observation, info = env.reset()

    # Warm-up
    for _ in range(10):
        action = agent.predict(observation)
        observation, _, _, _, _ = env.step(action)

    # Test prediction speed
    n_predictions = 1000

    print(f"\nTesting prediction speed ({n_predictions} predictions)...")

    # Run multiple tests
    times = []
    for test in range(n_tests):
        start_time = time.time()

        for _ in range(n_predictions):
            action = agent.predict(observation)
            observation, _, terminated, truncated, _ = env.step(action)

            if terminated or truncated:
                observation, _ = env.reset()

        elapsed = time.time() - start_time
        predictions_per_sec = n_predictions / elapsed
        times.append(elapsed)

        print(f"  Test {test+1}: {elapsed:.2f}s ({predictions_per_sec:.2f} predictions/second)")

    avg_time = np.mean(times)
    avg_predictions_per_sec = n_predictions / avg_time

    print(f"\nAverage: {avg_time:.2f}s ({avg_predictions_per_sec:.2f} predictions/second)")

    return avg_predictions_per_sec

# %%
# Main testing function
def run_all_tests():
    """Run all A2CAgent tests."""
    # Set random seed for reproducibility
    set_random_seed(42)

    # Generate test data
    df = generate_test_data(n_tickers=5, n_days=500)

    # 1. Test initialization
    agent, _, _ = test_agent_initialization(df)

    # 2. Test training
    agent, _ = test_agent_training(agent, total_timesteps=5000)

    # 3. Test prediction
    test_agent_prediction(agent, df)

    # 4. Test save/load
    loaded_agent = test_save_load(agent, df)

    # 5. Compare against baselines
    compare_strategies(agent, df)

    # 6. Test performance
    test_performance(agent, df)

    print("\n==== All A2CAgent Tests Completed ====")

# %%
# Run the tests
if __name__ == "__main__":
    run_all_tests()

