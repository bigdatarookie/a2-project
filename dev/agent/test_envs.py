# %% [markdown]
# # This is dedicated to testing custom environments

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
from packages.agent.custom_envs import MultiTickerPortfolioEnv

# %%
# Test the environment
def test_environment(daily_data, minute_data, feature_list=None):
    """
    Test the MultiTickerPortfolioEnv with real market data

    Args:
        daily_data (pd.DataFrame): Daily market data
        minute_data (pd.DataFrame): Minute market data
        feature_list (list): List of features to include in the environment
    """
    if feature_list is None:
        feature_list = ['volume']

    print("=" * 80)
    print("ENVIRONMENT TESTING SUITE")
    print("=" * 80)

    # Part 1: Basic Environment Initialization
    print("\n1. Testing environment initialization...")

    # Test with daily data
    try:
        env_daily = MultiTickerPortfolioEnv(
            df=daily_data,
            history_length=10,
            initial_cash=10000,
            feature_columns=feature_list,
            transaction_cost=0.001  # 0.1% transaction cost
        )

        print(f"✓ Daily data environment created successfully")
        print(f"  - Tickers: {env_daily.tickers[:5]}{'...' if len(env_daily.tickers) > 5 else ''}")
        print(f"  - Data shape: {env_daily.prices.shape}")
        print(f"  - Features: {list(env_daily.feature_data.keys())}")
        print(f"  - Auto-detected annualization factor: {env_daily.annualization_factor}")

        # Test observation and action spaces
        print(f"  - Observation space: {env_daily.observation_space}")
        print(f"  - Action space: {env_daily.action_space}")

    except Exception as e:
        print(f"✗ Failed to create daily environment: {e}")
        return

    # Test with minute data if provided
    if minute_data is not None:
        try:
            env_minute = MultiTickerPortfolioEnv(
                df=minute_data,
                history_length=10,
                initial_cash=10000,
                feature_columns=feature_list,
                transaction_cost=0.001
            )

            print(f"✓ Minute data environment created successfully")
            print(f"  - Data shape: {env_minute.prices.shape}")
            print(f"  - Auto-detected annualization factor: {env_minute.annualization_factor}")

        except Exception as e:
            print(f"✗ Failed to create minute environment: {e}")
            env_minute = None
    else:
        env_minute = None

    # Part 2: Test Reset Function
    print("\n2. Testing environment reset...")

    obs, info = env_daily.reset()
    print(f"✓ Environment reset succeeded")
    print(f"  - Observation shape: {obs.shape}")
    print(f"  - Initial info: {', '.join([f'{k}: {v}' for k, v in info.items() if not isinstance(v, (list, np.ndarray))])}")

    # Part 3: Test Step Function with Different Action Types
    print("\n3. Testing step function with different actions...")

    # Create an empty plot for portfolio values
    plt.figure(figsize=(12, 8))

    # Test different action strategies
    strategies = {
        "Equal Weight": lambda n_stocks: np.ones(n_stocks) / n_stocks,
        "First Stock Only": lambda n_stocks: np.array([1.0] + [0.0] * (n_stocks - 1)),
        "Cash Only": lambda n_stocks: np.zeros(n_stocks),
        "Random Weights": lambda n_stocks: np.random.random(n_stocks)
    }

    results = {}

    for name, strategy_fn in strategies.items():
        print(f"\n  Testing strategy: {name}")

        # Reset the environment
        obs, info = env_daily.reset(seed=42)

        # Run for a number of steps
        max_steps = min(100, env_daily.n_steps - env_daily.history_length - 1)
        portfolio_values = [info['portfolio_value']]
        portfolio_returns = []
        total_rewards = 0

        for i in range(max_steps):
            # Generate action based on strategy
            action = strategy_fn(env_daily.n_stocks)

            # Normalize if sum > 1
            if np.sum(action) > 1.0:
                action = action / np.sum(action)

            # Take a step
            obs, reward, terminated, truncated, info = env_daily.step(action)

            # Record values
            portfolio_values.append(info['portfolio_value'])
            if i > 0:
                ret = (portfolio_values[-1] / portfolio_values[-2]) - 1
                portfolio_returns.append(ret)

            total_rewards += reward

            if terminated or truncated:
                break

        # Store results
        final_value = portfolio_values[-1]
        total_return = (final_value / portfolio_values[0]) - 1
        avg_return = np.mean(portfolio_returns) if portfolio_returns else 0
        volatility = np.std(portfolio_returns) if len(portfolio_returns) > 1 else 0
        sharpe = avg_return / volatility if volatility > 0 else 0
        max_drawdown = calculate_max_drawdown(portfolio_values)

        results[name] = {
            'portfolio_values': portfolio_values,
            'total_return': total_return,
            'avg_return': avg_return,
            'volatility': volatility,
            'sharpe': sharpe,
            'max_drawdown': max_drawdown,
            'final_value': final_value,
            'total_rewards': total_rewards
        }

        # Print results
        print(f"    - Final portfolio value: ${final_value:.2f}")
        print(f"    - Total return: {total_return:.2%}")
        print(f"    - Sharpe ratio: {sharpe:.4f}")
        print(f"    - Max drawdown: {max_drawdown:.2%}")
        print(f"    - Total rewards: {total_rewards:.4f}")

        # Plot portfolio values
        plt.plot(portfolio_values, label=name)

    # Finish the portfolio value plot
    plt.title('Portfolio Value Comparison Across Strategies')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Part 4: Compare Strategy Performance
    print("\n4. Strategy Performance Comparison:")

    # Create performance comparison bar chart
    metrics = ['total_return', 'sharpe', 'max_drawdown']
    metric_labels = ['Total Return', 'Sharpe Ratio', 'Max Drawdown']

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    for i, (metric, label) in enumerate(zip(metrics, metric_labels)):
        values = [results[strategy][metric] for strategy in strategies.keys()]
        colors = plt.cm.viridis(np.linspace(0, 1, len(strategies)))
        axes[i].bar(strategies.keys(), values, color=colors)
        axes[i].set_title(label)
        axes[i].set_xticklabels(strategies.keys(), rotation=45, ha='right')
        axes[i].grid(axis='y')

    plt.tight_layout()
    plt.show()

    # Part 5: Test Transaction Costs Impact
    print("\n5. Testing transaction cost impact...")

    transaction_costs = [0.0, 0.001, 0.003, 0.01]
    tc_results = {}

    plt.figure(figsize=(12, 8))

    for tc in transaction_costs:
        # Create environment with specific transaction cost
        env_tc = MultiTickerPortfolioEnv(
            df=daily_data,
            history_length=10,
            initial_cash=10000,
            feature_columns=feature_list,
            transaction_cost=tc
        )

        # Reset environment
        obs, info = env_tc.reset(seed=42)

        # Run for a number of steps with the equal weight strategy
        max_steps = min(100, env_tc.n_steps - env_tc.history_length - 1)
        portfolio_values = [info['portfolio_value']]
        transaction_cost_total = 0

        for i in range(max_steps):
            # Equal weight strategy with slight random variation for more trading
            action = np.ones(env_tc.n_stocks) / env_tc.n_stocks + np.random.normal(0, 0.01, env_tc.n_stocks)
            action = np.clip(action, 0, 1)
            if np.sum(action) > 1.0:
                action = action / np.sum(action)

            # Take a step
            obs, reward, terminated, truncated, info = env_tc.step(action)

            # Record values
            portfolio_values.append(info['portfolio_value'])
            transaction_cost_total += info['transaction_cost']

            if terminated or truncated:
                break

        # Store results
        final_value = portfolio_values[-1]
        total_return = (final_value / portfolio_values[0]) - 1

        tc_results[f"{tc:.1%}"] = {
            'portfolio_values': portfolio_values,
            'total_return': total_return,
            'final_value': final_value,
            'transaction_cost_total': transaction_cost_total
        }

        # Plot portfolio values
        plt.plot(portfolio_values, label=f"TC = {tc:.1%}")

        print(f"  - Transaction cost {tc:.1%}: Return = {total_return:.2%}, Total TC = ${transaction_cost_total:.2f}")

    # Finish the transaction cost plot
    plt.title('Impact of Transaction Costs on Portfolio Value')
    plt.xlabel('Steps')
    plt.ylabel('Portfolio Value ($)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Part 6: Test Observation and State Consistency
    print("\n6. Testing observation and state consistency...")

    # Reset environment
    env = MultiTickerPortfolioEnv(
        df=daily_data,
        history_length=10,
        initial_cash=10000,
        feature_columns=feature_list
    )

    obs, info = env.reset(seed=42)

    # Take a series of random actions
    actions = []
    observations = []

    for i in range(10):
        # Random action
        action = np.random.random(env.n_stocks)
        if np.sum(action) > 1.0:
            action = action / np.sum(action)

        actions.append(action)
        observations.append(obs)

        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            break

    # Check observation shapes
    all_same_shape = all(o.shape == observations[0].shape for o in observations)
    print(f"✓ All observations have consistent shape: {all_same_shape}")

    # Check for NaN or infinity values
    has_nan = any(np.isnan(o).any() for o in observations)
    has_inf = any(np.isinf(o).any() for o in observations)

    if has_nan:
        print("✗ Warning: NaN values detected in observations")
    else:
        print("✓ No NaN values in observations")

    if has_inf:
        print("✗ Warning: Infinity values detected in observations")
    else:
        print("✓ No infinity values in observations")

    # Part 7: Performance Benchmark
    print("\n7. Environment performance benchmark:")

    # Reset environment
    env = MultiTickerPortfolioEnv(
        df=daily_data,
        history_length=10,
        initial_cash=10000,
        feature_columns=feature_list
    )

    obs, info = env.reset()

    # Time 1000 random steps
    n_steps = 1000

    start_time = time.time()
    for i in range(n_steps):
        # Random action
        action = np.random.random(env.n_stocks)
        if np.sum(action) > 1.0:
            action = action / np.sum(action)

        # Take a step
        obs, reward, terminated, truncated, info = env.step(action)

        if terminated or truncated:
            obs, info = env.reset()

    elapsed = time.time() - start_time
    steps_per_second = n_steps / elapsed

    print(f"  - Completed {n_steps} steps in {elapsed:.2f} seconds")
    print(f"  - Performance: {steps_per_second:.2f} steps/second")

    print("\n" + "=" * 80)
    print("ENVIRONMENT TESTING COMPLETE")
    print("=" * 80)

    return results

def calculate_max_drawdown(portfolio_values):
    """Calculate maximum drawdown from a series of portfolio values"""
    portfolio_values = np.array(portfolio_values)
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    return np.max(drawdown) if len(drawdown) > 0 else 0

# %%
# Run the test when executed directly
if __name__ == "__main__":
    # Define features to include
    features = ['volume', 'open', 'high', 'low', 'obv', 'rsi', 'atr']

    # Data importing
    processed_data = pd.read_csv('data/rts_1d_2023_2024_indicators.csv')
    processed_minute_data = pd.read_csv('data/rts_1min_2024_dec_indicators.csv')

    # Test the environment
    test_environment(processed_data, processed_minute_data, features)
