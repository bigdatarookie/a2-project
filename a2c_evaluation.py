# %% [markdown]
# # RL Portfolio Strategy Testing and Visualization
#
# This notebook loads and evaluates models tracked by MLflow, focusing on visualization and performance analysis.

# %%
# Import libraries
import gymnasium as gym
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import mplfinance as mpf
from datetime import datetime, timedelta
from scipy import stats
import os
from typing import Dict, List, Tuple, Optional, Callable, Union

# Stable Baselines
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy

# MLflow for model tracking and loading
import mlflow
from mlflow.tracking import MlflowClient

# Import our environment class from the first notebook
from a2c_environment import MultiTickerPortfolioEnv

# %% [markdown]
# ## 1. MLflow Model Loading Utilities

# %%
def list_mlflow_models(model_directory="data/models"):
    """
    List all models saved by MLflow in the specified directory.

    Args:
        model_directory: Path to the MLflow models directory

    Returns:
        A list of model info dictionaries with model_path, run_id, and metadata
    """
    try:
        # Set up MLflow client
        client = MlflowClient()

        # Set the tracking URI to the local directory
        mlflow.set_tracking_uri(f"file:{os.path.abspath(model_directory)}")

        # Get all experiments
        experiments = client.search_experiments()

        if not experiments:
            print(f"No experiments found in {model_directory}")
            return []

        models = []

        # For each experiment, get all runs
        for experiment in experiments:
            exp_id = experiment.experiment_id
            runs = client.search_runs(experiment_ids=[exp_id])

            for run in runs:
                run_id = run.info.run_id

                # Get artifacts for this run
                artifacts = client.list_artifacts(run_id)

                # Find model artifacts
                for artifact in artifacts:
                    if artifact.path == "model":
                        # Get run data
                        run_data = client.get_run(run_id).data

                        # Extract parameters
                        params = run_data.params
                        metrics = run_data.metrics

                        # Create model info
                        model_info = {
                            "run_id": run_id,
                            "model_path": f"{model_directory}/{run_id}/artifacts/model",
                            "params": params,
                            "metrics": metrics,
                            "experiment_id": exp_id,
                            "experiment_name": experiment.name,
                            "date": datetime.fromtimestamp(run.info.end_time/1000.0).strftime('%Y-%m-%d %H:%M:%S')
                        }

                        models.append(model_info)

        # Sort models by date (newest first)
        models.sort(key=lambda x: x["date"], reverse=True)

        return models

    except Exception as e:
        print(f"Error listing MLflow models: {e}")
        return []

# %%
def load_model_from_mlflow(model_info):
    """
    Load a model from MLflow tracking.

    Args:
        model_info: Dictionary containing model metadata, including model_path

    Returns:
        Loaded model
    """
    try:
        # Load the model
        model_path = model_info["model_path"]
        model = mlflow.sklearn.load_model(model_path)

        print(f"Loaded model from {model_path}")
        print(f"Model parameters:")
        for key, value in model_info["params"].items():
            print(f"  {key}: {value}")

        return model

    except Exception as e:
        print(f"Error loading model: {e}")
        return None

# %%
def load_best_model(model_directory="data/models", metric_name="mean_reward"):
    """
    Load the best model based on a specified metric.

    Args:
        model_directory: Path to the MLflow models directory
        metric_name: Metric to use for comparison (higher is better)

    Returns:
        Tuple of (model, model_info)
    """
    # List all models
    models = list_mlflow_models(model_directory)

    if not models:
        print("No models found")
        return None, None

    # Find model with best metric
    best_model_info = None
    best_metric = float('-inf')

    for model_info in models:
        if metric_name in model_info["metrics"]:
            metric_value = float(model_info["metrics"][metric_name])
            if metric_value > best_metric:
                best_metric = metric_value
                best_model_info = model_info

    if best_model_info:
        # Load the best model
        model = load_model_from_mlflow(best_model_info)
        print(f"Best model had {metric_name} = {best_metric:.4f}")
        return model, best_model_info
    else:
        print(f"No models found with metric {metric_name}")
        return None, None

# %% [markdown]
# ## 2. Period Length Estimation and Metric Calculation

# %%
def estimate_period_length(dates):
    """
    Estimate the period length from timestamp data and calculate
    the number of periods per year for annualization.

    Args:
        dates: List of date strings or timestamp objects

    Returns:
        A tuple of (period_length, periods_per_year)
        where period_length is a timedelta object and
        periods_per_year is a float representing the number of periods in a year
    """
    # Convert dates to datetime objects if they're strings
    if isinstance(dates[0], str):
        dates = [pd.to_datetime(d) for d in dates]

    # Calculate the differences between consecutive timestamps
    diffs = []
    for i in range(1, len(dates)):
        diff = dates[i] - dates[i-1]
        if diff.total_seconds() > 0:  # Only count positive differences
            diffs.append(diff)

    if not diffs:
        print("Warning: Could not estimate period length. Using daily as default.")
        return timedelta(days=1), 252  # Default to daily data (252 trading days/year)

    # Find the median difference to handle potential irregularities
    median_diff = sorted(diffs)[len(diffs) // 2]

    # Calculate periods per year
    seconds_per_year = 365.25 * 24 * 60 * 60
    periods_per_year = seconds_per_year / median_diff.total_seconds()

    print(f"Estimated period length: {median_diff}")
    print(f"Estimated periods per year for annualization: {periods_per_year:.2f}")

    return median_diff, periods_per_year

# %% [markdown]
# ## 3. Strategy Testing Framework

# %%
def run_strategy(env, strategy_fn, initial_cash=10000, name="Strategy"):
    """
    Run a trading strategy in the environment and return performance metrics

    Parameters:
    - env: The MultiTickerPortfolioEnv environment
    - strategy_fn: Function that takes observation and returns action
    - initial_cash: Starting capital
    - name: Name of the strategy for display

    Returns:
    - Dictionary containing performance metrics and time series
    """
    # Reset the environment
    observation, info = env.reset(seed=42)

    # Lists to store metrics
    dates = []
    portfolio_values = []
    returns = []
    asset_allocations = []
    actions_taken = []
    cash_values = []
    reward_components = []

    # Run the strategy
    while True:
        # Get action from strategy
        action = strategy_fn(observation)

        # Take step in environment
        observation, reward, terminated, truncated, info = env.step(action)

        # Record metrics
        dates.append(info['date'])
        portfolio_values.append(info['portfolio_value'])
        cash_values.append(info['cash'])
        asset_allocations.append(info['allocations'] if 'allocations' in info else info['shares'].copy())
        actions_taken.append(action)

        # Store reward components if available
        if 'sharpe_component' in info:
            reward_components.append({
                'sharpe': info['sharpe_component'],
                'consistency': info['consistency_bonus'],
                'diversification': info['diversification_score'],
                'drawdown_penalty': info['drawdown_penalty']
            })

        # Calculate daily return if we have at least two values
        if len(portfolio_values) > 1:
            daily_return = (portfolio_values[-1] / portfolio_values[-2]) - 1
            returns.append(daily_return)

        if terminated or truncated:
            break

    # Convert lists to numpy arrays
    portfolio_values = np.array(portfolio_values)
    returns = np.array(returns)
    asset_allocations = np.array(asset_allocations)

    # Estimate period length from date data for proper annualization
    period_length, periods_per_year = estimate_period_length(dates)

    # Calculate performance metrics
    total_return = (portfolio_values[-1] / initial_cash) - 1
    annualized_return = (1 + total_return) ** (periods_per_year / len(portfolio_values)) - 1

    # Risk metrics - use estimated periods per year for annualization
    volatility = np.std(returns) * np.sqrt(periods_per_year)  # Annualized

    # Calculate max drawdown properly with numpy operations
    peak = np.maximum.accumulate(portfolio_values)
    drawdown = (peak - portfolio_values) / peak
    max_drawdown = np.max(drawdown) if len(drawdown) > 0 else 0

    # Sharpe Ratio (assuming risk-free rate of 0 for simplicity)
    sharpe_ratio = annualized_return / volatility if volatility > 0 else 0

    # Sortino Ratio (penalizes only downside volatility)
    downside_returns = returns[returns < 0]
    downside_volatility = np.std(downside_returns) * np.sqrt(periods_per_year) if len(downside_returns) > 0 else 1e-6
    sortino_ratio = annualized_return / downside_volatility

    # Create a DataFrame for time series data
    df_results = pd.DataFrame({
        'date': dates,
        'portfolio_value': portfolio_values,
        'cash': cash_values
    })

    # Add allocation for each asset
    for i, ticker in enumerate(env.tickers):
        if asset_allocations.shape[1] == len(env.tickers):
            df_results[f'allocation_{ticker}'] = asset_allocations[:, i]

    # Add reward components if available
    if reward_components:
        df_results['sharpe_component'] = [rc['sharpe'] for rc in reward_components]
        df_results['consistency_bonus'] = [rc['consistency'] for rc in reward_components]
        df_results['diversification_score'] = [rc['diversification'] for rc in reward_components]
        df_results['drawdown_penalty'] = [rc['drawdown_penalty'] for rc in reward_components]

    # Set date as index
    df_results['date'] = pd.to_datetime(df_results['date'])
    df_results.set_index('date', inplace=True)

    # Return dictionary with all metrics and results
    return {
        'name': name,
        'time_series': df_results,
        'total_return': total_return,
        'annualized_return': annualized_return,
        'volatility': volatility,
        'max_drawdown': max_drawdown,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'final_value': portfolio_values[-1],
        'actions': actions_taken,
        'periods_per_year': periods_per_year,
        'period_length': period_length
    }

# %% [markdown]
# ## 4. Benchmark Strategies

# %%
def a2c_strategy(model):
    """Factory function to create a strategy using a trained A2C agent"""
    def strategy(observation):
        action, _ = model.predict(observation, deterministic=True)
        return action
    return strategy

def equal_weight_strategy(observation):
    """Simple buy and hold strategy with equal allocation to all assets"""
    # Extract the environment parameters from the observation
    # Number of stocks in the environment
    n_stocks = (len(observation) - 1) // (12)  # Estimation based on observation space

    # Check current allocations (at the end of the observation)
    portfolio_allocations = observation[-n_stocks-1:-1]  # Last n_stocks values before cash

    # If we have minimal allocation (below 1%), buy all stocks equally
    if np.sum(portfolio_allocations) < 0.01:
        return 0  # This will buy all stocks with the default 10% of cash
    else:
        # Otherwise hold
        return 1

def momentum_strategy(observation):
    """Strategy that buys stocks with positive momentum and sells those with negative momentum"""
    # Extract parameters from observation structure
    n_stocks = (len(observation) - 1) // (12)  # Estimation based on observation space

    # Extract momentum indicators from the observation
    # Momentum indicators start after the price history
    price_history_len = 10 * n_stocks  # Assuming history_length=10
    momentum_indicators = observation[price_history_len:price_history_len+n_stocks]

    # Determine actions for each stock based on momentum
    stock_actions = []
    for i in range(n_stocks):
        if momentum_indicators[i] > 0.01:  # Positive momentum of at least 1%
            stock_actions.append(0)  # Buy
        elif momentum_indicators[i] < -0.01:  # Negative momentum of at least -1%
            stock_actions.append(2)  # Sell
        else:
            stock_actions.append(1)  # Hold

    # Encode actions into a single action index
    action_index = 0
    for i, action in enumerate(stock_actions):
        action_index += action * (3 ** i)

    return action_index

# %% [markdown]
# ## 5. Load Data and Create Environment

# %%
# Load the data
def load_data(file_path='data/rts_daily_jan.csv'):
    df = pd.read_csv(file_path)
    print(f"Loaded data with shape: {df.shape}")
    return df

# %%
# Load data and create environment
df = load_data()
print("Sample data:")
print(df.head())

# Create evaluation environment
env_params = {
    "history_length": 24,  # Will be overridden by model parameters
    "initial_cash": 10000,
    "risk_aversion": 2.0   # Will be overridden by model parameters
}

# %% [markdown]
# ## 6. Model Evaluation

# %%
# Load the best model from MLflow
best_model, best_model_info = load_best_model()

if best_model_info and 'params' in best_model_info:
    # Update environment parameters based on model hyperparameters
    if 'history_length' in best_model_info['params']:
        env_params['history_length'] = int(best_model_info['params']['history_length'])
    if 'risk_aversion' in best_model_info['params']:
        env_params['risk_aversion'] = float(best_model_info['params']['risk_aversion'])

# Create environment with model parameters
env = MultiTickerPortfolioEnv(
    df,
    history_length=env_params['history_length'],
    initial_cash=env_params['initial_cash'],
    risk_aversion=env_params['risk_aversion']
)

# %% [markdown]
# ## 7. Run Model and Benchmark Evaluations

# %%
# Run all strategies for comparison
if best_model:
    print("Running A2C agent strategy evaluation...")
    a2c_results = run_strategy(env, a2c_strategy(best_model), name="A2C Agent")

    print("Running equal weight benchmark strategy evaluation...")
    env.reset(seed=42)  # Reset environment for benchmark
    equal_weight_results = run_strategy(env, equal_weight_strategy, name="Equal Weight")

    print("Running momentum strategy evaluation...")
    env.reset(seed=42)  # Reset environment for momentum strategy
    momentum_results = run_strategy(env, momentum_strategy, name="Momentum")

# %% [markdown]
# ## 8. Visualization

# %%
# 1. Performance Comparison: Portfolio Value Over Time
def plot_portfolio_comparison(results_list):
    plt.figure(figsize=(16, 10))

    # Format y-axis as currency
    def currency_formatter(x, pos):
        return f'${x:,.0f}'

    plt.subplot(2, 1, 1)
    for results in results_list:
        plt.plot(results['time_series'].index, results['time_series']['portfolio_value'], label=results['name'])

    plt.gca().yaxis.set_major_formatter(FuncFormatter(currency_formatter))
    plt.title('Portfolio Value Comparison', fontsize=14)
    plt.ylabel('Portfolio Value ($)', fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    # 2. Drawdown Comparison
    plt.subplot(2, 1, 2)
    for results in results_list:
        equity_curve = results['time_series']['portfolio_value']
        running_max = np.maximum.accumulate(equity_curve)
        drawdown = (running_max - equity_curve) / running_max
        plt.plot(results['time_series'].index, drawdown, label=f"{results['name']} Drawdown")

    plt.title('Drawdown Comparison', fontsize=14)
    plt.ylabel('Drawdown (%)', fontsize=12)
    plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'{x:.1%}'))
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=12)

    plt.tight_layout()
    plt.show()

# %%
# 2. Asset Allocation Over Time
def plot_asset_allocation(results):
    plt.figure(figsize=(16, 8))

    # Plot allocation for each asset
    allocation_columns = [col for col in results['time_series'].columns if 'allocation_' in col]
    if allocation_columns:
        allocation_data = results['time_series'][allocation_columns].copy()
        allocation_data.columns = [col.replace('allocation_', '') for col in allocation_data.columns]

        # Add cash allocation
        allocation_data['Cash'] = results['time_series']['cash'] / results['time_series']['portfolio_value']

        # Plot as area chart
        allocation_data.plot.area(figsize=(16, 8), alpha=0.7)
        plt.title(f'Asset Allocation Over Time ({results["name"]})', fontsize=14)
        plt.ylabel('Allocation (%)', fontsize=12)
        plt.xlabel('Date', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(loc='center left', bbox_to_anchor=(1, 0.5), fontsize=12)
        plt.tight_layout()
        plt.show()

# %%
# 3. Performance Metrics Comparison
def display_performance_metrics(results_list):
    metrics = ['total_return', 'annualized_return', 'volatility', 'max_drawdown', 'sharpe_ratio', 'sortino_ratio', 'periods_per_year']
    metrics_labels = ['Total Return', 'Annualized Return', 'Volatility', 'Max Drawdown', 'Sharpe Ratio', 'Sortino Ratio', 'Periods Per Year']

    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'Metric': metrics_labels
    })

    for results in results_list:
        comparison[results['name']] = [results[m] for m in metrics]

    # Format percentages
    for col in comparison.columns:
        if col != 'Metric':
            for i, metric in enumerate(metrics):
                if metric in ['total_return', 'annualized_return', 'volatility', 'max_drawdown']:
                    comparison.loc[i, col] = f"{comparison.loc[i, col]:.2%}"
                elif metric == 'periods_per_year':
                    comparison.loc[i, col] = f"{float(comparison.loc[i, col]):.2f}"

    # Display metrics comparison
    print("\n=== Performance Metrics Comparison ===")
    return comparison.set_index('Metric')

# %%
# 4. Reward Components Analysis
def plot_reward_components(results):
    if 'sharpe_component' in results['time_series'].columns:
        plt.figure(figsize=(16, 10))

        # Plot all reward components
        plt.subplot(2, 1, 1)
        plt.plot(results['time_series'].index, results['time_series']['sharpe_component'], label='Sharpe Component')
        plt.plot(results['time_series'].index, results['time_series']['consistency_bonus'], label='Consistency Bonus')
        plt.plot(results['time_series'].index, results['time_series']['diversification_score'], label='Diversification Score')
        plt.plot(results['time_series'].index, -results['time_series']['drawdown_penalty'], label='Drawdown Penalty (Negative)')

        plt.title(f'Reward Components Over Time ({results["name"]})', fontsize=14)
        plt.ylabel('Component Value', fontsize=12)
        plt.grid(True, alpha=0.3)
        plt.legend(fontsize=12)

        # Plot portfolio value alongside reward components
        plt.subplot(2, 1, 2)
        plt.plot(results['time_series'].index, results['time_series']['portfolio_value'], label='Portfolio Value', color='green')
        plt.gca().yaxis.set_major_formatter(FuncFormatter(lambda x, pos: f'${x:,.0f}'))
        plt.title('Portfolio Value Over Time', fontsize=14)
        plt.ylabel('Portfolio Value ($)', fontsize=12)
        plt.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.show()

# %%
# 5. Summary Statistics Table
def display_summary_table(results_list):
    """Display summary statistics for all strategies"""

    # Find best strategy
    best_strategy = max(results_list, key=lambda x: x['total_return'])

    print("\n=== Final Results ===")

    for results in results_list:
        period_info = f"(Period length: {results['period_length']}, {results['periods_per_year']:.2f} periods/year)"
        print(f"{results['name']} {period_info}:")
        print(f"- Starting Value: ${env_params['initial_cash']:.2f}")
        print(f"- Final Value: ${results['final_value']:.2f}")
        print(f"- Total Return: {results['total_return']:.2%}")
        print(f"- Annualized Return: {results['annualized_return']:.2%}")
        print(f"- Sharpe Ratio: {results['sharpe_ratio']:.2f}")
        print(f"- Max Drawdown: {results['max_drawdown']:.2%}")
        print("")

    print(f"Best performing strategy: {best_strategy['name']} with a return of {best_strategy['total_return']:.2%}")

    # Compare A2C vs equal weight
    a2c_result = next((r for r in results_list if r['name'] == 'A2C Agent'), None)
    equal_weight_result = next((r for r in results_list if r['name'] == 'Equal Weight'), None)

    if a2c_result and equal_weight_result:
        if a2c_result['final_value'] > equal_weight_result['final_value']:
            outperformance = (a2c_result['final_value'] / equal_weight_result['final_value'] - 1) * 100
            print(f"The A2C agent outperformed the Equal Weight benchmark by {outperformance:.2f}%")
        else:
            underperformance = (equal_weight_result['final_value'] / a2c_result['final_value'] - 1) * 100
            print(f"The A2C agent underperformed the Equal Weight benchmark by {underperformance:.2f}%")

# %% [markdown]
# ## 9. Run Complete Visualization

# %%
if best_model:
    # Run all visualizations
    all_results = [a2c_results, equal_weight_results, momentum_results]

    print("Plotting portfolio value comparison...")
    plot_portfolio_comparison(all_results)

    print("Plotting A2C agent asset allocation...")
    plot_asset_allocation(a2c_results)

    print("Displaying performance metrics...")
    metrics_comparison = display_performance_metrics(all_results)
    print(metrics_comparison)

    print("Plotting reward components...")
    plot_reward_components(a2c_results)

    print("Displaying summary statistics...")
    display_summary_table(all_results)
else:
    print("No model was loaded. Please train a model first using a2c_environment.py.")

# %% [markdown]
# ## 10. Model Comparison (Multiple Versions)

# %%
# Load all available models and compare their performance
def compare_multiple_models(model_directory="data/models", top_n=3):
    """
    Compare the performance of multiple models

    Args:
        model_directory: Path to the MLflow models directory
        top_n: Number of top models to compare

    Returns:
        List of results for each model
    """
    # List all models
    all_model_infos = list_mlflow_models(model_directory)

    if not all_model_infos:
        print("No models found")
        return []

    # Take top N models based on mean_reward
    all_model_infos.sort(key=lambda x: float(x["metrics"].get("mean_reward", 0)), reverse=True)
    model_infos = all_model_infos[:top_n]

    all_results = []

    for i, model_info in enumerate(model_infos):
        print(f"\nEvaluating model {i+1}/{len(model_infos)}...")

        # Load the model
        model = load_model_from_mlflow(model_info)

        if model:
            # Create environment with model parameters
            history_length = int(model_info['params'].get('history_length', 24))
            risk_aversion = float(model_info['params'].get('risk_aversion', 2.0))

            env = MultiTickerPortfolioEnv(
                df,
                history_length=history_length,
                initial_cash=10000,
                risk_aversion=risk_aversion
            )

            # Run the strategy
            run_id = model_info['run_id'][:8]  # Use short run ID for name
            strategy_results = run_strategy(
                env,
                a2c_strategy(model),
                name=f"Model {run_id}"
            )

            all_results.append(strategy_results)

    # Compare model performances
    if all_results:
        # Plot performance comparison
        plot_portfolio_comparison(all_results)

        # Display metrics
        metrics_comparison = display_performance_metrics(all_results)
        print(metrics_comparison)

    return all_results

# %%
# Run model comparison if we have models
print("Comparing multiple model versions...")
multiple_model_results = compare_multiple_models(top_n=3)

# %% [markdown]
# ## 11. Custom Parameter Testing

# %%
def test_custom_parameters(history_length=24, risk_aversion=2.0):
    """
    Create and test an environment with custom parameters

    Args:
        history_length: History length for environment
        risk_aversion: Risk aversion parameter

    Returns:
        Results dictionary
    """
    print(f"Testing custom parameters: history_length={history_length}, risk_aversion={risk_aversion}")

    # Create environment with custom parameters
    env = MultiTickerPortfolioEnv(
        df,
        history_length=history_length,
        initial_cash=10000,
        risk_aversion=risk_aversion
    )

    # Create a custom model or load best model
    if best_model:
        # Run the A2C strategy
        custom_results = run_strategy(
            env,
            a2c_strategy(best_model),
            name=f"Custom (h={history_length}, r={risk_aversion})"
        )

        # Run benchmark for comparison
        env.reset(seed=42)
        equal_weight_results = run_strategy(env, equal_weight_strategy, name="Equal Weight")

        # Visualize results
        plot_portfolio_comparison([custom_results, equal_weight_results])

        # Display metrics
        metrics_comparison = display_performance_metrics([custom_results, equal_weight_results])
        print(metrics_comparison)

        return custom_results
    else:
        print("No model available. Please train a model first.")
        return None

# %%
# Uncomment to test custom parameters
# custom_results = test_custom_parameters(history_length=30, risk_aversion=3.5)
