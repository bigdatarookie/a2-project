# Portfolio Management Module

The agent package now includes comprehensive portfolio management capabilities for reinforcement learning research and algorithmic trading.

## Features

- **OptimizedPortfolioEnv**: Memory-efficient Gym-compatible environment for portfolio management
- **Trading Strategies**: Multiple built-in strategies with consistent interfaces
- **Performance Analysis**: Comprehensive metrics and visualization tools
- **Flexible Configuration**: Customizable parameters for research and production use

## Quick Start

```python
from agent import quick_compare, EqualWeightStrategy, GreedyStrategy

# Define strategies to compare
agents = {
    'Equal Weight': EqualWeightStrategy(),
    'Greedy Momentum': GreedyStrategy(lookback=5),
}

# Run comparison (includes visualization)
results = quick_compare(df, agents, n_assets=5, max_steps=100)
```

## Detailed Usage

### 1. Environment Setup

```python
from agent import OptimizedPortfolioEnv

# Create environment
env = OptimizedPortfolioEnv(
    df=market_data,           # DataFrame with OHLCV + indicators  
    window_size=10,           # Observation window size
    n_assets=5,               # Number of assets to trade
    transaction_cost=0.001,   # Transaction cost (0.1%)
    enable_rendering=True,    # Enable visualization
    render_interval=50        # Render every 50 steps
)

# Reset and run
obs, info = env.reset()
action = np.array([0.2, 0.2, 0.2, 0.2, 0.2])  # Equal weights
obs, reward, done, truncated, info = env.step(action)
```

### 2. Trading Strategies

```python
from agent import create_strategy, EqualWeightStrategy, GreedyStrategy, MomentumStrategy

# Built-in strategies
equal_weight = EqualWeightStrategy()
greedy = GreedyStrategy(lookback=10, temperature=1.0)
momentum = MomentumStrategy(lookback=20, rsi_weight=0.3, macd_weight=0.3)

# Factory creation
strategy = create_strategy('momentum', lookback=15)

# Custom strategy
class MyStrategy(BaseStrategy):
    def get_action(self, observation):
        # Your strategy logic here
        return portfolio_weights
```

### 3. Episode Execution

```python
from agent import EpisodeRunner

# Single episode with detailed tracking
runner = EpisodeRunner(env)
results = runner.run_episode(strategy, max_steps=500, verbose=True)

# Results include:
print(f"Final Value: {results['final_value']:.2f}")
print(f"Total Return: {results['total_return']:.2f}%") 
print(f"Sharpe Ratio: {results['sharpe_ratio']:.3f}")
```

### 4. Agent Comparison

```python
from agent import AgentComparator, PortfolioVisualizer

# Compare multiple agents
comparator = AgentComparator(df)
agents_dict = {
    'Equal Weight': EqualWeightStrategy(),
    'Momentum': MomentumStrategy(lookback=10),
    'Random': RandomStrategy(seed=42)
}

results = comparator.compare_agents(
    agents_dict, 
    n_assets=8, 
    max_steps=200,
    enable_rendering=False  # Disable for speed
)

# Visualize results
PortfolioVisualizer.plot_comparison(results)
PortfolioVisualizer.plot_weights_evolution(results, tickers)
```

## Data Requirements

The portfolio environment expects a DataFrame with the following columns:

- `tic`: Ticker symbol
- `open`, `high`, `low`, `close`: OHLC prices  
- `volume`: Trading volume
- `rsi`: Relative Strength Index
- `macd_line`: MACD line
- `bb_middle`: Bollinger Bands middle line

The DataFrame should have a datetime index for proper time series handling.

## Configuration Options

### Environment Parameters

- `window_size`: Number of timesteps in observation window (default: 10)
- `n_assets`: Number of assets to trade (default: all available)
- `transaction_cost`: Transaction cost as fraction (default: 0.001)
- `use_vectorized`: Use vectorized operations for speed (default: True)
- `enable_rendering`: Enable visualization (default: False)
- `render_interval`: Steps between renders (default: 10)

### Strategy Parameters

Each strategy has specific parameters:

- **GreedyStrategy**: `lookback`, `temperature`
- **MomentumStrategy**: `lookback`, `rsi_weight`, `macd_weight`, `price_weight`
- **RandomStrategy**: `seed`

## Performance Considerations

- **Memory Efficiency**: Vectorized operations and efficient data structures
- **Rendering Control**: Optional visualization with configurable intervals
- **Batch Processing**: Support for multiple agent comparison
- **Resource Cleanup**: Automatic memory management and cleanup methods

## Testing

Run the comprehensive test suite:

```python
# From the project root
python -m pytest packages/agent/test_portfolio.py -v

# Or run manually in notebook
from packages.agent.test_portfolio import *
test_df = create_test_data(n_tickers=5, n_timesteps=100)
# Run individual test classes...
```

## Examples

See the usage examples in the next section for complete workflows and advanced features.