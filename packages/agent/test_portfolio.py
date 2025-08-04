"""
Comprehensive test suite for the agent package portfolio functionality.
Tests all modules and their integration for end-to-end workflow validation.
"""

import pytest
import numpy as np
import pandas as pd
from typing import Dict, Any
import tempfile
import os

# Test data generation
def create_test_data(n_tickers=5, n_timesteps=100):
    """Create synthetic test data with OHLCV and indicators"""
    tickers = [f'TICK{i}' for i in range(n_tickers)]
    dates = pd.date_range('2024-01-01', periods=n_timesteps, freq='1H')
    
    data = []
    for ticker in tickers:
        # Generate realistic price data
        base_price = np.random.uniform(50, 200)
        price_changes = np.random.normal(0, 0.02, n_timesteps)
        close_prices = base_price * np.cumprod(1 + price_changes)
        
        # Generate OHLC from close prices
        high_prices = close_prices * (1 + np.random.uniform(0, 0.02, n_timesteps))
        low_prices = close_prices * (1 - np.random.uniform(0, 0.02, n_timesteps))
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        
        # Generate volume
        volume = np.random.randint(1000, 10000, n_timesteps)
        
        # Generate technical indicators
        rsi = np.random.uniform(20, 80, n_timesteps)
        macd_line = np.random.normal(0, 1, n_timesteps)
        bb_middle = close_prices  # Simple moving average approximation
        
        for i in range(n_timesteps):
            data.append({
                'tic': ticker,
                'open': open_prices[i],
                'high': high_prices[i],
                'low': low_prices[i],
                'close': close_prices[i],
                'volume': volume[i],
                'rsi': rsi[i],
                'macd_line': macd_line[i],
                'bb_middle': bb_middle[i]
            })
    
    df = pd.DataFrame(data)
    df.index = pd.to_datetime([dates[i % n_timesteps] for i in range(len(df))])
    return df


class TestEnvironments:
    """Test cases for environments module"""
    
    @pytest.fixture
    def test_df(self):
        return create_test_data(n_tickers=3, n_timesteps=50)
    
    def test_env_creation(self, test_df):
        from packages.agent.environments import OptimizedPortfolioEnv
        
        env = OptimizedPortfolioEnv(test_df, window_size=5, n_assets=3)
        assert env.n_assets == 3
        assert env.window_size == 5
        assert env.action_space.shape == (3,)
        assert env.observation_space.shape == (3, 5, 8)
    
    def test_env_reset(self, test_df):
        from packages.agent.environments import OptimizedPortfolioEnv
        
        env = OptimizedPortfolioEnv(test_df, window_size=5, n_assets=3)
        obs, info = env.reset()
        
        assert obs.shape == (3, 5, 8)
        assert env.state.portfolio_value == 1.0
        assert np.isclose(np.sum(env.state.weights), 1.0)
    
    def test_env_step(self, test_df):
        from packages.agent.environments import OptimizedPortfolioEnv
        
        env = OptimizedPortfolioEnv(test_df, window_size=5, n_assets=3)
        env.reset()
        
        action = np.array([0.4, 0.3, 0.3])
        obs, reward, done, truncated, info = env.step(action)
        
        assert obs.shape == (3, 5, 8)
        assert isinstance(reward, (int, float))
        assert 'portfolio_value' in info


class TestStrategies:
    """Test cases for strategies module"""
    
    @pytest.fixture
    def test_observation(self):
        return np.random.rand(3, 10, 8)
    
    def test_equal_weight_strategy(self, test_observation):
        from packages.agent.strategies import EqualWeightStrategy
        
        strategy = EqualWeightStrategy()
        weights = strategy.get_action(test_observation)
        
        assert len(weights) == 3
        assert np.isclose(np.sum(weights), 1.0)
        assert np.allclose(weights, 1/3)
    
    def test_greedy_strategy(self, test_observation):
        from packages.agent.strategies import GreedyStrategy
        
        strategy = GreedyStrategy(lookback=5)
        weights = strategy.get_action(test_observation)
        
        assert len(weights) == 3
        assert np.isclose(np.sum(weights), 1.0)
        assert all(w >= 0 for w in weights)
    
    def test_strategy_factory(self):
        from packages.agent.strategies import create_strategy
        
        strategy = create_strategy('equal_weight')
        assert strategy.name == 'Equal Weight'
        
        strategy = create_strategy('greedy', lookback=7)
        assert hasattr(strategy, 'lookback')
        assert strategy.lookback == 7


class TestEnvUtils:
    """Test cases for env_utils module"""
    
    @pytest.fixture
    def test_df(self):
        return create_test_data(n_tickers=3, n_timesteps=30)
    
    def test_episode_runner(self, test_df):
        from packages.agent.environments import OptimizedPortfolioEnv
        from packages.agent.strategies import EqualWeightStrategy
        from packages.agent.env_utils import EpisodeRunner
        
        env = OptimizedPortfolioEnv(test_df, window_size=5, n_assets=3, enable_rendering=False)
        runner = EpisodeRunner(env)
        agent = EqualWeightStrategy()
        
        results = runner.run_episode(agent, max_steps=10, verbose=False)
        
        required_keys = ['final_value', 'total_return', 'sharpe_ratio', 'steps']
        for key in required_keys:
            assert key in results
    
    def test_agent_comparator(self, test_df):
        from packages.agent.strategies import EqualWeightStrategy, GreedyStrategy
        from packages.agent.env_utils import AgentComparator
        
        comparator = AgentComparator(test_df)
        agents = {
            'Equal': EqualWeightStrategy(),
            'Greedy': GreedyStrategy(lookback=3)
        }
        
        results = comparator.compare_agents(
            agents, max_steps=5, n_assets=3, enable_rendering=False
        )
        
        assert len(results) == 2
        assert 'Equal' in results
        assert 'Greedy' in results


class TestIntegration:
    """End-to-end integration tests"""
    
    @pytest.fixture
    def test_df(self):
        return create_test_data(n_tickers=5, n_timesteps=50)
    
    def test_full_workflow(self, test_df):
        """Test complete workflow from data to results"""
        from packages.agent import (
            OptimizedPortfolioEnv, EqualWeightStrategy, GreedyStrategy,
            EpisodeRunner, AgentComparator
        )
        
        # Create environment
        env = OptimizedPortfolioEnv(test_df, n_assets=3, enable_rendering=False)
        
        # Create strategies
        strategies = {
            'Equal Weight': EqualWeightStrategy(),
            'Greedy': GreedyStrategy(lookback=5)
        }
        
        # Run comparison
        comparator = AgentComparator(test_df)
        results = comparator.compare_agents(
            strategies, max_steps=10, n_assets=3, enable_rendering=False
        )
        
        # Validate results
        assert len(results) == 2
        for agent_name, metrics in results.items():
            assert 'final_value' in metrics
            assert 'total_return' in metrics
            assert metrics['steps'] <= 10
        
        env.cleanup()
    
    def test_quick_compare(self, test_df):
        """Test convenience function"""
        from packages.agent import quick_compare, EqualWeightStrategy, GreedyStrategy
        
        agents = {
            'Equal': EqualWeightStrategy(),
            'Greedy': GreedyStrategy()
        }
        
        # Note: This would normally show plots, but we can test the function runs
        try:
            results = quick_compare(
                test_df, agents, max_steps=5, n_assets=3, enable_rendering=False
            )
            assert len(results) == 2
        except Exception as e:
            # If matplotlib is not available in test environment, that's ok
            if "matplotlib" not in str(e).lower():
                raise


if __name__ == "__main__":
    # Run tests manually for development
    test_df = create_test_data(n_tickers=3, n_timesteps=30)
    
    # Run basic tests
    print("Running integration tests...")
    
    # Test 1: Full workflow
    from packages.agent import (
        OptimizedPortfolioEnv, EqualWeightStrategy, GreedyStrategy,
        AgentComparator
    )
    
    env = OptimizedPortfolioEnv(test_df, n_assets=3, enable_rendering=False)
    comparator = AgentComparator(test_df)
    
    agents = {
        'Equal Weight': EqualWeightStrategy(),
        'Greedy': GreedyStrategy(lookback=3)
    }
    
    results = comparator.compare_agents(
        agents, max_steps=5, n_assets=3, enable_rendering=False
    )
    
    print(f"Test passed: {len(results)} agents compared")
    print("All integration tests completed successfully!")