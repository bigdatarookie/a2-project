"""
agent - Reinforcement Learning Agents

A package for training and evaluating reinforcement learning agents for different tasks.
Now includes portfolio management environments and trading strategies.

Usage:
    # Training functionality
    from agent import train_from_config, train_from_dict

    # Portfolio environment and strategies
    from agent import OptimizedPortfolioEnv, EqualWeightStrategy, GreedyStrategy
    from agent import run_single_episode, quick_compare

    # Advanced usage
    from agent.environments import OptimizedPortfolioEnv, EnvState
    from agent.strategies import create_strategy, STRATEGY_REGISTRY
    from agent.env_utils import EpisodeRunner, AgentComparator, PortfolioVisualizer

Examples:
    # Quick portfolio analysis
    from agent import quick_compare, EqualWeightStrategy, GreedyStrategy
    
    agents = {
        'Equal Weight': EqualWeightStrategy(),
        'Greedy': GreedyStrategy(lookback=5)
    }
    results = quick_compare(df, agents, n_assets=5)

    # Custom environment setup
    from agent import OptimizedPortfolioEnv, EpisodeRunner
    
    env = OptimizedPortfolioEnv(df, n_assets=10, enable_rendering=True)
    runner = EpisodeRunner(env)
    metrics = runner.run_episode(strategy)
"""

__version__ = '0.2.0'

# Import training functionality
from packages.agent.trainer import train_from_config, train_from_dict

# Import portfolio environment components
from packages.agent.environments import OptimizedPortfolioEnv, EnvState
from packages.agent.strategies import (
    BaseStrategy, EqualWeightStrategy, GreedyStrategy, MomentumStrategy, 
    RandomStrategy, create_strategy, STRATEGY_REGISTRY
)
from packages.agent.env_utils import (
    EpisodeRunner, AgentComparator, PortfolioVisualizer,
    run_single_episode, quick_compare
)

# Define public API
__all__ = [
    # Training
    'train_from_config',
    'train_from_dict',
    
    # Portfolio Environment
    'OptimizedPortfolioEnv',
    'EnvState',
    
    # Trading Strategies
    'BaseStrategy',
    'EqualWeightStrategy',
    'GreedyStrategy', 
    'MomentumStrategy',
    'RandomStrategy',
    'create_strategy',
    'STRATEGY_REGISTRY',
    
    # Utilities
    'EpisodeRunner',
    'AgentComparator',
    'PortfolioVisualizer',
    'run_single_episode',
    'quick_compare'
]