"""
agent - Reinforcement Learning Agents

A package for training and evaluating reinforcement learning agents for different tasks.

Usage:
    from agent import train_from_config, train_from_dict

    # Train agent
    run_artifact_dir = train_from_config("config_name")
    run_artifact_dir = train_from_dict(exp_conf: Dict[str, Any])

"""

__version__ = '0.1.0'

# Import main functionality for easier access
from packages.agent.a2c_trainer import train_from_config
from packages.agent.a2c_trainer import train_from_dict

# Define public API
__all__ = [
    'train_from_config',
    'train_from_dict'
]
