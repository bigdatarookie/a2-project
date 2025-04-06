"""
Training module for custom agents and environments with logging to mlflow.
"""
# Import libraries
import os
import time
import numpy as np
import mlflow
from dotenv import load_dotenv
from typing import Dict, Any, List, Optional, Tuple
import json
import shutil
from packages.agent.agents import A2CAgent
from packages.agent.custom_envs import MultiTickerPortfolioEnv

# Load environment variables
load_dotenv()

# Get data directory
DATA_DIR = os.getenv('DATA_DIR', 'data')
MODEL_DIR = os.path.join(DATA_DIR, 'models')
RUNS_DIR = os.path.join(DATA_DIR, 'runs')
ARTIFACTS_DIR = os.path.join(DATA_DIR, 'artifacts')

# Ensure directories exist
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(RUNS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

def setup_mlflow(experiment_name: str) -> str:
    """
    Setup mlflow tracking for experiment.

    Args:
        experiment_name: Name of the experiment

    Returns:
        run_id: The ID of the created mlflow run
    """
    # Set tracking URI to local file system
    mlflow.set_tracking_uri(f"file:{RUNS_DIR}")

    # Get or create experiment
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id

    # Start new run
    run = mlflow.start_run(experiment_id=experiment_id)
    return run.info.run_id

def log_metrics(metrics: Dict[str, float], step: int) -> None:
    """
    Log metrics to mlflow.

    Args:
        metrics: Dictionary of metrics to log
        step: Current training step
    """
    mlflow.log_metrics(metrics, step=step)

def log_parameters(params: Dict[str, Any]) -> None:
    """
    Log parameters to mlflow.

    Args:
        params: Dictionary of parameters to log
    """
    mlflow.log_params(params)

def save_model(agent: A2CAgent, model_name: str, version: str) -> str:
    """
    Save trained model with versioning.

    Args:
        agent: Trained agent
        model_name: Name of the model
        version: Version of the model

    Returns:
        model_path: Path to the saved model
    """
    # Create model directory with versioning
    model_dir = os.path.join(MODEL_DIR, model_name, version)
    os.makedirs(model_dir, exist_ok=True)

    # Save model artifacts
    model_path = os.path.join(model_dir, "model.h5")
    agent.save(model_path)

    # Log model to mlflow
    mlflow.log_artifact(model_path)

    # Save model configuration
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'w') as f:
        json.dump(agent.get_config(), f)

    mlflow.log_artifact(config_path)

    return model_path

def evaluate_agent(agent: A2CAgent, env: MultiTickerPortfolioEnv,
                  n_episodes: int = 5) -> Dict[str, float]:
    """
    Evaluate agent on test environment.

    Args:
        agent: Agent to evaluate
        env: Test environment
        n_episodes: Number of episodes to evaluate

    Returns:
        metrics: Dictionary of evaluation metrics
    """
    total_rewards = []
    portfolio_values = []

    for episode in range(n_episodes):
        state = env.reset()
        done = False
        episode_reward = 0

        while not done:
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            state = next_state

            if done:
                portfolio_values.append(info.get('portfolio_value', 0))

        total_rewards.append(episode_reward)

    # Calculate evaluation metrics
    metrics = {
        'mean_reward': np.mean(total_rewards),
        'std_reward': np.std(total_rewards),
        'max_reward': np.max(total_rewards),
        'mean_portfolio_value': np.mean(portfolio_values),
        'final_portfolio_value': portfolio_values[-1] if portfolio_values else 0
    }

    return metrics

def train_agent(
    env_params: Dict[str, Any],
    agent_params: Dict[str, Any],
    training_params: Dict[str, Any],
    experiment_name: str,
    model_name: str,
    version: Optional[str] = None
) -> Tuple[A2CAgent, str]:
    """
    Train an agent with mlflow tracking.

    Args:
        env_params: Parameters for environment creation
        agent_params: Parameters for agent creation
        training_params: Parameters for training process
        experiment_name: Name of the experiment for mlflow
        model_name: Name of the model for saving
        version: Version string (defaults to timestamp if None)

    Returns:
        agent: Trained agent
        model_path: Path to saved model
    """
    # Generate version if not provided
    if version is None:
        version = f"v_{int(time.time())}"

    # Setup mlflow tracking
    run_id = setup_mlflow(experiment_name)

    # Create environment
    env = MultiTickerPortfolioEnv(**env_params)

    # Create evaluation environment (with different seed)
    eval_env_params = env_params.copy()
    eval_env_params['seed'] = env_params.get('seed', 42) + 100
    eval_env = MultiTickerPortfolioEnv(**eval_env_params)

    # Get state dimensions from environment
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0] if hasattr(env.action_space, 'shape') else env.action_space.n

    # Update agent params with state and action dimensions
    agent_params.update({
        'state_dim': state_dim,
        'action_dim': action_dim
    })

    # Create agent
    agent = A2CAgent(**agent_params)

    # Extract training parameters
    n_episodes = training_params.get('n_episodes', 1000)
    max_steps = training_params.get('max_steps', 1000)
    eval_frequency = training_params.get('eval_frequency', 50)

    # Log all parameters
    all_params = {
        **env_params,
        **agent_params,
        **training_params,
        'model_name': model_name,
        'version': version
    }
    log_parameters(all_params)

    # Training loop
    total_steps = 0

    for episode in range(n_episodes):
        state = env.reset()
        episode_reward = 0
        done = False
        step = 0

        while not done and step < max_steps:
            # Agent selects action
            action = agent.act(state)

            # Environment step
            next_state, reward, done, info = env.step(action)

            # Agent learns
            agent.learn(state, action, reward, next_state, done)

            # Update state and counters
            state = next_state
            episode_reward += reward
            step += 1
            total_steps += 1

        # Log episode metrics
        episode_metrics = {
            'episode_reward': episode_reward,
            'episode_length': step
        }

        if info:
            for key, value in info.items():
                if isinstance(value, (int, float)):
                    episode_metrics[f'info_{key}'] = value

        log_metrics(episode_metrics, episode)

        # Periodic evaluation
        if episode % eval_frequency == 0 or episode == n_episodes - 1:
            eval_metrics = evaluate_agent(agent, eval_env)

            # Add prefix to evaluation metrics
            eval_metrics_prefixed = {f'eval_{k}': v for k, v in eval_metrics.items()}
            log_metrics(eval_metrics_prefixed, episode)

            print(f"Episode {episode}/{n_episodes}: Reward={episode_reward:.2f}, "
                  f"Eval Reward={eval_metrics['mean_reward']:.2f}")

    # Save the trained model
    model_path = save_model(agent, model_name, version)

    # End mlflow run
    mlflow.end_run()

    return agent, model_path

def load_agent(model_name: str, version: str) -> A2CAgent:
    """
    Load a trained agent from disk.

    Args:
        model_name: Name of the model
        version: Version of the model

    Returns:
        agent: Loaded agent
    """
    model_dir = os.path.join(MODEL_DIR, model_name, version)

    # Load configuration
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, 'r') as f:
        config = json.load(f)

    # Create agent with loaded configuration
    agent = A2CAgent(**config)

    # Load model weights
    model_path = os.path.join(model_dir, "model.h5")
    agent.load(model_path)

    return agent

def get_model_versions(model_name: str) -> List[str]:
    """
    Get list of available versions for a model.

    Args:
        model_name: Name of the model

    Returns:
        versions: List of available versions
    """
    model_dir = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_dir):
        return []

    versions = [d for d in os.listdir(model_dir)
                if os.path.isdir(os.path.join(model_dir, d))]

    return sorted(versions)

def delete_model(model_name: str, version: Optional[str] = None) -> None:
    """
    Delete a model version or all versions of a model.

    Args:
        model_name: Name of the model
        version: Specific version to delete (if None, deletes all versions)
    """
    model_dir = os.path.join(MODEL_DIR, model_name)

    if not os.path.exists(model_dir):
        print(f"Model {model_name} does not exist.")
        return

    if version is None:
        # Delete entire model directory
        shutil.rmtree(model_dir)
        print(f"Deleted all versions of model {model_name}.")
    else:
        # Delete specific version
        version_dir = os.path.join(model_dir, version)
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
            print(f"Deleted version {version} of model {model_name}.")
        else:
            print(f"Version {version} of model {model_name} does not exist.")
