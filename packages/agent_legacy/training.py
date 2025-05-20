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
from contextlib import contextmanager
import traceback
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

def json_serializable(value):
    """Check if a value can be serialized to JSON."""
    try:
        json.dumps(value)
        return True
    except (TypeError, OverflowError):
        return False

class TrainingManager:
    """
    Class for managing agent training, evaluation, and model versioning
    with MLflow integration for tracking experiments.
    """

    def __init__(self, experiment_name: str, model_name: str):
        """
        Initialize the training manager.

        Args:
            experiment_name: Name of the experiment for MLflow
            model_name: Name of the model for saving
        """
        self.experiment_name = experiment_name
        self.model_name = model_name
        self.run_id = None
        self.active_run = None

        # Set MLflow tracking URI
        mlflow.set_tracking_uri(f"file:{RUNS_DIR}")

        # Get or create experiment
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            self.experiment_id = mlflow.create_experiment(experiment_name)
        else:
            self.experiment_id = experiment.experiment_id

    @contextmanager
    def _mlflow_run_context(self, run_name: Optional[str] = None):
        """
        Context manager for MLflow runs to ensure proper cleanup.

        Args:
            run_name: Optional name for the run
        """
        try:
            self.active_run = mlflow.start_run(
                experiment_id=self.experiment_id,
                run_name=run_name
            )
            self.run_id = self.active_run.info.run_id
            yield self.active_run
        except Exception as e:
            print(f"Error in MLflow run: {str(e)}")
            traceback.print_exc()
            raise
        finally:
            if mlflow.active_run():
                mlflow.end_run()
                self.active_run = None

    def log_metrics(self, metrics: Dict[str, float], step: int) -> None:
        """
        Log metrics to MLflow.

        Args:
            metrics: Dictionary of metrics to log
            step: Current training step
        """
        if mlflow.active_run():
            mlflow.log_metrics(metrics, step=step)
        else:
            print("Warning: Attempted to log metrics without active run")

    def log_parameters(self, params: Dict[str, Any]) -> None:
        """
        Log parameters to MLflow.

        Args:
            params: Dictionary of parameters to log
        """
        if mlflow.active_run():
            mlflow.log_params(params)
        else:
            print("Warning: Attempted to log parameters without active run")

    def save_model(self, agent: A2CAgent, version: str) -> str:
        """
        Save trained model with versioning.

        Args:
            agent: Trained agent
            version: Version of the model

        Returns:
            model_path: Path to the saved model
        """
        # Create model directory with versioning
        model_dir = os.path.join(MODEL_DIR, self.model_name, version)
        os.makedirs(model_dir, exist_ok=True)

        # Save model artifacts - use the model's save method
        model_path = os.path.join(model_dir, "model")
        agent.save(model_path)

        # Save environment configuration for reproducibility
        env_config_path = os.path.join(model_dir, "env_config.json")

        # Get env_kwargs from agent using get_config to ensure we get the actual config used
        config = agent.get_config()
        env_kwargs = config.get('env_kwargs', {})

        # Handle non-serializable objects in env_kwargs (like DataFrame)
        serializable_env_kwargs = {}
        for key, value in env_kwargs.items():
            # Skip large/complex objects that shouldn't be serialized
            if key == 'df':
                # For DataFrame, just save metadata
                if hasattr(value, 'shape'):
                    serializable_env_kwargs[f"{key}_metadata"] = {
                        'shape': list(value.shape),
                        'columns': list(value.columns) if hasattr(value, 'columns') else None
                    }
            else:
                # Try to serialize, skip if not serializable
                try:
                    json.dumps({key: value})
                    serializable_env_kwargs[key] = value
                except (TypeError, OverflowError):
                    print(f"Warning: Could not serialize '{key}' in env_kwargs")
                    serializable_env_kwargs[f"{key}_type"] = str(type(value))

        with open(env_config_path, 'w') as f:
            json.dump(serializable_env_kwargs, f)

        # Log artifacts to MLflow if active run exists
        if mlflow.active_run():
            mlflow.log_artifact(f"{model_path}.zip")  # SB3 adds .zip extension
            mlflow.log_artifact(env_config_path)

            # Also log model configuration
            model_config_path = os.path.join(model_dir, "model_config.json")
            with open(model_config_path, 'w') as f:
                # Save other configuration items except env_kwargs which we handled separately
                serializable_config = {
                    k: v for k, v in config.items()
                    if k != 'env_kwargs' and json_serializable(v)
                }
                json.dump(serializable_config, f)

            mlflow.log_artifact(model_config_path)

        return model_path

    def evaluate_agent(self, agent: A2CAgent, env: MultiTickerPortfolioEnv,
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
            # Handle gym vs gymnasium API differences
            reset_result = env.reset()
            if isinstance(reset_result, tuple):
                # New Gymnasium API returns (obs, info)
                state, _ = reset_result
            else:
                # Old Gym API just returns obs
                state = reset_result

            done = False
            episode_reward = 0

            while not done:
                action = agent.predict(state, deterministic=True)

                # Handle step return properly
                step_result = env.step(action)
                if len(step_result) == 4:
                    # Old gym: next_state, reward, done, info
                    next_state, reward, done, info = step_result
                    truncated = False
                else:
                    # New gymnasium: next_state, reward, terminated, truncated, info
                    next_state, reward, terminated, truncated, info = step_result
                    done = terminated or truncated

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
        self,
        env_params: Dict[str, Any],
        agent_params: Dict[str, Any],
        training_params: Dict[str, Any],
        version: Optional[str] = None,
        run_name: Optional[str] = None
    ) -> Tuple[A2CAgent, str]:
        """
        Train an agent with MLflow tracking using SB3's built-in functionality.

        Args:
            env_params: Parameters for environment creation
            agent_params: Parameters for agent creation
            training_params: Parameters for training process
            version: Version string (defaults to timestamp if None)
            run_name: Optional name for the MLflow run

        Returns:
            agent: Trained agent
            model_path: Path to saved model
        """
        # Generate version if not provided
        if version is None:
            version = f"v_{int(time.time())}"

        # Log parameters
        all_params = {
            'model_name': self.model_name,
            'version': version
        }

        # Add flattened parameters with prefixes
        for prefix, param_dict in [
            ('env_', env_params),
            ('agent_', agent_params),
            ('train_', training_params)
        ]:
            for k, v in param_dict.items():
                try:
                    # Only log serializable parameters
                    json.dumps({k: v})
                    all_params[f"{prefix}{k}"] = v
                except (TypeError, OverflowError):
                    pass  # Skip non-serializable values

        try:
            with self._mlflow_run_context(run_name):
                # Log parameters
                self.log_parameters(all_params)

                # Extract training parameters
                total_timesteps = training_params.get('n_episodes', 1000) * training_params.get('max_steps', 1000)
                eval_freq = training_params.get('eval_frequency', 50) * 100  # Convert to steps

                # Create a single environment for evaluation
                eval_env = MultiTickerPortfolioEnv(**env_params)

                # Extract agent parameters
                verbose = agent_params.pop('verbose', 0)
                seed = agent_params.pop('seed', None)

                # Create agent
                structured_agent_params = {
                    'env_kwargs': env_params,
                    'model_kwargs': agent_params,
                    'verbose': verbose,
                    'seed': seed
                }

                # Create agent with its own vectorized environment
                agent = A2CAgent(**structured_agent_params)

                # Import SB3 callbacks
                from stable_baselines3.common.callbacks import CallbackList, EvalCallback
                from stable_baselines3.common.vec_env import DummyVecEnv

                # Create a vectorized eval env for SB3's EvalCallback
                eval_vec_env = DummyVecEnv([lambda: MultiTickerPortfolioEnv(**env_params)])

                # Create SB3's built-in evaluation callback
                eval_callback = EvalCallback(
                    eval_vec_env,
                    best_model_save_path=os.path.join(MODEL_DIR, self.model_name, f"{version}_best"),
                    log_path=os.path.join(RUNS_DIR, self.model_name),
                    eval_freq=eval_freq,
                    deterministic=True,
                    render=False,
                    verbose=1
                )

                # Combine callbacks
                callback = eval_callback

                # Train the agent
                print(f"Starting training for {total_timesteps} total steps")
                agent.train(
                    total_timesteps=total_timesteps,
                    callback=callback,
                    log_interval=max(1, total_timesteps // 20)  # Print ~20 logs during training
                )

                # Save the final model
                model_path = os.path.join(MODEL_DIR, self.model_name, version, "final_model")
                os.makedirs(os.path.dirname(model_path), exist_ok=True)
                agent.save(model_path)

                # Log the final model
                mlflow.log_artifact(f"{model_path}.zip")

                return agent, model_path

        except Exception as e:
            print(f"Error during training: {str(e)}")
            traceback.print_exc()
            raise

    def load_agent(self, version: str) -> A2CAgent:
        """
        Load a trained agent from disk.

        Args:
            version: Version of the model

        Returns:
            agent: Loaded agent
        """
        model_dir = os.path.join(MODEL_DIR, self.model_name, version)

        # Check if model directory exists
        if not os.path.exists(model_dir):
            raise FileNotFoundError(f"Model version {version} not found at {model_dir}")

        # Load environment configuration
        env_config_path = os.path.join(model_dir, "env_config.json")

        # If serialized config exists, load it
        if os.path.exists(env_config_path):
            with open(env_config_path, 'r') as f:
                serialized_env_kwargs = json.load(f)

            # Reconstruct env_kwargs - this is partial, the DataFrame needs to be provided again
            env_kwargs = serialized_env_kwargs

            # Check for metadata on skipped objects
            if 'df_metadata' in env_kwargs:
                print("DataFrame metadata found but actual data needs to be provided again.")
                print(f"Expected DataFrame with shape: {env_kwargs['df_metadata'].get('shape')}")
                print(f"And columns: {env_kwargs['df_metadata'].get('columns')}")

                # Remove the metadata to avoid confusion
                env_kwargs.pop('df_metadata', None)
        else:
            # No config available, will need to provide it
            print("Warning: No environment configuration found. You'll need to provide all parameters.")
            env_kwargs = {}

        # Check if model file exists
        model_path = os.path.join(model_dir, "model.zip")
        if not os.path.exists(model_path):
            model_path = os.path.join(model_dir, "model")  # Try without .zip
            if not os.path.exists(model_path + ".zip"):
                raise FileNotFoundError(f"Model file not found at {model_path}.zip")

        # Use the A2CAgent.load class method to load the agent
        try:
            agent = A2CAgent.load(
                path=model_path,
                env_kwargs=env_kwargs
            )
            return agent
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            print("You may need to provide the correct environment parameters manually.")
            traceback.print_exc()
            raise

    def get_model_versions(self) -> List[str]:
        """
        Get list of available versions for this model.

        Returns:
            versions: List of available versions
        """
        model_dir = os.path.join(MODEL_DIR, self.model_name)

        if not os.path.exists(model_dir):
            return []

        versions = [d for d in os.listdir(model_dir)
                    if os.path.isdir(os.path.join(model_dir, d))]

        return sorted(versions)

    def delete_model_version(self, version: str) -> bool:
        """
        Delete a specific version of this model.

        Args:
            version: Specific version to delete

        Returns:
            success: Whether deletion was successful
        """
        version_dir = os.path.join(MODEL_DIR, self.model_name, version)
        if os.path.exists(version_dir):
            shutil.rmtree(version_dir)
            print(f"Deleted version {version} of model {self.model_name}.")
            return True
        else:
            print(f"Version {version} of model {self.model_name} does not exist.")
            return False

    def delete_all_versions(self) -> bool:
        """
        Delete all versions of this model.

        Returns:
            success: Whether deletion was successful
        """
        model_dir = os.path.join(MODEL_DIR, self.model_name)

        if not os.path.exists(model_dir):
            print(f"Model {self.model_name} does not exist.")
            return False

        # Delete entire model directory
        shutil.rmtree(model_dir)
        print(f"Deleted all versions of model {self.model_name}.")
        return True
