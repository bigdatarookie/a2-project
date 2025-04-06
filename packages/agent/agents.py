"""
Custom agents for portfolio management.
"""
# Import libraries
import logging
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from packages.agent.custom_envs import MultiTickerPortfolioEnv
from stable_baselines3.common.vec_env import VecEnv
from typing import Optional, Dict, Any, Type
import os

# Configure logger
logger = logging.getLogger(__name__)


class A2CAgent:
    """
    Wrapper for the Stable Baselines3 A2C agent.

    Provides a consistent interface for training, prediction, saving, and loading.
    """
    def __init__(
        self,
        env_kwargs: Dict[str, Any],
        model_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None
    ):
        """
        Initialize the A2C agent wrapper.

        Args:
            env_kwargs (dict): Keyword arguments to pass to the MultiTickerPortfolioEnv environment.
                               Must include 'df'.
            model_kwargs (dict, optional): Keyword arguments for the A2C model. Defaults to None.
            verbose (int): Verbosity level (0: no output, 1: info).
            seed (int, optional): Random seed for reproducibility.
        """
        self.env_kwargs = env_kwargs
        self.model_kwargs = model_kwargs or {}
        self.verbose = verbose
        self.seed = seed

        # Create the vectorized environment
        self.env = self._create_env()

        # Initialize the A2C model
        self.model = A2C(
            "MlpPolicy",  # Use Multi-Layer Perceptron policy
            self.env,
            verbose=self.verbose,
            seed=self.seed,
            **self.model_kwargs
        )
        logger.info(f"A2C agent initialized with env_kwargs: {env_kwargs} and model_kwargs: {self.model_kwargs}")

    def _create_env(self) -> VecEnv:
        """Creates the vectorized environment."""
        if 'df' not in self.env_kwargs:
            raise ValueError("Environment kwargs must include 'df' (DataFrame).")

        # Use make_vec_env to automatically wrap the environment
        env = make_vec_env(
            lambda: MultiTickerPortfolioEnv(**self.env_kwargs),
            n_envs=12,
            seed=self.seed
        )
        return env

    def train(
        self,
        total_timesteps: int,
        log_interval: int = 1,
        callback: Optional[Any] = None,
        tb_log_name: str = "A2C"
    ):
        """
        Train the A2C agent.

        Args:
            total_timesteps (int): The total number of samples (env steps) to train on.
            log_interval (int): The number of episodes between logging.
            callback (callable, optional): Callback function for monitoring training.
            tb_log_name (str): The name of the run for logging.
        """
        logger.info(f"Starting A2C training for {total_timesteps} timesteps...")
        self.model.learn(
            total_timesteps=total_timesteps,
            log_interval=log_interval,
            callback=callback,
            tb_log_name=tb_log_name,
            reset_num_timesteps=False # Allow continued training
        )
        logger.info("A2C training finished.")

    def predict(self, observation: Any, deterministic: bool = True) -> Any:
        """
        Get the agent's action for a given observation.

        Args:
            observation: The current environment observation.
            deterministic (bool): Whether to return deterministic or stochastic actions.

        Returns:
            The action chosen by the agent.
        """
        action, _states = self.model.predict(observation, deterministic=deterministic)
        return action

    def save(self, path: str):
        """
        Save the trained model to a file.

        Args:
            path (str): The file path (without extension) to save the model to.
                         The '.zip' extension will be added automatically.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.model.save(path)
        logger.info(f"A2C model saved to {path}.zip")

    @classmethod
    def load(
        cls,
        path: str,
        env_kwargs: Dict[str, Any],
        device: str = "auto"
    ) -> 'A2CAgent':
        """
        Load a trained model from a file.

        Args:
            path (str): The file path (without the '.zip' extension) to load the model from.
            env_kwargs (dict): Keyword arguments for recreating the environment. Must match training.
            device (str): Device onto which the policy should be loaded ('auto', 'cpu', 'cuda', ...).

        Returns:
            An instance of A2CAgent with the loaded model.
        """
        # Create a dummy environment instance needed for loading
        # Use make_vec_env to ensure consistency
        temp_env = make_vec_env(lambda: MultiTickerPortfolioEnv(**env_kwargs), n_envs=1)

        # Load the A2C model
        loaded_a2c = A2C.load(path, env=temp_env, device=device)
        logger.info(f"A2C model loaded from {path}.zip")

        # Create a new A2CAgent instance but replace the model and env
        agent_instance = cls(env_kwargs=env_kwargs) # env_kwargs needed for init
        agent_instance.model = loaded_a2c
        agent_instance.env = loaded_a2c.get_env() # Use env associated with the loaded model

        return agent_instance
