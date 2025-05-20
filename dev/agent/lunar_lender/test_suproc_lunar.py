import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import os
import io
import glob
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from collections import deque
import time
import sys
from typing import Any, Dict, Tuple, Union


class TrainingVisualizationCallback(BaseCallback):
    """
    Custom callback for visualizing training progress.
    Creates and updates plots for key metrics during training.
    """
    def __init__(
        self,
        log_dir: str,
        window_size: int = 100,
        plot_freq: int = 1000,
        verbose: int = 0
    ):
        super().__init__(verbose)
        self.log_dir = log_dir
        self.window_size = window_size
        self.plot_freq = plot_freq
        self.metrics = {
            'episode_rewards': deque(maxlen=window_size),  # Stores (timestep, value)
            'episode_lengths': deque(maxlen=window_size), # Stores (timestep, value)
            'value_losses': deque(maxlen=window_size),   # Stores (timestep, value)
            'policy_losses': deque(maxlen=window_size),  # Stores (timestep, value)
            'entropy': deque(maxlen=window_size)         # Stores (timestep, value)
        }
        self.start_time = time.time()
        self.has_data = False  # Flag to track if we have collected any data

    def _on_step(self) -> bool:
        if self.n_calls > 0 and self.n_calls % self.plot_freq == 0:
            self._update_metrics()
            if self.has_data:  # Only update plots if we have data
                self._update_plots()
        return True

    def _update_metrics(self):
        """Update metrics from the training environment's logger."""
        if hasattr(self.model, 'logger') and hasattr(self.model.logger, 'name_to_value'):
            logged_metrics = self.model.logger.name_to_value
            current_timestep = self.n_calls

            def try_append_metric(log_key, metric_name):
                value = logged_metrics.get(log_key)
                if value is not None and not np.isnan(float(value)): # Ensure value is float for np.isnan
                    self.metrics[metric_name].append((current_timestep, float(value)))

            try_append_metric('rollout/ep_rew_mean', 'episode_rewards')
            try_append_metric('rollout/ep_len_mean', 'episode_lengths')
            try_append_metric('train/value_loss', 'value_losses')
            try_append_metric('train/policy_loss', 'policy_losses')
            try_append_metric('train/entropy', 'entropy_loss')


class MLflowOutputFormat(KVWriter):
    """
    Dumps key/value pairs into MLflow's numeric format.
    """
    def write(
        self,
        key_values: Dict[str, Any],
        key_excluded: Dict[str, Union[str, Tuple[str, ...]]],
        step: int = 0,
    ) -> None:
        for (key, value), (_, excluded) in zip(
            sorted(key_values.items()), sorted(key_excluded.items())
        ):
            if excluded is not None and "mlflow" in excluded:
                continue

            if isinstance(value, np.ScalarType):
                if not isinstance(value, str):
                    mlflow.log_metric(key, value, step)


class ExpConf:
    EXPERIMENT_NAME = "lunar_lander_a2c_v2"
    # Environment
    ENV_ID = "LunarLander-v2"
    SEED = 31
    NUM_THREADS = 12
    # Model
    MODEL_ID = "A2C"
    POLICY = "MlpPolicy"
    # Training
    TOTAL_TIMESTEPS = 300000
    LEARNING_RATE = 0.001
    N_STEPS = 12
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    ENT_COEF = 0.01


if __name__ == '__main__':
    # Configure MLFlow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        experiment = mlflow.get_experiment_by_name(ExpConf.EXPERIMENT_NAME)
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment with id: {experiment_id}")
    except:
        experiment_id = mlflow.create_experiment(ExpConf.EXPERIMENT_NAME)
        print(f"Created new experiment with id: {experiment_id}")

    # Run training and evaluation with MLFlow logging
    with mlflow.start_run(experiment_id=experiment_id, run_name="a2c_training") as run:
        run_id = run.info.run_id
        mlflow.log_params(ExpConf.__dict__)

        # Create directories
        run_artifact_dir = f"/app/data/models/mlflow_experiments/{experiment_id}/{run_id}/artifacts"
        os.makedirs(run_artifact_dir, exist_ok=True)
        best_model_dir = f"{run_artifact_dir}/best_model"
        final_model_dir = f"{run_artifact_dir}/final_model"
        log_dir = f"{run_artifact_dir}/logs"  # Directory for logs and plots
        os.makedirs(log_dir, exist_ok=True)


        # Create environments
        train_env = make_vec_env(ExpConf.ENV_ID, n_envs=ExpConf.NUM_THREADS, seed=ExpConf.SEED, vec_env_cls=SubprocVecEnv)
        eval_env = Monitor(gym.make(ExpConf.ENV_ID))

        # Setup callbacks
        eval_callback = EvalCallback(
            eval_env,
            best_model_save_path=run_artifact_dir,
            log_path=log_dir,
            eval_freq=5000,
            deterministic=True,
            render=False,
            verbose=1
        )

        # Add visualization callback
        vis_callback = TrainingVisualizationCallback(
            log_dir=log_dir,
            window_size=100,
            plot_freq=10000
        )

        # Create model with custom MLFlow logger
        model = A2C(
            ExpConf.POLICY,
            train_env,
            learning_rate=ExpConf.LEARNING_RATE,
            n_steps=ExpConf.N_STEPS,
            gamma=ExpConf.GAMMA,
            gae_lambda=ExpConf.GAE_LAMBDA,
            ent_coef=ExpConf.ENT_COEF,
            verbose=1
        )

        # Set custom logger for MLFlow integration
        loggers = Logger(
            folder=None, # No disk logging from here, MLflow handles it
            output_formats=[MLflowOutputFormat()], # Only MLflow for console/programmatic output
        )
        model.set_logger(loggers)

        # Train model with both callbacks
        print("Starting training")
        model.learn(
            total_timesteps=ExpConf.TOTAL_TIMESTEPS,
            callback=[eval_callback, vis_callback]
        )

        # Save model and log it to MLFlow
        model.save(final_model_dir)
        # Final evaluation
        print("\nPerforming final evaluation...")
        mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10, render=False)
        mlflow.log_metrics({
            "final_mean_reward": mean_reward,
            "final_std_reward": std_reward
        })

        print(f"\nTraining complete!")
        print(f"Final mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")
