import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import os
import io
import glob
import argparse
import time
import sys
import yaml
from pathlib import Path
from gymnasium.spaces import Box, Discrete
from stable_baselines3 import A2C, PPO, SAC
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from collections import deque
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

def train_from_config(config_path: str) -> str:
    config_path = Path(config_path)
    with config_path.open('r') as f:
        exp_conf = yaml.safe_load(f)
    return train_from_dict(exp_conf)

def train_from_dict(exp_conf: Dict[str, Any]) -> str:
    # Configure MLFlow
    mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
    try:
        experiment = mlflow.get_experiment_by_name(exp_conf["EXPERIMENT_NAME"])
        experiment_id = experiment.experiment_id
        print(f"Using existing experiment with id: {experiment_id}")
    except:
        experiment_id = mlflow.create_experiment(exp_conf["EXPERIMENT_NAME"])
        print(f"Created new experiment with id: {experiment_id}")

    # Resources to clean up
    train_env = None
    eval_env = None
    model = None

    # Run training and evaluation with MLFlow logging
    with mlflow.start_run(experiment_id=experiment_id, run_name="a2c_training") as run:
        try:
            run_id = run.info.run_id
            mlflow.log_params(exp_conf)

            # Create directories
            run_artifact_dir = f"/app/data/models/mlflow_experiments/{experiment_id}/{run_id}/artifacts"
            os.makedirs(run_artifact_dir, exist_ok=True)
            best_model_dir = f"{run_artifact_dir}/best_model"
            final_model_dir = f"{run_artifact_dir}/final_model"
            log_dir = f"{run_artifact_dir}/logs"  # Directory for logs and plots
            os.makedirs(log_dir, exist_ok=True)

            # Create environments
            train_env = make_vec_env(exp_conf["ENV_ID"], n_envs=exp_conf["NUM_THREADS"], seed=exp_conf["SEED"], vec_env_cls=SubprocVecEnv)
            eval_env = Monitor(gym.make(exp_conf["ENV_ID"]))

            # Setup callbacks
            eval_callback = EvalCallback(
                eval_env,
                best_model_save_path=run_artifact_dir,
                log_path=log_dir,
                eval_freq=5000,
                deterministic=True,
                render=False
            )

            # Add visualization callback
            vis_callback = TrainingVisualizationCallback(
                log_dir=log_dir,
                window_size=100,
                plot_freq=5000
            )

            # Create model with custom MLFlow logger
            if exp_conf["MODEL_ID"] == "A2C" :
                model = A2C(
                    exp_conf["POLICY"],
                    train_env,
                    learning_rate=exp_conf["LEARNING_RATE"],
                    n_steps=exp_conf["N_STEPS"],
                    gamma=exp_conf["GAMMA"],
                    gae_lambda=exp_conf["GAE_LAMBDA"],
                    ent_coef=exp_conf["ENT_COEF"],
                    vf_coef=exp_conf.get("VF_COEF", 0.5),  # Default if not in config
                    max_grad_norm=exp_conf.get("MAX_GRAD_NORM", 0.5),  # Default if not in config
                    normalize_advantage=exp_conf.get("NORMALIZE_ADVANTAGE", True), # Default if not in config
                    policy_kwargs=exp_conf.get("POLICY_KWARGS", None) # Default if not in config
                )
            elif exp_conf["MODEL_ID"] == "PPO" :
                model = PPO(
                    exp_conf["POLICY"],
                    train_env,
                    learning_rate=exp_conf["LEARNING_RATE"],
                    n_steps=exp_conf["N_STEPS"],
                    batch_size=exp_conf["BATCH_SIZE"],
                    n_epochs=exp_conf.get("N_EPOCHS", 10),
                    gamma=exp_conf["GAMMA"],
                    gae_lambda=exp_conf["GAE_LAMBDA"],
                    clip_range=exp_conf.get("CLIP_RANGE", 0.2),
                    clip_range_vf=exp_conf.get("CLIP_RANGE_VF", None),
                    normalize_advantage=exp_conf.get("NORMALIZE_ADVANTAGE", True),
                    ent_coef=exp_conf["ENT_COEF"],
                    vf_coef=exp_conf["VF_COEF"],
                    max_grad_norm=exp_conf.get("MAX_GRAD_NORM", 0.5),
                    use_sde=exp_conf.get("USE_SDE", False),
                    sde_sample_freq=exp_conf.get("SDE_SAMPLE_FREQ", -1),
                    rollout_buffer_class=exp_conf.get("ROLLOUT_BUFFER_CLASS", None),
                    rollout_buffer_kwargs=exp_conf.get("ROLLOUT_BUFFER_KWARGS", None),
                    target_kl=exp_conf.get("TARGET_KL", None),
                    policy_kwargs=exp_conf.get("POLICY_KWARGS", None),
                    seed=exp_conf.get("SEED", None)
                )
            elif exp_conf["MODEL_ID"] == "SAC" :
                model = SAC(
                    exp_conf["POLICY"],
                    train_env,
                    learning_rate=exp_conf.get("LEARNING_RATE", 0.0003),
                    buffer_size=exp_conf.get("BUFFER_SIZE", 1000000),
                    learning_starts=exp_conf.get("LEARNING_STARTS", 100),
                    batch_size=exp_conf.get("BATCH_SIZE", 256),
                    tau=exp_conf.get("TAU", 0.005),
                    gamma=exp_conf.get("GAMMA", 0.99),
                    train_freq=exp_conf.get("TRAIN_FREQ", 1),
                    gradient_steps=exp_conf.get("GRADIENT_STEPS", 1),
                    action_noise=exp_conf.get("ACTION_NOISE", None),
                    replay_buffer_class=exp_conf.get("REPLAY_BUFFER_CLASS", None),
                    replay_buffer_kwargs=exp_conf.get("REPLAY_BUFFER_KWARGS", None),
                    optimize_memory_usage=exp_conf.get("OPTIMIZE_MEMORY_USAGE", False),
                    ent_coef=exp_conf.get("ENT_COEF", "auto"),
                    target_update_interval=exp_conf.get("TARGET_UPDATE_INTERVAL", 1),
                    target_entropy=exp_conf.get("TARGET_ENTROPY", "auto"),
                    use_sde=exp_conf.get("USE_SDE", False),
                    sde_sample_freq=exp_conf.get("SDE_SAMPLE_FREQ", -1),
                    use_sde_at_warmup=exp_conf.get("USE_SDE_AT_WARMUP", False),
                    policy_kwargs=exp_conf.get("POLICY_KWARGS", None),
                    seed=exp_conf.get("SEED", None),
                    device=exp_conf.get("DEVICE", "auto"),
                    _init_setup_model=exp_conf.get("_INIT_SETUP_MODEL", True)
                )
            else :
                raise ValueError(f"Model {exp_conf['MODEL_ID']} not supported")

            # Set custom logger for MLFlow integration
            loggers = Logger(
                folder=None, # No disk logging from here, MLflow handles it
                output_formats=[MLflowOutputFormat()], # Only MLflow for console/programmatic output
            )
            model.set_logger(loggers)

            # Train model with both callbacks
            print("Starting training")
            model.learn(
                total_timesteps=exp_conf["TOTAL_TIMESTEPS"],
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

            return str(run_artifact_dir)
        finally:
            # Clean up resources
            print("Cleaning up resources...")
            if train_env is not None:
                train_env.close()
            if eval_env is not None:
                eval_env.close()
            # Free model resources if needed
            if model is not None:
                del model
