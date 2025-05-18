# %% [markdown]
# # Lunar Lender v2
#
# This notebook implements a Lunar Lender environment using Gymnasium with a MLFlow logging configuration.

# %%
# Import libraries
import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import os
import io
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from IPython.display import clear_output, display, HTML, display_html, Video
from PIL import Image
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv

# %%
# Define experiment config
class ExperimentConfig:
    EXPERIMENT_NAME = "lunar_lander_a2c_v2"
    # Environment
    ENV_ID = "LunarLander-v2"
    SEED = 31
    NUM_THREADS = 12
    # Model
    MODEL_ID = "A2C"
    POLICY = "MlpPolicy"
    # Training
    TOTAL_TIMESTEPS = 100000
    LEARNING_RATE = 0.001
    N_STEPS = 12
    GAMMA = 0.99
    GAE_LAMBDA = 0.95
    ENT_COEF = 0.01

# %%
# Configure MLFlow
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

try:
    experiment = mlflow.get_experiment_by_name(ExperimentConfig.EXPERIMENT_NAME)
    experiment_id = experiment.experiment_id
    print(f"Using existing experiment with id: {experiment_id}")
except:
    experiment_id = mlflow.create_experiment(ExperimentConfig.EXPERIMENT_NAME)
    print(f"Created new experiment with id: {experiment_id}")

# %%
# Run training and evaluation with MLFlow logging
with mlflow.start_run(experiment_id=experiment_id, run_name="a2c_training") as run:
    run_id = run.info.run_id
    mlflow.log_params(ExperimentConfig.__dict__)

    run_artifact_dir = f"/app/data/models/mlflow_experiments/{experiment_id}/{run_id}/artifacts"
    os.makedirs(run_artifact_dir, exist_ok=True)
    best_model_dir = f"{run_artifact_dir}/best_model"
    final_model_dir = f"{run_artifact_dir}/final_model"

    # Create trainig environment
    train_env = make_vec_env(ExperimentConfig.ENV_ID, n_envs=ExperimentConfig.NUM_THREADS, seed=ExperimentConfig.SEED, vec_env_cls=DummyVecEnv)

    # Create model
    model = A2C(ExperimentConfig.POLICY, train_env)

    # Train model
    model.learn(total_timesteps=ExperimentConfig.TOTAL_TIMESTEPS)

    # Save model
    model.save(final_model_dir)

    # Evaluate model
    mean_reward, std_reward = evaluate_policy(model, train_env, n_eval_episodes=10, render=False)
    mlflow.log_metrics({
        "mean_reward": mean_reward,
        "std_reward": std_reward
    })
