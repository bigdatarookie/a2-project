# %% [markdown]
# # Lunar Lender v3
#
# This notebook implements a Lunar Lender environment using Gymnasium with a MLFlow logging configuration and automatic hyperparameter tuning.

# %%
# Import libraries
import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import os
import io
import glob
import matplotlib.pyplot as plt
import time
import sys
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
from stable_baselines3.common.logger import HumanOutputFormat, KVWriter, Logger
from collections import deque
from typing import Any, Dict, Tuple, Union
from contextlib import redirect_stdout
from agent import train_from_config, train_from_dict


# %%
# Create and display episode recording function
def record_episode(model, run_artifact_dir, env=None, episode_num=None):
    """Record a single episode and return reward and length.
    If env is provided, uses it instead of creating a new one.
    If episode_num is provided, uses it in the video name."""
    recording_dir = f"{run_artifact_dir}/recordings"
    os.makedirs(recording_dir, exist_ok=True)

    if env is None:
        # Create new environment if none provided
        env = gym.make(ExpConf.ENV_ID, render_mode="rgb_array")

    with open(os.devnull, 'w') as f, redirect_stdout(f):
        # Wrap with RecordVideo
        record_env = RecordVideo(
            env=env,
            video_folder=recording_dir,
            episode_trigger=lambda episode_id: True,  # Record the episode
            name_prefix=f"lunar_lander_episode_{episode_num if episode_num is not None else 'test'}"
        )

        # Run and record a single episode
        obs, _ = record_env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = record_env.step(action)
            episode_reward += reward
            episode_length += 1
            done = terminated or truncated
        record_env.close()

    return episode_reward, episode_length, recording_dir

# %%
# Run evaluation from best_model_dir and record all episodes
run_dir = f"/app/data/models/mlflow_experiments/510490392254633461/7d53d02e25c14ecfab1e92143703e700/artifacts"
# run_dir = run_artifact_dir
# Create environment with render mode for recording
eval_env = gym.make("LunarLander-v2", render_mode="rgb_array")
eval_env = Monitor(eval_env)

# Load model
model = A2C.load(f"{run_dir}/best_model")

# Lists to store episode metrics
episode_rewards = []
episode_lengths = []
n_eval_episodes = 10

# Evaluate and record each episode
for episode in range(n_eval_episodes):
    reward, length, _ = record_episode(model, run_dir, env=eval_env, episode_num=episode)
    episode_rewards.append(reward)
    episode_lengths.append(length)

# Create DataFrame with metrics
metrics_df = pd.DataFrame({
    'Episode': range(1, n_eval_episodes + 1),
    'Reward': episode_rewards,
    'Length': episode_lengths
})

# Add summary statistics
summary_df = pd.DataFrame({
    'Episode': ['Mean', 'Std'],
    'Reward': [metrics_df['Reward'].mean(), metrics_df['Reward'].std()],
    'Length': [metrics_df['Length'].mean(), metrics_df['Length'].std()]
})

# Display metrics table with summary
print("\nEvaluation Results:")
print(metrics_df.to_string(index=False))
print("\nSummary Statistics:")
print(summary_df.to_string(index=False))

# %%
# Display the selected episode's recording
episode_to_show = 4

video_path = glob.glob(f"{run_dir}/recordings/*episode_{episode_to_show-1}*.mp4")[0]
display(Video(video_path, embed=True, width=800, height=600))
print(f"Episode {episode_to_show} reward: {episode_rewards[episode_to_show-1]:.2f}")
print(f"Episode {episode_to_show} length: {episode_lengths[episode_to_show-1]}")
