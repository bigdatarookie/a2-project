# %% [markdown]
# # Lunar Lender
#
# This notebook implements a Lunar Lender environment using Gymnasium with a MLFlow logging configuration.

# %%
# Import libraries
import gymnasium as gym
import mlflow
import numpy as np
import pandas as pd
import os
import torch
import glob
from IPython.display import Video
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback

# %% [markdown]
# ## Setup MLFlow Tracking

# %%
# Set MLFlow tracking URI
os.environ["MLFLOW_TRACKING_URI"] = "file:///app/data/models/mlflow_experiments"
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])

# Create experiment
experiment_name = "lunar_lander_a2c"
try:
    experiment_id = mlflow.create_experiment(experiment_name)
    print(f"Created new experiment with id: {experiment_id}")
except:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    experiment_id = experiment.experiment_id
    print(f"Using existing experiment with id: {experiment_id}")

# Create base directories
base_dir = "/app/data/models/mlflow_experiments"
os.makedirs(base_dir, exist_ok=True)

# %% [markdown]
# ## Create Environment

# %%
# Create vectorized environment with parallel environments
env_id = "LunarLander-v2"
n_envs = 10

vec_env = make_vec_env(
    env_id=env_id,
    n_envs=n_envs,
    seed=0
)

# Get environment specs
observation_space = vec_env.observation_space
action_space = vec_env.action_space

print(f"Observation space: {observation_space}")
print(f"Action space: {action_space}")

# %% [markdown]
# ## Configure and Train A2C Agent

# %%
# Start MLflow run for training
with mlflow.start_run(experiment_id=experiment_id, run_name="a2c_training") as run:
    # Get run ID for storing artifacts
    run_id = run.info.run_id
    run_artifact_dir = f"/app/data/models/mlflow_experiments/{experiment_id}/{run_id}/artifacts"
    os.makedirs(run_artifact_dir, exist_ok=True)

    # Model output directories
    best_model_dir = f"{run_artifact_dir}/best_model"
    checkpoint_dir = f"{run_artifact_dir}/checkpoints"
    log_dir = f"{run_artifact_dir}/logs"

    os.makedirs(best_model_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    # A2C hyperparameters
    total_timesteps = 500000
    learning_rate = 0.0007
    n_steps = 5
    gamma = 0.99
    gae_lambda = 0.95
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5

    # Log environment parameters
    mlflow.log_param("algorithm", "A2C")
    mlflow.log_param("environment", env_id)
    mlflow.log_param("n_envs", n_envs)

    # Log hyperparameters
    mlflow.log_param("learning_rate", learning_rate)
    mlflow.log_param("n_steps", n_steps)
    mlflow.log_param("gamma", gamma)
    mlflow.log_param("gae_lambda", gae_lambda)
    mlflow.log_param("ent_coef", ent_coef)
    mlflow.log_param("vf_coef", vf_coef)
    mlflow.log_param("max_grad_norm", max_grad_norm)

    # Setup callbacks
    eval_env = gym.make(env_id)
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=5000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=10000,
        save_path=checkpoint_dir,
        name_prefix="a2c_lunar_lander"
    )

    # Create and train the agent
    model = A2C(
        policy="MlpPolicy",
        env=vec_env,
        learning_rate=learning_rate,
        n_steps=n_steps,
        gamma=gamma,
        gae_lambda=gae_lambda,
        ent_coef=ent_coef,
        vf_coef=vf_coef,
        max_grad_norm=max_grad_norm,
        verbose=1
    )

    # Log model parameters
    for param_name, param_value in model.policy.state_dict().items():
        if isinstance(param_value, torch.Tensor):
            mlflow.log_param(f"policy_{param_name}_shape", list(param_value.shape))

    # Train the agent
    mlflow.log_param("total_timesteps", total_timesteps)

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback]
    )

# %% [markdown]
# ## Evaluate Trained Agent and Record Episode

# %%
# Use the run_id from the training step to load the best model
best_model_path = f"{run_artifact_dir}/best_model/best_model.zip"
model = A2C.load(best_model_path)

print(f"Loaded model from {best_model_path}")

# Evaluate the model in a new MLflow run
with mlflow.start_run(experiment_id=experiment_id, run_name="a2c_evaluation") as eval_run:
    eval_run_id = eval_run.info.run_id
    eval_artifact_dir = f"/app/data/models/mlflow_experiments/{experiment_id}/{eval_run_id}/artifacts"
    os.makedirs(eval_artifact_dir, exist_ok=True)

    # Log which model we're evaluating
    mlflow.log_param("model_path", best_model_path)
    mlflow.log_param("training_run_id", run_id)

    # First evaluate performance to get the mean reward
    n_eval_episodes = 10
    print(f"Evaluating model performance across {n_eval_episodes} episodes...")

    # Create evaluation environment
    eval_env = gym.make(env_id)

    # Evaluate policy and get mean reward
    mean_reward, std_reward = evaluate_policy(
        model,
        eval_env,
        n_eval_episodes=n_eval_episodes,
        deterministic=True
    )

    # Log metrics
    mlflow.log_metric("mean_reward", mean_reward)
    mlflow.log_metric("std_reward", std_reward)

    print(f"Evaluation metrics - Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Create recording directory
    recording_dir = f"{eval_artifact_dir}/recordings"
    os.makedirs(recording_dir, exist_ok=True)

    # Create recording environment
    print(f"Recording a demonstration episode...")
    record_env = gym.make(env_id, render_mode="rgb_array")
    record_env = RecordVideo(
        env=record_env,
        video_folder=recording_dir,
        episode_trigger=lambda episode_id: True,  # Record the episode
        name_prefix="lunar_lander"
    )

    # Run and record a single episode
    obs, _ = record_env.reset()
    done = False
    episode_reward = 0
    step_count = 0

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = record_env.step(action)
        episode_reward += reward
        step_count += 1
        done = terminated or truncated

    record_env.close()

    # Log the episode results
    mlflow.log_metric("recorded_episode_reward", episode_reward)
    mlflow.log_metric("recorded_episode_steps", step_count)

# %% [markdown]
# ## Display Recorded Episode

# %%
# Display the recorded episode
video_files = sorted(glob.glob(f"{recording_dir}/*.mp4"))
if video_files:
    video_path = video_files[0]
    print(f"Displaying recorded episode: {video_path}")
    print(f"Episode reward: {episode_reward:.2f}")

    display(Video(video_path, embed=True, width=800, height=600))
else:
    print("No video available to display")

# %% [markdown]
# ## Areas for Improvement
#
# 1. **Hyperparameter Tuning**: Implement systematic tuning using grid search or Bayesian optimization
# 2. **Policy Network Architecture**: Experiment with deeper/wider networks or different activation functions
# 3. **Reward Engineering**: Design custom rewards to encourage specific behaviors
# 5. **Frame Stacking**: Use multiple frames as input to help with partial observability
# 6. **Curriculum Learning**: Start with easier versions of the task and gradually increase difficulty
# 7. **Ensemble Methods**: Train multiple agents and use voting or averaging for decisions
# 8. **Exploration Strategies**: Implement more sophisticated exploration like parameter noise
# 9. **Regularization Techniques**: Add entropy regularization or L2 regularization to prevent overfitting
# 10. **Custom Neural Network Architectures**: Implement attention mechanisms or residual connections

