# ## This file contains simple A2C agent training with MLflow tracking

# %%
import mlflow
import gymnasium as gym
from stable_baselines3 import A2C
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines3.common.env_util import make_vec_env
import numpy as np
import os
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

# %% Create or set the MLflow experiment
experiment_name = "A2C_Acrobot"
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
run_name = f"a2c_acrobot_{timestamp}"

# Creating base_dir variable from MLFLOW_ARTIFACT_ROOT environment variable
base_dir = os.getenv("MLFLOW_ARTIFACT_ROOT")

# Create a unique directory for this experiment run
experiment_dir = os.path.join(base_dir, experiment_name, timestamp)
os.makedirs(experiment_dir, exist_ok=True)

# Set up the experiment with a specific artifact location
artifact_location = os.path.abspath(experiment_dir)
mlflow.set_experiment(experiment_name)

# %% Start a new MLflow run
with mlflow.start_run(run_name=run_name):
    # Log parameters
    hyperparams = {
        "policy_type": "MlpPolicy",
        "env_id": "Acrobot-v1",
        "learning_rate": 7e-4,
        "n_steps": 5,
        "gamma": 0.99,
        "ent_coef": 0.01,
        "total_timesteps": 100000,
        "n_envs": 4  # Number of parallel environments
    }

    mlflow.log_params(hyperparams)

    # Create a specific log directory within the experiment directory
    log_dir = os.path.join(experiment_dir, "logs")
    os.makedirs(log_dir, exist_ok=True)

    # Use DummyVecEnv instead of SubprocVecEnv to avoid path resolution issues in Docker
    vec_env = make_vec_env(
        env_id=hyperparams["env_id"],
        n_envs=hyperparams["n_envs"],
        vec_env_cls=DummyVecEnv,  # Changed from SubprocVecEnv to DummyVecEnv
        monitor_dir=log_dir
    )

    # Create single environment for evaluation
    eval_env = Monitor(gym.make(hyperparams["env_id"]), os.path.join(log_dir, "eval"))

    # Create the model
    model = A2C(
        hyperparams["policy_type"],
        vec_env,
        learning_rate=hyperparams["learning_rate"],
        n_steps=hyperparams["n_steps"],
        gamma=hyperparams["gamma"],
        ent_coef=hyperparams["ent_coef"],
        verbose=1
    )

    # Set up callbacks to log metrics during training
    from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

    # Evaluation callback
    eval_freq = 5000  # Evaluate every 5000 steps
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=log_dir,
        log_path=log_dir,
        eval_freq=eval_freq,
        deterministic=True,
        render=False
    )

    # Checkpoint callback
    checkpoint_freq = 10000
    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=log_dir,
        name_prefix="a2c_model"
    )

    # Train the model
    model.learn(
        total_timesteps=hyperparams["total_timesteps"],
        callback=[eval_callback, checkpoint_callback]
    )

    # Save the final model
    final_model_path = os.path.join(log_dir, "final_model")
    model.save(final_model_path)
    mlflow.log_artifact(final_model_path + ".zip", "models")

    # Evaluate the model
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=10)
    mlflow.log_metrics({
        "mean_reward": mean_reward,
        "std_reward": std_reward
    })
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Log the env info
    env_info = {
        "env_id": hyperparams["env_id"],
        "n_envs": hyperparams["n_envs"],
        "observation_space": str(vec_env.observation_space),
        "action_space": str(vec_env.action_space)
    }
    mlflow.log_params(env_info)

    # Plot and log learning curve if available
    try:
        # For vectorized environments, we need to consolidate logs
        # from multiple monitor files (one per environment)
        results = []
        for i in range(hyperparams["n_envs"]):
            try:
                log_file = os.path.join(log_dir, f"{i}.monitor.csv")
                if os.path.exists(log_file):
                    df = pd.read_csv(log_file, skiprows=1)
                    df['env_id'] = i
                    results.append(df)
            except Exception as e:
                print(f"Couldn't read log for env {i}: {e}")

        if results:
            # Combine all environment results
            all_df = pd.concat(results, ignore_index=True)
            all_df = all_df.sort_values('t')

            # Plot rewards over time
            plt.figure(figsize=(10, 6))
            for env_id, group in all_df.groupby('env_id'):
                plt.plot(group['t'], group['r'], alpha=0.3, label=f'Env {env_id}')

            # Also plot mean rewards
            mean_rewards = all_df.groupby('t')['r'].mean().reset_index()
            plt.plot(mean_rewards['t'], mean_rewards['r'], 'k-', linewidth=2, label='Mean')

            plt.xlabel("Timesteps")
            plt.ylabel("Rewards")
            plt.title("Learning Curve - All Environments")
            plt.legend()
            plt.grid(True)

            # Save the figure and log it as an artifact
            fig_path = os.path.join(log_dir, "learning_curve_all_envs.png")
            plt.savefig(fig_path)
            plt.close()
            mlflow.log_artifact(fig_path, "figures")

            # Also plot mean rewards only for clarity
            plt.figure(figsize=(10, 6))
            plt.plot(mean_rewards['t'], mean_rewards['r'], 'b-', linewidth=2)
            plt.xlabel("Timesteps")
            plt.ylabel("Mean Reward")
            plt.title("Learning Curve - Mean Reward")
            plt.grid(True)

            # Save the figure and log it as an artifact
            fig_path = os.path.join(log_dir, "learning_curve_mean.png")
            plt.savefig(fig_path)
            plt.close()
            mlflow.log_artifact(fig_path, "figures")
    except Exception as e:
        print(f"Could not generate learning curves: {e}")

    # Close environments
    vec_env.close()
    eval_env.close()
