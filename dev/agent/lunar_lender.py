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
import time
import psutil
import matplotlib.pyplot as plt
from IPython.display import clear_output, display, HTML, display_html
from matplotlib.figure import Figure
import io
from PIL import Image
import threading
import ipywidgets as widgets
from IPython.display import Video
from gymnasium.spaces import Box, Discrete
from gymnasium.wrappers import RecordVideo
from stable_baselines3 import A2C
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import EvalCallback, CheckpointCallback, BaseCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv


# Configure PyTorch for optimal CPU performance
torch.set_num_threads(12)
os.environ["OMP_NUM_THREADS"] = "12"
os.environ["MKL_NUM_THREADS"] = "12"
torch.backends.cudnn.benchmark = True

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
n_envs = 12

# Create environments directly with vectorization for better performance
envs = [lambda: gym.make(env_id) for _ in range(n_envs)]
vec_env = DummyVecEnv(envs)  # DummyVecEnv is more efficient for CPU-only workloads

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
    total_timesteps = 1000000
    learning_rate = 0.001
    n_steps = 12
    gamma = 0.99
    gae_lambda = 0.95
    ent_coef = 0.01
    vf_coef = 0.5
    max_grad_norm = 0.5
    verbose = 1  # Set to verbose mode to get more detailed logs

    # Optimize batch size for better CPU utilization
    policy_kwargs = dict(
        net_arch=[64, 64],
        activation_fn=torch.nn.ReLU,
        ortho_init=True
    )

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

    # Create evaluation environment for monitoring
    eval_env = Monitor(gym.make(env_id))

    # Setup callbacks
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=best_model_dir,
        log_path=log_dir,
        eval_freq=50000,
        deterministic=True,
        render=False
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=100000,
        save_path=checkpoint_dir,
        name_prefix="a2c_lunar_lander"
    )

    # Create a custom callback for live training visualization
    class LiveMetricsCallback(BaseCallback):
        def __init__(self, eval_env, eval_freq=10000, n_eval_episodes=5, verbose=1):
            super(LiveMetricsCallback, self).__init__(verbose)
            self.eval_env = eval_env
            self.eval_freq = eval_freq
            self.n_eval_episodes = n_eval_episodes

            # Metrics storage
            self.training_steps = []
            self.rewards = []
            self.episode_lengths = []
            self.policy_losses = []
            self.value_losses = []
            self.entropy_losses = []
            self.learning_rates = []
            self.cpu_usages = []
            self.eval_rewards = []
            self.eval_steps = []
            self.fps_values = []

            # Initialize display objects
            self.output_widget = None
            self.fig = None
            self.start_time = None
            self.last_update_time = 0
            self.update_interval = 2.0  # Update plot at most every 2 seconds

        def _on_training_start(self):
            self.start_time = time.time()

            # Create a persistent output widget for the visualizations
            self.output_widget = widgets.Output()
            display(HTML("<h3>Training Progress Visualization</h3>"))
            display(self.output_widget)

            # Create the initial figure inside the widget
            with self.output_widget:
                self.fig, self.axs = plt.subplots(3, 2, figsize=(15, 12))
                self.fig.tight_layout(pad=3.0)

                # Initialize empty plots with legends
                if len(self.training_steps) == 0:
                    # Add dummy data point to initialize plots
                    self.training_steps = [0]
                    self.rewards = [0]
                    self.episode_lengths = [0]
                    self.policy_losses = [0]
                    self.value_losses = [0]
                    self.entropy_losses = [0]
                    self.learning_rates = [0]
                    self.cpu_usages = [0]
                    self.fps_values = [0]

                    # Initialize all plots
                    self._update_plot()

                plt.show()

            print("Starting training with live metrics visualization...")

        def _on_step(self):
            current_time = time.time()

            # Collect data every step regardless of update frequency
            if hasattr(self.model, 'logger'):
                # Store metrics - add explicit logging for debug
                if len(self.model.logger.name_to_value) > 0:
                    # Get training metrics if available
                    rewards = self.model.logger.name_to_value.get('rollout/ep_rew_mean', None)
                    ep_len = self.model.logger.name_to_value.get('rollout/ep_len_mean', None)
                    policy_loss = self.model.logger.name_to_value.get('train/policy_loss', None)
                    value_loss = self.model.logger.name_to_value.get('train/value_loss', None)
                    entropy = self.model.logger.name_to_value.get('train/entropy_loss', None)
                    learning_rate = self.model.logger.name_to_value.get('train/learning_rate', None)
                    fps = self.model.logger.name_to_value.get('time/fps', None)

                    # Only add data points if they're available (not None)
                    if all(x is not None for x in [rewards, ep_len, policy_loss, value_loss, entropy]):
                        self.training_steps.append(self.n_calls)
                        self.rewards.append(rewards)
                        self.episode_lengths.append(ep_len)
                        self.policy_losses.append(policy_loss)
                        self.value_losses.append(value_loss)
                        self.entropy_losses.append(entropy)
                        self.learning_rates.append(learning_rate if learning_rate is not None else 0)
                        self.fps_values.append(fps if fps is not None else 0)

                        # Get CPU usage
                        try:
                            self.cpu_usages.append(sum(psutil.cpu_percent(percpu=True, interval=0.1)))
                        except:
                            # Fallback if psutil fails
                            self.cpu_usages.append(0)

                        # Print debug info every 10000 steps
                        if self.n_calls % 10000 == 0:
                            print(f"Debug: Step {self.n_calls}, collected {len(self.training_steps)} data points")

            # Only update plot if enough time has passed since last update (to avoid flicker)
            if current_time - self.last_update_time > self.update_interval and len(self.training_steps) > 0:
                # Update the plot in the output widget
                self._update_plot()
                self.last_update_time = current_time

            # Perform evaluation
            if self.n_calls % self.eval_freq == 0:
                mean_reward, std_reward = evaluate_policy(
                    self.model,
                    self.eval_env,
                    n_eval_episodes=self.n_eval_episodes,
                    deterministic=True
                )
                self.eval_rewards.append(mean_reward)
                self.eval_steps.append(self.n_calls)

                # Log to MLflow
                self.logger.record("eval/mean_reward", mean_reward)
                self.logger.record("eval/std_reward", std_reward)

                # Print current evaluation (outside of the widget)
                elapsed_time = time.time() - self.start_time
                print(f"\nEvaluation at step {self.n_calls}: Reward = {mean_reward:.2f} ± {std_reward:.2f}, Time: {elapsed_time:.2f}s\n")

            return True

        def _update_plot(self):
            """Update the plot in the existing widget"""
            # Skip if no data
            if len(self.training_steps) == 0:
                return

            # Clear all subplots but keep the figure
            for ax in self.axs.flat:
                ax.clear()

            # Plot Episode Rewards with explicit legend
            ax = self.axs[0, 0]
            train_line, = ax.plot(self.training_steps, self.rewards, 'b-', linewidth=2, label='Training')
            ax.set_title('Episode Rewards', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Reward')
            ax.grid(True)

            # Add evaluation rewards if available
            eval_line = None
            if len(self.eval_steps) > 0:
                eval_line, = ax.plot(self.eval_steps, self.eval_rewards, 'r-', linewidth=2, label='Evaluation')
                ax.legend(handles=[train_line, eval_line], loc='best')
            else:
                ax.legend(handles=[train_line], loc='best')

            # Plot Episode Lengths
            ax = self.axs[0, 1]
            ep_len_line, = ax.plot(self.training_steps, self.episode_lengths, 'g-', linewidth=2, label='Episode Length')
            ax.set_title('Episode Length', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Steps')
            ax.legend(handles=[ep_len_line], loc='best')
            ax.grid(True)

            # Plot Policy Loss
            ax = self.axs[1, 0]
            policy_line, = ax.plot(self.training_steps, self.policy_losses, 'r-', linewidth=2, label='Policy Loss')
            ax.set_title('Policy Loss', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.legend(handles=[policy_line], loc='best')
            ax.grid(True)

            # Plot Value Loss
            ax = self.axs[1, 1]
            value_line, = ax.plot(self.training_steps, self.value_losses, 'c-', linewidth=2, label='Value Loss')
            ax.set_title('Value Loss', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.legend(handles=[value_line], loc='best')
            ax.grid(True)

            # Plot Entropy Loss
            ax = self.axs[2, 0]
            entropy_line, = ax.plot(self.training_steps, self.entropy_losses, 'm-', linewidth=2, label='Entropy Loss')
            ax.set_title('Entropy Loss', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.legend(handles=[entropy_line], loc='best')
            ax.grid(True)

            # Plot FPS and CPU Usage
            ax = self.axs[2, 1]

            # Create a twin axis for the FPS
            ax2 = ax.twinx()

            # Plot CPU on primary y-axis
            cpu_line, = ax.plot(self.training_steps, self.cpu_usages, 'y-', linewidth=2, label='CPU %')
            ax.set_title('Performance Metrics', fontsize=12, fontweight='bold')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('CPU Usage (%)', color='y')
            ax.tick_params(axis='y', labelcolor='y')

            # Plot FPS on secondary y-axis
            fps_line, = ax2.plot(self.training_steps, self.fps_values, 'g--', linewidth=2, label='FPS')
            ax2.set_ylabel('Frames per Second', color='g')
            ax2.tick_params(axis='y', labelcolor='g')

            # Create a combined legend
            lines = [cpu_line, fps_line]
            labels = [line.get_label() for line in lines]
            ax.legend(lines, labels, loc='upper left')

            # Update layout
            self.fig.tight_layout()

            # Redraw the figure in the widget
            with self.output_widget:
                self.fig.canvas.draw()
                plt.pause(0.001)  # Small pause to allow the plot to update

        def _on_training_end(self):
            """Save final plots and metrics to MLflow"""
            # Create a final non-interactive figure for MLflow
            final_fig = Figure(figsize=(15, 12))

            # Add the same plots to the final figure
            axes = final_fig.subplots(3, 2)

            # Plot Episode Rewards
            ax = axes[0, 0]
            ax.plot(self.training_steps, self.rewards, 'b-', label='Training')
            if len(self.eval_steps) > 0:
                ax.plot(self.eval_steps, self.eval_rewards, 'r-', label='Evaluation')
            ax.set_title('Episode Rewards')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Reward')
            ax.legend()
            ax.grid(True)

            # Plot Episode Lengths
            ax = axes[0, 1]
            ax.plot(self.training_steps, self.episode_lengths, 'g-')
            ax.set_title('Episode Length')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Steps')
            ax.grid(True)

            # Plot Policy and Value Loss
            ax = axes[1, 0]
            ax.plot(self.training_steps, self.policy_losses, 'r-', label='Policy Loss')
            ax.set_title('Policy Loss')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.grid(True)

            ax = axes[1, 1]
            ax.plot(self.training_steps, self.value_losses, 'c-', label='Value Loss')
            ax.set_title('Value Loss')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.grid(True)

            # Plot Entropy and CPU Usage
            ax = axes[2, 0]
            ax.plot(self.training_steps, self.entropy_losses, 'm-')
            ax.set_title('Entropy Loss')
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.grid(True)

            # Plot FPS and CPU
            ax = axes[2, 1]
            ax1 = ax
            ax2 = ax.twinx()

            # Plot CPU on primary y-axis
            cpu_line, = ax1.plot(self.training_steps, self.cpu_usages, 'y-', label='CPU %')
            ax1.set_title('Performance Metrics')
            ax1.set_xlabel('Training Steps')
            ax1.set_ylabel('CPU Usage (%)')

            # Plot FPS on secondary y-axis
            fps_line, = ax2.plot(self.training_steps, self.fps_values, 'g--', label='FPS')
            ax2.set_ylabel('Frames per Second')

            # Create a combined legend
            lines = [cpu_line, fps_line]
            labels = [line.get_label() for line in lines]
            ax1.legend(lines, labels, loc='upper left')

            ax1.grid(True)

            final_fig.tight_layout()

            # Save the figure to a file
            metrics_plot_path = f"{run_artifact_dir}/training_metrics.png"
            final_fig.savefig(metrics_plot_path)

            # Log the figure to MLflow
            mlflow.log_artifact(metrics_plot_path)

            # Save metrics data to CSV
            metrics_df = pd.DataFrame({
                'step': self.training_steps,
                'reward': self.rewards,
                'episode_length': self.episode_lengths,
                'policy_loss': self.policy_losses,
                'value_loss': self.value_losses,
                'entropy_loss': self.entropy_losses,
                'learning_rate': self.learning_rates,
                'cpu_usage': self.cpu_usages,
                'fps': self.fps_values
            })

            eval_df = pd.DataFrame({
                'step': self.eval_steps,
                'eval_reward': self.eval_rewards
            })

            metrics_csv_path = f"{run_artifact_dir}/training_metrics.csv"
            eval_csv_path = f"{run_artifact_dir}/eval_metrics.csv"

            metrics_df.to_csv(metrics_csv_path, index=False)
            eval_df.to_csv(eval_csv_path, index=False)

            # Log CSV files to MLflow
            mlflow.log_artifact(metrics_csv_path)
            mlflow.log_artifact(eval_csv_path)

            print(f"Training completed. Final metrics saved to {metrics_csv_path} and {eval_csv_path}")
            print(f"Training visualization saved to {metrics_plot_path}")

    # Add live metrics callback
    live_metrics_callback = LiveMetricsCallback(
        eval_env=eval_env,
        eval_freq=20000,  # Evaluate more frequently for the live plots
        n_eval_episodes=10,
        verbose=1
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
        policy_kwargs=policy_kwargs,
        normalize_advantage=True,  # Normalize advantages for stability
        use_rms_prop=True,  # Use RMSProp optimizer for potentially better performance
        verbose=1  # Set to 1 to collect metrics but not print them too often
    )

    # Log model parameters
    for param_name, param_value in model.policy.state_dict().items():
        if isinstance(param_value, torch.Tensor):
            mlflow.log_param(f"policy_{param_name}_shape", list(param_value.shape))

    # Train the agent
    mlflow.log_param("total_timesteps", total_timesteps)

    # Log all hyperparameters
    mlflow.log_params({
        "algorithm": "A2C",
        "environment": env_id,
        "n_envs": n_envs,
        "learning_rate": learning_rate,
        "n_steps": n_steps,
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "ent_coef": ent_coef,
        "vf_coef": vf_coef,
        "max_grad_norm": max_grad_norm,
        "policy_network": str(policy_kwargs),
        "total_timesteps": total_timesteps
    })

    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, checkpoint_callback, live_metrics_callback]
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

