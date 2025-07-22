# %% [markdown]
# ### This notebook is dedicated to test full training pipeline of custom agent on real data

# %%
# Import libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from packages.agent.custom_envs import MultiTickerPortfolioEnv
from packages.agent.agents import A2CAgent
from packages.agent.training import TrainingManager

# %%
# Run the test when executed directly
if __name__ == "__main__":
    # Define features to include
    features = ['volume', 'open', 'high', 'low', 'obv', 'rsi', 'atr']

    # Data importing
    processed_data = pd.read_csv('data/rts_1d_2023_2024_indicators.csv')

    # %% [markdown]
    # ## 1. Data Preparation

    # %%
    # Inspect data
    print(f"Data shape: {processed_data.shape}")
    print(f"Available columns: {processed_data.columns.tolist()}")
    print(f"Available tickers: {processed_data['tic'].unique().tolist()}")

    # Basic data preprocessing
    processed_data['date'] = pd.to_datetime(processed_data['date'])

    # Split data into training and testing
    # Use 80% of data for training, 20% for testing
    train_cutoff = int(len(processed_data['date'].unique()) * 0.8)
    train_dates = sorted(processed_data['date'].unique())[:train_cutoff]
    test_dates = sorted(processed_data['date'].unique())[train_cutoff:]

    # Create training and testing datasets
    train_data = processed_data[processed_data['date'].isin(train_dates)]
    test_data = processed_data[processed_data['date'].isin(test_dates)]

    print(f"Training data date range: {train_data['date'].min()} to {train_data['date'].max()}")
    print(f"Testing data date range: {test_data['date'].min()} to {test_data['date'].max()}")

    # %% [markdown]
    # ## 2. Environment Setup

    # %%
    # Define environment parameters
    env_kwargs = {
        'df': train_data,
        'history_length': 10,
        'initial_cash': 10000,
        'risk_aversion': 1.5,
        'price_column': 'close',
        'date_column': 'date',
        'ticker_column': 'tic',
        'feature_columns': features,
        'transaction_cost': 0.001  # 0.1% transaction cost
    }

    # Create test environment
    test_env = MultiTickerPortfolioEnv(**env_kwargs)
    print(f"Created environment with {len(test_env.tickers)} tickers")
    print(f"Observation space shape: {test_env.observation_space.shape}")
    print(f"Action space shape: {test_env.action_space.shape}")

    # %% [markdown]
    # ## 3. Agent Configuration

    # %%
    # Define agent parameters - structure them according to A2CAgent expectations
    agent_params = {
        # Common parameters for A2CAgent constructor
        'verbose': 1,
        'seed': 42,

        # Model specific parameters for A2C algorithm
        'learning_rate': 0.0007,
        'gamma': 0.99,
        'gae_lambda': 0.95,
        'ent_coef': 0.01,
        'vf_coef': 0.5,
        'normalize_advantage': True,
        'policy_kwargs': {
            'net_arch': [128, 128]
        }
    }

    # %% [markdown]
    # ## 4. Training Pipeline Setup

    # %%
    # Create training manager
    experiment_name = "portfolio_optimization_test"
    model_name = "a2c_portfolio_rts"
    trainer = TrainingManager(experiment_name=experiment_name, model_name=model_name)

    # Define training parameters
    training_params = {
        'n_episodes': 1000,  # Reduced for testing
        'max_steps': 252,   # Typical trading days in a year
        'eval_frequency': 10
    }

    # %% [markdown]
    # ## 5. Training Process

    # %%
    # Train the agent
    print("Starting agent training...")

    # Wrap agent training in try-except to handle potential errors
    try:
        # Set the version for this run
        version = "test_v1"

        # Train agent
        agent, model_path = trainer.train_agent(
            env_params=env_kwargs,
            agent_params=agent_params,
            training_params=training_params,
            version=version,
            run_name="initial_test_run"
        )

        print(f"Training completed successfully!")
        print(f"Model saved to: {model_path}")

    except Exception as e:
        print(f"Error during training: {str(e)}")
        raise

    # %% [markdown]
    # ## 6. Model Version Management

    # %%
    # List available model versions
    versions = trainer.get_model_versions()
    print(f"Available model versions: {versions}")

    # %% [markdown]
    # ## 7. Model Evaluation

    # %%
    # Create evaluation environment with test data
    eval_env_kwargs = env_kwargs.copy()
    eval_env_kwargs['df'] = test_data
    eval_env = MultiTickerPortfolioEnv(**eval_env_kwargs)

    # Evaluate the trained agent
    print("Evaluating trained agent on test data...")
    eval_metrics = trainer.evaluate_agent(agent, eval_env, n_episodes=5)

    # Print evaluation metrics
    print("\nEvaluation Results:")
    for metric, value in eval_metrics.items():
        print(f"{metric}: {value:.4f}")

    # %% [markdown]
    # ## 8. Test Model Loading

    # %%
    # Test loading a trained model
    print("\nTesting model loading functionality...")

    try:
        # Load the most recent model version
        recent_version = versions[-1] if versions else version
        loaded_agent = trainer.load_agent(recent_version)
        print(f"Successfully loaded model version: {recent_version}")

        # Verify the loaded model works by running a quick evaluation
        quick_eval = trainer.evaluate_agent(loaded_agent, eval_env, n_episodes=1)
        print("Loaded model evaluation (1 episode):", quick_eval['mean_reward'])

    except Exception as e:
        print(f"Error loading model: {str(e)}")

    # %% [markdown]
    # ## 9. Visualize Results

    # %%
    # Run a full episode with the trained agent and visualize the results
    print("\nRunning visualization episode...")

    # Reset the environment and run one full episode
    reset_result = eval_env.reset()
    # Handle gymnasium API that returns (obs, info) tuple
    if isinstance(reset_result, tuple):
        observation, info = reset_result
    else:
        observation = reset_result
        info = eval_env._get_info()  # Get info directly if using old Gym API

    done = False
    truncated = False
    portfolio_values = [info['portfolio_value']]
    dates = [info['date']]
    allocations_history = []

    while not done and not truncated:
        action = agent.predict(observation, deterministic=True)

        # Handle step return properly for both Gym and Gymnasium APIs
        step_result = eval_env.step(action)
        if len(step_result) == 4:
            # Old gym: next_state, reward, done, info
            observation, reward, done, info = step_result
            truncated = False
        else:
            # New gymnasium: next_state, reward, terminated, truncated, info
            observation, reward, done, truncated, info = step_result

        portfolio_values.append(info['portfolio_value'])
        dates.append(info['date'])
        allocations_history.append(info['allocations'])

    # Convert dates from string to datetime for plotting
    plot_dates = pd.to_datetime(dates)

    # Plot portfolio value over time
    plt.figure(figsize=(12, 6))
    plt.plot(plot_dates, portfolio_values)
    plt.title('Portfolio Value Over Time')
    plt.xlabel('Date')
    plt.ylabel('Portfolio Value ($)')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/portfolio_performance.png')
    plt.show()

    # Plot asset allocations over time
    allocations_history = np.array(allocations_history)
    plt.figure(figsize=(12, 6))

    # Plot stacked area chart of allocations
    plt.stackplot(plot_dates[1:], allocations_history.T,
                 labels=eval_env.tickers, alpha=0.8)

    plt.title('Asset Allocation Over Time')
    plt.xlabel('Date')
    plt.ylabel('Allocation')
    plt.legend(loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('data/asset_allocation.png')
    plt.show()

    # %% [markdown]
    # ## 10. Clean Up Test Artifacts (Optional)

    # %%
    # Uncomment to delete test model version
    # print("Cleaning up test artifacts...")
    # trainer.delete_model_version("test_v1")

    print("Test completed successfully!")
