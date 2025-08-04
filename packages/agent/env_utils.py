"""
Environment Utilities Module

This module contains utilities for running episodes, comparing agents, and visualizing results.
Provides a clean interface for portfolio environment operations and analysis.
"""

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, Any, List, Callable, Union
from .environments import OptimizedPortfolioEnv
from .strategies import BaseStrategy


class EpisodeRunner:
    """
    Handles episode execution with comprehensive tracking and analysis.
    """
    
    def __init__(self, env: OptimizedPortfolioEnv):
        self.env = env
        self.episode_data = {}
    
    def run_episode(self, agent: Union[BaseStrategy, Callable], max_steps: int = 1000, 
                   verbose: bool = True) -> Dict[str, Any]:
        """
        Run a single episode with detailed tracking.
        
        Args:
            agent: Trading strategy or callable that takes observation and returns action
            max_steps: Maximum number of steps to run
            verbose: Whether to print progress information
            
        Returns:
            Dictionary containing episode metrics and data
        """
        # Set up agent environment reference if needed
        if hasattr(agent, 'set_env'):
            agent.set_env(self.env)
        
        observation, _ = self.env.reset()
        done = False
        steps = 0
        episode_rewards = []
        episode_actions = []
        
        if verbose:
            print(f"Episode parameters:")
            print(f"  Max steps: {max_steps}")
            print(f"  Window size: {self.env.window_size}")
            print(f"  Assets: {self.env.n_assets}")
            print(f"  Rendering: {'Enabled' if self.env.enable_rendering else 'Disabled'}")
        
        start_time = time.time()
        
        while not done and steps < max_steps:
            steps += 1
            
            # Get action from agent
            if hasattr(agent, 'get_action'):
                action = agent.get_action(observation)
            else:
                action = agent(observation)
            
            # Execute step
            observation, reward, done, truncated, info = self.env.step(action)
            
            # Store episode data
            episode_rewards.append(reward)
            episode_actions.append(action.copy())
            
            # Progress updates
            if verbose and steps % 100 == 0:
                print(f"  Step {steps}/{max_steps} - Portfolio Value: ${info['portfolio_value']:.2f}")
        
        end_time = time.time()
        
        # Calculate episode metrics
        total_return = (info['portfolio_value'] - 1.0) * 100
        avg_daily_return = np.mean(self.env.state.daily_returns) if self.env.state.daily_returns else 0.0
        volatility = np.std(self.env.state.daily_returns) if len(self.env.state.daily_returns) > 1 else 0.0
        sharpe_ratio = avg_daily_return / volatility if volatility > 0 else 0.0
        
        episode_metrics = {
            'returns_history': self.env.state.returns_history,
            'daily_returns': self.env.state.daily_returns,
            'weights_history': self.env.state.weights_history,
            'final_value': info['portfolio_value'],
            'total_return': total_return,
            'avg_daily_return': avg_daily_return,
            'volatility': volatility,
            'sharpe_ratio': sharpe_ratio,
            'execution_time': end_time - start_time,
            'steps': steps,
            'episode_rewards': episode_rewards,
            'episode_actions': episode_actions
        }
        
        if verbose:
            print(f"\\nEpisode completed:")
            print(f"  Duration: {end_time - start_time:.2f}s")
            print(f"  Final value: ${info['portfolio_value']:.2f}")
            print(f"  Total return: {total_return:.2f}%")
            print(f"  Sharpe ratio: {sharpe_ratio:.3f}")
        
        return episode_metrics


class AgentComparator:
    """
    Handles comparison of multiple trading agents.
    """
    
    def __init__(self, df: pd.DataFrame):
        self.df = df
        self.results = {}
    
    def compare_agents(self, agents_dict: Dict[str, Union[BaseStrategy, Callable]], 
                      max_steps: int = None, n_assets: int = 5, 
                      enable_rendering: bool = False, render_interval: int = 10,
                      **env_kwargs) -> Dict[str, Dict[str, Any]]:
        """
        Compare multiple agents on the same environment setup.
        
        Args:
            agents_dict: Dictionary mapping agent names to agent instances
            max_steps: Maximum steps per episode
            n_assets: Number of assets to trade
            enable_rendering: Whether to enable visualization
            render_interval: Steps between renders
            **env_kwargs: Additional environment parameters
            
        Returns:
            Dictionary containing results for each agent
        """
        results = {}
        
        # Set default max_steps
        if max_steps is None:
            max_steps = len(self.df.index.unique()) - 20
        
        print(f"Running agent comparison:")
        print(f"  Assets: {n_assets}")
        print(f"  Max steps: {max_steps}")
        print(f"  Rendering: {'Every ' + str(render_interval) + ' steps' if enable_rendering else 'Disabled'}")
        print("-" * 50)
        
        for agent_name, agent in agents_dict.items():
            print(f"\\nRunning {agent_name}...")
            
            # Create fresh environment for each agent
            env = OptimizedPortfolioEnv(
                df=self.df,
                n_assets=n_assets,
                enable_rendering=enable_rendering,
                render_interval=render_interval,
                **env_kwargs
            )
            
            # Run episode
            runner = EpisodeRunner(env)
            metrics = runner.run_episode(agent, max_steps, verbose=True)
            results[agent_name] = metrics
            
            # Cleanup
            env.cleanup()
        
        # Print summary
        self._print_comparison_summary(results)
        self.results = results
        return results
    
    def _print_comparison_summary(self, results: Dict[str, Dict[str, Any]]) -> None:
        """Print formatted comparison summary"""
        print("\\n" + "="*60)
        print("RESULTS SUMMARY")
        print("="*60)
        
        for agent_name, metrics in results.items():
            total_return = metrics['total_return']
            sharpe = metrics['sharpe_ratio']
            time_taken = metrics['execution_time']
            print(f"{agent_name:15s}: ${metrics['final_value']:.2f} ({total_return:+.2f}%) "
                  f"Sharpe: {sharpe:.3f} Time: {time_taken:.1f}s")
        
        # Find best performers
        best_return = max(results.keys(), key=lambda x: results[x]['total_return'])
        best_sharpe = max(results.keys(), key=lambda x: results[x]['sharpe_ratio'])
        
        print(f"\\nBest Return: {best_return}")
        print(f"Best Sharpe: {best_sharpe}")


class PortfolioVisualizer:
    """
    Handles visualization of portfolio analysis results.
    """
    
    @staticmethod
    def plot_comparison(results: Dict[str, Dict[str, Any]], figsize: tuple = (15, 10)) -> None:
        """
        Create comprehensive comparison plots for multiple agents.
        
        Args:
            results: Results dictionary from AgentComparator
            figsize: Figure size tuple
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Agent Performance Comparison', fontsize=16)
        
        # Portfolio values over time
        for agent_name, metrics in results.items():
            axes[0, 0].plot(metrics['returns_history'], label=agent_name, linewidth=2)
        axes[0, 0].set_title('Portfolio Value Over Time')
        axes[0, 0].set_ylabel('Portfolio Value ($)')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Final values comparison
        agents = list(results.keys())
        final_values = [results[agent]['final_value'] for agent in agents]
        colors = ['skyblue', 'lightcoral', 'lightgreen', 'gold', 'lightpink'][:len(agents)]
        bars = axes[0, 1].bar(agents, final_values, color=colors)
        axes[0, 1].set_title('Final Portfolio Values')
        axes[0, 1].set_ylabel('Final Value ($)')
        axes[0, 1].tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar, value in zip(bars, final_values):
            axes[0, 1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                           f'${value:.2f}', ha='center', va='bottom')
        
        # Daily returns distribution
        for agent_name, metrics in results.items():
            if len(metrics['daily_returns']) > 0:
                axes[1, 0].hist(metrics['daily_returns'], alpha=0.6, label=agent_name, bins=20)
        axes[1, 0].set_title('Daily Returns Distribution')
        axes[1, 0].set_xlabel('Daily Return')
        axes[1, 0].set_ylabel('Frequency')
        axes[1, 0].legend()
        axes[1, 0].grid(True, alpha=0.3)
        
        # Risk-Return scatter
        returns = [results[agent]['total_return'] for agent in agents]
        volatilities = [results[agent]['volatility'] * 100 for agent in agents]  # Convert to percentage
        
        scatter = axes[1, 1].scatter(volatilities, returns, s=100, alpha=0.7, c=colors)
        for i, agent in enumerate(agents):
            axes[1, 1].annotate(agent, (volatilities[i], returns[i]), 
                               xytext=(5, 5), textcoords='offset points')
        axes[1, 1].set_title('Risk vs Return')
        axes[1, 1].set_xlabel('Volatility (%)')
        axes[1, 1].set_ylabel('Total Return (%)')
        axes[1, 1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        plt.close(fig)
    
    @staticmethod
    def plot_weights_evolution(results: Dict[str, Dict[str, Any]], tickers: List[str], 
                              agent_name: str = None) -> None:
        """
        Plot portfolio weights evolution over time.
        
        Args:
            results: Results dictionary from AgentComparator
            tickers: List of ticker symbols
            agent_name: Specific agent to plot (None for all)
        """
        agents_to_plot = [agent_name] if agent_name else list(results.keys())
        
        for agent in agents_to_plot:
            if agent not in results:
                continue
                
            weights_history = results[agent]['weights_history']
            if not weights_history:
                continue
            
            weights_array = np.array(weights_history)
            steps = range(len(weights_history))
            
            plt.figure(figsize=(12, 8))
            
            # Stacked area plot
            plt.stackplot(steps, *weights_array.T, labels=tickers[:weights_array.shape[1]], alpha=0.7)
            plt.title(f'Portfolio Weights Evolution - {agent}')
            plt.xlabel('Time Step')
            plt.ylabel('Portfolio Weight')
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.show()
            plt.close()


# Convenience functions for easy usage
def run_single_episode(df: pd.DataFrame, agent: Union[BaseStrategy, Callable], 
                      **env_kwargs) -> Dict[str, Any]:
    """
    Run a single episode with default settings.
    
    Args:
        df: Market data DataFrame
        agent: Trading strategy
        **env_kwargs: Environment parameters
        
    Returns:
        Episode metrics dictionary
    """
    env = OptimizedPortfolioEnv(df, **env_kwargs)
    runner = EpisodeRunner(env)
    results = runner.run_episode(agent)
    env.cleanup()
    return results


def quick_compare(df: pd.DataFrame, agents_dict: Dict[str, Union[BaseStrategy, Callable]], 
                 **kwargs) -> Dict[str, Dict[str, Any]]:
    """
    Quick agent comparison with default settings.
    
    Args:
        df: Market data DataFrame
        agents_dict: Dictionary of agents to compare
        **kwargs: Additional parameters
        
    Returns:
        Comparison results dictionary
    """
    comparator = AgentComparator(df)
    results = comparator.compare_agents(agents_dict, **kwargs)
    PortfolioVisualizer.plot_comparison(results)
    return results