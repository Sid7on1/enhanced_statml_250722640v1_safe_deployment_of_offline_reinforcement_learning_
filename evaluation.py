import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from gym import spaces

    from project_env import ProjectEnv

except ImportError as e:
    logger.error(
        "Error importing required modules. Please ensure all necessary packages are installed."
    )
    raise e

# Constants and configuration
class EvaluationError(Exception):
    pass


class Evaluation:
    def __init__(
        self,
        env: ProjectEnv,
        render: bool = False,
        seed: Optional[int] = None,
        episode_limit: int = 1000,
        max_steps: int = 100,
    ):
        """
        Initialize the evaluation class.

        :param env: The ProjectEnv environment to evaluate.
        :param render: Whether to render the environment during evaluation.
        :param seed: Optional random seed for reproducibility.
        :param episode_limit: Maximum number of episodes for evaluation.
        :param max_steps: Maximum steps per episode.
        """
        self.env = env
        self.render = render
        self.seed = seed
        self.episode_limit = episode_limit
        self.max_steps = max_steps

        self.total_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_episodes = 0
        self.total_steps = 0

        # Seed the environment
        if seed is not None:
            self.seed_environment(seed)

    def seed_environment(self, seed: int) -> None:
        """
        Seed the environment for reproducibility.

        :param seed: Random seed to set.
        """
        self.env.seed(seed)
        np.random.seed(seed)

    def reset_environment(self) -> np.ndarray:
        """
        Reset the environment to initial state.

        :return: Initial state observation.
        """
        return self.env.reset()

    def evaluate(self) -> Dict[str, float]:
        """
        Evaluate the agent's performance in the given environment.

        :return: Dictionary of evaluation metrics.
        """
        self.total_reward = 0
        self.episode_rewards = []
        self.episode_lengths = []
        self.total_episodes = 0
        self.total_steps = 0

        while self.total_episodes < self.episode_limit:
            episode_reward, episode_length = self.run_episode()
            self.episode_rewards.append(episode_reward)
            self.episode_lengths.append(episode_length)
            self.total_episodes += 1

            logger.info(
                f"Episode {self.total_episodes}/{self.episode_limit} | "
                f"Reward: {episode_reward:.2f} | "
                f"Length: {episode_length} steps"
            )

        avg_reward = np.mean(self.episode_rewards)
        max_reward = np.max(self.episode_rewards)
        min_reward = np.min(self->episode_rewards)
        avg_length = np.mean(self.episode_lengths)
        max_length = np.max(self.episode_lengths)
        min_length = np.min(self.episode_lengths)

        return {
            "total_episodes": self.total_episodes,
            "total_steps": self.total_steps,
            "avg_reward": avg_reward,
            "max_reward": max_reward,
            "min_reward": min_reward,
            "avg_length": avg_length,
            "max_length": max_length,
            "min_length": min_length,
        }

    def run_episode(
        self, max_steps: Optional[int] = None
    ) -> Tuple[float, int]:
        """
        Run a single episode of evaluation.

        :param max_steps: Maximum steps for this episode.
                        If None, use the class-level max_steps value.
        :return: Tuple of (episode reward, episode length).
        """
        if max_steps is None:
            max_steps = self.max_steps

        state = self.reset_environment()
        reward_sum = 0
        step_count = 0

        while step_count < max_steps:
            action = self.select_action(state)
            next_state, reward, done, _ = self.env.step(action)

            if self.render:
                self.env.render()

            reward_sum += reward
            step_count += 1
            state = next_state

            if done:
                break

        self.total_reward += reward_sum
        self.total_steps += step_count

        return reward_sum, step_count

    def select_action(self, state: np.ndarray) -> int:
        """
        Select an action for the given state using the agent's policy.

        :param state: The current state of the environment.
        :return: Selected action as an integer.
        """
        # TODO: Implement your action selection strategy here
        # This example returns a random action
        return self.env.action_space.sample()

# Helper classes and utilities
class ProjectEnv:
    def __init__(self):
        # TODO: Implement the ProjectEnv class
        # This class should represent the environment for your project
        # It should inherit from gym.Env and define the necessary methods
        # such as reset(), step(), render(), etc.
        pass

# Exception classes
class ActionValidationError(EvaluationError):
    pass


# Data structures/models
class EvaluationResults:
    def __init__(self, metrics: Dict[str, float]):
        self.metrics = metrics

    def to_dataframe(self) -> pd.DataFrame:
        """
        Convert evaluation results to a pandas DataFrame.

        :return: DataFrame containing evaluation metrics.
        """
        return pd.DataFrame([self.metrics], index=["Metrics"])

# Validation functions
def validate_action(action: int, action_space: spaces.Discrete) -> None:
    """
    Validate that an action is within the valid range of the action space.

    :param action: Action to validate.
    :param action_space: Discrete action space of the environment.
    :raise ActionValidationError: If the action is invalid.
    """
    if not action_space.contains(action):
        raise ActionValidationError(
            f"Invalid action: {action}. Must be within {action_space.n} discrete actions."
        )

# Utility methods
def setup_evaluation(
    env: ProjectEnv,
    render: bool = False,
    seed: Optional[int] = None,
    episode_limit: int = 1000,
    max_steps: int = 100,
) -> Evaluation:
    """
    Convenience function to setup and perform evaluation.

    :param env: The ProjectEnv environment to evaluate.
    :param render: Whether to render the environment during evaluation.
    :param seed: Optional random seed for reproducibility.
    :param episode_limit: Maximum number of episodes for evaluation.
    :param max_steps: Maximum steps per episode.
    :return: Evaluation results as a dictionary.
    """
    evaluator = Evaluation(env, render, seed, episode_limit, max_steps)
    return evaluator.evaluate()

# Integration interfaces
# Example integration with an agent
class ExampleAgent:
    def act(self, state: np.ndarray) -> int:
        # TODO: Implement your agent's action selection strategy
        # This method should take in a state and return an action
        pass

def evaluate_agent(
    agent: ExampleAgent,
    env: ProjectEnv,
    render: bool = False,
    seed: Optional[int] = None,
    episode_limit: int = 1000,
    max_steps: int = 100,
) -> Dict[str, float]:
    """
    Evaluate the performance of an agent in the given environment.

    :param agent: The agent to evaluate.
    :param env: The ProjectEnv environment to use for evaluation.
    :param render: Whether to render the environment during evaluation.
    :param seed: Optional random seed for reproducibility.
    :param episode_limit: Maximum number of episodes for evaluation.
    :param max_steps: Maximum steps per episode.
    :return: Dictionary of evaluation metrics.
    """
    evaluator = Evaluation(env, render, seed, episode_limit, max_steps)

    while evaluator.total_episodes < evaluator.episode_limit:
        state = evaluator.reset_environment()

        episode_reward, episode_length = evaluator.run_episode_with_agent(agent, state)
        evaluator.episode_rewards.append(episode_reward)
        evaluator.episode_lengths.append(episode_length)
        evaluator.total_episodes += 1

        logger.info(
            f"Episode {evaluator.total_episodes}/{evaluator.episode_limit} | "
            f"Reward: {episode_reward:.2f} | "
            f"Length: {episode_length} steps"
        )

    return evaluator.compute_metrics()

def run_episode_with_agent(
    self, agent: ExampleAgent, initial_state: np.ndarray
) -> Tuple[float, int]:
    """
    Run a single episode of evaluation with an agent.

    :param agent: The agent to evaluate.
    :param initial_state: Initial state of the environment.
    :return: Tuple of (episode reward, episode length).
    """
    state = initial_state
    reward_sum = 0
    step_count = 0

    while step_count < self.max_steps:
        action = agent.act(state)
        validate_action(action, self.env.action_space)

        next_state, reward, done, _ = self.env.step(action)

        if self.render:
            self.env.render()

        reward_sum += reward
        step_count += 1
        state = next_state

        if done:
            break

    self.total_reward += reward_sum
    self.total_steps += step_count

    return reward_sum, step_count

# Main function for standalone execution
if __name__ == "__main__":
    # Example usage:
    # Create your ProjectEnv environment here
    # env = ProjectEnv()

    # Create your agent here
    # agent = ExampleAgent()

    # Perform evaluation
    # results = evaluate_agent(agent, env)

    # Print or use evaluation results as needed
    # print(results)