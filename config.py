import logging
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import torch
from numpy.typing import ArrayLike
from torch import Tensor

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Configuration class for the agent and environment
class Config:
    def __init__(self, config_dict: Dict[str, Any]):
        """
        Initializes the configuration settings for the agent and environment.

        Args:
            config_dict (Dict[str, Any]): A dictionary containing the configuration settings.
        """
        self.config_dict = config_dict
        self.validate_config()  # Validate the configuration settings

        # Set default device to CPU
        self.device = torch.device("cpu")
        if torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            logger.info("CUDA is available. Using GPU for computations.")

        # Algorithm parameters
        self.gamma: float = self.config_dict["algorithm"]["gamma"]
        self.buffer_size: int = self.config_dict["algorithm"]["buffer_size"]
        self.batch_size: int = self.config_dict["algorithm"]["batch_size"]
        self.tau: float = self.config_dict["algorithm"]["tau"]

        # Agent parameters
        self.agent_name: str = self.config_dict["agent"]["name"]
        self.agent_type: str = self.config_dict["agent"]["type"]
        self.input_size: Tuple[int, ...] = self.config_dict["agent"]["input_size"]
        self.output_size: int = self.config_dict["environment"]["action_space"]["n"]
        self.hidden_sizes: List[int] = self.config_dict["agent"]["hidden_sizes"]
        self.activation: str = self.config_dict["agent"]["activation"]
        self.optimizer: str = self.config_dict["agent"]["optimizer"]
        self.learning_rate: float = self.config_dict["agent"]["learning_rate"]

        # Environment parameters
        self.env_name: str = self.config_dict["environment"]["name"]
        self.state_size: Tuple[int, ...] = self.config_dict["environment"]["observation_space"]["shape"]
        self.action_size: int = self.config_dict["environment"]["action_space"]["n"]
        self.reward_range: Tuple[float, float] = (
            self.config_dict["environment"]["reward_range"]
        )

        # Training parameters
        self.num_episodes: int = self.config_dict["training"]["num_episodes"]
        self.max_steps: int = self.config_dict["training"]["max_steps"]
        self.save_interval: int = self.config_dict["training"]["save_interval"]
        self.log_interval: int = self.config_dict["training"]["log_interval"]
        self.eval_interval: int = self.config_dict["training"]["eval_interval"]

        # Model directory
        self.model_dir: str = self.config_dict["model_dir"]
        os.makedirs(self.model_dir, exist_ok=True)

    def validate_config(self) -> None:
        """
        Validates the configuration settings and raises an error if any issues are found.
        """
        # Perform comprehensive validation of configuration settings here
        # Raise errors or warnings for invalid settings
        # Example:
        if not 0 < self.gamma <= 1:
            raise ValueError("Invalid gamma value. Must be between 0 and 1.")

    # Other methods and properties for configuration, saving, loading, etc.
    # ...

# Example usage
if __name__ == "__main__":
    # Example configuration dictionary
    config_dict = {
        "algorithm": {
            "gamma": 0.99,
            "buffer_size": 10000,
            "batch_size": 64,
            "tau": 0.005,
        },
        "agent": {
            "name": "MyAgent",
            "type": "actor_critic",
            "input_size": (1, 32, 32),
            "hidden_sizes": [256, 128, 64],
            "activation": "relu",
            "optimizer": "adam",
            "learning_rate": 0.001,
        },
        "environment": {
            "name": "MyEnv",
            "observation_space": {"shape": (4,)},
            "action_space": {"n": 2},
            "reward_range": (-1, 1),
        },
        "training": {
            "num_episodes": 1000,
            "max_steps": 1000,
            "save_interval": 10,
            "log_interval": 100,
            "eval_interval": 100,
        },
        "model_dir": "models/",
    }

    # Create configuration object
    config = Config(config_dict)

    # Access configuration settings
    logger.info(f"Agent name: {config.agent_name}")
    logger.info(f"Environment name: {config.env_name}")
    logger.info(f"Learning rate: {config.learning_rate}")
    # ...

    # Example of saving and loading configuration
    config.save("config.pkl")  # Implement save method to save configuration
    loaded_config = Config.load("config.pkl")  # Implement load method to load configuration