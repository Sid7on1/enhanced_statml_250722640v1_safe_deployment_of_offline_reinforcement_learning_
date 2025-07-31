import logging
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from policy_config import PolicyConfig
from utils import get_logger, get_device
from models import PolicyNetwork
from data_structures import PolicyData

# Set up logging
logger = get_logger(__name__)

class Policy(ABC):
    """Base policy class"""
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize the policy
        
        Args:
        config (PolicyConfig): Policy configuration
        """
        self.config = config
        self.device = get_device()
        self.policy_network = PolicyNetwork(self.config)
        self.optimizer = optim.Adam(self.policy_network.parameters(), lr=self.config.lr)
        self.criterion = nn.MSELoss()
        
    @abstractmethod
    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Take an action given the current state
        
        Args:
        state (np.ndarray): Current state
        
        Returns:
        np.ndarray: Action taken
        """
        pass
    
    def train(self, batch: PolicyData):
        """
        Train the policy on a batch of data
        
        Args:
        batch (PolicyData): Batch of data
        """
        states = batch.states.to(self.device)
        actions = batch.actions.to(self.device)
        rewards = batch.rewards.to(self.device)
        
        # Zero the gradients
        self.optimizer.zero_grad()
        
        # Forward pass
        outputs = self.policy_network(states)
        loss = self.criterion(outputs, actions)
        
        # Backward pass
        loss.backward()
        
        # Update the policy network
        self.optimizer.step()
        
        # Log the loss
        logger.info(f"Loss: {loss.item():.4f}")
    
    def save(self, path: str):
        """
        Save the policy to a file
        
        Args:
        path (str): Path to save the policy
        """
        torch.save(self.policy_network.state_dict(), path)
    
    def load(self, path: str):
        """
        Load the policy from a file
        
        Args:
        path (str): Path to load the policy
        """
        self.policy_network.load_state_dict(torch.load(path))

class VelocityThresholdPolicy(Policy):
    """Velocity threshold policy"""
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize the velocity threshold policy
        
        Args:
        config (PolicyConfig): Policy configuration
        """
        super().__init__(config)
        
    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Take an action given the current state
        
        Args:
        state (np.ndarray): Current state
        
        Returns:
        np.ndarray: Action taken
        """
        # Calculate the velocity
        velocity = np.linalg.norm(state[:2])
        
        # Check if the velocity is above the threshold
        if velocity > self.config.velocity_threshold:
            # Take an action to reduce the velocity
            return np.array([0, 0, -1])
        else:
            # Take a random action
            return np.random.uniform(-1, 1, size=3)

class FlowTheoryPolicy(Policy):
    """Flow theory policy"""
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize the flow theory policy
        
        Args:
        config (PolicyConfig): Policy configuration
        """
        super().__init__(config)
        
    def act(self, state: np.ndarray) -> np.ndarray:
        """
        Take an action given the current state
        
        Args:
        state (np.ndarray): Current state
        
        Returns:
        np.ndarray: Action taken
        """
        # Calculate the flow rate
        flow_rate = np.dot(state[:2], self.config.flow_direction)
        
        # Check if the flow rate is above the threshold
        if flow_rate > self.config.flow_threshold:
            # Take an action to reduce the flow rate
            return np.array([0, 0, -1])
        else:
            # Take a random action
            return np.random.uniform(-1, 1, size=3)

class PolicyNetwork(nn.Module):
    """Policy network"""
    
    def __init__(self, config: PolicyConfig):
        """
        Initialize the policy network
        
        Args:
        config (PolicyConfig): Policy configuration
        """
        super().__init__()
        self.fc1 = nn.Linear(config.state_dim, config.hidden_dim)
        self.fc2 = nn.Linear(config.hidden_dim, config.action_dim)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass
        
        Args:
        x (torch.Tensor): Input tensor
        
        Returns:
        torch.Tensor: Output tensor
        """
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

class PolicyConfig:
    """Policy configuration"""
    
    def __init__(self):
        """
        Initialize the policy configuration
        """
        self.state_dim = 3
        self.action_dim = 3
        self.hidden_dim = 128
        self.lr = 0.001
        self.velocity_threshold = 1.0
        self.flow_direction = np.array([1, 0, 0])
        self.flow_threshold = 1.0

class PolicyData:
    """Policy data"""
    
    def __init__(self, states: np.ndarray, actions: np.ndarray, rewards: np.ndarray):
        """
        Initialize the policy data
        
        Args:
        states (np.ndarray): States
        actions (np.ndarray): Actions
        rewards (np.ndarray): Rewards
        """
        self.states = states
        self.actions = actions
        self.rewards = rewards

def get_logger(name: str) -> logging.Logger:
    """
    Get a logger
    
    Args:
    name (str): Logger name
    
    Returns:
    logging.Logger: Logger
    """
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    handler = logging.StreamHandler()
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    return logger

def get_device() -> torch.device:
    """
    Get the device (GPU or CPU)
    
    Returns:
    torch.device: Device
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    else:
        return torch.device('cpu')

if __name__ == "__main__":
    # Create a policy configuration
    config = PolicyConfig()
    
    # Create a policy
    policy = VelocityThresholdPolicy(config)
    
    # Create a policy data
    states = np.random.uniform(-1, 1, size=(100, 3))
    actions = np.random.uniform(-1, 1, size=(100, 3))
    rewards = np.random.uniform(-1, 1, size=(100,))
    policy_data = PolicyData(states, actions, rewards)
    
    # Train the policy
    policy.train(policy_data)