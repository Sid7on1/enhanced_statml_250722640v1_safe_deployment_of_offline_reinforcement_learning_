import logging
import numpy as np
import torch
from typing import Dict, List, Tuple
from abc import ABC, abstractmethod
from collections import deque
from config import Config

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class Memory:
    def __init__(self, config: Config):
        self.config = config
        self.memory = deque(maxlen=self.config.memory_size)
        self.experience = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": []
        }

    def add_experience(self, state, action, reward, next_state, done):
        self.experience["states"].append(state)
        self.experience["actions"].append(action)
        self.experience["rewards"].append(reward)
        self.experience["next_states"].append(next_state)
        self.experience["dones"].append(done)

    def sample_experience(self, batch_size: int) -> Dict[str, torch.Tensor]:
        indices = np.random.choice(len(self.experience["states"]), batch_size, replace=False)
        states = torch.tensor([self.experience["states"][i] for i in indices], dtype=torch.float32)
        actions = torch.tensor([self.experience["actions"][i] for i in indices], dtype=torch.long)
        rewards = torch.tensor([self.experience["rewards"][i] for i in indices], dtype=torch.float32)
        next_states = torch.tensor([self.experience["next_states"][i] for i in indices], dtype=torch.float32)
        dones = torch.tensor([self.experience["dones"][i] for i in indices], dtype=torch.bool)
        return {
            "states": states,
            "actions": actions,
            "rewards": rewards,
            "next_states": next_states,
            "dones": dones
        }

    def get_memory_size(self) -> int:
        return len(self.experience["states"])

    def get_experience(self) -> Dict[str, List]:
        return self.experience

    def clear_experience(self):
        self.experience = {
            "states": [],
            "actions": [],
            "rewards": [],
            "next_states": [],
            "dones": []
        }

class ExperienceReplayBuffer(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_experience(self, batch_size: int) -> Tuple[List, List, List, List, List]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = [self.buffer[i][0] for i in indices]
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        next_states = [self.buffer[i][3] for i in indices]
        dones = [self.buffer[i][4] for i in indices]
        return states, actions, rewards, next_states, dones

class PrioritizedExperienceReplayBuffer(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)
        self.priorities = deque(maxlen=self.config.memory_size)

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(1.0)

    def sample_experience(self, batch_size: int) -> Tuple[List, List, List, List, List, List]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=[p**self.config.priority_alpha for p in self.priorities])
        states = [self.buffer[i][0] for i in indices]
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        next_states = [self.buffer[i][3] for i in indices]
        dones = [self.buffer[i][4] for i in indices]
        priorities = [self.priorities[i] for i in indices]
        return states, actions, rewards, next_states, dones, priorities

class ExperienceReplayBufferWithPriorities(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)
        self.priorities = deque(maxlen=self.config.memory_size)

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(1.0)

    def sample_experience(self, batch_size: int) -> Tuple[List, List, List, List, List, List]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=[p**self.config.priority_alpha for p in self.priorities])
        states = [self.buffer[i][0] for i in indices]
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        next_states = [self.buffer[i][3] for i in indices]
        dones = [self.buffer[i][4] for i in indices]
        priorities = [self.priorities[i] for i in indices]
        return states, actions, rewards, next_states, dones, priorities

    def update_priority(self, index: int, priority: float):
        self.priorities[index] = priority

class ExperienceReplayBufferWithStateActionRewardNextStateDone(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample_experience(self, batch_size: int) -> Tuple[List, List, List, List, List]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        states = [self.buffer[i][0] for i in indices]
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        next_states = [self.buffer[i][3] for i in indices]
        dones = [self.buffer[i][4] for i in indices]
        return states, actions, rewards, next_states, dones

class ExperienceReplayBufferWithStateActionRewardNextStateDoneAndPriority(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)
        self.priorities = deque(maxlen=self.config.memory_size)

    def add_experience(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
        self.priorities.append(1.0)

    def sample_experience(self, batch_size: int) -> Tuple[List, List, List, List, List, List]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=[p**self.config.priority_alpha for p in self.priorities])
        states = [self.buffer[i][0] for i in indices]
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        next_states = [self.buffer[i][3] for i in indices]
        dones = [self.buffer[i][4] for i in indices]
        priorities = [self.priorities[i] for i in indices]
        return states, actions, rewards, next_states, dones, priorities

    def update_priority(self, index: int, priority: float):
        self.priorities[index] = priority

class ExperienceReplayBufferWithStateActionRewardNextStateDoneAndPriorityAndTransition(Memory):
    def __init__(self, config: Config):
        super().__init__(config)
        self.buffer = deque(maxlen=self.config.memory_size)
        self.priorities = deque(maxlen=self.config.memory_size)

    def add_experience(self, state, action, reward, next_state, done, transition):
        self.buffer.append((state, action, reward, next_state, done, transition))
        self.priorities.append(1.0)

    def sample_experience(self, batch_size: int) -> Tuple[List, List, List, List, List, List, List]:
        indices = np.random.choice(len(self.buffer), batch_size, replace=False, p=[p**self.config.priority_alpha for p in self.priorities])
        states = [self.buffer[i][0] for i in indices]
        actions = [self.buffer[i][1] for i in indices]
        rewards = [self.buffer[i][2] for i in indices]
        next_states = [self.buffer[i][3] for i in indices]
        dones = [self.buffer[i][4] for i in indices]
        transitions = [self.buffer[i][5] for i in indices]
        priorities = [self.priorities[i] for i in indices]
        return states, actions, rewards, next_states, dones, transitions, priorities

    def update_priority(self, index: int, priority: float):
        self.priorities[index] = priority