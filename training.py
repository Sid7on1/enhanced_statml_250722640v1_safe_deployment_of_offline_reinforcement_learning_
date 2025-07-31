import logging
import os
import sys
import time
from typing import Dict, List, Optional

import gym
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from gym import spaces
from gym.utils import seeding
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Constants and configuration
CONFIG = {
    "env_name": "ContinuousStirredTankReactor-v0",
    "num_episodes": 1000,
    "max_steps": 100,
    "batch_size": 32,
    "learning_rate": 0.001,
    "gamma": 0.99,
    "tau": 0.001,
    "buffer_size": 10000,
    "seed": 42,
}

# Exception classes
class TrainingError(Exception):
    pass

class EnvironmentError(Exception):
    pass

# Data structures/models
class Episode:
    def __init__(self, env, seed):
        self.env = env
        self.seed = seed
        self.episode_reward = 0
        self.episode_steps = 0
        self.state = env.reset()
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

    def step(self, action):
        next_state, reward, done, _ = self.env.step(action)
        self.episode_reward += reward
        self.episode_steps += 1
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        return next_state, reward, done

    def reset(self):
        self.state = self.env.reset()
        self.episode_reward = 0
        self.episode_steps = 0
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []

class Buffer:
    def __init__(self, buffer_size):
        self.buffer_size = buffer_size
        self.buffer = []

    def add(self, episode):
        self.buffer.append(episode)

    def sample(self, batch_size):
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        return [self.buffer[i] for i in indices]

class Agent(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Agent, self).__init__()
        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x, action):
        x = torch.cat((x, action), dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Utility methods
def seed_torch(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def create_env(env_name):
    env = gym.make(env_name)
    env.seed(CONFIG["seed"])
    return env

def create_buffer(buffer_size):
    buffer = Buffer(buffer_size)
    return buffer

def create_agent(state_dim, action_dim):
    agent = Agent(state_dim, action_dim)
    return agent

def create_critic(state_dim, action_dim):
    critic = Critic(state_dim, action_dim)
    return critic

def train_agent(env, agent, critic, buffer, num_episodes, max_steps, batch_size, learning_rate, gamma, tau):
    for episode in range(num_episodes):
        episode_reward = 0
        episode_steps = 0
        state = env.reset()
        done = False

        while not done and episode_steps < max_steps:
            action = agent(state)
            next_state, reward, done, _ = env.step(action)
            episode_reward += reward
            episode_steps += 1

            buffer.add(Episode(env, CONFIG["seed"]))

            if len(buffer.buffer) >= batch_size:
                batch = buffer.sample(batch_size)
                states = torch.tensor([e.state for e in batch], dtype=torch.float32)
                actions = torch.tensor([e.actions[-1] for e in batch], dtype=torch.float32)
                rewards = torch.tensor([e.rewards[-1] for e in batch], dtype=torch.float32)
                next_states = torch.tensor([e.next_states[-1] for e in batch], dtype=torch.float32)
                dones = torch.tensor([e.dones[-1] for e in batch], dtype=torch.float32)

                critic_loss = 0
                for i in range(batch_size):
                    critic_loss += (critic(states[i], actions[i]) - rewards[i]).pow(2).mean()

                critic_optimizer = optim.Adam(critic.parameters(), lr=learning_rate)
                critic_optimizer.zero_grad()
                critic_loss.backward()
                critic_optimizer.step()

                actor_loss = 0
                for i in range(batch_size):
                    actor_loss += (agent(states[i]) - actions[i]).pow(2).mean()

                actor_optimizer = optim.Adam(agent.parameters(), lr=learning_rate)
                actor_optimizer.zero_grad()
                actor_loss.backward()
                actor_optimizer.step()

        logger.info(f"Episode {episode+1}, Reward: {episode_reward}, Steps: {episode_steps}")

def main():
    env = create_env(CONFIG["env_name"])
    buffer = create_buffer(CONFIG["buffer_size"])
    agent = create_agent(env.observation_space.shape[0], env.action_space.shape[0])
    critic = create_critic(env.observation_space.shape[0], env.action_space.shape[0])

    train_agent(env, agent, critic, buffer, CONFIG["num_episodes"], CONFIG["max_steps"], CONFIG["batch_size"], CONFIG["learning_rate"], CONFIG["gamma"], CONFIG["tau"])

if __name__ == "__main__":
    seed_torch(CONFIG["seed"])
    main()