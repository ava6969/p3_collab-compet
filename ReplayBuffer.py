from collections import deque, namedtuple
import random
from pandas.core.common import flatten
import torch
import numpy as np


class ReplayBuffer:
    def __init__(self, num_agents, size, seed):
        self.size = size
        self.buffer = [deque(maxlen=self.size) for _ in range(num_agents)]
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)

    def push(self, agent_idx, transition):
        """push into the buffer"""
        state, action, reward, next_state, done = transition
        action = list(flatten(action))
        e = self.experience(state, action, reward, next_state, done)
        self.buffer[agent_idx].append(e)

    def sample(self, batch_size, agent_idx, device):
        """sample from the buffer"""
        experiences = random.sample(self.buffer[agent_idx], batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(
            device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(
            device)
        return np.array([states, actions, rewards, next_states, dones])

    def length(self, idx):
        return len(self.buffer[idx])




