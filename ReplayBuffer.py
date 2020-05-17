import random
from collections import deque, namedtuple
import numpy as np
import torch


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, action_size, buffer_size, batch_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            batch_size (int): size of each training batch
        """
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  # internal memory (deque)
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "priority"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, priority):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, priority)
        self.memory.append(e)

    def sample(self, device):
        """Use priority to resample probablistically resample, weighted towards cases with higher error"""
        # get priorities
        priorities = [self.memory[i].priority for i in range(len(self))]

        # get sample numbers by priority
        cumsum_priorities = np.cumsum(priorities)
        stopping_values = [random.random()*sum(priorities) for i in range(self.batch_size)]
        stopping_values.sort()

        experience_idx = []
        experiences = []
        for i in range(len(cumsum_priorities)-1):
            if len(stopping_values) <= 0:
                break
            if stopping_values[0] < cumsum_priorities[i+1]:
                experience_idx.append(i)
                experiences.append(self.memory[i])
                stopping_values.pop(0)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)

        return states, actions, rewards, next_states, dones


    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)