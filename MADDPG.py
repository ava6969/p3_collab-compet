from DDPG import DDPG
import numpy as np


class MADDPG:
    def __init__(self, num_agents, state_size, action_size, random_seed):
        self.agents = [DDPG(state_size, action_size, num_agents=1, random_seed=random_seed, agent_id=i)
                       for i in range(num_agents)]
        self.num_agents = num_agents
        self.action_size = action_size

    def act(self, states, add_noise):
        actions = [a.act(states, add_noise) for a in self.agents]
        return np.reshape(actions, (1, self.num_agents * self.action_size))

    def step(self, states, actions, rewards, next_states, done):
        for i, agent in enumerate(self.agents):
            agent.step(states, actions, rewards[i], next_states, done[i])

    def reset(self):
        for a in self.agents:
            a.reset()

    def save(self):
        for a in self.agents:
            a.save()

    def load(self):
        for a in self.agents:
            a.load()

