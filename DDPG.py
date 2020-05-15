# individual network settings for each actor + critic pair
# see networkforall for details

from model import Network
from utilities import hard_update, gumbel_softmax, onehot_from_logits
from torch.optim import Adam
import torch
import numpy as np
from OUNoise import OUNoise


class DDPGAgent:
    def __init__(self, in_actor, hidden_in_actor, hidden_out_actor, out_actor, in_critic, hidden_in_critic,
                 hidden_out_critic, device,lr_actor=1.0e-2, lr_critic=1.0e-2):
        super(DDPGAgent, self).__init__()

        self.actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)
        self.target_actor = Network(in_actor, hidden_in_actor, hidden_out_actor, out_actor, actor=True).to(device)
        self.target_critic = Network(in_critic, hidden_in_critic, hidden_out_critic, 1).to(device)

        self.noise = OUNoise(out_actor, scale=1.0)

        # initialize targets same as original networks
        hard_update(self.target_actor, self.actor)
        hard_update(self.target_critic, self.critic)

        self.actor_optimizer = Adam(self.actor.parameters(), lr=lr_actor)
        self.critic_optimizer = Adam(self.critic.parameters(), lr=lr_critic, weight_decay=1.e-5)

    def act(self, obs, device, noise=0.0, train=False):

        if train:
            return self.target_act(obs, device)

        obs = torch.from_numpy(obs).float().to(device)
        self.actor.eval()
        with torch.no_grad():
            action = self.actor(obs) + noise * self.noise.noise().to(device)
        self.actor.train()
        action = action.cpu().data.numpy()
        return np.clip(action, -1, 1)

    def reset(self):
        self.noise.reset()

    def target_act(self, obs, device, noise=0.0):
        action = self.target_actor(obs) + noise * self.noise.noise().to(device)
        action = action.cpu().data.numpy()
        action = np.clip(action, -1, 1)
        return torch.from_numpy(action).to(device)