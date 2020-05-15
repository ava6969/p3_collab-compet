# main code that contains the neural network setup
# policy + critic updates
# see ddpg.py for other details in the network

from DDPG import DDPGAgent
import torch
import numpy as np

from ReplayBuffer import ReplayBuffer
from utilities import soft_update, transpose_to_tensor, transpose_list

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class MADPG:
    def __init__(self, state_size,
                 num_agents,
                 action_size,
                 random_seed,
                 buffer_size=int(1e6),
                 discount_factor=0.95,
                 tau=1e-3,
                 lr_actor=1e-4,
                 lr_critic=1e-3,
                 update_every=5,
                 epochs=1,
                 batch_size=128
                 ):
        super(MADPG, self).__init__()

        # critic input = obs_full + actions = state_size + num_agents*actions = 20
        critic_state_size = state_size + num_agents*action_size
        self.maddpg_agent = [DDPGAgent(state_size, 256, 128,
                                       action_size, critic_state_size, 512, 128,
                                       lr_actor=lr_actor,
                                       lr_critic=lr_critic,
                                       device=device
                                       )
                             for _ in range(num_agents)]

        self.discount_factor = discount_factor
        self.tau = tau
        self.seed = np.random.seed(random_seed)
        self.buffer = ReplayBuffer(num_agents, buffer_size, random_seed)
        self.update_every = update_every
        self.epochs = epochs
        self.batch_size = batch_size

    def get_actors(self):
        """get actors of all the agents in the MADDPG object"""
        actors = [ddpg_agent.actor for ddpg_agent in self.maddpg_agent]
        return actors

    def get_target_actors(self):
        """get target_actors of all the agents in the MADDPG object"""
        target_actors = [ddpg_agent.target_actor for ddpg_agent in self.maddpg_agent]
        return target_actors

    def act(self, obs_all_agents, noise=0.0, train=False):
        """get actions from all agents in the MADDPG object"""
        if train:
            return [agent.act(obs_all_agents, device, noise, train) for agent in self.maddpg_agent]
        actions = [agent.act(obs, device, noise, train) for agent, obs in zip(self.maddpg_agent, obs_all_agents)]
        return actions

    def target_act(self, obs_all_agents, noise=0.0):
        """get target network actions from all the agents in the MADDPG object """
        target_actions = [agent.target_act(obs_all_agents, device, noise) for agent in self.maddpg_agent]
        return target_actions

    def reset(self):
        for agent in self.maddpg_agent:
            agent.reset()

    def save(self):
        for i, agent in enumerate(self.maddpg_agent):
            torch.save(agent.actor.state_dict(), 'actor{}.pth'.format(i))
            torch.save(agent.critic.state_dict(), 'critic{}.pth'.format(i))

    def load(self, actors_file_path, critics_file_path):
        assert len(actors_file_path) == len(self.maddpg_agent) == len(critics_file_path)
        for i, agent in enumerate(self.maddpg_agent):
            agent.actor.load_state_dict(torch.load(actors_file_path[i]))
            agent.actor.load_state_dict(torch.load(critics_file_path[i]))

    def update_all(self, time_step, samples):
        # add data to buffer
        states, actions, rewards, next_states, dones = samples
        for i, agent in enumerate(self.maddpg_agent):
            sample = states[i], actions, rewards[i], next_states[i], dones[i]
            self.buffer.push(i, sample)
            if time_step % self.update_every != 0:
                break

            # Learn, if enough samples are available in memory
            if self.buffer.length(i) > self.batch_size:
                for _ in range(self.epochs):
                    experiences = self.buffer.sample(self.batch_size, i, device)
                    self.update(experiences, agent, i)

    def update(self, samples, agent, agent_number):
        """update the critics and actors of all the agents """

        state = samples[0]
        action = samples[1]
        reward = samples[2]
        next_state = samples[3]
        done = samples[4]

        agent.critic_optimizer.zero_grad()

        # critic loss = batch mean of (y- Q(s,a) from target network)^2
        # y = reward of this timestep + discount * Q(st+1,at+1) from target network
        actions = self.target_act(next_state)
        target_actions = torch.cat(actions, dim=1).to(device)

        target_critic_input = torch.cat((next_state, target_actions), dim=1).to(device)

        with torch.no_grad():
            q_next = agent.target_critic(target_critic_input)

        y = reward + self.discount_factor * q_next * (1 - done)

        critic_input = torch.cat((state, action), dim=1).to(device)
        q = agent.critic(critic_input)

        huber_loss = torch.nn.SmoothL1Loss()
        critic_loss = huber_loss(q, y.detach())
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.critic.parameters(), 0.5)
        agent.critic_optimizer.step()

        # update actor network using policy gradient
        agent.actor_optimizer.zero_grad()
        # make input to agent
        # detach the other agents to save computation
        # saves some time for computing derivative
        q_input = self.act(state, train=True)

        q_input = torch.cat(q_input, dim=1).to(device)
        # combine all the actions and observations for input to critic
        # many of the obs are redundant, and obs[1] contains all useful information already
        q_input2 = torch.cat((state, q_input), dim=1).to(device)

        # get the policy gradient
        actor_loss = -agent.critic(q_input2).mean()
        actor_loss.backward()
        # torch.nn.utils.clip_grad_norm_(agent.actor.parameters(), 0.5)
        agent.actor_optimizer.step()

    def update_targets(self):
        """soft update targets"""
        self.iter += 1
        for ddpg_agent in self.maddpg_agent:
            soft_update(ddpg_agent.target_actor, ddpg_agent.actor, self.tau)
            soft_update(ddpg_agent.target_critic, ddpg_agent.critic, self.tau)


