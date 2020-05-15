
from unityagents import UnityEnvironment
import numpy as np
from MADPG import MADPG
from collections import deque
import matplotlib.pyplot as plt

seed = 4
env = UnityEnvironment(file_name="Tennis",seed=seed)

# get the default brain
brain_name = env.brain_names[0]
brain = env.brains[brain_name]

# reset the environment
env_info = env.reset(train_mode=True)[brain_name]
# number of agents 
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size
states = env_info.vector_observations
state_size = states.shape[1]


print_every = 500
save_every = 500
agent = MADPG(state_size, num_agents, action_size, seed)

scores_max_hist = []
scores_mean_hist = []


def train(n_episodes=10000, max_t=1000, eps_start=1.0, eps_end=0.1, eps_decay=0.000001):
    """DPG.
    
    Params
    ======
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """

    scores_window = deque(maxlen=print_every)  # last 5 scores
    eps = eps_start                    # initialize epsilon
    
    for i_episode in range(1, n_episodes+1):
        env_info = env.reset(train_mode=True)[brain_name] # reset the environment
        states = env_info.vector_observations          # get the current state
        agent.reset()
        scores = np.zeros(num_agents)
        
        for t in range(max_t):
            actions = agent.act(states, eps)
            env_info = env.step(actions)[brain_name]        # send the action to the environment
            next_states = env_info.vector_observations
            rewards = env_info.rewards
            dones = env_info.local_done
            scores += rewards
            samples = states, actions, rewards, next_states, dones
            agent.update_all(t, samples)
            states = next_states

            if np.any(dones):
                break

        score_max = np.max(scores)
        scores_window.append(score_max)
        score_mean = np.mean(scores_window)

        scores_max_hist.append(score_max)
        scores_mean_hist.append(score_mean)
        eps = max(eps_end, eps-eps_decay) # decrease epsilon

        if i_episode % print_every == 0:
            print('\rEpisode {}\tAverage Score: {:.5f}, Max Scored: {:.5f}'
                  .format(i_episode, np.mean(scores_window), score_max))

        if i_episode % save_every == 0:
            agent.save()

        if np.mean(scores_window) >= 30.0:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.5f}'.format(i_episode-save_every, np.mean(scores_window)))
            agent.save()
            break


train()
fig = plt.figure()
ax = fig.add_subplot(111)
plt.plot(np.arange(1, len(scores_max_hist)+1), scores_max_hist, label='score')
plt.plot(np.arange(1, len(scores_mean_hist)+1), scores_mean_hist, label='average score')
plt.ylabel('Score')
plt.xlabel('Episode #')
plt.legend(loc='upper left')
plt.show()
env.close()
