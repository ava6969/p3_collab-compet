
from unityagents import UnityEnvironment
import numpy as np
from collections import deque
import matplotlib.pyplot as plt

from MADDPG import MADDPG

seed = 4
env = UnityEnvironment(file_name="Tennis", seed=seed, worker_id=2)

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


MAX_EPISODES = 5000
SOLVED_SCORE = 0.5    # FOR TESTING ONLY - should be 0.5
WINDOW_SIZE = 100       # FOR TESTING ONLY - should be 100
PRINT_EVERY = 10
ADD_NOISE = True
STOP_IF_NO_IMPROVEMENT_OVER_EPISODES = 300
RAND_SEED = 6

agents = MADDPG(num_agents, state_size, action_size, seed)


def maddpg():
    # initialize scores
    scores_window = deque(maxlen=WINDOW_SIZE)
    scores_all = []
    moving_average = []
    best_score = -np.inf
    best_episode = 0
    env_solved = False

    for i_episode in range(1, MAX_EPISODES + 1):
        env_info = env.reset(train_mode=True)[brain_name]  # reset the environment
        states = np.reshape(env_info.vector_observations, (1, num_agents * state_size))  # flatten states
        agents.reset()
        scores = np.zeros(num_agents)
        while True:
            actions = agents.act(states, ADD_NOISE)  # choose agent actions and flatten them
            env_info = env.step(actions)[brain_name]  # send both agents' actions to the environment
            next_states = np.reshape(env_info.vector_observations, (1, num_agents * state_size))  # flatten next states
            rewards = env_info.rewards  # get rewards
            done = env_info.local_done  # see if the episode finished
            agents.step(states, actions, rewards, next_states, done)  # perform the learning step
            scores += np.max(rewards)  # update scores with best reward
            states = next_states  # roll over states to next time step
            if np.any(done):  # exit loop if episode finished
                break

        ep_best_score = np.max(scores)  # record best score for episode
        scores_window.append(ep_best_score)  # add score to recent scores
        scores_all.append(ep_best_score)  # add score to history of all scores
        moving_average.append(np.mean(scores_window))  # recalculate moving average

        # save best score
        if ep_best_score > best_score and ep_best_score > 0:
            best_score = ep_best_score
            best_episode = i_episode
            print('New best score found on episode {:d}; score = {:.3f}'.format(best_episode, best_score))

        # print results
        if i_episode % PRINT_EVERY == 0:
            print('Episodes {:0>4d}-{:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
                i_episode - PRINT_EVERY, i_episode, np.max(scores_all[-PRINT_EVERY:]), moving_average[-1]))
            agents.save()

        # determine if environment is solved and keep best performing models
        if moving_average[-1] >= SOLVED_SCORE:
            if not env_solved:
                print('<<< Environment solved in {:d} episodes! \
                \n<<< Moving Average: {:.3f} over past {:d} episodes'.format(
                    i_episode - WINDOW_SIZE, moving_average[-1], WINDOW_SIZE))
                env_solved = True

                # save weights
                agents.save()
                break

            elif ep_best_score >= best_score:
                print('<<< Best episode so far!\
                \nEpisode {:0>4d}\tMax Reward: {:.3f}\tMoving Average: {:.3f}'.format(
                    i_episode, ep_best_score, moving_average[-1]))

                # save weights
                agents.save()

            # stop training if model stops improving
            elif (i_episode - best_episode) >= STOP_IF_NO_IMPROVEMENT_OVER_EPISODES:
                print('<-- Training stopped. Best score not matched or exceeded for',
                      STOP_IF_NO_IMPROVEMENT_OVER_EPISODES, 'episodes')
                break
            else:
                continue

    return scores_all, moving_average


scores, avgs = maddpg()

plt.figure()
plt.plot(scores, label='MADDPG')
plt.plot(avgs, c='r', label='moving avg')
plt.ylabel('Score')
plt.xlabel('Episode')
plt.legend(loc='upper left')
plt.show()
plt.show()
env.close()
