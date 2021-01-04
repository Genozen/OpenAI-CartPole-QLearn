import gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import math
import time
#create env
env = gym.make("CartPole-v1")
env.reset()

start_q_table = "qtable-cartpole-1607941327.pickle" # start of as nothing, or from filename

with open(start_q_table, "rb") as f:
    q_table = pickle.load(f)

EPISODES = 100
ALLOWED_STEPS = 1000
LEARNING_RATE = 0.1
DISCOUNT = 0.95

ep_rewards = []
aggr_ep_rewards = {'ep':[],'avg':[],'min':[],'max':[]}

time_tracked = []

buckets=(6, 6, 6, 6)
#convert the continuous value (float) into integers that fits in the discrete space...(chunking)
def get_discrete_state(state):
    upper_bounds = [env.observation_space.high[0], 0.5, env.observation_space.high[2], math.radians(50)]
    lower_bounds = [env.observation_space.low[0], -0.5, env.observation_space.low[2], -math.radians(50)]
    ratios = [(state[i] + abs(lower_bounds[i])) / (upper_bounds[i] - lower_bounds[i]) for i in range(len(state))]
    new_obs = [int(round((buckets[i] - 1) * ratios[i])) for i in range(len(state))]
    new_obs = [min(buckets[i] - 1, max(0, new_obs[i])) for i in range(len(state))]
    return tuple(new_obs)

#set the amount of episodes to train
for episode in range(EPISODES):
    start_time = time.time()
    discrete_state = get_discrete_state(env.reset()) #initialize the env state every new episode
    done = False
    duration = 0
    #for steps in range(ALLOWED_STEPS):
    while not done:
        action = np.argmax(q_table[discrete_state])

        new_state, reward, done, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        duration += reward
        #env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = current_q+LEARNING_RATE*(duration+DISCOUNT*max_future_q - current_q)
            q_table[discrete_state+(action,)] = new_q

        if not -math.radians(12) <= new_state[2] <= math.radians(12) or not -2.4 <= new_state[0] <= 2.4:
            break

        discrete_state = new_discrete_state

    # ep_rewards.append(duration)
    #
    # aggr_ep_rewards['ep'].append(episode)
    # aggr_ep_rewards['avg'].append(duration)
    end_time = time.time()

    time_tracked.append(end_time-start_time)
#print(q_table)
env.close()

plt.plot(time_tracked)
plt.show()

# plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = "avg")
# plt.legend(loc = 4) #legend at lower right
# plt.show()
