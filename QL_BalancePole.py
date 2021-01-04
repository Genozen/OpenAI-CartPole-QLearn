import gym
import numpy as np
import matplotlib.pyplot as plt
import math
import time
import pickle #to save q_table

#create env
env = gym.make("CartPole-v1")
env.reset()

# print(f"env space high: {env.observation_space.high}, length = {len(env.observation_space.high)}")
# print(f"env space low: {env.observation_space.low}")# [position of cart, velocity of cart, angle of pole, rotation rate of pole]
# print(f"env space n: {env.observation_space.n}") # how many actions we can take

EPISODES = 3001
ALLOWED_STEPS = 500
SHOW_EVERY = 3000 #render every 500 episodes

#QL variables
LEARNING_RATE = 0.1
DISCOUNT = 0.95

DISCRETE_OS_SIZE = [20] * len(env.observation_space.high)# [20 20 20 20]
discrete_os_win_size = (env.observation_space.high - env.observation_space.low)/DISCRETE_OS_SIZE #chunks

epsilon = 0.5
START_EPSILON_DECAYING = 1
END_EPSILON_DECAYING = EPISODES//2
epsilon_decay_value = epsilon/(END_EPSILON_DECAYING-START_EPSILON_DECAYING)

ep_rewards = []
aggr_ep_rewards = {'ep':[],'avg':[],'min':[],'max':[]}


q_table = np.random.uniform(low=-2, high = 0, size=(DISCRETE_OS_SIZE + [env.action_space.n])) #create a 20 x 20 table with 2 actions available (aciton_space.n)

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
    print(f"Trining episode {episode}")
    #allows limited steps for each episode
    discrete_state = get_discrete_state(env.reset()) #initialize the env state every new episode
    done = False
    duration = 0 #duration of pole keeping upright
    for steps in range(ALLOWED_STEPS):

        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = env.action_space.sample() #take random action

        new_state, reward, done, info = env.step(action)
        new_discrete_state = get_discrete_state(new_state)
        duration += reward


        if -math.radians(12) <= new_state[2] <= math.radians(12) and -1.2 <= new_state[0] <= 1.2:
            duration+=0.3
        #print(new_state[2])
        if(episode % SHOW_EVERY == 0):
            env.render()

        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = current_q+LEARNING_RATE*(duration+DISCOUNT*max_future_q - current_q)
            q_table[discrete_state+(action,)] = new_q

        #Goal
        elif duration >= ALLOWED_STEPS * (1 - 0.1):
            print(f"We made it at episode {episode}")
            q_table[discrete_state+(action,)] += 100

        if not -math.radians(12) <= new_state[2] <= math.radians(12) or not -2.4 <= new_state[0] <= 2.4:
            q_table[discrete_state+(action,)] = -2
            break
        elif steps == ALLOWED_STEPS:
            q_table[discrete_state+(action,)] = 100 + duration
        discrete_state = new_discrete_state

    if END_EPSILON_DECAYING >= episode >= START_EPSILON_DECAYING:
        epsilon -= epsilon_decay_value

    ep_rewards.append(duration)

    average_reward = sum(ep_rewards[-SHOW_EVERY:])/len(ep_rewards[-SHOW_EVERY:])
    aggr_ep_rewards['ep'].append(episode)
    aggr_ep_rewards['avg'].append(average_reward)
    aggr_ep_rewards['min'].append(min(ep_rewards[-SHOW_EVERY:]))
    aggr_ep_rewards['max'].append(max(ep_rewards[-SHOW_EVERY:]))
    #print(f"Episode: {episode} avg: {average_reward} min: {min(ep_rewards[-SHOW_EVERY:])} max: {max(ep_rewards[-SHOW_EVERY:])}")
env.close()


with open(f"qtable-cartpole-{int(time.time())}.pickle", "wb") as f:
     pickle.dump(q_table,f)

plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['avg'], label = "avg")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['min'], label = "min")
plt.plot(aggr_ep_rewards['ep'], aggr_ep_rewards['max'], label = "max")
plt.legend(loc = 4) #legend at lower right
plt.show()
