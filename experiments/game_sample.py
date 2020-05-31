## http://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/
import matplotlib.pyplot as plt

from matplotlib import animation
from time import gmtime, strftime
import random
import cv2
import sys
import numpy as np 

import gym
import gym_super_mario_bros

from utils.params_manager import ParamsManager
from environments import make_train_env

env_name = "SuperMarioBros-1-1-v0"
env_params = "super_mario"
action = 2

params_manager = ParamsManager.getInstance()
env_params = params_manager.get_env_params(env_params)
env_params['useful_region'] = env_params['useful_region']['Default'] 
env_params['env_name'] = env_name

env, num_states, num_actions = make_train_env(env_params)

env.reset()

observation, reward, done, info = env.step(env.action_space.sample())

print("After processing: " + str(np.array(observation).shape))

plt.imshow(np.array(np.concatenate(observation[0],0)))
plt.show()
