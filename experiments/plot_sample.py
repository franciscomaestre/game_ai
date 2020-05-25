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

from params_manager import ParamsManager

def process_frame_84(observation, frame_conf):
    observation = cv2.cvtColor(cv2.resize(observation, ( frame_conf['scale'] + frame_conf["crop_x"], frame_conf['scale'] + frame_conf["crop_y"])), cv2.COLOR_RGB2GRAY)
    observation = observation[frame_conf["crop_y"]:frame_conf["crop_y"]+frame_conf['scale'],frame_conf["crop_x"]:frame_conf["crop_x"]+frame_conf['scale']]
    observation = cv2.resize(observation, (84, 84))
    if frame_conf['binary']:
        ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,( 1, 84, 84))

#env = gym.make("SpaceInvaders-v0")
env = gym.make("SuperMarioBros-1-1-v0")

env.reset()

params_manager = ParamsManager.getInstance()
env_params = params_manager.get_env_params('super_mario')
env_params['useful_region'] = env_params['useful_region']['Default'] 


frame, reward0, terminal, info = env.step(0)
observation0 = process_frame_84(frame, env_params['useful_region'] )
print("After processing: " + str(np.array(observation0).shape))

plt.imshow(np.array(np.squeeze(frame)))
plt.show()

plt.imshow(np.array(np.squeeze(observation0)))
plt.show()

