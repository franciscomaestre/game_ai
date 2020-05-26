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

env = gym.make("SpaceInvaders-v0")
#env = gym.make("SuperMarioBros-1-1-v0")

env.reset()

params_manager = ParamsManager.getInstance()
env_params = params_manager.get_env_params('atari')
env_params['useful_region'] = env_params['useful_region']['SpaceInvaders'] 

action = 0

for i in range(185):
    env.step(action)

frame, reward0, terminal, info = env.step(action)
observation0 = process_frame_84(frame, env_params['useful_region'] )
print("After processing: " + str(np.array(observation0).shape))

## Imagen original
plt.imshow(np.array(np.squeeze(frame)))
plt.show()

## Imagen Modificada
plt.imshow(np.array(np.squeeze(observation0)))
plt.show()

done = False

## Imagen Apilada
observations_list = []
for i in range(4):
    if not done:
        observation, reward, done, info = env.step(action)
        observations_list.append(process_frame_84(observation, env_params['useful_region'] ))
    else:
        observations_list.append(observation)
observations_list = np.concatenate(observations_list, 0)[None, :, :, :]

print("After joining 4: " + str(np.array(observations_list).shape))

plt.imshow(np.array(np.squeeze(observations_list)))
plt.show()
