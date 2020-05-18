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

def preprocess(observation):
    observation = cv2.cvtColor(cv2.resize(observation, (84, 110)), cv2.COLOR_RGB2GRAY)
    observation = observation[26:110,:]
    #ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,( 1, 84, 84))

#env = gym.make("SpaceInvaders-v0")
env = gym.make("SuperMarioBros-1-1-v0")

env.reset()

frame, reward0, terminal, info = env.step(0)
observation0 = preprocess(frame)
print("After processing: " + str(np.array(observation0).shape))
plt.imshow(np.array(np.squeeze(observation0)))
plt.show()

frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
plt.imshow(np.array(np.squeeze(frame)))
plt.show()
