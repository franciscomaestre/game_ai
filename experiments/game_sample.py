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

import gym
import cv2
import numpy as np
import math
import subprocess as sp

from gym import Wrapper
from gym.spaces import Box
from collections import deque

class Monitor:
    def __init__(self, width, height, saved_path):

        self.command = ["ffmpeg", "-y", "-f", "rawvideo", "-vcodec", "rawvideo", "-s", "{}X{}".format(width, height),
                        "-pix_fmt", "rgb24", "-r", "80", "-i", "-", "-an", "-vcodec", "mpeg4", saved_path]
        try:
            self.pipe = sp.Popen(self.command, stdin=sp.PIPE, stderr=sp.PIPE)
        except FileNotFoundError:
            pass

    def record(self, image_array):
        self.pipe.stdin.write(image_array.tostring())

env_name = "SuperMarioBros-1-1-v0"
env_params = "super_mario"
action = 2

params_manager = ParamsManager.getInstance()
env_params = params_manager.get_env_params(env_params)
env_params['useful_region'] = env_params['useful_region']['Default'] 
env_params['env_name'] = env_name
env_params['record'] = True

env, num_states, num_actions = make_train_env(env_params)

env.reset()

for _ in range(10000):
    observation, reward, done, info = env.step(env.action_space.sample())
    env.render()
