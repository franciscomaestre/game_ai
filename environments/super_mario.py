"""
@author: Francisco Maestre. Modified version of Viet Nguyen
"""

import gym
import cv2
import numpy as np
import math
import subprocess as sp
import gym_super_mario_bros

from gym import Wrapper
from gym.spaces import Box
from collections import deque
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY

from .utils import ObservationEnv, MonitorEnv


def make_train_env(env_params):
    env = gym_super_mario_bros.make(env_params['env_name'])

    env = JoypadSpace(env, RIGHT_ONLY)

    '''
    A la hora de ejecutar los env, se ejecuta primero el último añadido (CustomStack) y de ahí a los anteriores
    '''

    if 'record' in env_params.keys() and env_params['record']:
        env = MonitorEnv(env, video_path="{}/video_{}.mp4".format(env_params['video_dir'], env_params['env_name']))
    
    env = CustomReward(env)
    env = ObservationEnv(env, frame_conf=env_params['useful_region'])        
    env = CustomSkipFrame(env, skip=env_params['skip_rate'])
    env = CustomStackFrame(env, stack=env_params['num_frames_to_stack'])

    return env, env.observation_space.shape[0], len(RIGHT_ONLY)            


class CustomReward(Wrapper):
    
    def __init__(self, env=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=1., shape=(1, 84, 84))
        self.curr_x_pos = 0
        self.curr_score = 0
        self.curr_life = 2

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self.reward(reward, done, info), done, info

    def reward(self, reward, done, info):
        reward = 0
        ## Si sube nuestro score damos un pequeño reward
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]

        ## Aprox. La x_pos llega a 4000. Aquí ponderamos para que la x_pos de hasta 20 puntos de reward
        reward += (info["x_pos"] - self.curr_x_pos) / 20.
        self.curr_x_pos = info["x_pos"]

        ## Penalizamos fuertemente perder una vida
        if info["life"] < self.curr_life:
            reward -= 100
        self.curr_life = info["life"]

        ## En caso de terminar la partida, si es porque hemos ganado, damos un premio, sino penalizamos
        if done:
            if info["flag_get"]:
                reward += 300

        return reward / 10.

    def reset(self):
        self.curr_x_pos = 0
        self.curr_life = 2
        self.curr_score = 0
        return self.env.reset()

class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=0):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=1., shape=(1, 84, 84))
        self.skip = skip

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        total_reward = reward
        for _skip in range(self.skip):
            if not done:
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
        return observation, total_reward, done, info

    def reset(self):
        return self.env.reset()

class CustomStackFrame(Wrapper):
    def __init__(self, env, stack=4, skip=0):
        super(CustomStackFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=1., shape=(4, 84, 84))
        self.stack = stack
        self.skip = skip

    def step(self, action):
        observations_list = []
        observation, reward, done, info = self.env.step(action)
        total_reward = reward
        observations_list.append(observation)
        for _stack in range(self.stack-1):
            if not done:
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                observations_list.append(observation)
            else:
                observations_list.append(observation)
        observations_list = np.concatenate(observations_list, 0)[None, :, :, :]
        return observations_list.astype(np.float32), total_reward, done, info

    def reset(self):
        observation = self.env.reset()
        observations_list = np.concatenate([observation for _ in range(self.stack)], 0)[None, :, :, :]
        return observations_list.astype(np.float32)
