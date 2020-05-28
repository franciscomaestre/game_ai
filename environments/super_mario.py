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

from .utils import ObservationEnv, Monitor


def make_train_env(env_params):
    env = gym_super_mario_bros.make(env_params['env_name'])
    
    if 'video' in env_params.keys() and env_params['video']:
        video_path = "{}/video_{}.mp4".format(env_params['video_dir'], env_params['env_name'])
        monitor = Monitor(256, 240, video_path)
    else:
        monitor = None

    env = JoypadSpace(env, RIGHT_ONLY)
    env = CustomReward(env)
    env = ObservationEnv(env, env_params['useful_region'], monitor, version=2)
    env = CustomSkipFrame(env)


    return env, env.observation_space.shape[0], len(RIGHT_ONLY)            

'''
class CustomReward(Wrapper):
    
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_x_pos = 0
        self.curr_y_pos = 0
        self.curr_score = 0
        self.curr_life = 2
        self.curr_time = 400
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def _process_observation(self,frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
            return frame
        else:
            return np.zeros((1, 84, 84))
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        frame_observation = self._process_observation(observation)

        return frame_observation, self._reward(reward, done, info) , done, info

    def _reward(self, reward, done, info):

        #print(info)

        ## Penalizamos el que pase el tiempo en una posición sin moverse (se ha quedado bloqueado)
        if info["time"] < self.curr_time:
            if info["x_pos"] == self.curr_x_pos and info["y_pos"] == self.curr_y_pos:
                reward -= 1
        self.curr_curr_time = info["time"]

        ## Si sube nuestro score damos un pequeño reward. Si ha sido gracias a coger la bandera, el reward es mayor
        if info["score"] > self.curr_score:
            if info["score"] > (self.curr_score+1000):
                reward += 100
            else:
                reward += 5
        self.curr_score = info["score"]

        ## Aprox. La x_pos llega a 4000. Aquí ponderamos para que la x_pos de hasta 200 puntos de reward
        reward += (info["x_pos"] - self.curr_x_pos) / 20.
        self.curr_x_pos = info["x_pos"]

        ## No vamos a modificar su comportamiento con la Y, pero si vamos a controlar su valor
        self.curr_y_pos = info["y_pos"]

        ## Penalizamos fuertemente perder una vida
        if info["life"] < self.curr_life:
            reward -= 100
        self.curr_life = info["life"]

        ## En caso de terminar la partida, si es porque hemos ganado, damos un premio, sino penalizamos
        if done:
            if info["flag_get"]:
                reward += 100
            else:
                reward -= 100

        return reward / 10.

    def reset(self):
        self.curr_x_pos = 0
        self.curr_y_pos = 0
        self.curr_life = 2
        self.curr_score = 0
        self.curr_time = 400
        return self._process_observation(self.env.reset())
'''


class CustomReward(Wrapper):
    
    def __init__(self, env=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_x_pos = 0
        self.curr_y_pos = 0
        self.curr_score = 0
        self.curr_life = 2
        self.curr_time = 400

    def step(self, action):
        observation, reward, done, info = self.env.step(action)

        return observation, self._reward(reward, done, info) , done, info

    def _reward(self, reward, done, info):

        #print(info)

        ## Penalizamos el que pase el tiempo en una posición sin moverse (se ha quedado bloqueado)
        if info["time"] < self.curr_time:
            if info["x_pos"] == self.curr_x_pos and info["y_pos"] == self.curr_y_pos:
                reward -= 1
        self.curr_curr_time = info["time"]

        ## Si sube nuestro score damos un pequeño reward. Si ha sido gracias a coger la bandera, el reward es mayor
        if info["score"] > self.curr_score:
            if info["score"] > (self.curr_score+1000):
                reward += 100
            else:
                reward += 5
        self.curr_score = info["score"]

        ## Aprox. La x_pos llega a 4000. Aquí ponderamos para que la x_pos de hasta 200 puntos de reward
        reward += (info["x_pos"] - self.curr_x_pos) / 20.
        self.curr_x_pos = info["x_pos"]

        ## No vamos a modificar su comportamiento con la Y, pero si vamos a controlar su valor
        self.curr_y_pos = info["y_pos"]

        ## Penalizamos fuertemente perder una vida
        if info["life"] < self.curr_life:
            reward -= 100
        self.curr_life = info["life"]

        ## En caso de terminar la partida, si es porque hemos ganado, damos un premio, sino penalizamos
        if done:
            if info["flag_get"]:
                reward += 100
            else:
                reward -= 100

        return reward / 10.

    def reset(self):
        self.curr_x_pos = 0
        self.curr_y_pos = 0
        self.curr_life = 2
        self.curr_score = 0
        self.curr_time = 400
        return self.env.reset()



class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        observations_list = []
        observation, reward, done, info = self.env.step(action)
        for _ in range(self.skip):
            if not done:
                observation, reward, done, info = self.env.step(action)
                total_reward += reward
                observations_list.append(observation)
            else:
                observations_list.append(observation)
        observations_list = np.concatenate(observations_list, 0)[None, :, :, :]
        return observations_list.astype(np.float32), reward, done, info

    def reset(self):
        observation = self.env.reset()
        observations_list = np.concatenate([observation for _ in range(self.skip)], 0)[None, :, :, :]
        return observations_list.astype(np.float32)
