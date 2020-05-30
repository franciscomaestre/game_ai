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

    if env_params['escenario'] == 'escenario_1':
        env = CustomRewardBasic(env)
        env = ObservationEnv(env, env_params['useful_region'], monitor, version=1)

    if env_params['escenario'] == 'escenario_2':
        env = CustomRewardComplete(env)
    
    if env_params['escenario'] == 'escenario_3':
        env = CustomRewardBasic(env)
        env_params['useful_region']['crop_x'] = 0
        env_params['useful_region']['crop_y'] = 0
        env = ObservationEnv(env, env_params['useful_region'], monitor, version=2)
        
    if env_params['escenario'] == 'escenario_4':
        env = CustomRewardBasic(env)
        env_params['useful_region']['crop_x'] = 0
        env_params['useful_region']['crop_y'] = 20
        env = ObservationEnv(env, env_params['useful_region'], monitor, version=2)

    if env_params['escenario'] == 'escenario_5':
        env_params['useful_region']['crop_x'] = 0
        env_params['useful_region']['crop_y'] = 0
        env = CustomRewardCompleteCrop(env, env_params['useful_region'])
    
    if env_params['escenario'] == 'escenario_6':
        env_params['useful_region']['crop_x'] = 0
        env_params['useful_region']['crop_y'] = 20
        env = CustomRewardCompleteCrop(env, env_params['useful_region'])

    env = CustomSkipFrame(env)

    return env, env.observation_space.shape[0], len(RIGHT_ONLY)            


class CustomRewardCompleteCrop(Wrapper):
    
    def __init__(self, env=None, frame_conf=None):
        super(CustomRewardCompleteCrop, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_x_pos = 0
        self.curr_y_pos = 0
        self.curr_score = 0
        self.curr_life = 2
        self.curr_time = 400
        self.frame_conf = frame_conf

    def _process_observation(self,observation):
        observation = cv2.cvtColor(cv2.resize(observation, ( self.frame_conf['scale'] + self.frame_conf["crop_x"], self.frame_conf['scale'] + self.frame_conf["crop_y"])), cv2.COLOR_RGB2GRAY)
        observation = observation[self.frame_conf["crop_y"]:self.frame_conf["crop_y"]+self.frame_conf['scale'],self.frame_conf["crop_x"]:self.frame_conf["crop_x"]+self.frame_conf['scale']]
        observation = cv2.resize(observation, (84, 84))[None, :, :] / 255.
        if self.frame_conf['binary']:
            ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
        return np.reshape(observation,( 1, 84, 84))
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        
        frame_observation = self._process_observation(observation)

        return frame_observation, self._reward(reward, done, info) , done, info

    def _reward(self, reward, done, info):

        ## Si sube nuestro score damos un pequeño reward. Si ha sido gracias a coger la bandera, el reward es mayor
        reward += (info['score']-self.curr_score)/20.
        self.curr_score = info["score"]

        ## Hacemos que la X
        reward += (info["x_pos"] - self.curr_x_pos)/40.
        self.curr_x_pos = info["x_pos"]

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

class CustomRewardComplete(Wrapper):
    
    def __init__(self, env=None):
        super(CustomRewardComplete, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_x_pos = 0
        self.curr_y_pos = 0
        self.curr_score = 0
        self.curr_life = 2
        self.curr_time = 400

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

        ## Si sube nuestro score damos un pequeño reward. Si ha sido gracias a coger la bandera, el reward es mayor
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]

        ## Aprox. La x_pos llega a 4000. Aquí ponderamos para que la x_pos de hasta 200 puntos de reward
        reward += (info["x_pos"] - self.curr_x_pos) / 20.
        self.curr_x_pos = info["x_pos"]

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


class CustomRewardBasic(Wrapper):
    
    def __init__(self, env=None):
        super(CustomRewardBasic, self).__init__(env)
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

        ## Si sube nuestro score damos un pequeño reward. Si ha sido gracias a coger la bandera, el reward es mayor
        reward += (info["score"] - self.curr_score) / 40.
        self.curr_score = info["score"]

        ## Aprox. La x_pos llega a 4000. Aquí ponderamos para que la x_pos de hasta 200 puntos de reward
        reward += (info["x_pos"] - self.curr_x_pos) / 20.
        self.curr_x_pos = info["x_pos"]

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
