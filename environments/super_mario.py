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


'''
La clase CustomReward es un Wrapper que sobreescribe ciertas funciones del Env, pero sigue siendo el mismo. Este Wrapper está pensado para Super Mario.
Básicamente, sobreescribimos la forma en la que va a detectar la puntucación y de esta forma el saber si está avanzando correctamente.

Se podría haber usado el que hay por defecto, pero ese no está pintando por pantalla. Aquí hemos mezclado el step con el que este salga por pantalla.

'''

"""
Preparamos el enviroment haciendo uso de los wrapper preparados para el SuperMario
"""

def make_train_env(env_params):
    env = gym_super_mario_bros.make(env_params['env_name'])
    
    if 'video' in env_params.keys() and env_params['video']:
        video_path = "{}video_{}.mp4".format(env_params['video_dir'], env_params['env_name'])
        monitor = Monitor(256, 240, video_path)
    else:
        monitor = None

    env = ObservationEnv(env, env_params['useful_region'], monitor)
    
    env = JoypadSpace(env, SIMPLE_MOVEMENT)
    env = CustomReward(env)
    env = CustomSkipFrame(env)
    return env, env.observation_space.shape[0], len(SIMPLE_MOVEMENT)

    return env

class CustomReward(Wrapper):
    
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_x_pos = 0
        self.curr_score = 0
        self.curr_life = 2
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        '''Aquí adaptamos la observacion para pintarla'''
        
        if self.monitor:
            self.monitor.record(observation)

        return observation, self._reward(reward, done, info) , done, info

    def _reward(self, reward, done, info):

        ## Si sube nuestro score damos un pequeño reward. Si ha sido gracias a coger la bandera, el reward es mayor
        reward += (info['score']-self.curr_score)/40.
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
        self.curr_life = 2
        self.curr_score = 0
        return self.env.reset()


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        observations_list = []
        observation, reward, done, info = self.env.step(action)
        for i in range(self.skip):
            if not done:
                observation, reward, done, info = self.env.step(action)
                observations_list.append(observation)
            else:
                observations_list.append(observation)
        observations_list = np.concatenate(observations_list, 0)[None, :, :, :]
        return observations_list.astype(np.float32), reward, done, info

    def reset(self):
        observation = self.env.reset()
        observations_list = np.concatenate([observation for _ in range(self.skip)], 0)[None, :, :, :]
        return observations_list.astype(np.float32)


