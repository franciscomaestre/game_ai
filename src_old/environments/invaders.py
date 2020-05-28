"""
@author: Francisco Maestre. Modified version of Viet Nguyen
"""

import gym
from nes_py.wrappers import JoypadSpace
from gym.spaces import Box
from gym import Wrapper
import cv2
import numpy as np
import subprocess as sp
from .common import Monitor

'''
La clase CustomReward es un Wrapper que sobreescribe ciertas funciones del Env, pero sigue siendo el mismo. Este Wrapper está pensado para Super Mario.
Básicamente, sobreescribimos la forma en la que va a detectar la puntucación y de esta forma el saber si está avanzando correctamente.

Se podría haber usado el que hay por defecto, pero ese no está pintando por pantalla. Aquí hemos mezclado el step con el que este salga por pantalla.

'''

class CustomReward(Wrapper):
    
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_score = 0
        self.curr_lives = 3
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None

    def _preprocess_observation(self,frame):
        if frame is not None:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            frame = cv2.resize(frame, (84, 84))[None, :, :] / 255.
            return frame
        else:
            return np.zeros((1, 84, 84))
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        '''Aquí adaptamos la observacion para pintarla'''
        
        if self.monitor:
            self.monitor.record(observation)
        frame_observation = self._preprocess_observation(observation)

        return frame_observation, self._reward(reward, done, info) , done, info

    def _reward(self, reward, done, info):
        ## Podemos usar info['ale.lives']
        if self.curr_lives > info['ale.lives']:
            reward -=100
        self.curr_lives = info['ale.lives']
        if done:
            if info['ale.lives'] > 0:
                reward += 100
            else:
                reward -= 100
        return reward

    def reset(self):
        self.curr_score = 0
        self.curr_lives = 3
        return self._preprocess_observation(self.env.reset())


class CustomSkipFrame(Wrapper):
    def __init__(self, env, skip=4):
        super(CustomSkipFrame, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(4, 84, 84))
        self.skip = skip

    def step(self, action):
        total_reward = 0
        observations_list = []
        observation, reward, done, info = self.env.step(action)
        for i in range(self.skip):
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

"""
Preparamos el enviroment haciendo uso de los wrapper preparados para el SuperMario
"""

def create_train_env(world, stage, action_type, output_path=None):
    env = gym.make("SpaceInvaders-v0")
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None

    if action_type == "right":
        actions = env.env.get_action_meanings()
    else:
        actions = env.env.get_action_meanings()
    
    #env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    return env, env.observation_space.shape[0], len(actions)
