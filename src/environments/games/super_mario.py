"""
@author: Francisco Maestre. Modified version of Viet Nguyen
"""

import gym_super_mario_bros
from gym.spaces import Box
from gym import Wrapper
from nes_py.wrappers import JoypadSpace
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT, COMPLEX_MOVEMENT, RIGHT_ONLY
import cv2
import numpy as np
import math
import subprocess as sp
from src.environments.common import Monitor

'''
La clase CustomReward es un Wrapper que sobreescribe ciertas funciones del Env, pero sigue siendo el mismo. Este Wrapper está pensado para Super Mario.
Básicamente, sobreescribimos la forma en la que va a detectar la puntucación y de esta forma el saber si está avanzando correctamente.

Se podría haber usado el que hay por defecto, pero ese no está pintando por pantalla. Aquí hemos mezclado el step con el que este salga por pantalla.

'''

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

    def _process_observation(self,frame):
        if frame is not None:
            frame = cv2.cvtColor(cv2.resize(frame, (84, 110)), cv2.COLOR_RGB2GRAY)
            frame = frame[26:110,:]
            return np.reshape(frame,( 1, 84, 84))
        else:
            return np.zeros((1, 84, 84))
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        '''Aquí adaptamos la observacion para pintarla'''
        
        if self.monitor:
            self.monitor.record(observation)
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
        self.curr_life = 2
        self.curr_score = 0
        return self._process_observation(self.env.reset())


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

"""
Preparamos el enviroment haciendo uso de los wrapper preparados para el SuperMario
"""

def create_train_env(world, stage, action_type, output_path=None):
    env = gym_super_mario_bros.make("SuperMarioBros-{}-{}-v0".format(world, stage))
    if output_path:
        monitor = Monitor(256, 240, output_path)
    else:
        monitor = None
    if action_type == "right":
        actions = RIGHT_ONLY
    elif action_type == "simple":
        actions = SIMPLE_MOVEMENT
    elif action_type == "complex":
        actions = COMPLEX_MOVEMENT
    else:
        actions = RIGHT_ONLY

    env = JoypadSpace(env, actions)
    env = CustomReward(env, monitor)
    env = CustomSkipFrame(env)
    return env, env.observation_space.shape[0], len(actions)