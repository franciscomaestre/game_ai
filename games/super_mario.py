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

from .utils import ObservationEnv, Monitor, FrameStack, LazyFrames, MaxAndSkipEnv

"""
Preparamos el enviroment haciendo uso de los wrapper preparados para el SuperMario
"""

def make_train_env(env_name, env_conf):
    env = gym_super_mario_bros.make(env_name)
    
    if 'video' in env_conf.keys() and env_conf['video']:
        video_path = "{}video_{}.mp4".format(env_conf['video_dir'], env_name)
        monitor = Monitor(256, 240, video_path)
    else:
        monitor = None

    if env_conf['episodic_life']:
        env = EpisodicLifeEnv(env)

    env = ObservationEnv(env, env_conf['useful_region'], monitor)

    env = FrameStack(env, env_conf['num_frames_to_stack'])

    env = CustomReward(env, monitor)

    return env

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done = True

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = True
        # check current lives, make loss of life terminal,
        # then update lives to handle bonus lives
        lives = info['life']
        if lives < self.lives and lives > 0:
            # for Qbert sometimes we stay in lives == 0 condition for a few frames
            # so its important to keep lives > 0, so that we only reset once
            # the environment advertises done.
            done = True
            self.was_real_done = False
        self.lives = lives
        return obs, reward, done, info

    def reset(self):
        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """
        if self.was_real_done:
            obs = self.env.reset()
            self.lives = 0
        else:
            # no-op step to advance from terminal/lost life state
            obs, _, _, info = self.env.step(0)
            self.lives = info['life']
        return obs

class CustomReward(Wrapper):
    
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.curr_x_pos = 0
        self.curr_score = 0
        self.curr_life = 2
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self._reward(reward, done, info) , done, info

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
        return self.env.reset()
