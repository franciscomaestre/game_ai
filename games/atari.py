"""
@author: Francisco Maestre. Modified version of Viet Nguyen
"""

import gym
import cv2
import random
import atari_py
import numpy as np
import subprocess as sp

from gym import Wrapper
from gym.spaces import Box
from collections import deque
from nes_py.wrappers import JoypadSpace

from .utils import ObservationEnv, Monitor, FrameStack, LazyFrames, MaxAndSkipEnv


"""
Preparamos el enviroment haciendo uso de los wrapper preparados para el SuperMario
"""

def make_train_env(env_name, env_conf):
    env = gym.make(env_name)

    if 'NoFrameskip' in env_name:
        assert 'NoFrameskip' in env.spec.id
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=env_conf['skip_rate'])

    if 'video' in env_conf.keys():
        video_path = "{}video_{}.mp4".format(env_conf['video_dir'], env_name)
        monitor = Monitor(256, 240, video_path)
    else:
        monitor = None

    if env_conf['episodic_life']:
        env = EpisodicLifeEnv(env)

    try:
        if 'FIRE' in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
    except AttributeError:
        pass

    env = ObservationEnv(env, env_conf['useful_region'], monitor)

    env = FrameStack(env, env_conf['num_frames_to_stack'])
    
    env = CustomReward(env)

    return env

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        """Sample initial states by taking random number of no-ops on reset.
        No-op is assumed to be action 0.
        """
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.noop_action = 0
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def reset(self):
        """ Do no-op action for a number of steps in [1, noop_max]."""
        self.env.reset()
        noops = random.randrange(1, self.noop_max + 1)  # pylint: disable=E1101
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(self.noop_action)
        return obs

    def step(self, ac):
        return self.env.step(ac)

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        """Take action on reset for environments that are fixed until firing."""
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

    def step(self, ac):
        return self.env.step(ac)

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
        lives = info['ale.lives']
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
            self.lives = info['ale.lives']
        return obs

class CustomReward(Wrapper):
    
    def __init__(self, env=None):
        super(CustomReward, self).__init__(env)
        self.curr_score = 0
        
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        return observation, self._reward(reward, done, info) , done, info

    def _reward(self, reward, done, info):
        #Ahora mismo no estamos haciendo nada
        return reward

    def reset(self):
        self.curr_score = 0
        return self.env.reset()


