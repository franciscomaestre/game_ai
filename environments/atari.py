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

from .utils import ObservationEnv, Monitor

def make_train_env(env_params):
    env = gym.make(env_params['env_name'])

    actions = env.env.get_action_meanings()

    env = CustomReward(env)
    env = ObservationEnv(env, frame_conf=env_params['useful_region'])      
    env = CustomSkipFrame(env, skip=env_params['skip_rate'])
    env = CustomStackFrame(env, stack=env_params['num_frames_to_stack'])

    return env, env.observation_space.shape[0], len(actions)

class CustomReward(Wrapper):
    
    def __init__(self, env=None, monitor=None):
        super(CustomReward, self).__init__(env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.curr_lives = 3
        if monitor:
            self.monitor = monitor
        else:
            self.monitor = None
    
    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        '''Aquí adaptamos la observacion para pintarla'''
        return observation, self._reward(reward, done, info) , done, info

    def _reward(self, reward, done, info): 
        if self.curr_lives > info['ale.lives']:
            reward -= 100
        self.curr_lives = info['ale.lives']
        if done:
            if info['ale.lives'] > 0:
                reward = info['ale.lives'] * 100
            else:
                reward -= 100

        return reward

    def reset(self):
        self.curr_lives = 3
        return self.env.reset()

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
