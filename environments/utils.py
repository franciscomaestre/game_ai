"""
@author: Francisco Maestre. Modified version of Viet Nguyen
"""
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


def process_frame_84(observation, frame_conf):
    observation = cv2.cvtColor(cv2.resize(observation, ( frame_conf['scale'] + frame_conf["crop_x"], frame_conf['scale'] + frame_conf["crop_y"])), cv2.COLOR_RGB2GRAY)
    observation = observation[frame_conf["crop_y"]:frame_conf["crop_y"]+frame_conf['scale'],frame_conf["crop_x"]:frame_conf["crop_x"]+frame_conf['scale']]
    observation = cv2.resize(observation, (84, 84))
    if frame_conf['binary']:
        ret, observation = cv2.threshold(observation,1,255,cv2.THRESH_BINARY)
    return np.reshape(observation,( 1, 84, 84))

class ObservationEnv(gym.ObservationWrapper):
    def __init__(self, env, frame_conf, monitor = None):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(0, 255, [1, 84, 84], dtype=np.uint8)
        self.frame_conf = frame_conf
        self.monitor = monitor
        self.steps = 0

    def observation(self, observation):
        if self.monitor:
            self.monitor.record(observation[0][0])
        self.steps += 1
        return process_frame_84(observation, self.frame_conf)


class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env=None, skip=4):
        """Return only every `skip`-th frame"""
        gym.Wrapper.__init__(self, env)
        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = deque(maxlen=2)
        self._skip = skip

    def step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break

        max_frame = np.max(np.stack(self._obs_buffer), axis=0)
        return max_frame, total_reward, done, info

    def reset(self):
        """Clear past frame buffer and init. to first obs. from inner env."""
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class FrameStack(gym.Wrapper):
    def __init__(self, env, k = 4):
        """Stack k last frames.
        Returns lazy array, which is much more memory efficient.
        From baselines atari_wrapper
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = Box(low=0, high=255, shape=(shp[0] * k , shp[1], shp[2]), dtype=np.uint8)

    def reset(self):
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return np.array(self._get_ob())

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))

class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        """
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=0)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[i]