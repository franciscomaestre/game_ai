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
        self.pipe.stdin.write(image_array)

class MonitorEnv(gym.ObservationWrapper):

    def __init__(self, env, video_path = None):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0, high=255, shape=(1, 84, 84))
        self.monitor = Monitor(256, 256, video_path)

    def observation(self, observation):
        if self.monitor:
            self.monitor.record(np.array(observation))
        return observation

class ObservationEnv(gym.ObservationWrapper):
    def __init__(self, env, frame_conf):
        gym.ObservationWrapper.__init__(self, env)
        self.observation_space = Box(low=0, high=1., shape=(1, 84, 84))
        self.frame_conf = frame_conf

    def observation(self, observation):
        #pasamos a escala de grises
        observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
        #hacemos un resize a la escala querida
        total_scale_x = 84 + self.frame_conf["crop_x_l"] + self.frame_conf["crop_x_r"]
        total_scale_y = 84 + self.frame_conf["crop_y_t"] + self.frame_conf["crop_y_d"]
        observation = cv2.resize(observation, (total_scale_x, total_scale_y))
        #cortamos lo que no queremos
        observation = observation[self.frame_conf["crop_y_t"]: total_scale_y-self.frame_conf["crop_y_d"], self.frame_conf["crop_x_l"]:total_scale_x-self.frame_conf['crop_x_r']]
        #tras el corte, volvemos a hacer un resize al tamaño que queremos
        observation = cv2.resize(observation, (84, 84))[None, :, :] / 255.
        #Pasamos la imagen a binario en caso de ser requerido
        if self.frame_conf['binary']:
            ret, observation = cv2.threshold(observation,0,1,cv2.THRESH_BINARY)
        return np.reshape(observation,( 1, 84, 84))



