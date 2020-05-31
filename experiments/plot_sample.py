## http://www.pinchofintelligence.com/openai-gym-part-3-playing-space-invaders-deep-reinforcement-learning/
import matplotlib.pyplot as plt

from matplotlib import animation
from time import gmtime, strftime
import random
import cv2
import sys
import numpy as np 

import gym
import gym_super_mario_bros

from params_manager import ParamsManager
'''
def process_frame_84(observation, frame_conf):
    #pasamos a escala de grises
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    #hacemos un resize para cortar una parte
    total_x = frame_conf['scale'] + frame_conf["crop_x"]
    total_y = frame_conf['scale'] + frame_conf["crop_y"]
    observation = cv2.resize(observation, (total_x, total_y))
    observation = observation[frame_conf["crop_y"]: total_y, frame_conf["crop_x"]:total_x]
    #tras el corte, volvemos a hacer un resize al tamaño que queremos
    observation = cv2.resize(observation, (84, 84))[None, :, :] / 255.
    #Pasamos la imagen a binario en caso de ser requerido
    if frame_conf['binary']:
        ret, observation = cv2.threshold(observation,0,1,cv2.THRESH_BINARY)
    return np.reshape(observation,( 1, 84, 84))
'''

def process_frame_84_v1(observation, frame_conf):
    #pasamos a escala de grises
    observation = cv2.cvtColor(observation, cv2.COLOR_RGB2GRAY)
    #hacemos un resize a la escala querida
    observation = cv2.resize(observation, (frame_conf['scale'], frame_conf['scale']))
    #cortamos lo que no queremos
    observation = observation[frame_conf["crop_y_u"]: frame_conf['scale']-frame_conf["crop_y_d"], frame_conf["crop_x_l"]:frame_conf['scale']]
    #tras el corte, volvemos a hacer un resize al tamaño que queremos
    observation = cv2.resize(observation, (frame_conf['scale'], frame_conf['scale']))[None, :, :] / 255.
    #Pasamos la imagen a binario en caso de ser requerido
    if frame_conf['binary']:
        ret, observation = cv2.threshold(observation,0,1,cv2.THRESH_BINARY)
    return np.reshape(observation,( 1, frame_conf['scale'], frame_conf['scale']))



env_name = "SpaceInvaders-v0"
env_name = "SuperMarioBros-1-1-v0"

env = gym.make(env_name)

env.reset()

params_manager = ParamsManager.getInstance()
if env_name == "SuperMarioBros-1-1-v0":
    env_params = params_manager.get_env_params('super_mario')
else:
    env_params = params_manager.get_env_params('atari')
env_params['useful_region'] = env_params['useful_region']['Default'] 

action = 0

for i in range(5):
    env.step(action)

frame, reward0, terminal, info = env.step(action)

## Imagen original
plt.imshow(np.array(np.squeeze(frame)))
plt.show(block=False)
plt.pause(2)
plt.close()

observation0 = process_frame_84_v1(frame, env_params['useful_region'] )
print("After processing: " + str(np.array(observation0).shape))

## Imagen Modificada
plt.imshow(np.array(np.squeeze(observation0)))
plt.show(block=False)
plt.pause(2)
plt.close()


observation0 = process_frame_84_v2(frame, env_params['useful_region'] )
print("After processing: " + str(np.array(observation0).shape))

## Imagen Modificada
plt.imshow(np.array(np.squeeze(observation0)))
plt.show(block=False)
plt.pause(2)
plt.close()


'''
observation1 = process_frame(frame)
print("After processing: " + str(np.array(observation1).shape))

## Imagen Modificada
plt.imshow(np.array(np.squeeze(observation1)))
plt.show()

print(np.array_equal(observation0, observation1))
'''

