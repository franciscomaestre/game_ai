"""
@author: Francisco Maestre. Modified version of Viet Nguyen
"""

import os
import torch
import argparse
import warnings
import torch.nn.functional as F

from environments import make_train_env
from models.actor_critic import ActorCritic
from utils.params_manager import ParamsManager
from processes.test import DiscreteActorCriticTestProcess

os.environ['OMP_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore")

def test():
    ## Obtenemos los argumentos
    args = get_args()

    #Cargamos los parametros
    agent_params, env_params = get_params(args)

    ## Dejamos una semilla fija
    torch.manual_seed(agent_params['seed'])

    ## Cargamos el modelo entrenado
    _, global_model = get_trained_model(agent_params, env_params)

    ## Empezamos a ejecutar el modelo entrenado
    process = DiscreteActorCriticTestProcess(0, agent_params, env_params, global_model)
    process.start()
    process.join()

def get_args():
    parser = argparse.ArgumentParser(
        """Implementacion de refuerzo A3C con el Super Mario Bros""")
    parser.add_argument("--env_name", help="Name of the Gym environment", type=str, default="SuperMarioBros-1-1-v0")
    parser.add_argument("--env_params", help="Name of the Parameters environment. It could be super_mario or atari", type=str, default="super_mario")
    args = parser.parse_args()
    return args

def get_params(args):
    params_manager= ParamsManager().getInstance()
    agent_params = params_manager.get_agent_params()
    env_params = params_manager.get_env_params(args.env_params.lower())
    env_params['env_name'] = args.env_name

    custom_region_available = False
    for key, value in env_params['useful_region'].items():
        if key in env_params['env_name']:
            env_params['useful_region'] = value
            custom_region_available = True
            break
    if custom_region_available is not True:
        env_params['useful_region'] = env_params['useful_region']['Default'] 

    return agent_params, env_params

def get_trained_model(agent_params, env_params):

    #Obtenemos el env de entrenamiento para el juego seleccionado
    env, num_states, num_actions = make_train_env(env_params)

    ## Cargamos el modelo
    model = ActorCritic(num_states, num_actions)

    ## Recuperamos la red entrenada
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/a3c_{}".format(agent_params['model_path'], env_params['env_name'])))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c_{}".format(agent_params['model_path'], env_params['env_name']),
                                         map_location=lambda storage, loc: storage))
    return env, model

def run_trained_model(env, model):
    model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    while True:
        if done:
            h_0 = torch.zeros((1, 512), dtype=torch.float)
            c_0 = torch.zeros((1, 512), dtype=torch.float)
            env.reset()
        else:
            h_0 = h_0.detach()
            c_0 = c_0.detach()
        if torch.cuda.is_available():
            h_0 = h_0.cuda()
            c_0 = c_0.cuda()
            state = state.cuda()

        logits, value, h_0, c_0 = model(state, h_0, c_0)
        policy = F.softmax(logits, dim=1)
        action = torch.argmax(policy).item()
        action = int(action)
        state, reward, done, info = env.step(action)
        state = torch.from_numpy(state)
        env.render()
        if done:
            print("{} - completed".format(opt.game))
            break

if __name__ == "__main__":
    test()
    
    
