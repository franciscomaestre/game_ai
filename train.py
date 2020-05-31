"""
@author: Francisco Maestre. Modified version of Viet Nguyen version
"""

import os
import torch
import shutil
import warnings
import argparse
import torch.multiprocessing as _mp

from environments import make_train_env
from models.discrete import ActorCritic, GlobalAdam

from utils.params_manager import ParamsManager
from processes.train import DiscreteActorCriticTrainProcess
from processes.test import DiscreteActorCriticTestProcess

os.environ['OMP_NUM_THREADS'] = '1'
warnings.filterwarnings("ignore")

def train(agent_params, env_params):
    
    ## Fijamos una semilla manual para poder estudiar bien los resultados
    torch.manual_seed(agent_params['seed'])

    ## Creamos las carpetas por defecto
    _make_default_folders(agent_params)    

    ## Arrancamos el modelo para entrenar (ya sea desde cero o con una versión previa)
    global_model = get_global_model(agent_params, env_params)

    ## Lanzamos el optimizador
    optimizer = GlobalAdam(global_model.parameters(), learning_rate=agent_params['learning_rate'])

    ## Lanzamos los procesos en paralelo para realizar el entrenamiento
    launch_processes(global_model, optimizer, agent_params, env_params)

def get_args():
    parser = argparse.ArgumentParser(
        """Implementacion de refuerzo A3C con el Super Mario Bros""")
    parser.add_argument("--env_name", help="Name of the Gym environment", type=str, default="SuperMarioBros-1-1-v0")
    parser.add_argument("--env_params", help="Name of the Parameters environment. It could be super_mario or atari", type=str, default="super_mario")
    parser.add_argument("--train_name", help="Name of the training. Put whatever you want", type=str, default="train_mario")
    args = parser.parse_args()
    return args

def get_params(args):
    params_manager= ParamsManager().getInstance()
    agent_params = params_manager.get_agent_params()
    env_params = params_manager.get_env_params(args.env_params.lower())
    env_params['env_name'] = args.env_name
    agent_params['train_name'] = args.train_name

    custom_region_available = False
    for key, value in env_params['useful_region'].items():
        if key in env_params['env_name']:
            env_params['useful_region'] = value
            custom_region_available = True
            break
    if custom_region_available is not True:
        env_params['useful_region'] = env_params['useful_region']['Default'] 

    return agent_params, env_params

def _make_default_folders(agent_params):
    if os.path.isdir(agent_params['log_path']):
        shutil.rmtree(agent_params['log_path'])
    os.makedirs(agent_params['log_path'])
    if not os.path.isdir(agent_params['model_path']):
        os.makedirs(agent_params['model_path'])

def get_global_model(agent_params, env_params):
    ## Creamos en env de entrenamiento
    env, num_states, num_actions = make_train_env(env_params)

    ## Creamos la red neuronal para que vaya aprendiendo
    global_model = ActorCritic(num_states, num_actions)
    if agent_params['use_gpu']:
        global_model.cuda()
    global_model.share_memory()

    return global_model

def launch_processes(global_model, optimizer, agent_params, env_params):
    ## Arrancamos el multiprocessing
    _mp.set_start_method('spawn')

    processes = []
    for index in range(agent_params['num_agents']):
        if index == 0:
            process = DiscreteActorCriticTrainProcess(index, agent_params, env_params, global_model, optimizer, True)
        else:
            process = DiscreteActorCriticTrainProcess(index, agent_params, env_params, global_model, optimizer, False)
        process.start()
        processes.append(process)

    process = DiscreteActorCriticTestProcess(agent_params['num_agents'], agent_params, env_params, global_model)
    process.start()
    processes.append(process)
    for process in processes:
        process.join()

if __name__ == "__main__":

    ## Obtenemos los argumentos
    args = get_args()
    
    #Cargamos los parametros
    agent_params, env_params = get_params(args)


    train(agent_params, env_params)
