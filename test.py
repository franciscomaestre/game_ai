"""
@author: Francisco Maestre. Modified version of Viet Nguyen
"""

import os
os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import torch
from src.models.actor_critic import ActorCritic
from src.environments import create_train_env
import torch.nn.functional as F

def get_args():
    parser = argparse.ArgumentParser(
        """Implementacion de refuerzo A3C con el Super Mario Bros""")
    parser.add_argument("--game", type=str, default="super_mario")    
    parser.add_argument("--world", type=int, default=1)
    parser.add_argument("--stage", type=int, default=1)
    parser.add_argument("--action_type", type=str, default="right")
    parser.add_argument("--saved", type=str, default="data/models")
    parser.add_argument("--output", type=str, default="data/output")
    args = parser.parse_args()
    return args

def get_trained_model(opt):

    #Obtenemos el env de entrenamiento para el juego seleccionado
    env, num_states, num_actions = create_train_env(opt=opt,video=True)

    ## Cargamos el modelo
    model = ActorCritic(num_states, num_actions)

    ## Recuperamos la red entrenada
    if torch.cuda.is_available():
        model.load_state_dict(torch.load("{}/a3c_{}_{}_{}_{}".format(opt.saved, opt.game, opt.world, opt.stage, opt.action_type)))
        model.cuda()
    else:
        model.load_state_dict(torch.load("{}/a3c_{}_{}_{}_{}".format(opt.saved, opt.game, opt.world, opt.stage, opt.action_type),
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
    ## Obtenemos los argumentos
    opt = get_args()

    ## Dejamos una semilla fija
    torch.manual_seed(123)

    ## Cargamos el modelo entrenado
    env, model = get_trained_model(opt)
    
    ## Empezamos a ejecutar el modelo entrenado
    run_trained_model(env, model)
