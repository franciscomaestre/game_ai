"""
@author: Francisco Maestre. Modified version of Viet Nguyen version
"""

import torch
import timeit
import torch.nn.functional as F
import torch.multiprocessing as _mp

from collections import deque
from tensorboardX import SummaryWriter
from environments import make_train_env
from models.actor_critic import ActorCritic
from torch.distributions import Categorical

class DiscreteActorCriticTestProcess(_mp.Process):

    def __init__(self, index, agent_params, env_params, global_model):
        super(DiscreteActorCriticTestProcess, self).__init__()
        self.index = index
        self.env_params = env_params
        self.agent_params = agent_params
        self.global_model = global_model

    def run(self):
        torch.manual_seed(self.agent_params['seed'] + self.index)
        env, num_states, num_actions = make_train_env(self.env_params)
        local_model = ActorCritic(num_states, num_actions)
        local_model.eval()
        state = torch.from_numpy(env.reset())
        done = True
        curr_step = 0
        actions = deque(maxlen=self.agent_params['max_actions'])

        episode_reward = 0
        best_reward = -999999

        while True:
            
            episode_reward = 0

            curr_step += 1
            if done:
                local_model.load_state_dict(self.global_model.state_dict())
            with torch.no_grad():
                if done:
                    h_0 = torch.zeros((1, 512), dtype=torch.float)
                    c_0 = torch.zeros((1, 512), dtype=torch.float)
                else:
                    h_0 = h_0.detach()
                    c_0 = c_0.detach()

            logits, value, h_0, c_0 = local_model(state, h_0, c_0)
            policy = F.softmax(logits, dim=1)
            action = torch.argmax(policy).item()
            state, reward, done, _ = env.step(action)
            episode_reward += reward
            env.render()
            actions.append(action)
            if curr_step > self.agent_params['num_global_steps'] or actions.count(actions[0]) == actions.maxlen:
                done = True
            
            if done:

                if episode_reward > best_reward:
                    best_reward = episode_reward

                curr_step = 0
                actions.clear()
                state = env.reset()
                print("Reward: {:.2f}\tBest Reward: {:.2f}\t".format(episode_reward, best_reward))
            state = torch.from_numpy(state)

    
