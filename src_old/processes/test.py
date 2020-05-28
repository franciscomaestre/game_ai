"""
@author: Francisco Maestre. Modified version of Viet Nguyen version
"""

import torch
from src.environments import create_train_env
from src.models.actor_critic import ActorCritic
import torch.nn.functional as F
from torch.distributions import Categorical
from collections import deque
from tensorboardX import SummaryWriter
import timeit
from environments import make_train_env


def local_test(index, opt, global_model, env_params):
    torch.manual_seed(123 + index)
    
    #env, num_states, num_actions = create_train_env(opt=opt,video=False)
    env, num_states, num_actions = make_train_env(env_params)
    
    local_model = ActorCritic(num_states, num_actions)
    local_model.eval()
    state = torch.from_numpy(env.reset())
    done = True
    curr_step = 0
    actions = deque(maxlen=opt.max_actions)
    while True:
        curr_step += 1
        if done:
            local_model.load_state_dict(global_model.state_dict())
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
        env.render()
        actions.append(action)
        if curr_step > opt.num_global_steps or actions.count(actions[0]) == actions.maxlen:
            done = True
        if done:
            curr_step = 0
            actions.clear()
            state = env.reset()
        state = torch.from_numpy(state)
