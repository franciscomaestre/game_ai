"""
@author: Francisco Maestre. Modified version of Viet Nguyen version
"""

import math
import torch
import timeit
import torch.nn.functional as F
import torch.multiprocessing as _mp

from collections import deque
from tensorboardX import SummaryWriter
from torch.distributions import Categorical

from environments import make_train_env
from models.actor_critic import ActorCritic

def _print_status(curr_episode, start_time, episode_reward, best_reward):
    control_time = timeit.default_timer()
    time_diff = int(control_time - start_time)
    seconds = time_diff%60
    time_diff -= seconds
    minutes = math.floor((time_diff)/60.)
    hours = math.floor((minutes)/60.)
    minutes -= hours*60
    print("Global Process -- Episode {}\tReward: {:.2f}\tBest Reward: {:.2f}\tThe code runs for {} h {} m {} s".format(curr_episode, episode_reward, best_reward, hours, minutes, seconds))

class DiscreteActorCriticTrainProcess(_mp.Process):
    
    def __init__(self, index, agent_params, env_params, global_model, optimizer, save=False):
        super(DiscreteActorCriticTrainProcess, self).__init__()
        self.index = index
        self.env_params = env_params
        self.agent_params = agent_params
        self.global_model = global_model
        self.optimizer = optimizer
        self.save = save

    def run(self):
        #Usamos una semilla distinta en cada iteración para que no todos los caminos sean iguales
        torch.manual_seed(self.agent_params['seed'] + self.index)

        if self.save:
            start_time = timeit.default_timer()
        writer = SummaryWriter(self.agent_params['log_path'])

        #Creamos el ambiente del juego
        env, num_states, num_actions = make_train_env(self.env_params)
        
        ## Creamos la red que vamos a entrenar y la configuramos
        local_model = ActorCritic(num_states, num_actions)
        if self.agent_params['use_gpu']:
            local_model.cuda()
        local_model.train()
        state = torch.from_numpy(env.reset())
        if self.agent_params['use_gpu']:
            state = state.cuda()
        
        
        done = True
        curr_step = 0
        curr_episode = 0

        episode_reward = 0
        best_reward = -999999

        while True:
                
            curr_episode += 1
            local_model.load_state_dict(self.global_model.state_dict())
            if done:
                h_0 = torch.zeros((1, 512), dtype=torch.float)
                c_0 = torch.zeros((1, 512), dtype=torch.float)
            else:
                h_0 = h_0.detach()
                c_0 = c_0.detach()
            if self.agent_params['use_gpu']:
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()

            log_policies = []
            values = []
            rewards = []
            entropies = []

            episode_reward = 0

            for _ in range(self.agent_params['num_local_steps']):
                curr_step += 1
                logits, value, h_0, c_0 = local_model(state, h_0, c_0)
                policy = F.softmax(logits, dim=1)
                log_policy = F.log_softmax(logits, dim=1)
                entropy = -(policy * log_policy).sum(1, keepdim=True)

                m = Categorical(policy)
                action = m.sample().item()

                state, reward, done, _ = env.step(action)
                state = torch.from_numpy(state)
                if self.agent_params['use_gpu']:
                    state = state.cuda()
                if curr_step > self.agent_params['num_global_steps']:
                    done = True

                if done:
                    curr_step = 0
                    state = torch.from_numpy(env.reset())
                    if self.agent_params['use_gpu']:
                        state = state.cuda()

                values.append(value)
                log_policies.append(log_policy[0, action])
                rewards.append(reward)
                entropies.append(entropy)

                episode_reward += reward

                if done:
                    break
            
            if best_reward < episode_reward:
                best_reward = episode_reward

            R = torch.zeros((1, 1), dtype=torch.float)
            if self.agent_params['use_gpu']:
                R = R.cuda()
            if not done:
                _, R, _, _ = local_model(state, h_0, c_0)

            gae = torch.zeros((1, 1), dtype=torch.float)
            if self.agent_params['use_gpu']:
                gae = gae.cuda()
            actor_loss = 0
            critic_loss = 0
            entropy_loss = 0
            next_value = R

            for value, log_policy, reward, entropy in list(zip(values, log_policies, rewards, entropies))[::-1]:
                gae = gae * self.agent_params['gamma'] * self.agent_params['tau']
                gae = gae + reward + self.agent_params['gamma'] * next_value.detach() - value.detach()
                next_value = value
                actor_loss = actor_loss + log_policy * gae
                R = R * self.agent_params['gamma'] + reward
                critic_loss = critic_loss + (R - value) ** 2 / 2
                entropy_loss = entropy_loss + entropy

            total_loss = -actor_loss + critic_loss - self.agent_params['beta'] * entropy_loss
            writer.add_scalar("Train_{}/Loss".format(self.index), total_loss, curr_episode)
            self.optimizer.zero_grad()
            total_loss.backward()

            for local_param, global_param in zip(local_model.parameters(), self.global_model.parameters()):
                if global_param.grad is not None:
                    break
                global_param._grad = local_param.grad

            self.optimizer.step()

            #Si hemos configurado que se guarde, cada X épocas se encargará de salvarlo. Esto actualizará el que vemos visualmente
            if self.save:
                if curr_episode % self.agent_params['save_internal'] == 0 and curr_episode > 0:
                    torch.save(self.global_model.state_dict(),
                            "{}/a3c_{}".format(self.agent_params['model_path'], self.env_params['env_name']))
                _print_status(curr_episode, start_time, episode_reward, best_reward)

            if curr_episode == int(self.agent_params['num_global_steps'] / self.agent_params['num_local_steps']):
                print("Training process {} terminated".format(self.index))
                if self.save:
                    end_time = timeit.default_timer()
                    print('The code runs for %.2f s ' % (end_time - start_time))
                return
    
