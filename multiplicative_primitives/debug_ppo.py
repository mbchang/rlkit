import argparse
import gym
import numpy as np
import os
import sys
from itertools import count
from collections import namedtuple

import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical

from log import RunningAverage
from rb import Memory
from ppo import PPO
from agent import BaseActionAgent



parser = argparse.ArgumentParser(description='PyTorch ppo example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--lr', type=float, default=5e-3, metavar='R',
                    help='learning rate (default: 5e-4)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--gpu-index', type=int, default=0,
                    help='gpu_index (default: 0)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=100, metavar='I',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()

torch.manual_seed(args.seed)
device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

eps = np.finfo(np.float32).eps.item()

class Policy(nn.Module):
    def __init__(self, dims):
        super(Policy, self).__init__()
        self.dims = dims
        self.layers = nn.ModuleList()
        for i in range(len(self.dims[:-2])):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
        self.action_head = nn.Linear(self.dims[-2], self.dims[-1])

    def forward(self, state):
        for layer in self.layers:
            state = F.relu(layer(state))
        action_scores = self.action_head(state)
        action_dist = F.softmax(action_scores, dim=-1)
        return action_dist

    def select_action(self, state):
        # volatile
        action_dist = self.forward(state)
        m = Categorical(action_dist)
        action = m.sample()
        return action.data

    def get_log_prob(self, state, action):
        # not volatile
        action_dist = self.forward(state)
        m = Categorical(action_dist)
        log_prob = m.log_prob(action)
        return log_prob

class ValueFn(nn.Module):
    def __init__(self, dims):
        super(ValueFn, self).__init__()
        self.dims = dims
        self.layers = nn.ModuleList()
        for i in range(len(self.dims[:-2])):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))
        self.value_head = nn.Linear(self.dims[-2], self.dims[-1])

    def forward(self, state):
        for layer in self.layers:
            state = F.relu(layer(state))
        state_values = self.value_head(state)
        return state_values

def sample_trajectory(agent, env):
    episode_data = []
    state = env.reset()
    for t in range(100):  # Don't infinite loop while learning
        action, log_prob, value = agent(torch.from_numpy(state).float())
        state, reward, done, _ = env.step(action)
        if args.render:
            env.render()
        mask = 0 if done else 1
        e = {'state': state,
             'action': action,
             'logprob': log_prob,
             'mask': mask,
             'reward': reward,
             'value': value}
        episode_data.append(e)
        agent.store_transition(e)
        if done:
            break
    returns = sum([e['reward'] for e in episode_data])
    return returns, t

def main():
    # env = gym.make('CartPole-v0')
    # env = gym.make('Ant-v2')
    import sys
    sys.path.append('../')
    from rlkit.envs.wrappers import NormalizedBoxEnv
    from rlkit.envs.ant_goal import AntGoalEnv
    env = AntGoalEnv(n_tasks=1, use_low_gear_ratio=True)
    tasks = env.get_all_task_idx()
    env.seed(args.seed)
    discrete = type(env.action_space) == gym.spaces.discrete.Discrete
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if discrete else env.action_space.shape[0]
    hdim = 256
    agent = BaseActionAgent(
        policy=Policy(dims=[obs_dim, hdim, hdim, act_dim]), 
        valuefn=ValueFn(dims=[obs_dim, hdim, hdim, 1]), 
        id=0, 
        device=device, 
        args=args).to(device)
    run_avg = RunningAverage()
    returns = []
    for i_episode in range(1, 100001):
    # for i_episode in range(1, 501):
        ret, t = sample_trajectory(agent, env)
        running_reward = run_avg.update_variable('reward', ret)
        if i_episode % 1000 == 0:
            print('Update Agent')
            agent.improve()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast return: {:.2f}\tAverage return: {:.2f}'.format(
                i_episode, ret, running_reward))
        if i_episode % (10*args.log_interval) == 0:
            returns.append(running_reward)
            pickle.dump(returns, open('log.p', 'wb'))  # this only starts logging after the first improve_every though!

    # np.save(returns, 'plot.npy')


# def experiment(variant):
#    task_params = variant[‘task_params’]
#    env = NormalizedBoxEnv(AntGoalEnv(n_tasks=task_params[‘n_tasks’], use_low_gear_ratio=task_params[‘low_gear’]))
#    ptu.set_gpu_mode(variant[‘use_gpu’], variant[‘gpu_id’])

#    tasks = env.get_all_task_idx()

#    obs_dim = int(np.prod(env.observation_space.shape))
#    action_dim = int(np.prod(env.action_space.shape))




if __name__ == '__main__':
    main()
