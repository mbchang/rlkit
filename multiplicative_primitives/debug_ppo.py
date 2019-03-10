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


from log import RunningAverage
from rb import Memory
from ppo import PPO
from agent import BaseActionAgent
from multigaussian import Policy, ValueFn

import sys
sys.path.append('../')
from rlkit.envs.wrappers import NormalizedBoxEnv
from rlkit.envs.ant_goal import AntGoalEnv

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
parser.add_argument('--save-every', type=int, default=1000,
                    help='interval between training status saves (default: 1000)')
parser.add_argument('--update-every', type=int, default=1000,
                    help='interval between updating agent (default: 1000)')
parser.add_argument('--maxeplen', type=int, default=100,
                    help='length of an episode (default: 100)')
parser.add_argument('--max-iters', type=int, default=int(1e7),
                    help='number of episodes total (default: 1e7)')
parser.add_argument('--debug', action='store_true',
                    help='debug')
args = parser.parse_args()

torch.manual_seed(args.seed)
device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

eps = np.finfo(np.float32).eps.item()

def process_args(args):
    if args.debug:
        args.max_iters = 100
        args.maxeplen = 50
        args.log_interval = 5
        args.save_every = 25
        args.update_every = 25
    return args

class Experiment():
    def __init__(self, agent, env, args):
        self.agent = agent
        self.env = env
        self.run_avg = RunningAverage()
        self.logger = None
        self.args = args

    def sample_trajectory(self):
        episode_data = []
        state = self.env.reset()
        for t in range(self.args.maxeplen):  # Don't infinite loop while learning
            action, log_prob, value = self.agent(torch.from_numpy(state).float())
            state, reward, done, _ = self.env.step(action)
            if args.render:
                self.env.render()
            mask = 0 if done else 1
            e = {'state': state,
                 'action': action,
                 'logprob': log_prob,
                 'mask': mask,
                 'reward': reward,
                 'value': value}
            episode_data.append(e)
            self.agent.store_transition(e)
            if done:
                break
        returns = sum([e['reward'] for e in episode_data])
        return returns, t

    def experiment(self):
        returns = []
        for i_episode in range(1, self.args.max_iters+1):
            ret, t = self.sample_trajectory()
            running_return = self.run_avg.update_variable('reward', ret)
            if i_episode % self.args.update_every == 0:
                print('Update Agent')
                self.agent.improve()
            if i_episode % self.args.log_interval == 0:
                self.log(i_episode, ret, running_return)
            if i_episode % self.args.save_every == 0:
                # this should just take the logger into account.
                print(running_return)
                print(self.run_avg.data)

                returns.append(running_return)
                pickle.dump(returns, open('log.p', 'wb'))  # this only starts logging after the first improve_every though!

    def log(self, i_episode, ret, running_return):
        print('Episode {}\tLast return: {:.2f}\tAverage return: {:.2f}'.format(
            i_episode, ret, running_return))

def main(args):
    args = process_args(args)
    env = gym.make('CartPole-v0')
    # env = gym.make('Ant-v2')
    # env = AntGoalEnv(n_tasks=1, use_low_gear_ratio=True)
    # tasks = env.get_all_task_idx()
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
    exp = Experiment(agent, env, args)
    exp.experiment()


if __name__ == '__main__':
    main(args)
