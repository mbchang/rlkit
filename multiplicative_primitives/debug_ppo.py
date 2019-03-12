import argparse
import gym
from gym.wrappers import Monitor
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os
import sys
from itertools import count
from collections import namedtuple

from moviepy.editor import ImageSequenceClip
import operator
import pickle
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import time

from log import RunningAverage, create_logger
from rb import Memory
from ppo import PPO
from agent import BaseActionAgent
from multigaussian import Policy, ValueFn, PrimitivePolicy

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
parser.add_argument('--hdim', type=int, default=256,
                    help='hdim (default: 256)')
parser.add_argument('--log-interval', type=int, default=100, metavar='I',
                    help='interval between training status logs (default: 10)')
parser.add_argument('--save-every', type=int, default=1000,
                    help='interval between training status saves (default: 1000)')
parser.add_argument('--update-every', type=int, default=1000,
                    help='interval between updating agent (default: 1000)')
parser.add_argument('--visualize-every', type=int, default=1000,
                    help='interval between visualizing agent (default: 1000)')
parser.add_argument('--maxeplen', type=int, default=100,
                    help='length of an episode (default: 100)')
parser.add_argument('--max-iters', type=int, default=int(1e7),
                    help='number of episodes total (default: 1e7)')
parser.add_argument('--debug', action='store_true',
                    help='debug')
parser.add_argument('--resume', action='store_true',
                    help='resume')
parser.add_argument('--transfer', action='store_true',
                    help='transfer')
parser.add_argument('--printf', action='store_true',
                    help='printf')
parser.add_argument('--outputdir', type=str, default='runs',
                    help='outputdir')
args = parser.parse_args()

np.random.seed(args.seed)
torch.manual_seed(args.seed)
device=torch.device('cuda', index=args.gpu_index) if torch.cuda.is_available() else torch.device('cpu')

eps = np.finfo(np.float32).eps.item()

def process_args(args):
    if args.debug:
        args.max_iters = 100
        args.maxeplen = 100
        args.log_interval = 5
        args.save_every = 25
        args.update_every = 25
        args.visualize_every = 25
    return args

class Experiment():
    def __init__(self, agent, env, logger, args):
        self.agent = agent
        self.env = env
        self.run_avg = RunningAverage()
        self.logger = logger
        self.args = args

    def sample_trajectory(self, render):
        # print('new_trajectory')
        episode_data = []
        state = self.env.reset()
        for t in range(self.args.maxeplen):  # Don't infinite loop while learning
            action, log_prob, value = self.agent(torch.from_numpy(state).float())
            state, reward, done, _ = self.env.step(action.to('cpu').numpy())
            mask = 0 if done else 1
            e = {'state': state,
                 'action': action.to('cpu'),
                 'logprob': log_prob,
                 'mask': mask,
                 'reward': reward,
                 'value': value}
            if render:
                # frame = plt.imsave('{}.png'.format(t), self.env.render(mode='rgb_array'))
                frame = self.env.render(mode='rgb_array')
                e['frame'] = frame
            episode_data.append(e)
            self.agent.store_transition(e)
            if done:
                break
        return episode_data

    def experiment(self):
        for i_episode in range(1, self.args.max_iters+1):
            visualize = i_episode % self.args.visualize_every == 0
            self.logger.update_variable('episode', i_episode)
            # ret, t = self.sample_trajectory()
            episode_data = self.sample_trajectory(render=visualize)
            ret = returns = sum([e['reward'] for e in episode_data])
            running_return = self.run_avg.update_variable('running_return', ret)
            self.logger.update_variable('running_return', running_return)
            if i_episode % self.args.update_every == 0:
                print('Update Agent')
                self.agent.improve()
            if i_episode % self.args.log_interval == 0:
                self.log(i_episode, ret, running_return)
            if i_episode % self.args.save_every == 0:
                current_metrics = {
                    'running_return': running_return
                }
                # the below may be redundant
                ckpt = {
                    'agent': self.logger.to_cpu(self.agent.state_dict()),
                    'episode': i_episode,
                    'running_return': running_return
                }
                self.logger.save_checkpoint(ckpt, current_metrics, i_episode, args, '_train')
                self.logger.plot('episode', 'running_return', self.logger.expname+'_running_return')
                self.logger.save(self.logger.expname)
            if visualize:
                frames = np.array([e['frame'] for e in episode_data])
                clip = ImageSequenceClip(list(frames), fps=30).resize(0.5)
                clip.write_gif('{}/{}.gif'.format(self.logger.logdir, i_episode), fps=30)
            del episode_data

    def log(self, i_episode, ret, running_return):
        print('Episode {}\tLast return: {:.2f}\tAverage return: {:.2f}'.format(
            i_episode, ret, running_return))

def build_expname(args):
    expname = 'debug'
    return expname

def initialize_logger(logger):
    logger.add_variables(['episode', 'running_return'])
    logger.add_metric('running_return', -np.inf, operator.ge)

def main(args):
    args = process_args(args)
    logger = create_logger(build_expname, args)
    initialize_logger(logger)
    # env = gym.make('CartPole-v0')
    # env = gym.make('Ant-v2')
    env = AntGoalEnv(n_tasks=1, use_low_gear_ratio=True)
    # tasks = env.get_all_task_idx()
    # env = Monitor(env, './video')
    env.seed(args.seed)
    discrete = type(env.action_space) == gym.spaces.discrete.Discrete
    obs_dim = env.observation_space.shape[0]
    act_dim = env.action_space.n if discrete else env.action_space.shape[0]
    policy = Policy if discrete else PrimitivePolicy
    agent = BaseActionAgent(
        policy=policy(dims=[obs_dim, args.hdim, args.hdim, act_dim]), 
        valuefn=ValueFn(dims=[obs_dim, args.hdim, args.hdim, 1]), 
        id=0, 
        device=device, 
        args=args).to(device)
    exp = Experiment(agent, env, logger, args)
    exp.experiment()


if __name__ == '__main__':
    main(args)
