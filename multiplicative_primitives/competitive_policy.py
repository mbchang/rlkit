"""
For the policy:
    instead of a network that maps from state to action,
    we will have k networks, one for each action, that
"""


import argparse
import gym
import numpy as np
from itertools import count
from collections import namedtuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.distributions import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal



parser = argparse.ArgumentParser(description='PyTorch actor-critic example')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor (default: 0.99)')
parser.add_argument('--seed', type=int, default=543, metavar='N',
                    help='random seed (default: 543)')
parser.add_argument('--render', action='store_true',
                    help='render the environment')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='interval between training status logs (default: 10)')
args = parser.parse_args()


env = gym.make('CartPole-v0')
env.seed(args.seed)
torch.manual_seed(args.seed)


SavedAction = namedtuple('SavedAction', ['log_prob', 'value'])


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.affine1 = nn.Linear(4, 128)
        self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        x = F.relu(self.affine1(x))
        action_scores = self.action_head(x)
        state_values = self.value_head(x)
        return F.softmax(action_scores, dim=-1), state_values

class GaussianParams(nn.Module):
    def __init__(self, hdim, zdim):
        super(GaussianParams, self).__init__()
        self.mu = nn.Linear(hdim, zdim)
        self.logstd = nn.Linear(hdim, zdim)

        # TODO: you can initialize them to standard normal.

    def forward(self, x):
        mu = self.mu(x)
        logstd = self.logstd(x)
        return mu, torch.exp(logstd)

class Recognizer(nn.Module):
    """
        x --> z
    """
    def __init__(self, dims):
        super(Recognizer, self).__init__()
        assert len(dims) >= 2
        self.dims = dims
        self.act = F.relu
        self.layers = nn.ModuleList()
        for i in range(len(self.dims)-2):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))

        # domain specific!
        self.hdim = self.dims[-2]
        self.zdim = self.dims[-1]
        self.parameter_producer = GaussianParams(hdim=self.hdim, zdim=self.zdim)

    def kl_standard_normal(self, dist):
        prior = self.standard_normal_prior()
        kl = torch.distributions.kl.kl_divergence(p=dist, q=prior).mean()
        return kl

    def standard_normal_prior(self):
        prior_mu = torch.zeros(self.zdim)
        prior_std = torch.ones(self.zdim)
        prior = MultivariateNormal(loc=prior_mu, scale_tril=torch.diag(prior_std))
        return prior

    def forward(self, x):
        for layer in self.layers:
            x = self.act(layer(x))
        # at this point we should output params, which are 
        mu, std = self.parameter_producer(x)
        dist = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))
        return dist

def normalize(vector):
    denominator = torch.sum(vector, dim=-1).unsqueeze(-1)
    normalized = vector / denominator
    return normalized


# How will they get rewarded?
class CompetitivePolicy(nn.Module):
    def __init__(self, encoders):
        super(CompetitivePolicy, self).__init__()
        self.encoders = encoders

        self.affine1 = nn.Linear(4, 128)
        # self.action_head = nn.Linear(128, 2)
        self.value_head = nn.Linear(128, 1)

        self.saved_actions = []
        self.rewards = []

    def forward(self, x):
        # action
        # should this be probabilities?
        action_scores = torch.Tensor([m.kl_standard_normal(m(x)) for m in self.encoders])
        assert action_scores.dim() == 1  # batch size of 1!
        action_dist = normalize(action_scores)
        # action_dist = F.softmax(action_scores, dim=-1)

        # value
        x = F.relu(self.affine1(x))
        state_values = self.value_head(x)
        return action_dist, state_values


encoders = nn.ModuleList([Recognizer(dims=[4, 32]) for i in range(2)])
model = CompetitivePolicy(encoders=encoders)
# model = Policy()
optimizer = optim.Adam(model.parameters(), lr=3e-2)
eps = np.finfo(np.float32).eps.item()


def select_action(state):
    state = torch.from_numpy(state).float()
    probs, state_value = model(state)
    m = Categorical(probs)
    action = m.sample()
    model.saved_actions.append(SavedAction(m.log_prob(action), state_value))
    return action.item()


def finish_episode():
    R = 0
    saved_actions = model.saved_actions
    policy_losses = []
    value_losses = []
    rewards = []
    for r in model.rewards[::-1]:
        R = r + args.gamma * R
        rewards.insert(0, R)
    rewards = torch.tensor(rewards)
    rewards = (rewards - rewards.mean()) / (rewards.std() + eps)
    for (log_prob, value), r in zip(saved_actions, rewards):
        reward = r - value.item()
        policy_losses.append(-log_prob * reward)
        value_losses.append(F.smooth_l1_loss(value, torch.tensor([r])))
    optimizer.zero_grad()
    loss = torch.stack(policy_losses).sum() + torch.stack(value_losses).sum()
    loss.backward()
    optimizer.step()
    del model.rewards[:]
    del model.saved_actions[:]




    


def main():
    running_reward = 10
    for i_episode in count(1):
        state = env.reset()
        for t in range(10000):  # Don't infinite loop while learning
            action = select_action(state)
            state, reward, done, _ = env.step(action)
            if args.render:
                env.render()
            model.rewards.append(reward)
            if done:
                break

        running_reward = running_reward * 0.99 + t * 0.01
        finish_episode()
        if i_episode % args.log_interval == 0:
            print('Episode {}\tLast length: {:5d}\tAverage length: {:.2f}'.format(
                i_episode, t, running_reward))
        if running_reward > env.spec.reward_threshold:
            print("Solved! Running reward is now {} and "
                  "the last episode runs to {} time steps!".format(running_reward, t))
            break


if __name__ == '__main__':
    main()
