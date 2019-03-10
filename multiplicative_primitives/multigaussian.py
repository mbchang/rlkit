import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Categorical

from functions import Trunk

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

class BasePolicy(nn.Module):
    def __init__(self):
        super(BaseGaussianPolicy, self).__init__()

    def select_action(self, state):
        dist = self.forward(state)
        action = dist.sample()
        return action.data

    def get_log_prob(self, state, action):
        dist = self.network(state)
        log_prob = dist.log_prob(action)
        return log_prob

class CompositePolicy(BasePolicy):
    def __init__(self, weightdims, primitives):
        super(CompositePolicy, self).__init__()
        self.weightdims = weightdims
        self.primitives = primitives
        self.weight_network = Trunk(self.weightdims[:-1])
        self.weight_head = nn.Linear(self.weightdims[-2], self.weightdims[-1])

    def forward(self, state):
        mus, logstds = zip(*[p(state) for p in self.primitives])  # list of length k of (bsize, zdim)
        mus = torch.stack(mus, dim=1)  # (bsize, k, zdim)
        logstds = torch.stack(logstds, dim=1)  # (bsize, k, zdim)
        stds = torch.exp(logstds)  # (bsize, k, zdim)
        bsize, k, zdim = stds.size()

        weights = F.sigmoid(self.weight_head(self.weight_network(state)))  # (bsize, k)
        broadcasted_weights = weights.view(bsize, k, 1)

        ##############################
        weights_over_variance = broadcasted_weights/(stds*stds)  # (bsize, k, zdim)
        inverse_variance = torch.sum(weights_over_variance, dim=1)  # (bsize, zdim)
        ##############################
        composite_std = 1.0/torch.sqrt(inverse_variance)
        composite_logstd = -0.5 * torch.log(inverse_variance)
        ##############################
        weighted_mus = weights_over_variance * mus
        composite_mu = torch.sum(weighted_mus, dim=1)/inverse_variance  # (bsize, zdim)
        ##############################
        dist = MultivariateNormal(loc=composite_mu, scale_tril=torch.diag_embed(composite_std))
        return dist

class PrimitivePolicy(BasePolicy):
    def __init__(self, dims):
        super(PrimitivePolicy, self).__init__()
        self.dims = dims
        self.network = Trunk(self.dims[:-1])
        self.mu_head = nn.Linear(self.dims[-2], self.dims[-1])
        self.logstd_head = nn.Linear(self.dims[-2], self.dims[-1])

    def forward(self, state):
        h = self.network(state)
        mu, std = self.mu_head(h), torch.exp(self.logstd_head(h))
        dist = MultivariateNormal(loc=mu, scale_tril=torch.diag_embed(std))
        return dist

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
        value = self.value_head(state)
        return value


def debug():
    BSIZE = 5
    K = 4
    ZDIM = 3

    # list of length k of (bsize, zdim)
    mus = [torch.rand(BSIZE, ZDIM) for k in range(K)]
    logstds = [torch.rand(BSIZE, ZDIM) for k in range(K)]
    weights = torch.rand(BSIZE, K)
    mus = torch.stack(mus, dim=1)  # (bsize, k, zdim)
    logstds = torch.stack(logstds, dim=1)  # (bsize, k, zdim)
    stds = torch.exp(logstds)  # (bsize, k, zdim)
    bsize, k, zdim = stds.size()
    broadcasted_weights = weights.view(bsize, k, 1)
    print('mus', mus.size())
    print('logstds', logstds.size())
    print('stds', stds.size())
    print('weights', weights.size())
    print('broadcasted_weights', broadcasted_weights.size())
    ##############################
    weights_over_variance = broadcasted_weights/(stds*stds)  # (bsize, k, zdim)
    inverse_variance = torch.sum(weights_over_variance, dim=1)  # (bsize, zdim)
    print('weights_over_variance', weights_over_variance.size())
    print('inverse_variance', inverse_variance.size())
    ##############################
    composite_std = 1.0/torch.sqrt(inverse_variance)  # (bsize, zdim)
    composite_logstd = -0.5 * torch.log(inverse_variance)  # (bsize, zdim)
    print('composite_std', composite_std.size())
    print('composite_logstd', composite_logstd.size())
    ##############################
    weighted_mus = weights_over_variance * mus  # (bsize, k, zsize)
    composite_mu = torch.sum(weighted_mus, dim=1)/inverse_variance  # (bsize, zdim)
    print('weighted_mus', weighted_mus.size())
    print('composite_mu', composite_mu.size())
    ##############################

    """
    mus torch.Size([5, 4, 3])
    logstds torch.Size([5, 4, 3])
    stds torch.Size([5, 4, 3])
    weights torch.Size([5, 4])
    broadcasted_weights torch.Size([5, 4, 1])
    weights_over_variance torch.Size([5, 4, 3])
    inverse_variance torch.Size([5, 3])
    composite_std torch.Size([5, 3])
    composite_logstd torch.Size([5, 3])
    weighted_mus torch.Size([5, 4, 3])
    composite_mu torch.Size([5, 3])
    """


if __name__ == '__main__':
    debug()
