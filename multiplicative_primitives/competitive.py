import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import pprint

from rb import Memory

from torch.distributions.multivariate_normal import MultivariateNormal

torch.manual_seed(6)


"""
    There are many possible variants. Note that this visualizes a "chain" of
    computation.
    ----------------------------------------------------------------------------
    (1) Below is probably more suited for domains where the modules can be
    customized to be more domain-specific. This approach is probably more 
    relevant for addressing questions like parametric generalization or 
    compositional generalization. Of course, we could also just do this way 
    if we are doing the generic thing.

    This one assumes that we initialize the modules outside:

        def __init__(self, encoders, computations, decoders, num_steps, args)

    Similar as above, but where the number of steps is self-decided:

        def __init__(self, encoders, computations, decoders, args)

    ----------------------------------------------------------------------------
    (2) Below is probably more suited for generic domains. This is more of a 
    "general computational model" type of framework.

    This one assumes that we initialize the modules inside:

        def __init__(self, indim, hdim, outdim, n_encoders, n_computations, n_decoders, num_steps, args)

    Similar as above, but where the number of steps is self-decided:

        def __init__(self, indim, hdim, outdim, n_encoders, n_computations, n_decoders, args)

    ----------------------------------------------------------------------------
    Conclusion: It is probably better to begin with option (1) because it 
    probably offers more flexibility and subsumbes option (2). Option (2) is 
    attractive if you want to think of what you are building as a generic 
    module, which also makes sense. Let's go with option (1).
"""

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


class Computation(nn.Module):
    """
        z --> w
    """
    def __init__(self, dims):
        super(Computation, self).__init__()
        assert len(dims) >= 2
        self.dims = dims
        self.act = F.relu
        self.layers = nn.ModuleList()
        for i in range(len(self.dims)-1):
            self.layers.append(nn.Linear(self.dims[i], self.dims[i+1]))

    def forward(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        x = self.layers[-1](x) # no activation
        return x

class ComputationalModule(nn.Module):
    """
        Consists of a recognizer and a computation
    """
    def __init__(self, recognizer, computation, id):
        super(ComputationalModule, self).__init__()
        self.recognizer = recognizer
        self.computation = computation
        self.id = id
        # self.updater = lambda x, delta: x + delta
        self.updater = lambda x, delta: delta  # TODO how to incorporate skip connections?

    def initialize_optimizer(self):
        pass

    def initialize_optimizer_scheduler(self):
        pass

    def forward(self, x):
        z = self.recognizer(x)
        w = self.computation(z)
        y = self.updater(x, w)
        return y

class BaseSociety(nn.Module):
    """
    Outline:
        Encoders, Modules, and Decoders should all be decentralized
            If the list of encoders or decoders contians only a single element, 
            then it is just directly used.
            If a module is used across all the data, it should encode 
            information shared across all "tasks."
            If a module is used for a particular "task," it need should only 
            encode information for that particular "task."
    """
    def __init__(self, encoders, computations, decoders, args):
        super(BaseSociety, self).__init__()
        self.encoders = encoders
        self.computations = computations
        self.decoders = decoders
        self.args = args

        self.hard = True

        self.initialize_memory()
        self.initialize_optimizers(args)
        self.initialize_optimizer_schedulers(args)

    def initialize_memory(self):
        self.computation_buffer = Memory(element='inputoutput')

    def initialize_optimizers(self, args):
        pass

    def initialize_optimizer_schedulers(self, args):
        pass

    def compute_hard(self, modules, h_t):
        """
            Executes one computation step

            1. All modules produce Gaussian distributions
            2. Compute KL
            3. Declare winner
            4. Winner sample from the Gaussian distribution
            5. Winner computes on the samples
            6. Produce output
        """
        # Step 1.
        z_dists = [m.recognizer(h_t) for m in modules]

        # print('z_dists loc')
        # pprint.pprint([z.loc for z in z_dists])

        # Step 2.
        z_kls = torch.stack([m.recognizer.kl_standard_normal(z_dist) for (m, z_dist) in zip(modules, z_dists)])  # (k, bsize)

        # print('z_kls')
        # pprint.pprint(z_kls)

        # Step 3. (argmax across bsize)
        winner = torch.argmax(z_kls, dim=0)  # (bsize)

        # print('winner')
        # print(winner)
        
        # Step 4.
        z_sample = []
        # iterating through the batch. This could be made more efficient.
        for batch_idx, w in enumerate(winner):
            z_sample.append(z_dists[w].rsample()[batch_idx])
            # z_sample.append(z_dists[w].loc[batch_idx])
        z_sample = torch.stack(z_sample)  # (bsize, zdim)

        # print('z_sample')
        # print(z_sample)

        # Step 5.
        h_deltas = []
        for batch_idx, w in enumerate(winner):
            h_deltas.append(modules[w].computation(z_sample[batch_idx]))
        h_deltas = torch.stack(h_deltas)  # (bsize, hdim)

        # print('h_deltas')
        # pprint.pprint(h_deltas)

        # Step 6.
        h_tp1 = []
        for batch_idx, w in enumerate(winner):
            h_tp1.append(modules[w].updater(h_t[batch_idx], h_deltas[batch_idx]))
        h_tp1 = torch.stack(h_tp1)

        # print('h_tp1')
        # pprint.pprint(h_tp1)

        return h_tp1, None


    def compute_soft(self, modules, h_t):
        """
            Executes one computation step

            1. All modules produce Gaussian distributions
            2. Compute KL
            3. Sample from the Gaussian distribution
            4. Compute on the samples
            5. Weight computations by KL
            6. Produce averaged output
        """
        # Step 1.
        z_dists = [m.recognizer(h_t) for m in modules]

        # Step 2.
        z_kls = [m.recognizer.kl_standard_normal(z_dist) for (m, z_dist) in zip(modules, z_dists)]  # k x (bsize)
        z_kl_sum = torch.sum(torch.stack(z_kls), dim=0)  # (bsize)

        # Step 3.
        z_samples = [z_dist.rsample() for z_dist in z_dists]  # k x (bsize, zdim)

        # Step 4.
        h_deltas = [m.computation(z_sample) for (m, z_sample) in zip(modules, z_samples)]  # k x (bsize, hdim)

        # Step 5. 
        h_deltas_weighted = [(z_kl/z_kl_sum).unsqueeze(-1) * h_delta for (z_kl, h_delta) in zip(z_kls, h_deltas)]  # k x (bsize, hdim)

        # Step 6.
        h_tp1 = torch.sum(torch.stack([m.updater(h_t, h_delta_weighted) for (m, h_delta_weighted) in zip(modules, h_deltas_weighted)], dim=0), dim=0)  # (bsize, hdim)

        # bookkeep:
        # we need to save the zs_kl Variable such that it gets taken into
        # account when we compute the gradient.
        return h_tp1, None

    def compute(self, modules, h_t):
        if self.hard:
            return self.compute_hard(modules, h_t)
        else:
            return self.compute_soft(modules, h_t)

    def forward(self, x):
        """
            One variant of this is that you include the decoders in the 
            computations

            One variant of this is that all computations are of the same 
            dimension

            self.trace gets rewritten for every forward pass
        """
        self.trace = Trace()
        counter = 0
        x, done = self.compute(self.encoders, x)
        print('x after encoder')
        print(x)
        while True:
            x, done = self.compute(self.computations, x)
            counter += 1
            print('x after computation step {}'.format(counter))
            print(x)
            # hacky:
            if counter == 2: done = True
            if done: break
        x, done = self.compute(self.decoders, x)
        print('x after decoder')
        print(x)
        return x

    def compute_supervision(self):
        # note that each module should get a different loss.
        # what this means is that each module HAS ITS OWN OPTIMIZER!
        pass


# this should probably be a configurable thing, like a dictionary
# the main thing is that it stores a list of dictionaries. Probably similar
# to replay buffer.

class Trace(object):
    def __init__(self):
        super(Trace, self).__init__()
        self.trace = []

    def append(self, step_data):
        """
            step_data: dict
        """
        self.trace.append(step_data)

    def read(self):
        return copy.deepcopy(self.trace)

    def __len__(self):
        return len(self.trace)



if __name__ == '__main__':
    bsize = 3
    k = 4
    indim = 5
    zdim = 6
    hdim = 5
    outdim = 5

    # first we will imagine that indim == hdim == outdim
    # later we will take care of the case where indim != hdim != outdim

    encoders = nn.ModuleList([ComputationalModule(
        recognizer=Recognizer(dims=[indim, zdim]), 
        computation=Computation(dims=[zdim, hdim]),
        id=kk)
            for kk in range(k)])
    computations = nn.ModuleList([ComputationalModule(
        recognizer=Recognizer(dims=[hdim, zdim]), 
        computation=Computation(dims=[zdim, hdim]),
        id=kk)
            for kk in range(k)])
    decoders = nn.ModuleList([ComputationalModule(
        recognizer=Recognizer(dims=[hdim, zdim]), 
        computation=Computation(dims=[zdim, outdim]),
        id=kk)
            for kk in range(k)])

    bs = BaseSociety(encoders, computations, decoders, {})
    print('bs')
    print(bs)

    x = torch.rand(bsize, indim)
    print('x')
    print(x)

    # forward pass
    x = bs(x)
    print('x')
    print(x)



