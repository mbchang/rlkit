import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from rb import Memory
from ppo import PPO

class BaseActionAgent(nn.Module):
    def __init__(self, policy, valuefn, id, device, args):
        super(BaseActionAgent, self).__init__()
        self.policy = policy
        self.valuefn = valuefn
        self.model = nn.ModuleList([self.policy, self.valuefn])
        self.id = id
        self.device=device
        self.args = args

        self.initialize_memory()
        self.initialize_optimizer(args)
        self.initialize_optimizer_scheduler(args)
        self.initialize_rl_alg(args)

    def initialize_memory(self):
        self.buffer = Memory(element='simplertransition')

    def initialize_optimizer(self, args):
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=self.args.lr)
        self.value_optimizer = optim.Adam(self.valuefn.parameters(), lr=self.args.lr)
        self.optimizer = {'policy_opt': self.policy_optimizer, 'value_opt': self.value_optimizer}

    def initialize_optimizer_scheduler(self, args):
        lr_lambda = lambda epoch: 1  # TODO
        self.po_scheduler = optim.lr_scheduler.LambdaLR(self.policy_optimizer, lr_lambda)
        self.vo_scheduler = optim.lr_scheduler.LambdaLR(self.value_optimizer, lr_lambda)

    def initialize_rl_alg(self, args):
        hyperparams = {
            'gamma': self.args.gamma,
        }

        self.rl_alg = PPO(
            policy=self.policy, 
            policy_optimizer=self.policy_optimizer, 
            valuefn=self.valuefn, 
            value_optimizer=self.value_optimizer, 
            replay_buffer=self.buffer,
            device=self.device,
            **hyperparams)

    def cuda(self):
        self.model.cuda()

    def forward(self, state):
        state = state.to(self.device)
        with torch.no_grad():
            action = self.policy.select_action(state.detach())
        log_prob = self.policy.get_log_prob(state.detach(), action)  # not sure about the gradients here. Detach?
        value = self.valuefn(state.detach())  # detach?
        return action, log_prob, value

    def store_transition(self, transition):
        self.buffer.push(
            transition['state'],
            transition['action'],
            transition['mask'],
            transition['reward'],
            )

    def improve(self):
        self.rl_alg.improve(args=self.args)

