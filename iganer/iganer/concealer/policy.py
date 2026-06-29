"""Concealer policy (factor A): detector state -> action distribution + value."""
import torch, torch.nn as nn
from torch.distributions import Categorical
class ConcealerPolicy(nn.Module):
    def __init__(self, state_dim, n_actions, hidden=256):
        super().__init__()
        self.body=nn.Sequential(nn.Linear(state_dim,hidden),nn.ReLU(),
                                nn.Linear(hidden,hidden),nn.ReLU())
        self.pi=nn.Linear(hidden,n_actions); self.v=nn.Linear(hidden,1)
    def forward(self,s):
        h=self.body(s); return self.pi(h), self.v(h).squeeze(-1)
    @torch.no_grad()
    def act(self,s):
        logits,v=self.forward(s); d=Categorical(logits=logits); a=d.sample()
        return a, d.log_prob(a), v
    def evaluate(self,s,a):
        logits,v=self.forward(s); d=Categorical(logits=logits)
        return d.log_prob(a), d.entropy(), v
