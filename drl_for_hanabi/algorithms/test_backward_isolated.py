import sys, os
sys.path.insert(0, '/home/percy/hanabi')
sys.path.insert(0, '/home/percy/hanabi/drl_for_hanabi/extra/decentralized')
import torch
from population_training import PopulationAgent
l = PopulationAgent()
print('learner built')
obs_vec = [0.0] * 136
obs_vec_t = torch.as_tensor(obs_vec, dtype=torch.float32)
logits = l.policy_network(obs_vec_t)
from torch.distributions.categorical import Categorical
dist = Categorical(logits=logits)
act = dist.sample()
log_prob = dist.log_prob(act)
print('forward pass done, log_prob:', log_prob.item())
loss = -log_prob
l.optimizer_pi.zero_grad()
loss.backward()
print('backward pass OK - no partner pool involved')
l.optimizer_pi.step()
print('optimizer step OK')
