import sys, os
sys.path.insert(0, '/home/percy/hanabi')
sys.path.insert(0, '/home/percy/hanabi/drl_for_hanabi/extra/decentralized')
import torch
torch.backends.mkldnn.enabled = False
torch.set_num_threads(1)
from population_training import PopulationAgent, build_partner_pool, run_one_episode
l = PopulationAgent()
p = build_partner_pool()
obs_vecs, act_ints, log_probs, rtg, score, illegal, partner = run_one_episode(l, p)
print('episode data collected, steps:', len(obs_vecs))
rtg_t = torch.as_tensor(rtg, dtype=torch.float32).to(l.device)
log_probs_t = torch.stack(log_probs)
loss_pi = -(log_probs_t * rtg_t).mean()
print('loss computed:', loss_pi.item())
l.optimizer_pi.zero_grad()
loss_pi.backward()
print('backward pass OK')
l.optimizer_pi.step()
print('optimizer step OK')
