import sys, os
sys.path.insert(0, '/home/percy/hanabi')
sys.path.insert(0, '/home/percy/hanabi/drl_for_hanabi/extra/decentralized')
from population_training import PopulationAgent, build_partner_pool
p = build_partner_pool()
print('partner pool built first')
l = PopulationAgent()
print('learner built second - matches main() order')
