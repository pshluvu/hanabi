"""Population-trained Agent -- deployment/inference wrapper.

Loads a checkpoint produced by drl_for_hanabi/algorithms/population_training.py
and exposes it through the same act(observation) interface every other agent
in this repo uses, so it can be seated in play_together.py alongside any
combination of the existing agents.

Architecture matches what population_training.py trains, so a checkpoint
saved by that script loads directly here with no conversion needed.
"""
import sys
import os

drl_folder = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.realpath(__file__))))))
if drl_folder not in sys.path:
    sys.path.insert(0, drl_folder)

import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

from hanabi_learning_environment.rl_env import Agent
from drl_for_hanabi.utils import basics, state_representation_dec, action_representation_dec


class Population_Dec_Agent(Agent):
    """Plays using a policy trained against a diverse partner pool, rather
    than pure self-play -- intended to generalize better to unfamiliar
    teammates than the self-play-trained ppo/vpg/spg agents do.
    """

    def __init__(self, config, *args, **kwargs):
        """Initialize the agent.

        Args:
            config: dict with keys:
                - checkpoint_path: str, direct path to a .pth file saved by
                  population_training.py (required).
                - hidden_layers_pi / hidden_layers_vf: must match the
                  architecture used during training (defaults match
                  population_training.py's defaults).
                - obs_vec_type: int, must match training (default 136).
                - num_actions: int, must match training (default 11).
        """
        self.hidden_layers_pi = config.get('hidden_layers_pi', [128, 128, 64])
        self.hidden_layers_vf = config.get('hidden_layers_vf', [128, 64, 32])
        self.obs_vec_type = config.get('obs_vec_type', 136)
        self.num_actions = config.get('num_actions', 11)
        checkpoint_path = config.get('checkpoint_path')
        if not checkpoint_path:
            raise ValueError(
                "Population_Dec_Agent requires config['checkpoint_path'] "
                "pointing at a .pth file saved by population_training.py.")

        obs_type_options = {
            7: state_representation_dec.obs_to_vec7_playability,
            37: state_representation_dec.obs_to_vec37_owncards,
            62: state_representation_dec.obs_to_vec62_discards,
            136: state_representation_dec.obs_to_vec136_binary,
            186: state_representation_dec.obs_to_vec186_others_cards,
        }
        self.obs_to_vec_func = obs_type_options[self.obs_vec_type]

        self.device = basics.get_default_device()
        self.policy_network = basics.neural_network(
            self.obs_vec_type, self.hidden_layers_pi, self.num_actions, nn.Tanh)
        self.value_network = basics.neural_network(
            self.obs_vec_type, self.hidden_layers_vf, 1, nn.Tanh)
        self.policy_network.to(self.device)
        self.value_network.to(self.device)
        self.optimizer_pi = Adam(self.policy_network.parameters())
        self.optimizer_vf = Adam(self.value_network.parameters())

        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        self.policy_network.load_state_dict(checkpoint["policy_network"])
        self.optimizer_pi.load_state_dict(checkpoint["optimizer_pi"])
        self.value_network.load_state_dict(checkpoint["value_network"])
        self.optimizer_vf.load_state_dict(checkpoint["optimizer_vf"])
        self.policy_network.eval()
        print(f"Population_Dec_Agent loaded policy parameters from: {checkpoint_path}")

    def observation_to_vector(self, obs):
        return self.obs_to_vec_func(obs)

    def act(self, observation):
        if observation['current_player_offset'] != 0:
            return None

        obs_vec = self.observation_to_vector(observation)
        obs_vec_t = torch.as_tensor(obs_vec, dtype=torch.float32).to(self.device)
        with torch.no_grad():
            logits = self.policy_network(obs_vec_t)
        dist = Categorical(logits=logits)
        act_int = torch.argmax(dist.probs).item()  # greedy at deployment time

        legals_int = observation['legal_moves_as_int']
        legals_HLE = observation['legal_moves']
        hint_possible = (observation['information_tokens'] > 0)

        if act_int == 10 and hint_possible:
            return action_representation_dec.get_random_hint(legals_HLE)
        if act_int in legals_int:
            return act_int
        # Fallback: the greedy pick wasn't legal in this state (can happen
        # with an undertrained or edge-case policy) -- take a random legal
        # move rather than crash the game, same convention the existing
        # dec agents use.
        import random
        return random.choice(legals_int)
