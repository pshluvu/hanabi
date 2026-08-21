"""
Agent registry.

This file does NOT reimplement or merge any agent. It only imports the
existing agent classes -- exactly as they already live in the repo -- and
gives each one a short name plus a default config, so that any combination
of agents can be seated in a single game by `play_together.py`.

Agents come from two places in the repo (both already present):
  1. hanabi_learning_environment/agents/            (Random, Simple)
  2. drl_for_hanabi/extra/decentralized/agents/      (Human, rule-based, SPG/VPG/PPO)

Nothing in either folder is changed. This file just points at them.
"""

# --- Agents from hanabi_learning_environment/agents/ ------------------------
from hanabi_learning_environment.agents.random_agent import RandomAgent
from hanabi_learning_environment.agents.simple_agent import SimpleAgent

# --- Agents from drl_for_hanabi/extra/decentralized/agents/ -----------------
from agents.human_player.human_player import HumanPlayer
from agents.rule_based.bram_simple_agent import BramSimpleAgent
from agents.rule_based.bram_bias import BramBiasAgent
from agents.rule_based.bram_best_rule_based_agent import BramsBestRuleBasedAgent
from agents.spg_dec.spg_dec_agent import SPG_Dec_Agent
from agents.vpg_dec.vpg_dec_agent import VPG_Dec_Agent
from agents.ppo_dec.ppo_dec_agent import PPO_Dec_Agent

# --- Optional: Rainbow DQN, wrapped to speak the same act() interface -------
# Rainbow lives in hanabi_learning_environment/agents/rainbow and normally
# plays via a different (Dopamine-style) begin_episode/step/end_episode loop.
# rainbow_adapter.py wraps it so it can sit in the same game as everyone else.
# It needs the legacy TF1 + gin + Dopamine stack and a trained checkpoint, so
# the import is guarded: if that stack isn't installed, Rainbow is just
# skipped from the registry instead of crashing every other agent.
try:
  from rainbow_adapter import RainbowActAgent
  _RAINBOW_AVAILABLE = True
except ImportError as _rainbow_import_error:
  _RAINBOW_AVAILABLE = False
  _RAINBOW_IMPORT_ERROR = _rainbow_import_error


# Same hyperparameter/checkpoint configs already used in hanabi_runner.py,
# copied here (not moved) so hanabi_runner.py keeps working untouched.
CONFIG_SPG = {
    'hidden_layers': [64, 64],
    'obs_vec_type': 62,
    'num_actions': 11,
    "load_policy": ["2021-02-26_shuf", 25, 5],
}

CONFIG_VPG = {
    'hidden_layers_pi': [128, 128, 64],
    'hidden_layers_vf': [128, 64, 32],
    'obs_vec_type': 136,
    'num_actions': 11,
    "load_policy": ["2021-04-13_combi_vpg", 9, 4],
}

CONFIG_PPO = {
    'hidden_layers_pi': [128, 128, 64],
    'hidden_layers_vf': [128, 64, 32],
    'obs_vec_type': 186,
    'num_actions': 11,
    "load_policy": ["2021-04-13_combi_ppo", 8, 1],
}

CONFIG_RAINBOW = {
    # Point this at a trained Rainbow checkpoint directory, e.g. one produced
    # by hanabi_learning_environment/agents/rainbow/train.py
    "checkpoint_dir": None,
    "gin_config": None,   # path to a .gin file, e.g. configs/hanabi_rainbow.gin
}


def _build_random(config):
  return RandomAgent(config=config or {})


def _build_simple(config):
  return SimpleAgent(config=config or {})


def _build_human(config):
  return HumanPlayer()


def _build_bram_simple(config):
  return BramSimpleAgent(config=config or {})


def _build_bram_bias(config):
  return BramBiasAgent(config=config or {'play_position': 0})


def _build_bram_best(config):
  return BramsBestRuleBasedAgent(config=config or {})


def _build_spg(config):
  return SPG_Dec_Agent(config=config or CONFIG_SPG)


def _build_vpg(config):
  return VPG_Dec_Agent(config=config or CONFIG_VPG)


def _build_ppo(config):
  return PPO_Dec_Agent(config=config or CONFIG_PPO)


def _build_rainbow(config):
  if not _RAINBOW_AVAILABLE:
    raise ImportError(
        "RainbowActAgent could not be imported (missing legacy TF1 / gin / "
        "Dopamine dependencies). Original error: {}".format(_RAINBOW_IMPORT_ERROR))
  return RainbowActAgent(config=config or CONFIG_RAINBOW)


# name -> (builder function, default config)
AGENT_REGISTRY = {
    "random": (_build_random, {}),
    "simple": (_build_simple, {}),
    "human": (_build_human, {}),
    "bram_simple": (_build_bram_simple, {}),
    "bram_bias": (_build_bram_bias, {'play_position': 0}),
    "bram_best": (_build_bram_best, {}),
    "spg": (_build_spg, CONFIG_SPG),
    "vpg": (_build_vpg, CONFIG_VPG),
    "ppo": (_build_ppo, CONFIG_PPO),
    "rainbow": (_build_rainbow, CONFIG_RAINBOW),
}


def build_agent(name, config=None):
  """Build a single agent instance by its registry name.

  Args:
    name: str, one of AGENT_REGISTRY's keys (see `list_agents()`).
    config: optional dict overriding that agent's default config.

  Returns:
    An Agent instance (subclass of hanabi_learning_environment.rl_env.Agent)
    implementing act(observation) -> action or None.
  """
  if name not in AGENT_REGISTRY:
    raise KeyError(
        "Unknown agent '{}'. Available agents: {}".format(
            name, sorted(AGENT_REGISTRY.keys())))
  builder, default_config = AGENT_REGISTRY[name]
  return builder(config if config is not None else default_config)


def list_agents():
  """Returns the sorted list of agent names available in the registry."""
  return sorted(AGENT_REGISTRY.keys())
