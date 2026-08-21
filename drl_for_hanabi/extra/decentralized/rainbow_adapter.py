"""
Rainbow adapter.

Rainbow (hanabi_learning_environment/agents/rainbow/dqn_agent.py,
rainbow_agent.py) is built for Dopamine-style self-play training, not the
act(observation) interface every other agent in this repo uses. It's driven
by run_experiment.py through a begin_episode / step / end_episode loop, and
it returns a plain int action rather than an action dict.

This file does NOT change dqn_agent.py or rainbow_agent.py. It just wraps a
loaded RainbowAgent so it can be seated in the same game as every other
agent through the usual act(observation) call:

  - env.step() already accepts either an action dict OR a legal int action
    (see rl_env.py's HanabiEnv.step docstring), so returning the raw int
    from Rainbow's begin_episode()/step() works without any translation.
  - observation['vectorized'] and observation['legal_moves_as_int'] are
    already present on every observation rl_env.py produces, which is
    exactly what Rainbow's begin_episode/step expect.

Requirements (not bundled -- these are Rainbow's own original requirements):
  - TensorFlow 1.x, gin-config, and the Dopamine helper modules already in
    hanabi_learning_environment/agents/rainbow/third_party/dopamine/
  - A trained checkpoint directory (produced by rainbow/train.py)

If those aren't available, agent_registry.py catches the ImportError and
simply leaves "rainbow" out of the available agents instead of crashing.
"""
import os
import sys
import numpy as np

# Make the rainbow/ folder importable, matching how run_experiment.py expects
# its sibling modules (replay_memory, third_party/dopamine, etc.) to resolve.
_RAINBOW_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(
        os.path.abspath(__file__))))),
    "hanabi_learning_environment", "agents", "rainbow")
if os.path.isdir(_RAINBOW_DIR) and _RAINBOW_DIR not in sys.path:
  sys.path.insert(0, _RAINBOW_DIR)

import gin.tf  # noqa: E402  (legacy TF1 + gin, same as dqn_agent.py)
import tensorflow as tf  # noqa: E402
import rainbow_agent  # noqa: E402  (hanabi_learning_environment/agents/rainbow/rainbow_agent.py)

from hanabi_learning_environment.rl_env import Agent


def _format_legal_moves(legal_moves, action_dim):
  """Same helper as run_experiment.py: -inf mask over illegal actions."""
  new_legal_moves = np.full(action_dim, -float('inf'))
  if legal_moves:
    new_legal_moves[legal_moves] = 0
  return new_legal_moves


class RainbowActAgent(Agent):
  """Wraps a trained RainbowAgent so it exposes act(observation)."""

  def __init__(self, config, *args, **kwargs):
    """Initialize the wrapper and restore a trained checkpoint.

    Args:
      config: dict with keys:
        - checkpoint_dir: str, directory containing the Rainbow checkpoint
          produced by rainbow/train.py (required).
        - gin_config: str, optional path to a .gin file (e.g.
          hanabi_learning_environment/agents/rainbow/configs/hanabi_rainbow.gin)
          to configure network/replay hyperparameters to match the checkpoint.
        - num_actions: int, size of the action space (environment.num_moves()).
        - observation_size: int, length of the vectorized observation.
        - num_players: int, number of players in the game.
    """
    checkpoint_dir = config.get("checkpoint_dir")
    if not checkpoint_dir:
      raise ValueError(
          "RainbowActAgent requires config['checkpoint_dir'] pointing at a "
          "trained Rainbow checkpoint (see rainbow/train.py).")

    gin_config = config.get("gin_config")
    if gin_config:
      gin.parse_config_file(gin_config)

    num_actions = config.get("num_actions")
    observation_size = config.get("observation_size")
    num_players = config.get("num_players")
    if num_actions is None or observation_size is None or num_players is None:
      raise ValueError(
          "RainbowActAgent requires 'num_actions', 'observation_size', and "
          "'num_players' in config (read them off the environment, e.g. "
          "environment.num_moves(), environment.vectorized_observation_shape()[0], "
          "and len(agents)).")

    self._action_dim = num_actions
    self._sess = tf.Session()
    self._agent = rainbow_agent.RainbowAgent(
        num_actions=num_actions,
        observation_size=observation_size,
        num_players=num_players)
    self._agent.eval_mode = True  # no training / no exploration noise

    checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    if checkpoint is None:
      raise ValueError(
          "No checkpoint found in directory: {}".format(checkpoint_dir))
    saver = tf.train.Saver()
    saver.restore(self._sess, checkpoint)
    self._agent._sess = self._sess  # dqn_agent.py builds its own session; use ours

    self._episode_started = False

  def reset(self, config=None):
    """Marks the next act() call as the start of a new episode."""
    self._episode_started = False

  def act(self, observation):
    """Act based on an observation, same contract as every other agent."""
    if observation['current_player_offset'] != 0:
      return None

    legal_moves = observation['legal_moves_as_int']
    legal_moves_vec = _format_legal_moves(legal_moves, self._action_dim)
    observation_vector = np.array(observation['vectorized'])

    if not self._episode_started:
      action = self._agent.begin_episode(
          current_player=0,
          legal_actions=legal_moves_vec,
          observation=observation_vector)
      self._episode_started = True
    else:
      # This wrapper doesn't track per-agent reward bookkeeping the way
      # run_experiment.py's self-play loop does, since here Rainbow is one
      # agent among possibly several *different* agent types. Pass 0 for
      # eval-only play; wire in real rewards here if you want online
      # learning while mixed in with other agents.
      action = self._agent.step(
          reward=0,
          current_player=0,
          legal_actions=legal_moves_vec,
          observation=observation_vector)

    # action is a legal, int-valued action; env.step() accepts ints directly.
    return int(action)
