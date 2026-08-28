""" Play any combination of existing agents together in one game.

This does not merge, rewrite, or duplicate any agent's code. It only:
  1. builds each requested agent via agent_registry.build_agent(name)
  2. steps them through ONE shared Hanabi environment, turn by turn
     (same loop pattern as hanabi_runner.py's Runner class)

Usage examples:
  python play_together.py --agents random simple
  python play_together.py --agents ppo vpg bram_best --episodes 10
  python play_together.py --agents human bram_simple
  python play_together.py --agents rainbow ppo \
      --rainbow-checkpoint /path/to/checkpoint_dir \
      --rainbow-gin hanabi_learning_environment/agents/rainbow/configs/hanabi_rainbow.gin

Run `python play_together.py --list-agents` to see every registered name.
"""
import argparse

from hanabi_learning_environment import rl_env_adjusted
from agents.human_player.human_player import HumanPlayer
from agent_registry import build_agent, list_agents, AGENT_REGISTRY


def parse_args():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument(
      "--agents", nargs="+", default=None,
      help="2 to 5 agent names to seat in the game, in turn order. "
           "Run --list-agents to see the options.")
  parser.add_argument(
      "--episodes", type=int, default=1,
      help="Number of episodes to play.")
  parser.add_argument(
      "--list-agents", action="store_true",
      help="Print available agent names and exit.")
  parser.add_argument(
      "--rainbow-checkpoint", default=None,
      help="Checkpoint directory for the 'rainbow' agent, if included.")
  parser.add_argument(
      "--rainbow-gin", default=None,
      help="Optional .gin config path for the 'rainbow' agent.")
  return parser.parse_args()


class Runner(object):
  """Steps a list of already-built agents through one shared environment."""

  def __init__(self, agents):
    game_config = {
        "colors": 5,
        "ranks": 5,
        "players": len(agents),
        "max_information_tokens": 8,
        "max_life_tokens": 3,
    }

    self.game_config = game_config
    self.environment = rl_env_adjusted.make(obs_type="CHEAT", config=game_config)
    self.agents = agents
    print()
    for idx, agent in enumerate(self.agents):
      print(f"Agent {idx} is {agent.__class__.__name__}")

  def run(self, nr_episodes=1):
    rewards = []
    for episode in range(nr_episodes):
      observations = self.environment.reset()
      agents = self.agents    

      for idx, agent in enumerate(agents):
        if isinstance(agent, HumanPlayer):
          print("\nNEW GAME:\n")
          agent.print_the_state(observations["player_observations"][idx])
          break

      done = False
      episode_reward = 0
      turn = 0
      current_player_action = None
      current_player = None
      while not done:
        turn += 1
        for agent_id, agent in enumerate(agents):
          observation = observations["player_observations"][agent_id]
          action = agent.act(observation)
          if observation["current_player"] == agent_id:
            assert action is not None
            current_player_action = action
            current_player = observation["current_player"]
          else:
            assert action is None

        print(f"Turn {turn}, agent {current_player}, action: {current_player_action}")
        observations, reward, done, unused_info = self.environment.step(
            current_player_action)
        episode_reward += reward

      for idx, agent in enumerate(agents):
        if isinstance(agent, HumanPlayer):
          print("\nGAME FINISHED:\n")
          agent.print_the_state(observations["player_observations"][idx])
          print(f"\nSCORE: {episode_reward}")
          if episode_reward == 25:
            print("\nPERFECT GAME! CONGRATULATIONS!")
          break

      rewards.append(episode_reward)
      print(f"\nFinished episode nr: {episode} with reward {episode_reward}")
      print(f"Max reward: {max(rewards):.3f}")
      avg = sum(rewards) / len(rewards)
      print(f"Average reward: {avg:.3f}")
    return rewards


def main():
  args = parse_args()

  if args.list_agents:
    print("Available agents:")
    for name in list_agents():
      print(f"  - {name}")
    return

  if not args.agents:
    raise SystemExit(
        "Pass --agents with 2 to 5 names, e.g. --agents random simple\n"
        "Or run --list-agents to see the options.")
  if not (2 <= len(args.agents) <= 5):
    raise SystemExit("Hanabi needs between 2 and 5 players.")

  built_agents = []
  for name in args.agents:
    if name == "rainbow":
      # Rainbow's config needs environment-derived sizes, filled in below
      # once we know num_players; build it after the environment exists.
      built_agents.append(("rainbow", None))
    else:
      built_agents.append((name, build_agent(name)))

  # If rainbow was requested, build it now that we know num_players, using a
  # throwaway env just to read num_moves()/observation size.
  if any(name == "rainbow" for name, _ in built_agents):
    if not args.rainbow_checkpoint:
      raise SystemExit(
          "The 'rainbow' agent needs --rainbow-checkpoint pointing at a "
          "trained checkpoint directory (see rainbow/train.py).")
    probe_config = {
        "colors": 5, "ranks": 5, "players": len(args.agents),
        "max_information_tokens": 8, "max_life_tokens": 3,
    }
    probe_env = rl_env_adjusted.make(obs_type="CHEAT", config=probe_config)
    rainbow_config = dict(AGENT_REGISTRY["rainbow"][1])
    rainbow_config.update({
        "checkpoint_dir": args.rainbow_checkpoint,
        "gin_config": args.rainbow_gin,
        "num_actions": probe_env.num_moves(),
        "observation_size": probe_env.vectorized_observation_shape()[0],
        "num_players": len(args.agents),
    })
    rainbow_agent_instance = build_agent("rainbow", rainbow_config)
    built_agents = [
        (name, rainbow_agent_instance if name == "rainbow" else agent)
        for name, agent in built_agents
    ]

  agents = [agent for _, agent in built_agents]
  runner = Runner(agents=agents)
  runner.run(nr_episodes=args.episodes)


if __name__ == "__main__":
  main()
