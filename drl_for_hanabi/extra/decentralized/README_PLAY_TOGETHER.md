# Playing all agents together

These 3 files add a shared runner so any combination of the repo's existing
agents can play one game together. **No existing agent file is modified.**

- `agent_registry.py` — imports every agent exactly as it already exists and
  gives each one a short name.
- `rainbow_adapter.py` — wraps the existing Rainbow DQN agent
  (`hanabi_learning_environment/agents/rainbow/`) so it exposes the same
  `act(observation)` call every other agent uses. Requires the legacy
  TF1 + gin + Dopamine stack Rainbow already needed, plus a trained
  checkpoint. If that stack isn't installed, "rainbow" is simply left out
  of the registry — nothing else breaks.
- `play_together.py` — the runner. Builds the requested agents and steps
  them through one shared environment, turn by turn.

## Where these files go

Drop all three (plus this README) into:

```
drl_for_hanabi/extra/decentralized/
```

right next to the existing `hanabi_runner.py` and `agents/` folder — they
rely on being run from that directory, the same way `hanabi_runner.py`
already does (`from agents.human_player.human_player import HumanPlayer`,
etc.).

## Usage

```bash
cd drl_for_hanabi/extra/decentralized

# See every agent name available
python play_together.py --list-agents

# Play two heuristic agents against each other
python play_together.py --agents random simple

# Mix rule-based and trained RL agents, 10 episodes
python play_together.py --agents ppo vpg bram_best --episodes 10

# Sit in yourself against a rule-based agent
python play_together.py --agents human bram_simple

# Include Rainbow (needs a trained checkpoint + the legacy TF1/gin stack)
python play_together.py --agents rainbow ppo \
    --rainbow-checkpoint /path/to/checkpoint_dir \
    --rainbow-gin ../../../hanabi_learning_environment/agents/rainbow/configs/hanabi_rainbow.gin
```

Available names: `random`, `simple`, `human`, `bram_simple`, `bram_bias`,
`bram_best`, `spg`, `vpg`, `ppo`, `rainbow`.

## Why Rainbow needed a wrapper

Every other agent already implements `act(observation) -> action or None` —
that's the shared interface `hanabi_runner.py` was built around, so those
agents already "just work" together.

Rainbow is the one exception: it's built for Dopamine-style self-play
training via `begin_episode` / `step` / `end_episode`
(`agents/rainbow/run_experiment.py`), and it returns a plain integer action.
`rainbow_adapter.py` wraps it so it fits the same `act()` contract as
everyone else — this works cleanly because `HanabiEnv.step()` already
accepts either an action dict *or* a legal int action, and every
observation already includes the `vectorized` and `legal_moves_as_int`
fields Rainbow expects. No changes to `dqn_agent.py` or `rainbow_agent.py`
were needed.
