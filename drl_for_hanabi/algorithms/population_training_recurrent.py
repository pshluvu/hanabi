"""
Population-based training for a Hanabi agent that can play with ANY of the
existing agents in this repo, not just a copy of itself -- RECURRENT version.

WHY THIS FILE EXISTS (vs. population_training.py):
The original population_training.py uses a plain feedforward (MLP) policy,
which has no memory: every decision is made from scratch off the current
observation, with nothing carried over from earlier in the game. Hanabi is
a POMDP -- a lot of what matters (how cautious has my partner been about
discarding, have they been hinting for playability or for safety, are they
following a convention or playing generically) is only visible as a PATTERN
across several turns, not in any single observation.

This version gives the learner a small GRU-based memory. The hidden state
is carried across the learner's own turns WITHIN one game (and reset at the
start of the next game), so it can implicitly build up "how is this partner
behaving so far" purely from the sequence of things it observes -- fireworks
changing, hints being given, discards happening. Crucially, the partner's
IDENTITY is never given to the network as an input anywhere. It only ever
sees the same observation vector as before; the difference is that it can
now remember what it's seen earlier in the same game. That keeps the
learner usable against a partner it has never met, which a from-a-lookup-
table "if partner == X do Y" approach would not.

This is a fresh, from-scratch training run -- the GRU-based network has a
different shape than the MLP in population_training.py, so it CANNOT resume
from an existing policy_epochN.pth checkpoint from that script. It saves to
its own SAVE_DIR (population_recurrent) so your existing MLP run and its
checkpoints are completely untouched.

Usage:
    cd drl_for_hanabi/algorithms
    CUDA_VISIBLE_DEVICES="" nohup python population_training_recurrent.py \
        > population_training_recurrent_log.txt 2>&1 &
    tail -f population_training_recurrent_log.txt
"""
import sys
import os

# Make the repo importable regardless of where this script is run from.
_ALGO_DIR = os.path.dirname(os.path.realpath(__file__))
_DRL_DIR = os.path.dirname(_ALGO_DIR)  # drl_for_hanabi/
_REPO_ROOT = os.path.dirname(_DRL_DIR)  # repo root, contains hanabi_learning_environment
_DECENTRALIZED_DIR = os.path.join(_DRL_DIR, "extra", "decentralized")
for _p in (_REPO_ROOT, _DECENTRALIZED_DIR):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import random
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical
from torch.optim import Adam

torch.set_num_threads(1)
torch.backends.mkldnn.enabled = False

from hanabi_learning_environment import rl_env_adjusted
from drl_for_hanabi.utils import basics, state_representation_dec, action_representation_dec
from agent_registry import build_agent  # noqa: E402  (lives in extra/decentralized/)


# ----------------------------------------------------------------------------
# Config -- tune these. Same starting point as population_training.py where
# it still applies; GRU-specific bits are new.
# ----------------------------------------------------------------------------
EPOCHS = 300000
                                # the two are directly comparable at 30k/60k/100k/200k
BATCH_SIZE = 1000              # minimum number of environment steps per epoch
OBS_VEC_TYPE = 136             # matches VPG's encoding: fireworks, own hint-knowledge, discards
NUM_ACTIONS = 11               # 0-4 discard, 5-9 play, 10 hint (same scheme as existing agents)
HIDDEN_SIZE_PI = 128           # GRU hidden size for the policy head
HIDDEN_SIZE_VF = 64            # GRU hidden size for the value head
LR_PI = 1e-3
LR_VF = 1e-3
GAMMA = 0.99
ENTROPY_COEF = 0.01             # same rationale as population_training.py: discourages
                                 # collapsing onto narrow partner-specific conventions
ILLEGAL_MOVE_PENALTY = -1.0
ILLEGAL_MOVE_MAX = 50          # safety cap: abandon episode if this many illegal picks happen
SAVE_PARAMS_EVERY = 50         # epochs between checkpoint saves (0 = only save at the very end)
SHOW_EPOCH_EVERY = 5

# hoad_piers (TF/Keras-based) is left out here: loading it crashes with a
# native "double free" heap-corruption error when TensorFlow's model loader
# and PyTorch's allocator are both active in one process -- a known flaky
# torch+TF interaction, not a bug in this script. The MLP run
# (population_training.py) already has hoad_piers in its pool, so that
# comparison point isn't lost; revisit adding it back here once the
# torch/TF conflict is resolved (e.g. isolating hoad_piers evaluation into
# its own subprocess).
PARTNER_NAMES = ["simple", "bram_simple", "bram_best", "ppo", "vpg", "spg"]

SAVE_DIR = os.path.join(_DRL_DIR, "experiments", "policy_params", "population_recurrent")
# Separate, fixed folder name -- deliberately NOT "population" so this never
# collides with or tries to resume from the MLP run's checkpoints.


def build_partner_pool():
    """Builds one instance of each partner agent, once, up front."""
    pool = {}
    for name in PARTNER_NAMES:
        try:
            pool[name] = build_agent(name)
            print(f"Partner pool: loaded '{name}'")
        except Exception as e:  # noqa: BLE001
            print(f"Partner pool: SKIPPED '{name}' (could not build: {e})")
    if not pool:
        raise RuntimeError("No partners could be built -- check PARTNER_NAMES / agent_registry.")
    return pool


class RecurrentHead(nn.Module):
    """linear-in -> GRUCell -> linear-out.

    The GRU's hidden state is the thing that carries information across an
    agent's own turns within a single episode. Nothing about the partner's
    identity is ever fed in -- only the same observation vector the MLP
    version used -- so anything the hidden state ends up encoding about
    "how this partner plays" has to be inferred purely from the sequence of
    observations, the same way a human player would pick up on a partner's
    habits over the course of a game.
    """

    def __init__(self, obs_dim, hidden_size, out_dim):
        super().__init__()
        self.hidden_size = hidden_size
        self.input_proj = nn.Linear(obs_dim, hidden_size)
        self.act = nn.Tanh()
        self.gru_cell = nn.GRUCell(hidden_size, hidden_size)
        self.out = nn.Linear(hidden_size, out_dim)

    def forward(self, obs_vec_t, h_prev):
        """obs_vec_t: (batch, obs_dim). h_prev: (batch, hidden_size)."""
        x = self.act(self.input_proj(obs_vec_t))
        h_next = self.gru_cell(x, h_prev)
        out = self.out(h_next)
        return out, h_next

    def init_hidden(self, device, batch_size=1):
        return torch.zeros(batch_size, self.hidden_size, device=device)


class PopulationAgentRecurrent:
    """The trainable learner, GRU-based policy and value networks."""

    def __init__(self):
        self.device = basics.get_default_device()
        self.obs_to_vec_func = state_representation_dec.obs_to_vec136_binary
        self.policy_net = RecurrentHead(OBS_VEC_TYPE, HIDDEN_SIZE_PI, NUM_ACTIONS)
        self.value_net = RecurrentHead(OBS_VEC_TYPE, HIDDEN_SIZE_VF, 1)
        self.policy_net.to(self.device)
        self.value_net.to(self.device)
        self.optimizer_pi = Adam(self.policy_net.parameters(), lr=LR_PI)
        self.optimizer_vf = Adam(self.value_net.parameters(), lr=LR_VF)

    def observation_to_vector(self, obs):
        return self.obs_to_vec_func(obs)

    def new_episode_hidden(self):
        """Fresh (zeroed) hidden state pair, one call at the start of every
        episode -- this is what makes sure information never leaks between
        games (the learner is not supposed to remember the PREVIOUS game's
        partner, only build up a picture of the CURRENT one)."""
        h_pi = self.policy_net.init_hidden(self.device)
        h_vf = self.value_net.init_hidden(self.device)
        return h_pi, h_vf

    def act_int_and_legal(self, obs, h_pi, h_vf, explore=True):
        """Same contract as population_training.py's version, plus it takes
        and returns the recurrent hidden states, and also returns the value
        estimate (computed live, since we can no longer cheaply recompute a
        stateful value later from a flat batch of stored observations)."""
        obs_vec = self.observation_to_vector(obs)
        obs_vec_t = torch.as_tensor(obs_vec, dtype=torch.float32, device=self.device).unsqueeze(0)

        logits, h_pi_next = self.policy_net(obs_vec_t, h_pi)
        value_pred, h_vf_next = self.value_net(obs_vec_t, h_vf)

        dist = Categorical(logits=logits.squeeze(0))
        act_int = dist.sample().item() if explore else torch.argmax(dist.probs).item()

        legals_int = obs['legal_moves_as_int']
        legals_HLE = obs['legal_moves']
        hint_possible = (obs['information_tokens'] > 0)
        legal = False
        action = None
        if act_int == 10:
            if hint_possible:
                action = action_representation_dec.get_random_hint(legals_HLE)
                legal = True
        elif act_int in legals_int:
            action = act_int
            legal = True

        act_int_t = torch.as_tensor(act_int, device=self.device)
        return (act_int, action, legal,
                dist.log_prob(act_int_t),
                dist.entropy(),
                value_pred.squeeze(),
                h_pi_next, h_vf_next)

    def save_policy(self, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            "policy_net": self.policy_net.state_dict(),
            "optimizer_pi": self.optimizer_pi.state_dict(),
            "value_net": self.value_net.state_dict(),
            "optimizer_vf": self.optimizer_vf.state_dict(),
        }, save_path)
        print(f"Saved checkpoint to {save_path}")

    def load_policy(self, load_path):
        checkpoint = torch.load(load_path, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net"])
        self.optimizer_pi.load_state_dict(checkpoint["optimizer_pi"])
        self.value_net.load_state_dict(checkpoint["value_net"])
        self.optimizer_vf.load_state_dict(checkpoint["optimizer_vf"])
        print(f"Resumed from checkpoint: {load_path}")


def find_latest_checkpoint():
    if not os.path.isdir(SAVE_DIR):
        return None, 0
    best_epoch, best_path = 0, None
    for fname in os.listdir(SAVE_DIR):
        if fname.startswith("policy_epoch") and fname.endswith(".pth"):
            try:
                epoch_num = int(fname[len("policy_epoch"):-len(".pth")])
            except ValueError:
                continue
            if epoch_num > best_epoch:
                best_epoch, best_path = epoch_num, os.path.join(SAVE_DIR, fname)
    return best_path, best_epoch


def run_one_episode(learner, partner_pool, illegal_move_penalty=ILLEGAL_MOVE_PENALTY):
    """Plays one 2-player game: learner at seat 0, a randomly sampled
    partner at seat 1. Returns training data for the learner's OWN turns
    only, plus the episode score for logging.

    The learner's hidden state (h_pi, h_vf) starts at zero for this episode
    and is threaded forward through every one of its own decisions in
    order -- that ordering is exactly what lets the GRU pick up on
    "what has this partner been doing so far".
    """
    partner_name = random.choice(list(partner_pool.keys()))
    partner = partner_pool[partner_name]

    game_config = {"colors": 5, "ranks": 5, "players": 2,
                    "max_information_tokens": 8, "max_life_tokens": 3}
    env = rl_env_adjusted.make(obs_type="CARD_KNOWLEDGE", config=game_config)
    observations = env.reset()
    done = False

    h_pi, h_vf = learner.new_episode_hidden()

    learner_log_probs, learner_entropies, learner_values = [], [], []
    all_rewards = []  # every step's reward, both seats, for team reward-to-go
    learner_step_positions = []  # index into all_rewards where a learner action was taken
    illegal_count = 0

    while not done:
        obs0 = observations["player_observations"][0]
        obs1 = observations["player_observations"][1]
        current_player = obs0["current_player"]

        if current_player == 0:
            (act_int, action, legal, log_prob, entropy, value_pred,
             h_pi, h_vf) = learner.act_int_and_legal(obs0, h_pi, h_vf)
            if legal:
                observations, rew, done, _ = env.step(action)
            else:
                illegal_count += 1
                rew = illegal_move_penalty
            learner_log_probs.append(log_prob)
            learner_entropies.append(entropy)
            learner_values.append(value_pred)
            all_rewards.append(rew)
            learner_step_positions.append(len(all_rewards) - 1)
            if illegal_count > ILLEGAL_MOVE_MAX:
                break
        else:
            action = partner.act(obs1)
            observations, rew, done, _ = env.step(action)
            all_rewards.append(rew)

    ep_lives = observations["player_observations"][0]["life_tokens"]
    ep_firework = sum(observations["player_observations"][0]["fireworks"].values())
    ep_score = ep_firework if ep_lives > 0 else 0

    rtg_full = basics.reward_to_go(all_rewards, gamma=GAMMA)
    learner_rtg = [rtg_full[i] for i in learner_step_positions]

    return (learner_log_probs, learner_entropies, learner_values,
            learner_rtg, ep_score, illegal_count, partner_name)


def train_one_epoch(learner, partner_pool, epoch_num):
    batch_log_probs, batch_entropies, batch_values, batch_rtg = [], [], [], []
    scores, illegal_counts = [], []
    steps_this_epoch = 0

    while steps_this_epoch < BATCH_SIZE:
        log_probs, entropies, values, rtg, score, illegal_count, partner_name = \
            run_one_episode(learner, partner_pool)
        batch_log_probs.extend(log_probs)
        batch_entropies.extend(entropies)
        batch_values.extend(values)
        batch_rtg.extend(rtg)
        scores.append(score)
        illegal_counts.append(illegal_count)
        steps_this_epoch += len(log_probs)

    rtg_t = torch.as_tensor(batch_rtg, dtype=torch.float32, device=learner.device)
    values_t = torch.stack(batch_values)

    # Value network: regress toward reward-to-go. Unlike the MLP version,
    # these value predictions were already computed live during rollout
    # (they had to be, to keep the GRU's hidden state in sync with the
    # policy's) -- so no separate re-forward pass over stored observations.
    loss_vf = ((values_t - rtg_t) ** 2).mean()
    learner.optimizer_vf.zero_grad()
    loss_vf.backward()
    learner.optimizer_vf.step()

    # Policy network: REINFORCE with the value estimates as a baseline
    # (detached so this loss doesn't also backprop into the value network).
    baseline = values_t.detach()
    advantage = rtg_t - baseline
    if advantage.std() > 1e-6:
        advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)
    log_probs_t = torch.stack(batch_log_probs)
    entropy_t = torch.stack(batch_entropies)
    mean_entropy = entropy_t.mean()
    loss_pi = -(log_probs_t * advantage).mean() - ENTROPY_COEF * mean_entropy
    learner.optimizer_pi.zero_grad()
    loss_pi.backward()
    learner.optimizer_pi.step()

    if epoch_num % SHOW_EPOCH_EVERY == 0:
        avg_score = sum(scores) / len(scores)
        avg_illegal = sum(illegal_counts) / len(illegal_counts)
        print(f"Epoch {epoch_num:5d} | episodes {len(scores):4d} | "
              f"avg score {avg_score:5.2f} | avg illegal/ep {avg_illegal:4.2f} | "
              f"entropy {mean_entropy.item():5.3f} | "
              f"loss_pi {loss_pi.item():7.4f} | loss_vf {loss_vf.item():7.4f}")

    return sum(scores) / len(scores)


def main():
    print(f"[RECURRENT] Training against partner pool: {PARTNER_NAMES}")
    partner_pool = build_partner_pool()
    print(f"Actually usable partners: {list(partner_pool.keys())}")

    learner = PopulationAgentRecurrent()

    resume_path, start_epoch = find_latest_checkpoint()
    if resume_path:
        learner.load_policy(resume_path)
        print(f"Resuming from epoch {start_epoch} (out of {EPOCHS} target)")
    else:
        print("No existing checkpoint found in population_recurrent/ -- starting fresh")

    for epoch in range(start_epoch + 1, EPOCHS + 1):
        train_one_epoch(learner, partner_pool, epoch)
        if SAVE_PARAMS_EVERY and epoch % SAVE_PARAMS_EVERY == 0:
            learner.save_policy(os.path.join(SAVE_DIR, f"policy_epoch{epoch}.pth"))

    final_path = os.path.join(SAVE_DIR, "policy_final.pth")
    learner.save_policy(final_path)
    print(f"Training complete. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
